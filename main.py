# main.py
"""
Snake RL pipeline

Order:
1) 10x10 EA short MLP
2) 10x10 EA short CNN
3) 10x10 MLP
4) 20x20 MLP
5) 10x10 CNN
6) 20x20 CNN
7) 20x20 EA from MLP
8) 20x20 EA from CNN

Logging:
- For each train_* run, every 10% of steps:
  [step/steps] max_len=... p90=...
"""

import os
import math
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import trange

from env_fast import TorchSnakeEnv
from models import DuelingMLP, DuelingCNN

# -------------------------
# Global config
# -------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONS = 3

STACK = 3
FEAT_DIM_SINGLE = 34
FEAT_DIM = FEAT_DIM_SINGLE * STACK

N_ENVS_DEFAULT = 8192
BATCH = 4096
REPLAY_SIZE = 300_000
WARMUP = 10_000

GAMMA = 0.99
LR = 5e-5
TARGET_UPDATE = 1000
UPDATES_PER_TRAIN = 2
N_STEP = 3

PER_ALPHA = 0.6
PER_EPS = 1e-4
BETA_START = 0.4
BETA_FRAMES = 1_000_000

TRAIN_EVERY = 4
ACCUM_STEPS = 2
BASE_SHAPING = 0.03
EPS_END = 0.05

CKPT_DIR = "saved_models"
os.makedirs(CKPT_DIR, exist_ok=True)


# -------------------------
# Utilities
# -------------------------
def make_epsilon_fn(total_train_steps, eps_start=0.9, eps_end=EPS_END, decay_frac=0.6):
    decay_steps = max(1, int(total_train_steps * float(decay_frac)))

    def eps_fn(step):
        t = min(1.0, step / decay_steps)
        return float(eps_start + (eps_end - eps_start) * t)

    return eps_fn


def flatten_params(model):
    with torch.no_grad():
        parts = []
        shapes = []
        for p in model.parameters():
            arr = p.detach().cpu().numpy().ravel()
            parts.append(arr)
            shapes.append(p.shape)
        flat = np.concatenate(parts).astype(np.float32)
    return flat, shapes


def unflatten_to_model(model, flat, shapes):
    with torch.no_grad():
        offset = 0
        for p, shape in zip(model.parameters(), shapes):
            size = int(np.prod(shape))
            chunk = flat[offset:offset + size].astype(np.float32)
            chunk_t = torch.from_numpy(chunk.reshape(shape)).to(p.device)
            p.copy_(chunk_t)
            offset += size


def sanity_step_vector(model, env, obs_stack):
    model.eval()
    with torch.no_grad():
        qvals = model(obs_stack)
        if qvals.shape[1] != ACTIONS:
            print(f"[sanity] wrong action dim: {qvals.shape}")
            return False
        act = qvals.argmax(1)
        nf, r, d = env.step(act)
        if nf.shape[0] != obs_stack.shape[0]:
            print(f"[sanity] wrong batch size from env: {nf.shape[0]}")
            return False
    return True


def sanity_step_cnn(model, env):
    model.eval()
    with torch.no_grad():
        grid = env.grid_observation().to(DEVICE)
        qvals = model(grid)
        if qvals.shape[1] != ACTIONS:
            print(f"[sanity CNN] wrong action dim: {qvals.shape}")
            return False
        act = qvals.argmax(1)
        nf, r, d = env.step(act)
        if nf.shape[0] != grid.shape[0]:
            print(f"[sanity CNN] wrong batch size from env: {nf.shape[0]}")
            return False
    return True


# -------------------------
# PER SumTree
# -------------------------
class SumTree:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.tree_size = 1
        while self.tree_size < self.capacity:
            self.tree_size *= 2
        self.tree = np.zeros(2 * self.tree_size, dtype=np.float32)
        self.size = 0

    def _propagate(self, idx, change):
        parent = idx // 2
        while parent >= 1:
            self.tree[parent] += change
            parent //= 2

    def update(self, idx, priority):
        tree_idx = idx + self.tree_size
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def add(self, idx, priority):
        self.update(idx, priority)
        if self.size < self.capacity:
            self.size += 1

    def total(self):
        return float(self.tree[1])

    def find_prefixsum_idx(self, s):
        idx = 1
        while idx < self.tree_size:
            left = 2 * idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx - self.tree_size


class PrioritizedReplaySumTree:
    def __init__(self, capacity, feat_dim, device=DEVICE, alpha=PER_ALPHA, eps=PER_EPS):
        self.capacity = int(capacity)
        self.device = device
        self.alpha = float(alpha)
        self.eps = float(eps)

        self.f = torch.zeros((self.capacity, feat_dim), dtype=torch.float16, device=device)
        self.nf = torch.zeros_like(self.f)
        self.a = torch.zeros(self.capacity, dtype=torch.long, device=device)
        self.r = torch.zeros(self.capacity, dtype=torch.float32, device=device)
        self.d = torch.zeros(self.capacity, dtype=torch.float32, device=device)

        self.sumtree = SumTree(self.capacity)
        self.next_idx = 0
        self.full = False

    def size(self):
        return self.capacity if self.full else self.next_idx

    def add_batch(self, f_batch, a_batch, r_batch, nf_batch, d_batch, priorities=None):
        b = f_batch.size(0)
        idxs = (torch.arange(b, device=self.device) + self.next_idx) % self.capacity
        self.f[idxs] = f_batch.to(torch.float16)
        self.nf[idxs] = nf_batch.to(torch.float16)
        self.a[idxs] = a_batch
        self.r[idxs] = r_batch
        self.d[idxs] = d_batch.float()

        if priorities is None:
            current_size = self.size()
            if current_size > 0:
                leaves = self.sumtree.tree[self.sumtree.tree_size:self.sumtree.tree_size + current_size]
                max_p = float(np.max(leaves)) if leaves.size > 0 else 1.0
                if max_p <= 0:
                    max_p = 1.0
            else:
                max_p = 1.0
            priorities = np.full((b,), max_p, dtype=np.float32)
        else:
            priorities = priorities.astype(np.float32)

        priorities = np.maximum(priorities, 1e-12).astype(np.float32)
        for i, p in enumerate(priorities):
            data_idx = (self.next_idx + i) % self.capacity
            self.sumtree.add(data_idx, float((abs(p) + self.eps) ** self.alpha))
        self.next_idx = (self.next_idx + b) % self.capacity
        if self.next_idx == 0:
            self.full = True

    def sample_batch(self, batch_size, beta=1.0):
        m = self.size()
        if m == 0:
            raise RuntimeError("Sampling from empty replay")

        total = self.sumtree.total()

        if total <= 0 or not np.isfinite(total):
            idxs = np.random.randint(0, m, size=(batch_size,), dtype=np.int64)
            priorities = np.ones(batch_size, dtype=np.float32)
        else:
            segment = total / batch_size
            idxs = []
            priorities = []
            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)
                s = np.random.uniform(a, b)
                data_idx = self.sumtree.find_prefixsum_idx(s)
                if data_idx < 0 or data_idx >= self.capacity:
                    data_idx = int(data_idx % m)
                else:
                    if data_idx >= m:
                        data_idx = int(data_idx % m)
                idxs.append(int(data_idx))
                priorities.append(float(self.sumtree.tree[self.sumtree.tree_size + data_idx]))
            idxs = np.array(idxs, dtype=np.int64)
            priorities = np.array(priorities, dtype=np.float32)

        probs = priorities / (total + 1e-12) if total > 0 else (np.ones_like(priorities, dtype=np.float32) / float(m))
        probs = np.maximum(probs, 1e-12)

        weights = (m * probs) ** (-beta)
        max_w = np.max(weights)
        if not np.isfinite(max_w) or max_w <= 0:
            max_w = 1.0
        weights = weights / (max_w + 1e-12)

        idxs_t = torch.from_numpy(idxs).to(torch.long).to(self.device)

        f_b = self.f[idxs_t].float()
        a_b = self.a[idxs_t]
        r_b = self.r[idxs_t]
        nf_b = self.nf[idxs_t].float()
        d_b = self.d[idxs_t]

        weights_t = torch.from_numpy(weights.astype(np.float32)).float().to(self.device)

        return f_b, a_b, r_b, nf_b, d_b, idxs, weights_t

    def update_priorities(self, idxs, td_errors):
        if isinstance(td_errors, torch.Tensor):
            td = td_errors.detach().cpu().numpy().astype(np.float32)
        else:
            td = np.asarray(td_errors, dtype=np.float32)
        for i, data_idx in enumerate(idxs):
            p = float((abs(td[i]) + self.eps) ** self.alpha)
            p = max(p, 1e-12)
            self.sumtree.update(int(data_idx), p)


# -------------------------
# MLP training (vector)
# -------------------------
def train_mlp(
    grid,
    steps,
    ckpt_path,
    load=None,
    n_envs=N_ENVS_DEFAULT,
    max_steps_scale=50,
    success_length_threshold=None,
    eps_fn=None,
    random_start=True,
    seed=None,
):
    print(f"\n=== TRAIN MLP GRID {grid} (n_envs={n_envs}) ===")
    print(f"≈ {(steps * n_envs) / 1e6:.1f}M env steps")

    max_steps = grid * grid * max(1, int(max_steps_scale))

    env = TorchSnakeEnv(
        n=n_envs,
        g=grid,
        max_steps=max_steps,
        device=DEVICE,
        shaping_scale=BASE_SHAPING * (grid / 10),
        eat_reward=1.5,
        length_reward_scale=0.12,
        no_eat_limit=400,
        success_length_threshold=success_length_threshold,
        random_start=random_start,
        seed=seed,
    )

    obs_single = env.reset().to(DEVICE)
    obs_stack = deque(maxlen=STACK)
    for _ in range(STACK):
        obs_stack.append(obs_single.clone())

    def get_stacked():
        return torch.cat(list(obs_stack), dim=1)

    obs_f = get_stacked().to(DEVICE).float()

    q = DuelingMLP(FEAT_DIM).to(DEVICE)
    tgt = DuelingMLP(FEAT_DIM).to(DEVICE)

    if load is not None and os.path.exists(load):
        ckpt = torch.load(load, map_location=DEVICE)
        if "q" in ckpt:
            q.load_state_dict(ckpt["q"])
        else:
            q.load_state_dict(ckpt)

    tgt.load_state_dict(q.state_dict())

    if not sanity_step_vector(q, env, obs_f):
        raise RuntimeError("MLP sanity check failed")

    opt = torch.optim.Adam(q.parameters(), lr=LR)
    scaler = GradScaler()

    replay = PrioritizedReplaySumTree(REPLAY_SIZE, FEAT_DIM, device=DEVICE, alpha=PER_ALPHA, eps=PER_EPS)

    reward_buf = torch.zeros((n_envs, N_STEP), dtype=torch.float32, device=DEVICE)
    done_buf = torch.zeros((n_envs, N_STEP), dtype=torch.float32, device=DEVICE)
    next_obs_buf = torch.zeros((n_envs, N_STEP, FEAT_DIM_SINGLE), dtype=torch.float32, device=DEVICE)
    first_obs_buf = torch.zeros((n_envs, FEAT_DIM), dtype=torch.float32, device=DEVICE)
    action_buf = torch.zeros((n_envs, N_STEP), dtype=torch.long, device=DEVICE)
    steps_since_reset = torch.zeros(n_envs, dtype=torch.long, device=DEVICE)
    prev_done = torch.zeros(n_envs, dtype=torch.bool, device=DEVICE)

    def beta_by_frame(frame_idx):
        return min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / max(1, BETA_FRAMES))

    def run_training_step(global_step):
        if replay.size() <= WARMUP:
            return None
        beta = beta_by_frame(global_step)
        mini_batch = BATCH // ACCUM_STEPS
        opt.zero_grad(set_to_none=True)
        all_idxs = []
        all_td = []
        for _ in range(ACCUM_STEPS):
            f_b, a_b, r_b, nf_b, done_b, idxs, is_weights = replay.sample_batch(mini_batch, beta=beta)
            with torch.no_grad(), autocast():
                na = q(nf_b).argmax(1)
                tq = tgt(nf_b).gather(1, na[:, None]).squeeze()
                y = r_b + (GAMMA ** N_STEP) * (1 - done_b) * tq
            with autocast():
                qv = q(f_b).gather(1, a_b[:, None]).squeeze()
                td = (qv - y).detach()
                loss = (is_weights * F.smooth_l1_loss(qv, y, reduction="none")).mean()
            all_td.append(td.abs().cpu().numpy())
            scaler.scale(loss / ACCUM_STEPS).backward()
            all_idxs.append(idxs)
        scaler.step(opt)
        scaler.update()
        flat_idxs = np.concatenate(all_idxs, axis=0)
        flat_td = np.concatenate(all_td, axis=0)
        replay.update_priorities(flat_idxs, torch.from_numpy(flat_td).to(DEVICE))
        return flat_td

    if eps_fn is None:
        eps_fn = make_epsilon_fn(steps)

    log_interval = max(1, steps // 10)

    pbar = trange(steps)
    for step in pbar:
        eps = eps_fn(step)
        obs_stacked = get_stacked().to(DEVICE).float()

        with torch.no_grad(), autocast():
            qvals = q(obs_stacked)
        act = qvals.argmax(1)
        rnd = torch.rand(n_envs, device=DEVICE) < eps
        if rnd.any():
            act[rnd] = torch.randint(0, ACTIONS, (rnd.sum().item(),), device=DEVICE)

        nf_single, r, d = env.step(act)
        nf_single = nf_single.to(DEVICE)
        r = r.to(DEVICE)
        d = d.to(DEVICE)

        if N_STEP > 1:
            reward_buf[:, :-1] = reward_buf[:, 1:]
            done_buf[:, :-1] = done_buf[:, 1:]
            next_obs_buf[:, :-1, :] = next_obs_buf[:, 1:, :]
            action_buf[:, :-1] = action_buf[:, 1:]
        reward_buf[:, -1] = r
        done_buf[:, -1] = d.float()
        next_obs_buf[:, -1, :] = nf_single
        action_buf[:, -1] = act

        mask_new = (steps_since_reset == 0)
        if mask_new.any():
            first_obs_buf[mask_new] = obs_stacked[mask_new]

        steps_since_reset = steps_since_reset + (~d).long()
        steps_since_reset[d] = 0

        ready_mask = (steps_since_reset >= N_STEP)
        if ready_mask.any():
            idxs_ready = torch.where(ready_mask)[0]
            rewards_k = reward_buf[idxs_ready]
            dones_k = done_buf[idxs_ready]

            discounts = torch.tensor([GAMMA ** i for i in range(N_STEP)], device=DEVICE).view(1, N_STEP)
            rewards_rev = torch.flip(rewards_k, dims=[1])
            dones_rev = torch.flip(dones_k, dims=[1])
            cont = 1.0 - dones_rev
            cumprod = torch.cumprod(cont, dim=1)
            prod_excl = torch.ones_like(cumprod)
            prod_excl[:, 1:] = cumprod[:, :-1]
            disc = discounts.to(DEVICE)
            ret = (rewards_rev * disc * prod_excl).sum(dim=1)

            next_state = next_obs_buf[idxs_ready, 0, :]
            done_n = dones_rev.any(dim=1).float()
            s0 = first_obs_buf[idxs_ready]

            if STACK > 1:
                last_frames = list(obs_stack)[-(STACK - 1):]
                last_frames_k = [fr[idxs_ready] for fr in last_frames]
                next_stack = torch.cat(last_frames_k + [next_state], dim=1)
            else:
                next_stack = next_state

            a_batch = action_buf[idxs_ready, 0].to(DEVICE)
            f_batch = s0.to(DEVICE)
            r_batch = ret.to(DEVICE)
            nf_batch = next_stack.to(DEVICE)
            d_batch = done_n.to(DEVICE)

            replay.add_batch(f_batch, a_batch, r_batch, nf_batch, d_batch)

            first_obs_buf[idxs_ready] = get_stacked()[idxs_ready]
            reward_buf[idxs_ready] = 0.0
            done_buf[idxs_ready] = 0.0
            next_obs_buf[idxs_ready] = 0.0
            action_buf[idxs_ready] = 0
            steps_since_reset[idxs_ready] = 0

        if (step % TRAIN_EVERY == 0) and (replay.size() > WARMUP):
            for _ in range(UPDATES_PER_TRAIN):
                run_training_step(step)

        just_done = d & (~prev_done)
        if just_done.any():
            mask = torch.zeros(n_envs, dtype=torch.bool, device=DEVICE)
            mask[torch.where(just_done)[0]] = True
            obs_after_reset = env.reset(mask=mask).to(DEVICE)
            frames = list(obs_stack)
            for fr in frames:
                fr[mask] = obs_after_reset[mask].clone()
            nf_single[mask] = obs_after_reset[mask].clone()
            steps_since_reset[mask] = 0
            first_obs_buf[mask] = get_stacked()[mask]

        prev_done = d.clone()
        obs_stack.append(nf_single.clone())

        if step % TARGET_UPDATE == 0:
            tgt.load_state_dict(q.state_dict())

        if step > 0 and step % log_interval == 0:
            lengths = env.length.cpu().numpy()
            max_len = int(lengths.max())
            p90 = float(np.percentile(lengths, 90))
            print(f"[MLP {grid}x{grid}] [{step}/{steps}] max_len={max_len} p90={p90:.2f}")

    torch.save({"q": q.state_dict()}, ckpt_path)
    print(f"✅ Saved MLP model to {ckpt_path}")
    return ckpt_path


# -------------------------
# CNN training (grid)
# -------------------------
def train_cnn(
    grid,
    steps,
    ckpt_path,
    load=None,
    n_envs=N_ENVS_DEFAULT,
    max_steps_scale=50,
    success_length_threshold=None,
    eps_fn=None,
    random_start=True,
    seed=None,
):
    print(f"\n=== TRAIN CNN GRID {grid} (n_envs={n_envs}) ===")
    print(f"≈ {(steps * n_envs) / 1e6:.1f}M env steps")

    max_steps = grid * grid * max(1, int(max_steps_scale))

    env = TorchSnakeEnv(
        n=n_envs,
        g=grid,
        max_steps=max_steps,
        device=DEVICE,
        shaping_scale=BASE_SHAPING * (grid / 10),
        eat_reward=1.5,
        length_reward_scale=0.12,
        no_eat_limit=400,
        success_length_threshold=success_length_threshold,
        random_start=random_start,
        seed=seed,
    )

    env.reset().to(DEVICE)

    q = DuelingCNN(in_channels=3, grid_size=grid).to(DEVICE)
    tgt = DuelingCNN(in_channels=3, grid_size=grid).to(DEVICE)

    if load is not None and os.path.exists(load):
        ckpt = torch.load(load, map_location=DEVICE)
        if "q" in ckpt:
            q.load_state_dict(ckpt["q"])
        else:
            q.load_state_dict(ckpt)

    tgt.load_state_dict(q.state_dict())

    if not sanity_step_cnn(q, env):
        raise RuntimeError("CNN sanity check failed")

    opt = torch.optim.Adam(q.parameters(), lr=LR)
    scaler = GradScaler()

    feat_dim = 3 * grid * grid
    replay = PrioritizedReplaySumTree(REPLAY_SIZE, feat_dim, device=DEVICE, alpha=PER_ALPHA, eps=PER_EPS)

    reward_buf = torch.zeros((n_envs, N_STEP), dtype=torch.float32, device=DEVICE)
    done_buf = torch.zeros((n_envs, N_STEP), dtype=torch.float32, device=DEVICE)
    next_obs_buf = torch.zeros((n_envs, N_STEP, 3, grid, grid), dtype=torch.float32, device=DEVICE)
    first_obs_buf = torch.zeros((n_envs, 3, grid, grid), dtype=torch.float32, device=DEVICE)
    action_buf = torch.zeros((n_envs, N_STEP), dtype=torch.long, device=DEVICE)
    steps_since_reset = torch.zeros(n_envs, dtype=torch.long, device=DEVICE)
    prev_done = torch.zeros(n_envs, dtype=torch.bool, device=DEVICE)

    def beta_by_frame(frame_idx):
        return min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / max(1, BETA_FRAMES))

    def run_training_step(global_step):
        if replay.size() <= WARMUP:
            return None
        beta = beta_by_frame(global_step)
        mini_batch = BATCH // ACCUM_STEPS
        opt.zero_grad(set_to_none=True)
        all_idxs = []
        all_td = []
        for _ in range(ACCUM_STEPS):
            f_b, a_b, r_b, nf_b, done_b, idxs, is_weights = replay.sample_batch(mini_batch, beta=beta)
            f_b = f_b.view(-1, 3, grid, grid)
            nf_b = nf_b.view(-1, 3, grid, grid)
            with torch.no_grad(), autocast():
                na = q(nf_b).argmax(1)
                tq = tgt(nf_b).gather(1, na[:, None]).squeeze()
                y = r_b + (GAMMA ** N_STEP) * (1 - done_b) * tq
            with autocast():
                qv = q(f_b).gather(1, a_b[:, None]).squeeze()
                td = (qv - y).detach()
                loss = (is_weights * F.smooth_l1_loss(qv, y, reduction="none")).mean()
            all_td.append(td.abs().cpu().numpy())
            scaler.scale(loss / ACCUM_STEPS).backward()
            all_idxs.append(idxs)
        scaler.step(opt)
        scaler.update()
        flat_idxs = np.concatenate(all_idxs, axis=0)
        flat_td = np.concatenate(all_td, axis=0)
        replay.update_priorities(flat_idxs, torch.from_numpy(flat_td).to(DEVICE))
        return flat_td

    if eps_fn is None:
        eps_fn = make_epsilon_fn(steps)

    log_interval = max(1, steps // 10)

    pbar = trange(steps)
    for step in pbar:
        eps = eps_fn(step)
        grid_obs = env.grid_observation().to(DEVICE)

        with torch.no_grad(), autocast():
            qvals = q(grid_obs)
        act = qvals.argmax(1)
        rnd = torch.rand(n_envs, device=DEVICE) < eps
        if rnd.any():
            act[rnd] = torch.randint(0, ACTIONS, (rnd.sum().item(),), device=DEVICE)

        nf, r, d = env.step(act)
        nf = nf.to(DEVICE)
        r = r.to(DEVICE)
        d = d.to(DEVICE)
        nf_grid = env.grid_observation().to(DEVICE)

        if N_STEP > 1:
            reward_buf[:, :-1] = reward_buf[:, 1:]
            done_buf[:, :-1] = done_buf[:, 1:]
            next_obs_buf[:, :-1, :, :, :] = next_obs_buf[:, 1:, :, :, :]
            action_buf[:, :-1] = action_buf[:, 1:]
        reward_buf[:, -1] = r
        done_buf[:, -1] = d.float()
        next_obs_buf[:, -1, :, :, :] = nf_grid
        action_buf[:, -1] = act

        mask_new = (steps_since_reset == 0)
        if mask_new.any():
            first_obs_buf[mask_new] = grid_obs[mask_new]

        steps_since_reset = steps_since_reset + (~d).long()
        steps_since_reset[d] = 0

        ready_mask = (steps_since_reset >= N_STEP)
        if ready_mask.any():
            idxs_ready = torch.where(ready_mask)[0]
            rewards_k = reward_buf[idxs_ready]
            dones_k = done_buf[idxs_ready]

            discounts = torch.tensor([GAMMA ** i for i in range(N_STEP)], device=DEVICE).view(1, N_STEP)
            rewards_rev = torch.flip(rewards_k, dims=[1])
            dones_rev = torch.flip(dones_k, dims=[1])
            cont = 1.0 - dones_rev
            cumprod = torch.cumprod(cont, dim=1)
            prod_excl = torch.ones_like(cumprod)
            prod_excl[:, 1:] = cumprod[:, :-1]
            disc = discounts.to(DEVICE)
            ret = (rewards_rev * disc * prod_excl).sum(dim=1)

            next_state = next_obs_buf[idxs_ready, 0, :, :, :]
            done_n = dones_rev.any(dim=1).float()
            s0 = first_obs_buf[idxs_ready]

            a_batch = action_buf[idxs_ready, 0].to(DEVICE)
            f_batch = s0.to(DEVICE).view(-1, 3 * grid * grid)
            r_batch = ret.to(DEVICE)
            nf_batch = next_state.to(DEVICE).view(-1, 3 * grid * grid)
            d_batch = done_n.to(DEVICE)

            replay.add_batch(f_batch, a_batch, r_batch, nf_batch, d_batch)

            first_obs_buf[idxs_ready] = env.grid_observation().to(DEVICE)[idxs_ready]
            reward_buf[idxs_ready] = 0.0
            done_buf[idxs_ready] = 0.0
            next_obs_buf[idxs_ready] = 0.0
            action_buf[idxs_ready] = 0
            steps_since_reset[idxs_ready] = 0

        if (step % TRAIN_EVERY == 0) and (replay.size() > WARMUP):
            for _ in range(UPDATES_PER_TRAIN):
                run_training_step(step)

        just_done = d & (~prev_done)
        if just_done.any():
            mask = torch.zeros(n_envs, dtype=torch.bool, device=DEVICE)
            mask[torch.where(just_done)[0]] = True
            obs_after_reset = env.reset(mask=mask).to(DEVICE)
            nf[mask] = obs_after_reset[mask].clone()
            steps_since_reset[mask] = 0
            first_obs_buf[mask] = env.grid_observation().to(DEVICE)[mask]

        prev_done = d.clone()

        if step % TARGET_UPDATE == 0:
            tgt.load_state_dict(q.state_dict())

        if step > 0 and step % log_interval == 0:
            lengths = env.length.cpu().numpy()
            max_len = int(lengths.max())
            p90 = float(np.percentile(lengths, 90))
            print(f"[CNN {grid}x{grid}] [{step}/{steps}] max_len={max_len} p90={p90:.2f}")

    torch.save({"q": q.state_dict()}, ckpt_path)
    print(f"✅ Saved CNN model to {ckpt_path}")
    return ckpt_path


# -------------------------
# EA helpers (MLP)
# -------------------------
def evaluate_mlp_individual(flat_weights, shapes, episodes, device, grid, batch_size, random_start=True):
    model = DuelingMLP(FEAT_DIM).to(device)
    unflatten_to_model(model, flat_weights, shapes)
    model.eval()

    max_steps = grid * grid * 50
    env = TorchSnakeEnv(n=episodes, g=grid, max_steps=max_steps, device=device, random_start=random_start)
    obs = env.reset().to(device)

    stack = deque(maxlen=STACK)
    for _ in range(STACK):
        stack.append(obs.clone())

    def get_stack():
        return torch.cat(list(stack), dim=1)

    done = torch.zeros(episodes, dtype=torch.bool, device=device)
    returns = torch.zeros(episodes, device=device)

    while not done.all():
        obs_f = get_stack().float()
        with torch.no_grad():
            qvals = model(obs_f)
        acts = qvals.argmax(1)
        nf, r, d = env.step(acts)
        returns += r
        done |= d
        stack.append(nf.to(device))

    lengths = env.length.cpu().numpy().astype(np.int32)
    mean_return = float(returns.mean().cpu().numpy())
    return mean_return, lengths


def run_ea_short_10x10_mlp(population=64, eval_episodes=16, generations=3, sigma=1e-2, elite_frac=0.1, grid=10):
    print(f"\n=== EA SHORT 10x10 MLP: pop={population} eval_eps={eval_episodes} gens={generations} ===")
    template = DuelingMLP(FEAT_DIM).to(DEVICE)
    base_flat, shapes = flatten_params(template)
    dim = base_flat.size

    pop = np.random.randn(population, dim).astype(np.float32) * sigma

    for gen in range(generations):
        print(f"\n-- EA MLP gen {gen + 1}/{generations}")
        fitness = np.zeros(population, dtype=np.float32)
        lengths_all = []
        for i in range(population):
            f, lengths = evaluate_mlp_individual(pop[i], shapes, eval_episodes, DEVICE, grid, BATCH)
            fitness[i] = f
            lengths_all.append(lengths)
        idx_sorted = np.argsort(-fitness)
        best_idx = int(idx_sorted[0])
        best_f = float(fitness[best_idx])
        mean_f = float(np.mean(fitness))
        all_lengths = np.concatenate(lengths_all, axis=0)
        p50 = float(np.median(all_lengths))
        p90 = float(np.percentile(all_lengths, 90))
        max_len = int(all_lengths.max())
        print(f"Gen {gen + 1}: mean_f={mean_f:.4f} best_f={best_f:.4f} median_len={p50:.2f} p90={p90:.2f} max_len={max_len}")

        n_elite = max(1, int(math.ceil(population * elite_frac)))
        elites = pop[idx_sorted[:n_elite]].copy()
        offspring = []
        while len(offspring) < population - n_elite:
            a, b, c = np.random.choice(population, 3, replace=False)
            parent = [a, b, c][np.argmax(fitness[[a, b, c]])]
            child = pop[parent].copy()
            child += np.random.randn(dim).astype(np.float32) * sigma
            offspring.append(child)
        pop = np.vstack([elites] + offspring)[:population]
        sigma *= 0.99

    fitness = np.zeros(population, dtype=np.float32)
    for i in range(population):
        f, _ = evaluate_mlp_individual(pop[i], shapes, eval_episodes, DEVICE, grid, BATCH)
        fitness[i] = f
    best_idx = int(np.argmax(fitness))
    best_flat = pop[best_idx]
    best_model = DuelingMLP(FEAT_DIM).to(DEVICE)
    unflatten_to_model(best_model, best_flat, shapes)
    path = os.path.join(CKPT_DIR, "evo_10x10_mlp.pt")
    torch.save({"q": best_model.state_dict()}, path)
    print(f"✅ Saved EA short 10x10 MLP model to {path}")
    return path


# -------------------------
# EA helpers (CNN)
# -------------------------
def evaluate_cnn_individual(flat_weights, shapes, episodes, device, grid, batch_size, random_start=True):
    model = DuelingCNN(in_channels=3, grid_size=grid).to(device)
    unflatten_to_model(model, flat_weights, shapes)
    model.eval()

    max_steps = grid * grid * 50
    env = TorchSnakeEnv(n=episodes, g=grid, max_steps=max_steps, device=device, random_start=random_start)
    env.reset().to(device)

    done = torch.zeros(episodes, dtype=torch.bool, device=device)
    returns = torch.zeros(episodes, device=device)

    while not done.all():
        grid_obs = env.grid_observation().to(device)
        with torch.no_grad():
            qvals = model(grid_obs)
        acts = qvals.argmax(1)
        nf, r, d = env.step(acts)
        returns += r
        done |= d

    lengths = env.length.cpu().numpy().astype(np.int32)
    mean_return = float(returns.mean().cpu().numpy())
    return mean_return, lengths


def run_ea_short_10x10_cnn(population=64, eval_episodes=16, generations=3, sigma=1e-2, elite_frac=0.1, grid=10):
    print(f"\n=== EA SHORT 10x10 CNN: pop={population} eval_eps={eval_episodes} gens={generations} ===")
    template = DuelingCNN(in_channels=3, grid_size=grid).to(DEVICE)
    base_flat, shapes = flatten_params(template)
    dim = base_flat.size

    pop = np.random.randn(population, dim).astype(np.float32) * sigma

    for gen in range(generations):
        print(f"\n-- EA CNN gen {gen + 1}/{generations}")
        fitness = np.zeros(population, dtype=np.float32)
        lengths_all = []
        for i in range(population):
            f, lengths = evaluate_cnn_individual(pop[i], shapes, eval_episodes, DEVICE, grid, BATCH)
            fitness[i] = f
            lengths_all.append(lengths)
        idx_sorted = np.argsort(-fitness)
        best_idx = int(idx_sorted[0])
        best_f = float(fitness[best_idx])
        mean_f = float(np.mean(fitness))
        all_lengths = np.concatenate(lengths_all, axis=0)
        p90 = float(np.percentile(all_lengths, 90))
        max_len = int(all_lengths.max())
        print(f"Gen {gen + 1}: mean_f={mean_f:.4f} best_f={best_f:.4f} p90_len={p90:.2f} max_len={max_len}")

        n_elite = max(1, int(math.ceil(population * elite_frac)))
        elites = pop[idx_sorted[:n_elite]].copy()
        offspring = []
        while len(offspring) < population - n_elite:
            a, b, c = np.random.choice(population, 3, replace=False)
            parent = [a, b, c][np.argmax(fitness[[a, b, c]])]
            child = pop[parent].copy()
            child += np.random.randn(dim).astype(np.float32) * sigma
            offspring.append(child)
        pop = np.vstack([elites] + offspring)[:population]
        sigma *= 0.99

    fitness = np.zeros(population, dtype=np.float32)
    for i in range(population):
        f, _ = evaluate_cnn_individual(pop[i], shapes, eval_episodes, DEVICE, grid, BATCH)
        fitness[i] = f
    best_idx = int(np.argmax(fitness))
    best_flat = pop[best_idx]
    best_model = DuelingCNN(in_channels=3, grid_size=grid).to(DEVICE)
    unflatten_to_model(best_model, best_flat, shapes)
    path = os.path.join(CKPT_DIR, "evo_10x10_cnn.pt")
    torch.save({"q": best_model.state_dict()}, path)
    print(f"✅ Saved EA short 10x10 CNN model to {path}")
    return path


# -------------------------
# EA from trained models (20x20)
# -------------------------
def run_ea_20x20_from_mlp(base_ckpt, population=64, eval_episodes=16, generations=3, sigma=5e-3, elite_frac=0.1, grid=20):
    print(f"\n=== EA 20x20 FROM MLP: base={base_ckpt} ===")
    base_model = DuelingMLP(FEAT_DIM).to(DEVICE)
    ckpt = torch.load(base_ckpt, map_location=DEVICE)
    if "q" in ckpt:
        base_model.load_state_dict(ckpt["q"])
    else:
        base_model.load_state_dict(ckpt)
    base_flat, shapes = flatten_params(base_model)
    dim = base_flat.size

    pop = np.tile(base_flat, (population, 1))
    pop += np.random.randn(population, dim).astype(np.float32) * sigma

    for gen in range(generations):
        print(f"\n-- EA 20x20 MLP gen {gen + 1}/{generations}")
        fitness = np.zeros(population, dtype=np.float32)
        for i in range(population):
            f, _ = evaluate_mlp_individual(pop[i], shapes, eval_episodes, DEVICE, grid, BATCH)
            fitness[i] = f
        idx_sorted = np.argsort(-fitness)
        best_idx = int(idx_sorted[0])
        best_f = float(fitness[best_idx])
        mean_f = float(np.mean(fitness))
        print(f"Gen {gen + 1}: mean_f={mean_f:.4f} best_f={best_f:.4f}")

        n_elite = max(1, int(math.ceil(population * elite_frac)))
        elites = pop[idx_sorted[:n_elite]].copy()
        offspring = []
        while len(offspring) < population - n_elite:
            a, b, c = np.random.choice(population, 3, replace=False)
            parent = [a, b, c][np.argmax(fitness[[a, b, c]])]
            child = pop[parent].copy()
            child += np.random.randn(dim).astype(np.float32) * sigma
            offspring.append(child)
        pop = np.vstack([elites] + offspring)[:population]
        sigma *= 0.99

    fitness = np.zeros(population, dtype=np.float32)
    for i in range(population):
        f, _ = evaluate_mlp_individual(pop[i], shapes, eval_episodes, DEVICE, grid, BATCH)
        fitness[i] = f
    best_idx = int(np.argmax(fitness))
    best_flat = pop[best_idx]
    best_model = DuelingMLP(FEAT_DIM).to(DEVICE)
    unflatten_to_model(best_model, best_flat, shapes)
    path = os.path.join(CKPT_DIR, "evo_20x20_from_mlp.pt")
    torch.save({"q": best_model.state_dict()}, path)
    print(f"✅ Saved EA 20x20 from MLP model to {path}")
    return path


def run_ea_20x20_from_cnn(base_ckpt, population=64, eval_episodes=16, generations=3, sigma=5e-3, elite_frac=0.1, grid=20):
    print(f"\n=== EA 20x20 FROM CNN: base={base_ckpt} ===")
    base_model = DuelingCNN(in_channels=3, grid_size=grid).to(DEVICE)
    ckpt = torch.load(base_ckpt, map_location=DEVICE)
    if "q" in ckpt:
        base_model.load_state_dict(ckpt["q"])
    else:
        base_model.load_state_dict(ckpt)
    base_flat, shapes = flatten_params(base_model)
    dim = base_flat.size

    pop = np.tile(base_flat, (population, 1))
    pop += np.random.randn(population, dim).astype(np.float32) * sigma

    for gen in range(generations):
        print(f"\n-- EA 20x20 CNN gen {gen + 1}/{generations}")
        fitness = np.zeros(population, dtype=np.float32)
        for i in range(population):
            f, _ = evaluate_cnn_individual(pop[i], shapes, eval_episodes, DEVICE, grid, BATCH)
            fitness[i] = f
        idx_sorted = np.argsort(-fitness)
        best_idx = int(idx_sorted[0])
        best_f = float(fitness[best_idx])
        mean_f = float(np.mean(fitness))
        print(f"Gen {gen + 1}: mean_f={mean_f:.4f} best_f={best_f:.4f}")

        n_elite = max(1, int(math.ceil(population * elite_frac)))
        elites = pop[idx_sorted[:n_elite]].copy()
        offspring = []
        while len(offspring) < population - n_elite:
            a, b, c = np.random.choice(population, 3, replace=False)
            parent = [a, b, c][np.argmax(fitness[[a, b, c]])]
            child = pop[parent].copy()
            child += np.random.randn(dim).astype(np.float32) * sigma
            offspring.append(child)
        pop = np.vstack([elites] + offspring)[:population]
        sigma *= 0.99

    fitness = np.zeros(population, dtype=np.float32)
    for i in range(population):
        f, _ = evaluate_cnn_individual(pop[i], shapes, eval_episodes, DEVICE, grid, BATCH)
        fitness[i] = f
    best_idx = int(np.argmax(fitness))
    best_flat = pop[best_idx]
    best_model = DuelingCNN(in_channels=3, grid_size=grid).to(DEVICE)
    unflatten_to_model(best_model, best_flat, shapes)
    path = os.path.join(CKPT_DIR, "evo_20x20_from_cnn.pt")
    torch.save({"q": best_model.state_dict()}, path)
    print(f"✅ Saved EA 20x20 from CNN model to {path}")
    return path


# -------------------------
# For app.py: QNet + valid_action_mask
# -------------------------
class QNet(DuelingCNN):
    def __init__(self, grid_size=20):
        super().__init__(in_channels=3, grid_size=grid_size)


def valid_action_mask(env):
    # all actions valid in this env
    return torch.ones(1, ACTIONS, dtype=torch.bool, device=env.device)


# -------------------------
# Orchestrator
# -------------------------
def main():
    # 1) 10x10 evolutionary short test (MLP)
    evo_10_mlp = run_ea_short_10x10_mlp()

    # 2) 10x10 evolutionary short test (CNN)
    evo_10_cnn = run_ea_short_10x10_cnn()

    # 3) 10x10 MLP
    mlp_10 = train_mlp(
        grid=10,
        steps=20_000,
        ckpt_path=os.path.join(CKPT_DIR, "mlp_10x10.pt"),
        load=evo_10_mlp,
        n_envs=2048,
    )

    # 4) 20x20 MLP
    mlp_20 = train_mlp(
        grid=20,
        steps=80_000,
        ckpt_path=os.path.join(CKPT_DIR, "mlp_20x20.pt"),
        load=mlp_10,
        n_envs=4096,
    )

    # 5) 10x10 CNN
    cnn_10 = train_cnn(
        grid=10,
        steps=20_000,
        ckpt_path=os.path.join(CKPT_DIR, "cnn_10x10.pt"),
        load=evo_10_cnn,
        n_envs=2048,
    )

    # 6) 20x20 CNN
    cnn_20 = train_cnn(
        grid=20,
        steps=80_000,
        ckpt_path=os.path.join(CKPT_DIR, "cnn_20x20.pt"),
        load=cnn_10,
        n_envs=4096,
    )

    # 7) 20x20 evolutionary from MLP
    run_ea_20x20_from_mlp(base_ckpt=mlp_20)

    # 8) 20x20 evolutionary from CNN
    run_ea_20x20_from_cnn(base_ckpt=cnn_20)

    print("\nAll runs complete.")


if __name__ == "__main__":
    main()
