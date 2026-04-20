# main.py
"""
Hands-off auto-benchmark and train runner for Snake RL (MLP and CNN).

Behavior (fully automatic, no CLI required):
- Runs a short automated benchmark over a built-in grid of (batch, n_envs) pairs.
- Repeats each config a small number of times to reduce noise.
- Uses a weighted score over median it/s, median p90, median p50 and mean TD-error
  to select the best config.
- Runs a small EA grid (pop_size × ea_envs) with small env counts and picks the best EA config.
- Runs final training for each model using the chosen training config and runs a final EA using the chosen EA config.
- Saves benchmark CSVs and temporary .pt checkpoints under benchmarks/.
- Saves final training and final EA checkpoints under saved_models/.
- Designed to run hands-off: just `python main.py`.

Notes:
- Place this file next to your `env_fast.py` and `models.py`.
- The script runs sequentially and is conservative about GPU usage (one training job at a time).
"""

import os
import time
import math
import csv
import copy
from collections import deque
from statistics import median

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import trange

# Local imports (must exist)
from env_fast import TorchSnakeEnv
from models import DuelingMLP, DuelingCNN

# -------------------------
# Directories
# -------------------------
CKPT_DIR = "saved_models"
BENCH_DIR = "benchmarks"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(BENCH_DIR, exist_ok=True)

# -------------------------
# Machine tuning (10 CPU cores)
# -------------------------
CPU_CORES = 10
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_CORES))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_CORES))
torch.set_num_threads(CPU_CORES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# RL defaults and hyperparams
# -------------------------
ACTIONS = 3
STACK = 3
FEAT_DIM_SINGLE = 34
FEAT_DIM = FEAT_DIM_SINGLE * STACK

# -------------------------
# Automatic grids (no CLI)
# -------------------------
# Training grid tuned for 5GB GPU + 10 CPU cores
TRAIN_BATCHES = [1024, 2048, 4096]   # training batch sizes to test
TRAIN_ENVS = [512, 1024]             # n_envs to test
REPEATS = 2                          # repeats per config (keeps runtime reasonable)
SHORT_STEPS = 1000                   # short-run steps per config

# EA grid (big pop sizes, small env counts as requested)
EA_POP_SIZES = [64, 128, 256]
EA_ENVS = [8, 16, 32]
EA_GENERATIONS = 2
EA_EVAL_STEPS = 100

REPLAY_SIZE = 1_000_000
WARMUP = 10_000

GAMMA = 0.99
LR = 3e-5
UPDATES_PER_TRAIN = 1
N_STEP = 3

PER_ALPHA = 0.7
PER_EPS = 1e-5
BETA_START = 0.3
BETA_FRAMES = 2_000_000

TRAIN_EVERY = 4
ACCUM_STEPS = 1
BASE_SHAPING = 0.03
EPS_END = 0.05

TAU = 0.005

# -------------------------
# Utilities
# -------------------------
def linear_anneal(start, end, progress):
    return float(start + (end - start) * max(0.0, min(1.0, progress)))

def shaping_coeffs(pretrain, step, total_steps, grid):
    if pretrain:
        lam_dist = 0.20
        lam_len = 0.12 if grid < 20 else 0.15
    else:
        lam_dist_start = 0.20
        lam_len_start = 0.12 if grid < 20 else 0.15
        lam_dist_target = 0.05
        lam_len_target = 0.05
        prog = min(1.0, step / max(1, int(total_steps * 0.3)))
        lam_dist = linear_anneal(lam_dist_start, lam_dist_target, prog)
        lam_len = linear_anneal(lam_len_start, lam_len_target, prog)
    lam_time = -0.001
    lam_cycle = -1.0
    return lam_dist, lam_len, lam_time, lam_cycle

def make_scaled_epsilon_fn(grid, total_steps, eps_start=1.0, eps_end=EPS_END):
    scale = grid / 10.0
    decay_steps = max(1, int(total_steps * 0.7 * scale))
    def eps_fn(step):
        t = min(1.0, step / decay_steps)
        return float(eps_start + (eps_end - eps_start) * t)
    return eps_fn

def soft_update(tgt, src, tau=TAU):
    with torch.no_grad():
        for tp, sp in zip(tgt.parameters(), src.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)

# -------------------------
# Minimal SumTree + PER replay
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
            val = float((abs(p) + self.eps) ** self.alpha)
            val = max(val, 1e-6)
            self.sumtree.add(data_idx, val)
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
            p = max(p, 1e-6)
            self.sumtree.update(int(data_idx), p)

# -------------------------
# Training loops (MLP + CNN)
# Each returns metrics dict: {"it_per_s", "final_p90", "final_p50", "final_p10", "final_max", "mean_td"}
# -------------------------
def train_mlp(grid, steps, ckpt_path, batch_size, n_envs, seed=None):
    max_steps = grid * grid * max(1, int(20))
    env = TorchSnakeEnv(
        n=n_envs, g=grid, max_steps=max_steps, device=DEVICE,
        shaping_scale=BASE_SHAPING * (grid / 10),
        eat_reward=1.5,
        length_reward_scale=0.12 if grid < 20 else 0.15,
        no_eat_limit=400 if grid < 20 else 600,
        random_start=True, seed=seed,
    )

    obs_single = env.reset().to(DEVICE)
    obs_stack = deque(maxlen=STACK)
    for _ in range(STACK):
        obs_stack.append(obs_single.clone())

    def get_stacked():
        return torch.cat(list(obs_stack), dim=1)

    q = DuelingMLP(FEAT_DIM).to(DEVICE)
    tgt = DuelingMLP(FEAT_DIM).to(DEVICE)
    tgt.load_state_dict(q.state_dict())

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
        mini_batch = max(1, batch_size // ACCUM_STEPS)
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

    eps_fn = make_scaled_epsilon_fn(grid, steps)
    log_interval = max(1, steps // 10)

    t_start = time.time()
    td_accum = []
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

        nf_single, r, d, info = env.step(act)
        nf_single = nf_single.to(DEVICE); r = r.to(DEVICE); d = d.to(DEVICE)

        pretrain = (grid == 10)
        lam_dist, lam_len, lam_time, lam_cycle = shaping_coeffs(pretrain, step, steps, grid)
        shaping_raw = info.get("shaping_raw") if isinstance(info, dict) else None
        if shaping_raw is None:
            shaping_raw = torch.zeros_like(r)
        eat_mask = (r > 0).float()
        shaped = r + lam_dist * shaping_raw + lam_len * eat_mask + lam_time

        if N_STEP > 1:
            reward_buf[:, :-1] = reward_buf[:, 1:]
            done_buf[:, :-1] = done_buf[:, 1:]
            next_obs_buf[:, :-1, :] = next_obs_buf[:, 1:, :]
            action_buf[:, :-1] = action_buf[:, 1:]
        reward_buf[:, -1] = shaped
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
            td = run_training_step(step)
            if td is not None:
                td_accum.append(np.mean(np.abs(td)))
            for _ in range(UPDATES_PER_TRAIN):
                soft_update(tgt, q)

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

        if step > 0 and step % log_interval == 0:
            lengths = env.length.cpu().numpy()
            p90 = float(np.percentile(lengths, 90))
            p50 = float(np.percentile(lengths, 50))
            pbar.set_description(f"MLP g{grid} B{batch_size} E{n_envs} step{step} p90={p90:.1f} p50={p50:.1f}")

    elapsed = time.time() - t_start
    it_per_s = steps / max(1e-9, elapsed)
    lengths = env.length.cpu().numpy()
    final_p90 = float(np.percentile(lengths, 90))
    final_p50 = float(np.percentile(lengths, 50))
    final_p10 = float(np.percentile(lengths, 10))
    final_max = float(lengths.max())
    mean_td = float(np.mean(td_accum)) if td_accum else float("nan")

    torch.save({"q": q.state_dict()}, ckpt_path)
    return {
        "it_per_s": it_per_s,
        "final_p90": final_p90,
        "final_p50": final_p50,
        "final_p10": final_p10,
        "final_max": final_max,
        "mean_td": mean_td,
    }

def train_cnn(grid, steps, ckpt_path, batch_size, n_envs, seed=None):
    max_steps = grid * grid * max(1, int(20))
    env = TorchSnakeEnv(
        n=n_envs, g=grid, max_steps=max_steps, device=DEVICE,
        shaping_scale=BASE_SHAPING * (grid / 10),
        eat_reward=1.5,
        length_reward_scale=0.12 if grid < 20 else 0.15,
        no_eat_limit=400 if grid < 20 else 600,
        random_start=True, seed=seed,
    )

    env.reset().to(DEVICE)
    q = DuelingCNN(in_channels=3, grid_size=grid).to(DEVICE)
    tgt = DuelingCNN(in_channels=3, grid_size=grid).to(DEVICE)
    tgt.load_state_dict(q.state_dict())

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
        mini_batch = max(1, batch_size // ACCUM_STEPS)
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

    eps_fn = make_scaled_epsilon_fn(grid, steps)
    log_interval = max(1, steps // 10)

    t_start = time.time()
    td_accum = []
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

        nf, r, d, info = env.step(act)
        nf = nf.to(DEVICE); r = r.to(DEVICE); d = d.to(DEVICE)
        nf_grid = env.grid_observation().to(DEVICE)

        pretrain = (grid == 10)
        lam_dist, lam_len, lam_time, lam_cycle = shaping_coeffs(pretrain, step, steps, grid)
        shaping_raw = info.get("shaping_raw") if isinstance(info, dict) else None
        if shaping_raw is None:
            shaping_raw = torch.zeros_like(r)
        eat_mask = (r > 0).float()
        shaped = r + lam_dist * shaping_raw + lam_len * eat_mask + lam_time

        if N_STEP > 1:
            reward_buf[:, :-1] = reward_buf[:, 1:]
            done_buf[:, :-1] = done_buf[:, 1:]
            next_obs_buf[:, :-1, :, :, :] = next_obs_buf[:, 1:, :, :, :]
            action_buf[:, :-1] = action_buf[:, 1:]
        reward_buf[:, -1] = shaped
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
            td = run_training_step(step)
            if td is not None:
                td_accum.append(np.mean(np.abs(td)))
            for _ in range(UPDATES_PER_TRAIN):
                soft_update(tgt, q)

        just_done = d & (~prev_done)
        if just_done.any():
            mask = torch.zeros(n_envs, dtype=torch.bool, device=DEVICE)
            mask[torch.where(just_done)[0]] = True
            obs_after_reset = env.reset(mask=mask).to(DEVICE)
            nf[mask] = obs_after_reset[mask].clone()
            steps_since_reset[mask] = 0
            first_obs_buf[mask] = env.grid_observation().to(DEVICE)[mask]

        prev_done = d.clone()

        if step > 0 and step % log_interval == 0:
            lengths = env.length.cpu().numpy()
            p90 = float(np.percentile(lengths, 90))
            p50 = float(np.percentile(lengths, 50))
            pbar.set_description(f"CNN g{grid} B{batch_size} E{n_envs} step{step} p90={p90:.1f} p50={p50:.1f}")

    elapsed = time.time() - t_start
    it_per_s = steps / max(1e-9, elapsed)
    lengths = env.length.cpu().numpy()
    final_p90 = float(np.percentile(lengths, 90))
    final_p50 = float(np.percentile(lengths, 50))
    final_p10 = float(np.percentile(lengths, 10))
    final_max = float(lengths.max())
    mean_td = float(np.mean(td_accum)) if td_accum else float("nan")

    torch.save({"q": q.state_dict()}, ckpt_path)
    return {
        "it_per_s": it_per_s,
        "final_p90": final_p90,
        "final_p50": final_p50,
        "final_p10": final_p10,
        "final_max": final_max,
        "mean_td": mean_td,
    }

# -------------------------
# EA helpers (configurable)
# -------------------------
def evaluate_model_on_env(model, env, steps_eval=100):
    model.eval()
    with torch.no_grad():
        for _ in range(steps_eval):
            obs = env.grid_observation().to(DEVICE)
            qvals = model(obs)
            act = qvals.argmax(1)
            nf, r, d, info = env.step(act)
    lengths = env.length.cpu().numpy()
    return float(np.percentile(lengths, 50))

def run_ea_short_mlp(grid, ea_n_envs, pop_size=64, generations=2, eval_steps=100, seed=None, base_model=None):
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)
    env = TorchSnakeEnv(n=ea_n_envs, g=grid, max_steps=grid*grid*20, device=DEVICE, seed=seed)
    base = DuelingMLP(FEAT_DIM).to(DEVICE) if base_model is None else base_model
    best_model = copy.deepcopy(base)
    best_score = evaluate_model_on_env(best_model, env, steps_eval=eval_steps)
    population = []
    for i in range(pop_size):
        m = copy.deepcopy(base)
        for p in m.parameters():
            p.data.add_(0.01 * torch.randn_like(p))
        population.append(m)
    for gen in range(generations):
        scores = []
        for m in population:
            s = evaluate_model_on_env(m, env, steps_eval=eval_steps)
            scores.append(s)
            if s > best_score:
                best_score = s
                best_model = copy.deepcopy(m)
        idxs = np.argsort(scores)[-2:]
        parents = [population[i] for i in idxs]
        new_pop = []
        for i in range(pop_size):
            parent = parents[i % 2]
            child = copy.deepcopy(parent)
            for p in child.parameters():
                p.data.add_(0.02 * torch.randn_like(p))
            new_pop.append(child)
        population = new_pop
    return {"best_score": best_score, "model": best_model}

def run_ea_short_cnn(grid, ea_n_envs, pop_size=64, generations=2, eval_steps=100, seed=None, base_model=None):
    if seed is not None:
        torch.manual_seed(seed); np.random_seed(seed)
    env = TorchSnakeEnv(n=ea_n_envs, g=grid, max_steps=grid*grid*20, device=DEVICE, seed=seed)
    base = DuelingCNN(in_channels=3, grid_size=grid).to(DEVICE) if base_model is None else base_model
    best_model = copy.deepcopy(base)
    best_score = evaluate_model_on_env(best_model, env, steps_eval=eval_steps)
    population = []
    for i in range(pop_size):
        m = copy.deepcopy(base)
        for p in m.parameters():
            p.data.add_(0.01 * torch.randn_like(p))
        population.append(m)
    for gen in range(generations):
        scores = []
        for m in population:
            s = evaluate_model_on_env(m, env, steps_eval=eval_steps)
            scores.append(s)
            if s > best_score:
                best_score = s
                best_model = copy.deepcopy(m)
        idxs = np.argsort(scores)[-2:]
        parents = [population[i] for i in idxs]
        new_pop = []
        for i in range(pop_size):
            parent = parents[i % 2]
            child = copy.deepcopy(parent)
            for p in child.parameters():
                p.data.add_(0.02 * torch.randn_like(p))
            new_pop.append(child)
        population = new_pop
    return {"best_score": best_score, "model": best_model}

# -------------------------
# Orchestration helpers
# -------------------------
def write_csv(path, rows, keys):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})
    os.replace(tmp, path)

def _norm(x, lo, hi):
    if hi <= lo:
        return 0.0
    return (x - lo) / (hi - lo)

def weighted_score(it_s, p90, p50, mean_td, stats):
    it_s_n = _norm(it_s, stats["it_s_min"], stats["it_s_max"])
    p90_n  = _norm(p90,  stats["p90_min"],  stats["p90_max"])
    p50_n  = _norm(p50,  stats["p50_min"],  stats["p50_max"])
    td_n   = _norm(mean_td, stats["td_min"], stats["td_max"])
    return (
        0.50 * it_s_n +
        0.30 * p90_n +
        0.15 * p50_n +
        0.05 * (1.0 - td_n)
    )

def evaluate_training_grid_auto(model_type, grid):
    rows = []
    agg_rows = []
    pairs = [(B, E) for B in TRAIN_BATCHES for E in TRAIN_ENVS]

    # Run all configs with repeats, save tmp checkpoints into benchmarks/
    for B, E in pairs:
        run_metrics = []
        for r in range(REPEATS):
            seed_run = int(time.time()) % (2**31)
            print(f"\nAuto-run {model_type.upper()} B={B} E={E} repeat={r+1}/{REPEATS}")
            ckpt = os.path.join(BENCH_DIR, f"tmp_{model_type}_g{grid}_B{B}_E{E}_r{r}.pt")
            try:
                if model_type == "mlp":
                    m = train_mlp(grid=grid, steps=SHORT_STEPS, ckpt_path=ckpt,
                                  batch_size=B, n_envs=E, seed=seed_run)
                else:
                    m = train_cnn(grid=grid, steps=SHORT_STEPS, ckpt_path=ckpt,
                                  batch_size=B, n_envs=E, seed=seed_run)
            except Exception as ex:
                print("Benchmark run failed:", ex)
                m = {"it_per_s": 0.0, "final_p90": 0.0, "final_p50": 0.0,
                     "final_p10": 0.0, "final_max": 0.0, "mean_td": float("nan")}
            row = {
                "model": model_type,
                "grid": grid,
                "batch": B,
                "n_envs": E,
                "repeat": r,
                "it_per_s": m["it_per_s"],
                "final_p90": m["final_p90"],
                "final_p50": m["final_p50"],
                "final_p10": m["final_p10"],
                "final_max": m["final_max"],
                "mean_td": m["mean_td"],
            }
            rows.append(row)
            run_metrics.append(m)

        # Aggregate per (B,E)
        it_list = [rm["it_per_s"] for rm in run_metrics]
        p90_list = [rm["final_p90"] for rm in run_metrics]
        p50_list = [rm["final_p50"] for rm in run_metrics]
        td_list = [rm["mean_td"] for rm in run_metrics if not math.isnan(rm["mean_td"])]

        agg_rows.append({
            "model": model_type,
            "grid": grid,
            "batch": B,
            "n_envs": E,
            "it_per_s_med": median(it_list) if it_list else 0.0,
            "final_p90_med": median(p90_list) if p90_list else 0.0,
            "final_p50_med": median(p50_list) if p50_list else 0.0,
            "mean_td_med": median(td_list) if td_list else float("nan"),
        })

    # Save raw benchmark CSV
    raw_keys = ["model", "grid", "batch", "n_envs", "repeat",
                "it_per_s", "final_p90", "final_p50", "final_p10", "final_max", "mean_td"]
    raw_path = os.path.join(BENCH_DIR, f"bench_{model_type}_g{grid}_raw.csv")
    write_csv(raw_path, rows, raw_keys)

    # Compute stats for normalization
    it_vals = [r["it_per_s_med"] for r in agg_rows]
    p90_vals = [r["final_p90_med"] for r in agg_rows]
    p50_vals = [r["final_p50_med"] for r in agg_rows]
    td_vals = [r["mean_td_med"] for r in agg_rows if not math.isnan(r["mean_td_med"])]

    stats = {
        "it_s_min": min(it_vals) if it_vals else 0.0,
        "it_s_max": max(it_vals) if it_vals else 1.0,
        "p90_min": min(p90_vals) if p90_vals else 0.0,
        "p90_max": max(p90_vals) if p90_vals else 1.0,
        "p50_min": min(p50_vals) if p50_vals else 0.0,
        "p50_max": max(p50_vals) if p50_vals else 1.0,
        "td_min": min(td_vals) if td_vals else 0.0,
        "td_max": max(td_vals) if td_vals else 1.0,
    }

    # Compute weighted scores and pick best
    best_cfg = None
    best_score = -1e9
    for r in agg_rows:
        score = weighted_score(
            r["it_per_s_med"],
            r["final_p90_med"],
            r["final_p50_med"],
            r["mean_td_med"] if not math.isnan(r["mean_td_med"]) else stats["td_max"],
            stats,
        )
        r["score"] = score
        if score > best_score:
            best_score = score
            best_cfg = r

    # Save aggregated CSV
    agg_keys = ["model", "grid", "batch", "n_envs",
                "it_per_s_med", "final_p90_med", "final_p50_med", "mean_td_med", "score"]
    agg_path = os.path.join(BENCH_DIR, f"bench_{model_type}_g{grid}_agg.csv")
    write_csv(agg_path, agg_rows, agg_keys)

    print(f"\nBest training config for {model_type.upper()} on grid {grid}: "
          f"batch={best_cfg['batch']} n_envs={best_cfg['n_envs']} score={best_cfg['score']:.4f}")

    return best_cfg

def evaluate_ea_grid_auto(model_type, grid):
    rows = []
    agg_rows = []
    pairs = [(p, e) for p in EA_POP_SIZES for e in EA_ENVS]

    for pop_size, ea_envs in pairs:
        run_scores = []
        for r in range(REPEATS):
            seed_run = int(time.time()) % (2**31)
            print(f"\nEA auto-run {model_type.upper()} pop={pop_size} ea_envs={ea_envs} repeat={r+1}/{REPEATS}")
            try:
                if model_type == "mlp":
                    res = run_ea_short_mlp(grid=grid, ea_n_envs=ea_envs,
                                           pop_size=pop_size, generations=EA_GENERATIONS,
                                           eval_steps=EA_EVAL_STEPS, seed=seed_run)
                else:
                    res = run_ea_short_cnn(grid=grid, ea_n_envs=ea_envs,
                                           pop_size=pop_size, generations=EA_GENERATIONS,
                                           eval_steps=EA_EVAL_STEPS, seed=seed_run)
                score = res["best_score"]
            except Exception as ex:
                print("EA benchmark failed:", ex)
                score = 0.0
            rows.append({
                "model": model_type,
                "grid": grid,
                "pop_size": pop_size,
                "ea_envs": ea_envs,
                "repeat": r,
                "best_score": score,
            })
            run_scores.append(score)

        agg_rows.append({
            "model": model_type,
            "grid": grid,
            "pop_size": pop_size,
            "ea_envs": ea_envs,
            "best_score_med": median(run_scores) if run_scores else 0.0,
        })

    # Save EA raw CSV
    raw_keys = ["model", "grid", "pop_size", "ea_envs", "repeat", "best_score"]
    raw_path = os.path.join(BENCH_DIR, f"ea_{model_type}_g{grid}_raw.csv")
    write_csv(raw_path, rows, raw_keys)

    # Pick best EA config by median score
    best_cfg = None
    best_score = -1e9
    for r in agg_rows:
        if r["best_score_med"] > best_score:
            best_score = r["best_score_med"]
            best_cfg = r

    # Save EA aggregated CSV
    agg_keys = ["model", "grid", "pop_size", "ea_envs", "best_score_med"]
    agg_path = os.path.join(BENCH_DIR, f"ea_{model_type}_g{grid}_agg.csv")
    write_csv(agg_path, agg_rows, agg_keys)

    print(f"\nBest EA config for {model_type.upper()} on grid {grid}: "
          f"pop_size={best_cfg['pop_size']} ea_envs={best_cfg['ea_envs']} "
          f"median_score={best_cfg['best_score_med']:.4f}")

    return best_cfg

# -------------------------
# Main orchestration
# -------------------------
def main_auto():
    grid = 10  # fixed grid size for now

    for model in ["mlp", "cnn"]:
        print(f"\n================ {model.upper()} AUTO-BENCHMARK ================")
        bt = evaluate_training_grid_auto(model, grid)
        be = evaluate_ea_grid_auto(model, grid)

        B = int(bt["batch"])
        E = int(bt["n_envs"])
        print(f"\n--- Final training for {model.upper()} with batch={B}, envs={E} ---")
        ckpt_final = os.path.join(CKPT_DIR, f"final_{model}_g{grid}_B{B}_E{E}.pt")
        try:
            if model == "mlp":
                train_mlp(grid=grid, steps=max(2000, SHORT_STEPS * 2),
                          ckpt_path=ckpt_final, batch_size=B, n_envs=E)
            else:
                train_cnn(grid=grid, steps=max(2000, SHORT_STEPS * 2),
                          ckpt_path=ckpt_final, batch_size=B, n_envs=E)
        except Exception as ex:
            print("Final training failed:", ex)

        pop_size = int(be["pop_size"])
        ea_envs = int(be["ea_envs"])
        print(f"\n--- Final EA for {model.upper()} pop={pop_size} ea_envs={ea_envs} ---")
        try:
            if model == "mlp":
                res = run_ea_short_mlp(grid=grid, ea_n_envs=ea_envs, pop_size=pop_size,
                                       generations=EA_GENERATIONS, eval_steps=EA_EVAL_STEPS)
            else:
                res = run_ea_short_cnn(grid=grid, ea_n_envs=ea_envs, pop_size=pop_size,
                                       generations=EA_GENERATIONS, eval_steps=EA_EVAL_STEPS)
            print("Final EA best score:", res["best_score"])

            # Save final EA model checkpoint for app.py-style loading
            ea_name = f"evo_{grid}x{grid}_from_{model}.pt"
            ea_path = os.path.join(CKPT_DIR, ea_name)
            torch.save({"q": res["model"].state_dict()}, ea_path)
            print(f"Saved final EA model to {ea_path}")
        except Exception as ex:
            print("Final EA failed:", ex)

    print("\nAll done. Benchmarks are in benchmarks/ and final checkpoints are in saved_models/.")

if __name__ == "__main__":
    main_auto()
