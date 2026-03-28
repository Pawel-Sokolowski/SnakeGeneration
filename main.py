import os
import time
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from tqdm import tqdm

import snake_gym_env   # ← ensures Snake-v0 is registered


SAVED_DIR = "saved_models"
os.makedirs(SAVED_DIR, exist_ok=True)


# ===============================================================
# DEVICE
# ===============================================================
def get_device():
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


# ===============================================================
# MODEL  (matches viewer's new architecture)
# ===============================================================
class SnakeQNet(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.net(x)


# ===============================================================
# REPLAY BUFFER
# ===============================================================
class ReplayBuffer:
    def __init__(self, max_length=200_000):
        self.states = deque(maxlen=max_length)
        self.actions = deque(maxlen=max_length)
        self.next_states = deque(maxlen=max_length)
        self.rewards = deque(maxlen=max_length)
        self.dones = deque(maxlen=max_length)

    def store(self, s, a, ns, r, d):
        self.states.append(s)
        self.actions.append(int(a))
        self.next_states.append(ns)
        self.rewards.append(float(r))
        self.dones.append(bool(d))

    def sample(self, batch_size):
        if len(self.dones) < batch_size:
            return None
        idx = np.random.choice(len(self.dones), batch_size, replace=False)
        s = np.stack([self.states[i] for i in idx])
        a = np.array([self.actions[i] for i in idx], dtype=np.int64)
        ns = np.stack([self.next_states[i] for i in idx])
        r = np.array([self.rewards[i] for i in idx], dtype=np.float32)
        d = np.array([self.dones[i] for i in idx], dtype=np.float32)
        return s, a, ns, r, d

    def __len__(self):
        return len(self.dones)


# ===============================================================
# UTILS
# ===============================================================
def obs_to_features(obs):
    # Works for both vectorized and single env observations
    return obs["features"].astype(np.float32)


@torch.no_grad()
def evaluate_policy(make_env, model, device, episodes=5):
    """
    Runs greedy (argmax) policy for a few episodes to get a stable score estimate.
    Uses a single non-vectorized env created by make_env().
    """
    model.eval()
    returns = []
    lengths = []

    for _ in range(episodes):
        env = make_env()
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0

        while True:
            feat = obs_to_features(obs)  # (48,)
            x = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
            q = model(x)
            act = int(torch.argmax(q, dim=1).item())

            obs, r, term, trunc, info = env.step(act)
            ep_ret += float(r)
            ep_len += 1

            if term or trunc:
                break

        env.close()
        returns.append(ep_ret)
        lengths.append(ep_len)

    avg_ret = float(np.mean(returns)) if returns else 0.0
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    return avg_ret, avg_len


def save_checkpoint(model, path, *, input_dim, num_actions, grid_size,
                    training_steps, episodes_completed, avg_eval_return):
    """
    Saves a checkpoint with state_dict and rich metadata so the viewer
    can always reconstruct dimensions and show diagnostics.
    """
    payload = {
        "model_state_dict": model.state_dict(),
        "input_dim": int(input_dim),
        "num_actions": int(num_actions),
        "grid_size": int(grid_size),
        "training_steps": int(training_steps),
        "episodes_completed": int(episodes_completed),
        "avg_eval_return": float(avg_eval_return),
        "timestamp": time.time(),
        "arch": [int(input_dim), 512, 256, int(num_actions)],
    }
    torch.save(payload, path)


# ===============================================================
# TRAIN LOOP
# ===============================================================
def train_model(
    grid_size=20,
    episodes=200_000,
    learning_rate=1e-4,
    batch_size=256,
    seed=42,
    eval_every=2000,     # ← save every eval
    n_envs=8,
    eval_episodes=5,     # how many episodes to average during eval
):

    # seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = get_device()

    # env factory (no render)
    def make_env():
        return gym.make("Snake-v0", grid_size=grid_size, render_mode=None)

    # vector env for training
    env = SyncVectorEnv([make_env for _ in range(n_envs)])

    obs, _ = env.reset()
    feat = obs_to_features(obs)          # shape: (n_envs, 48)
    input_dim = int(feat.shape[1])       # 48
    num_actions = int(env.single_action_space.n)  # 4

    model = SnakeQNet(input_dim, num_actions).to(device)
    target = SnakeQNet(input_dim, num_actions).to(device)
    target.load_state_dict(model.state_dict())

    opt = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()

    buffer = ReplayBuffer(max_length=200_000)

    gamma = 0.995
    eps_start, eps_end = 1.0, 0.05
    eps_decay = 2_000_000

    warmup = 10_000
    target_update = 2000

    training_steps = 0
    completed = 0

    returns = np.zeros(n_envs, dtype=np.float32)

    # eval/checkpoint bookkeeping
    next_eval_milestone = eval_every
    best_avg_eval_return = -float("inf")

    pbar = tqdm(total=episodes, desc="Episodes")

    while completed < episodes:
        feats = obs_to_features(obs)  # (n_envs, 48)

        epsilon = max(eps_end, eps_start - training_steps / eps_decay)

        with torch.no_grad():
            q = model(torch.tensor(feats, dtype=torch.float32, device=device))
            greedy = q.argmax(dim=1).cpu().numpy()

        random_mask = np.random.rand(n_envs) < epsilon
        random_actions = np.random.randint(num_actions, size=n_envs)
        actions = np.where(random_mask, random_actions, greedy)

        next_obs, reward, term, trunc, info = env.step(actions)
        done = np.logical_or(term, trunc)

        next_feats = obs_to_features(next_obs)

        for i in range(n_envs):
            buffer.store(feats[i], actions[i], next_feats[i], reward[i], done[i])

        returns += reward
        training_steps += n_envs

        # train
        if training_steps > warmup:
            train_step = buffer.sample(batch_size)
            if train_step is not None:
                # re-run train step using the helper to compute loss
                s, a, ns, r, d = train_step
                s = torch.tensor(s, dtype=torch.float32, device=device)
                ns = torch.tensor(ns, dtype=torch.float32, device=device)
                a = torch.tensor(a, dtype=torch.long, device=device).unsqueeze(1)
                r = torch.tensor(r, dtype=torch.float32, device=device)
                d = torch.tensor(d, dtype=torch.float32, device=device)

                with torch.no_grad():
                    next_action = model(ns).argmax(dim=1, keepdim=True)
                    next_q = target(ns).gather(1, next_action).squeeze(1)
                    targets = r + gamma * next_q * (1 - d)

                q = model(s).gather(1, a).squeeze(1)
                loss = loss_fn(q, targets)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                if training_steps % target_update == 0:
                    target.load_state_dict(model.state_dict())

        # episodes accounting and progress
        if np.any(done):
            for i in range(n_envs):
                if done[i]:
                    completed += 1
                    pbar.update(1)
                    pbar.set_postfix(R=float(returns[i]), eps=epsilon)
                    returns[i] = 0.0

                    # ---- EVAL & CHECKPOINT at milestones ----
                    if completed >= next_eval_milestone:
                        avg_eval_return, avg_eval_len = evaluate_policy(make_env, model, device, episodes=eval_episodes)

                        # File names
                        tag = f"g{grid_size}_ep{completed}_steps{training_steps}_R{avg_eval_return:.2f}"
                        ckpt_path = os.path.join(SAVED_DIR, f"snake_dqn_{tag}.pt")
                        latest_path = os.path.join(SAVED_DIR, "snake_dqn_latest.pt")
                        best_path = os.path.join(SAVED_DIR, "snake_dqn_best.pt")

                        # Save milestone checkpoint
                        save_checkpoint(
                            model, ckpt_path,
                            input_dim=input_dim,
                            num_actions=num_actions,
                            grid_size=grid_size,
                            training_steps=training_steps,
                            episodes_completed=completed,
                            avg_eval_return=avg_eval_return
                        )

                        # Save/overwrite "latest"
                        save_checkpoint(
                            model, latest_path,
                            input_dim=input_dim,
                            num_actions=num_actions,
                            grid_size=grid_size,
                            training_steps=training_steps,
                            episodes_completed=completed,
                            avg_eval_return=avg_eval_return
                        )

                        # Save/overwrite "best" if improved
                        if avg_eval_return > best_avg_eval_return:
                            best_avg_eval_return = avg_eval_return
                            save_checkpoint(
                                model, best_path,
                                input_dim=input_dim,
                                num_actions=num_actions,
                                grid_size=grid_size,
                                training_steps=training_steps,
                                episodes_completed=completed,
                                avg_eval_return=avg_eval_return
                            )

                        # move the milestone forward
                        next_eval_milestone += eval_every

        obs = next_obs

    env.close()
    pbar.close()

    # Final save when training finishes
    final_tag = f"g{grid_size}_ep{completed}_steps{training_steps}"
    final_path = os.path.join(SAVED_DIR, f"snake_dqn_{final_tag}_final.pt")
    save_checkpoint(
        model, final_path,
        input_dim=input_dim,
        num_actions=num_actions,
        grid_size=grid_size,
        training_steps=training_steps,
        episodes_completed=completed,
        avg_eval_return=float(best_avg_eval_return if best_avg_eval_return != -float('inf') else 0.0)
    )
    print("Saved final:", final_path)


if __name__ == "__main__":
    train_model()