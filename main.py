import os
import time
import random
from collections import deque
from typing import List, Optional

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

import snake_gym_env

SAVED_DIR = "saved_models"
os.makedirs(SAVED_DIR, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")
        torch.set_num_threads(8)
        torch.set_num_interop_threads(8)
    return device


class SnakeQNet(nn.Module):
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, max_length: int = 50_000):
        self.states = deque(maxlen=max_length)
        self.actions = deque(maxlen=max_length)
        self.next_states = deque(maxlen=max_length)
        self.rewards = deque(maxlen=max_length)
        self.dones = deque(maxlen=max_length)

    def store(self, state_np, action, next_state_np, reward, done):
        self.states.append(state_np)
        self.actions.append(int(action))
        self.next_states.append(next_state_np)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def sample(self, batch_size: int):
        if len(self.dones) < batch_size:
            return None
        idx = np.random.choice(len(self.dones), batch_size, replace=False)
        states = np.stack([self.states[i] for i in idx])
        actions = np.array([self.actions[i] for i in idx], dtype=np.int64)
        next_states = np.stack([self.next_states[i] for i in idx])
        rewards = np.array([self.rewards[i] for i in idx], dtype=np.float32)
        dones = np.array([self.dones[i] for i in idx], dtype=np.float32)
        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.dones)


def train_step_double_dqn(
    buffer: ReplayBuffer,
    model: nn.Module,
    target_model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    gamma: float,
    batch_size: int,
    device: torch.device,
):
    batch = buffer.sample(batch_size)
    if batch is None:
        return None

    states_np, actions_np, next_states_np, rewards_np, dones_np = batch

    states = torch.from_numpy(states_np).float().to(device)
    next_states = torch.from_numpy(next_states_np).float().to(device)
    actions = torch.from_numpy(actions_np).long().to(device).unsqueeze(1)
    rewards = torch.from_numpy(rewards_np).float().to(device)
    dones = torch.from_numpy(dones_np).float().to(device)

    with torch.no_grad():
        next_q_online = model(next_states)
        next_actions = next_q_online.argmax(dim=1, keepdim=True)
        next_q_target = target_model(next_states)
        next_q_selected = next_q_target.gather(1, next_actions).squeeze(1)
        targets = rewards + gamma * next_q_selected * (1.0 - dones)

    q_values = model(states)
    q_action = q_values.gather(1, actions).squeeze(1)

    loss = loss_fn(q_action, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return float(loss.item())


def obs_to_features(obs: dict) -> np.ndarray:
    coords = obs["coords"].astype(np.float32)
    feats = obs["features"].astype(np.float32)
    n_envs = coords.shape[0]
    coords_flat = coords.reshape(n_envs, -1)
    return np.concatenate([coords_flat, feats], axis=1)


def train_model(
    grid_size=20,
    episodes=2000,
    learning_rate=1e-4,
    batch_size=256,
    seed: Optional[int] = 42,
    eval_every=2000,
    n_envs: int = 8,
    save_dir=SAVED_DIR,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = get_device()

    def make_env():
        return gym.make("Snake-v0", render_mode=None, grid_size=grid_size)

    env = SyncVectorEnv([make_env for _ in range(n_envs)])

    obs, _ = env.reset()
    feat = obs_to_features(obs)
    input_dim = feat.shape[1]
    num_actions = env.single_action_space.n

    model = SnakeQNet(input_dim, num_actions).to(device)
    target_model = SnakeQNet(input_dim, num_actions).to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()
    buffer = ReplayBuffer(max_length=50_000)

    gamma = 0.99
    max_steps = 4000

    eps_start = 1.0
    eps_end = 0.05
    eps_decay_steps = 200_000

    training_steps = 0
    warmup_steps = 5_000
    target_update_freq = 1000

    rewards_history: List[float] = []
    per_env_returns = np.zeros(n_envs, dtype=np.float32)
    completed_episodes = 0
    best_eval = -float("inf")

    start_time = time.time()
    pbar = tqdm(total=episodes, desc="Episodes (across envs)")

    while completed_episodes < episodes:
        state_feats = obs_to_features(obs)

        epsilon = max(eps_end, eps_start - training_steps / eps_decay_steps)

        with torch.no_grad():
            state_tensor = torch.from_numpy(state_feats).float().to(device)
            q_values = model(state_tensor)
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()

        random_mask = np.random.rand(n_envs) < epsilon
        random_actions = np.random.randint(num_actions, size=n_envs)
        actions = np.where(random_mask, random_actions, greedy_actions)

        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        dones = np.logical_or(terminations, truncations)
        next_feats = obs_to_features(next_obs)

        for i in range(n_envs):
            buffer.store(
                state_feats[i],
                actions[i],
                next_feats[i],
                float(rewards[i]),
                bool(dones[i]),
            )

        per_env_returns += rewards.astype(np.float32)
        training_steps += n_envs

        if len(buffer) >= batch_size and training_steps > warmup_steps:
            loss_val = train_step_double_dqn(
                buffer,
                model,
                target_model,
                optimizer,
                loss_fn,
                gamma,
                batch_size,
                device,
            )
            if training_steps % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

        if np.any(dones):
            for i in range(n_envs):
                if dones[i]:
                    completed_episodes += 1
                    rewards_history.append(float(per_env_returns[i]))
                    pbar.update(1)
                    pbar.set_postfix({
                        "R": f"{per_env_returns[i]:.1f}",
                        "Eps": f"{epsilon:.3f}",
                        "Buf": len(buffer),
                    })
                    per_env_returns[i] = 0.0
                    if completed_episodes >= episodes:
                        break

        obs = next_obs

        if completed_episodes > 0 and completed_episodes % eval_every == 0:
            eval_rewards = []
            eval_env = gym.make("Snake-v0", render_mode=None, grid_size=grid_size)
            for _ in range(3):
                o, _ = eval_env.reset()
                feat_eval = obs_to_features(
                    {
                        "coords": o["coords"][None, ...],
                        "features": o["features"][None, ...],
                    }
                )[0]
                total_r = 0.0
                for _ in range(max_steps):
                    with torch.no_grad():
                        qvals = model(
                            torch.from_numpy(feat_eval)
                            .float()
                            .unsqueeze(0)
                            .to(device)
                        )
                        act = int(torch.argmax(qvals, dim=1).item())
                    o2, r_eval, term_eval, trunc_eval, _ = eval_env.step(act)
                    total_r += float(r_eval)
                    feat_eval = obs_to_features(
                        {
                            "coords": o2["coords"][None, ...],
                            "features": o2["features"][None, ...],
                        }
                    )[0]
                    if term_eval or trunc_eval:
                        break
                eval_rewards.append(total_r)
            eval_env.close()

            eval_mean = float(np.mean(eval_rewards))
            print(f"\n[Eval at {completed_episodes} episodes] mean {eval_mean:.2f}")

            if eval_mean > best_eval:
                best_eval = eval_mean
                best_path = os.path.join(save_dir, f"best_snake_grid{grid_size}.pt")
                torch.save(model.state_dict(), best_path)
                print(f"Saved new best model: {best_path}")

    pbar.close()
    env.close()

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed/3600:.2f} hours.")
    print(f"Total episodes: {completed_episodes}")

    if rewards_history:
        plt.figure(figsize=(8, 5))
        plt.plot(rewards_history, alpha=0.6)
        if len(rewards_history) >= 50:
            sm = np.convolve(rewards_history, np.ones(50) / 50, mode="valid")
            plt.plot(range(49, len(rewards_history)), sm, color="red", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Snake training (grid {grid_size}, {n_envs} envs)")
        plt.grid(True)
        plt.tight_layout()
        curve_path = os.path.join(save_dir, f"learning_curve_grid{grid_size}.png")
        plt.savefig(curve_path)
        plt.close()
        print(f"Learning curve saved to {curve_path}")


if __name__ == "__main__":
    train_model(
        grid_size=20,
        episodes=200_000,
        learning_rate=1e-4,
        batch_size=256,
        seed=42,
        eval_every=2000,
        n_envs=8,
    )