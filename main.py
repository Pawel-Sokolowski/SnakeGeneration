import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import amp
from torch.amp import autocast, GradScaler
from itertools import product
from collections import deque
import matplotlib.pyplot as plt
import gymnasium as gym
import snake_gym_env  # Make sure this module is available

# 🧠 Transformer-based Q-Network
class SnakeTransformerQNet(nn.Module):
    def __init__(self, seq_len, feature_dim, num_actions):
        super().__init__()
        self.embed = nn.Linear(feature_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)

# 🎒 Replay Buffer
class BatchReplayBuffer:
    def __init__(self, max_length=100_000):
        self.states, self.actions = deque(maxlen=max_length), deque(maxlen=max_length)
        self.next_states, self.rewards, self.dones = deque(maxlen=max_length), deque(maxlen=max_length), deque(maxlen=max_length)

    def store_batch(self, s, a, s_, r, d):
        self.states.extend(s)
        self.actions.extend(a)
        self.next_states.extend(s_)
        self.rewards.extend(r)
        self.dones.extend(d)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.dones), batch_size, replace=False)
        def batch(data, dtype): return torch.tensor(np.array([data[i] for i in idx]), dtype=dtype)
        return (
            batch(self.states, torch.float32),
            batch(self.actions, torch.int64),
            batch(self.next_states, torch.float32),
            batch(self.rewards, torch.float32),
            batch(self.dones, torch.float32)
        )

# 🎛️ Parameter Grid
param_grid = {
    "learning_rate": [0.001],
    "gamma": [0.99],
    "batch_size": [4096],
    "env_count": [64]
}

def param_grid_search(grid):
    keys, values = zip(*grid.items())
    for combo in product(*values):
        yield dict(zip(keys, combo))

# 🎯 Action Selection
def select_action(model, obs_batch, epsilon, num_actions, device):
    actions = []
    with torch.no_grad(), amp.autocast("cuda"):
        q_vals = model(obs_batch.to(device))
        for i in range(len(obs_batch)):
            if random.random() < epsilon:
                actions.append(random.randint(0, num_actions - 1))
            else:
                actions.append(torch.argmax(q_vals[i]).item())
    return actions

# 🔧 Training Step
def train_step(buffer, model, target_model, optimizer, loss_fn, gamma, batch_size, device, scaler):
    if len(buffer.dones) < batch_size: return
    s, a, s_, r, d = buffer.sample(batch_size)
    s, s_, a = s.to(device), s_.to(device), a.to(device).unsqueeze(1)
    r, d = r.to(device), d.to(device)

    with torch.no_grad(), amp.autocast("cuda"):
        target_q = target_model(s_)
        max_q = target_q.max(dim=1)[0]
        targets = r + gamma * max_q * (1 - d)

    with amp.autocast("cuda"):
        q_vals = model(s)
        q_action = q_vals.gather(1, a).squeeze()
        loss = loss_fn(q_action, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

# 🧵 DDP Setup
def setup(rank, world_size):
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup(): dist.destroy_process_group()

def train(rank, world_size, params):
    try:
        setup(rank, world_size)
        device = torch.device(f"cuda:{rank}")
        envs = [gym.make("Snake-v0", render_mode=None) for _ in range(params["env_count"])]
        seq_len, feature_dim = envs[0].observation_space.shape
        num_actions = envs[0].action_space.n

        model = SnakeTransformerQNet(seq_len, feature_dim, num_actions).to(device)
        target_model = SnakeTransformerQNet(seq_len, feature_dim, num_actions).to(device)
        target_model.load_state_dict(model.state_dict())
        model = DDP(model, device_ids=[rank])

        with torch.no_grad():
            dummy = torch.rand((params["env_count"], seq_len, feature_dim), dtype=torch.float32).to(device)
            model(dummy)
            target_model(dummy)

        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        loss_fn, scaler = nn.HuberLoss(), GradScaler()
        buffer = BatchReplayBuffer()

        eps, eps_min, eps_decay = 1.0, 0.05, (1.0 - 0.05) / 1_000_000
        gamma, batch_size = params["gamma"], params["batch_size"]
        train_freq, sync_freq, max_steps, n_episodes = 4, 1000, 500, 100_000
        step_count = 0
        rewards_history, all_rewards = deque(maxlen=100), []
        obs_batch = [env.reset()[0].astype(np.float32) for env in envs]
        start_time = time.time()

        for ep in range(n_episodes):
            ep_reward, ep_start = [0.0] * len(envs), time.time()
            for _ in range(max_steps):
                step_count += 1
                obs_tensor = torch.tensor(np.array(obs_batch), dtype=torch.float32)
                actions = select_action(model, obs_tensor, eps, num_actions, device)
                eps = max(eps - eps_decay, eps_min)

                next_obs_batch, rewards, dones = [], [], []
                for i, env in enumerate(envs):
                    obs2, r, term, trunc, _ = env.step(actions[i])
                    done = term or trunc
                    next_obs_batch.append(obs2.astype(np.float32))
                    rewards.append(r)
                    dones.append(done)
                    ep_reward[i] += r
                    if done:
                        next_obs_batch[i] = env.reset()[0].astype(np.float32)

                buffer.store_batch(obs_batch, actions, next_obs_batch, rewards, dones)
                obs_batch = next_obs_batch

                if step_count % train_freq == 0:
                    train_step(buffer, model, target_model, optimizer, loss_fn, gamma, batch_size, device, scaler)

                if step_count % sync_freq == 0:
                    target_model.load_state_dict(model.module.state_dict())

            avg_reward = np.mean(ep_reward)
            rewards_history.append(avg_reward)
            all_rewards.append(avg_reward)
            elapsed, ep_time = time.time() - start_time, time.time() - ep_start
            est_total = elapsed / (ep + 1) * n_episodes
            est_remain = est_total - elapsed
            mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**2

            if rank == 0 and ep % 10 == 0:
                print(f"[GPU {rank}] Ep {ep} | Avg: {avg_reward:.2f} | EpTime: {ep_time:.2f}s | ETA: {est_remain/60:.1f} min | Alloc: {mem_alloc:.1f} MB | Reserved: {mem_reserved:.1f} MB")

            if np.mean(rewards_history) > 40 and rank == 0:
                print(f"[GPU {rank}] Solved at episode {ep}!")
                break

    finally:
        if rank == 0:
            os.makedirs("saved_models", exist_ok=True)
            tag = f"lr{params['learning_rate']}_gamma{params['gamma']}_bs{params['batch_size']}_env{params['env_count']}"
            model_path = f"saved_models/transformer_snake_{tag}.pt"
            plot_path = f"saved_models/transformer_curve_{tag}.png"

            torch.save(model.module.state_dict(), model_path)

            plt.figure(figsize=(10, 6))
            plt.plot(all_rewards, label="Episode Avg Reward")
            if len(all_rewards) >= 100:
                smoothed = np.convolve(all_rewards, np.ones(100)/100, mode="valid")
                plt.plot(range(99, len(all_rewards)), smoothed, label="Moving Avg (100)", color="orange")
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.title(f"DDP Transformer Snake Training\n{tag}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

        cleanup()
        
def run_grid():
    world_size = torch.cuda.device_count()
    for params in param_grid_search(param_grid):
        tag = f"lr{params['learning_rate']}_gamma{params['gamma']}_bs{params['batch_size']}_env{params['env_count']}"
        print(f"\n🔁 Launching training with config: {tag}")
        mp.spawn(train, args=(world_size, params), nprocs=world_size, join=True)
        time.sleep(3)  # Safety buffer between sweeps

if __name__ == "__main__":
    run_grid()
