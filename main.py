import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
from collections import deque
import matplotlib.pyplot as plt
import gymnasium as gym
import snake_gym_env  # Make sure this module is available

# 🧠 Lightweight CNN Q-Network optimized for GTX 1070
class SnakeCNNQNet(nn.Module):
    def __init__(self, seq_len, feature_dim, num_actions):
        super().__init__()
        # Convert sequence to spatial representation for CNN
        # seq_len = max_snake_length + 1, feature_dim = 2 (x,y coordinates)
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Simple embedding for coordinates
        self.coord_embed = nn.Linear(feature_dim, 8)
        
        # 1D CNN to process sequence
        self.conv1 = nn.Conv1d(8, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(16)
        
        # Final layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # x shape: (batch, seq_len, feature_dim)
        x = self.coord_embed(x)  # (batch, seq_len, 8)
        x = x.transpose(1, 2)    # (batch, 8, seq_len)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)         # (batch, 64, 16)
        x = x.view(batch_size, -1)
        return self.fc(x)

# 🧠 Simple MLP Q-Network (backup option)
class SnakeMLPQNet(nn.Module):
    def __init__(self, seq_len, feature_dim, num_actions):
        super().__init__()
        input_size = seq_len * feature_dim
        
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

# 🎒 Optimized Replay Buffer for single GPU
class OptimizedReplayBuffer:
    def __init__(self, max_length=50_000):  # Reduced buffer size
        self.states = deque(maxlen=max_length)
        self.actions = deque(maxlen=max_length)
        self.next_states = deque(maxlen=max_length)
        self.rewards = deque(maxlen=max_length)
        self.dones = deque(maxlen=max_length)

    def store_batch(self, s, a, s_, r, d):
        for i in range(len(s)):
            self.states.append(s[i])
            self.actions.append(a[i])
            self.next_states.append(s_[i])
            self.rewards.append(r[i])
            self.dones.append(d[i])

    def sample(self, batch_size):
        if len(self.dones) < batch_size:
            return None
        idx = np.random.choice(len(self.dones), batch_size, replace=False)
        
        return (
            torch.stack([self.states[i] for i in idx]),
            torch.tensor([self.actions[i] for i in idx], dtype=torch.int64),
            torch.stack([self.next_states[i] for i in idx]),
            torch.tensor([self.rewards[i] for i in idx], dtype=torch.float32),
            torch.tensor([self.dones[i] for i in idx], dtype=torch.float32)
        )

    def __len__(self):
        return len(self.dones)

# 🎛️ Optimized Parameter Grid for GTX 1070
param_grid = {
    "model_type": ["cnn", "mlp"],  # Two model options
    "learning_rate": [0.001, 0.0005],
    "gamma": [0.99],
    "batch_size": [256, 512],  # Much smaller batch sizes
    "env_count": [8, 16],      # Fewer parallel environments
    "grid_size": [15, 20]      # Smaller grid sizes
}

def param_grid_search(grid):
    keys, values = zip(*grid.items())
    for combo in product(*values):
        yield dict(zip(keys, combo))

# 🎯 Optimized Action Selection
def select_action(model, obs_batch, epsilon, num_actions, device):
    actions = []
    with torch.no_grad():
        q_vals = model(obs_batch.to(device))
        for i in range(len(obs_batch)):
            if random.random() < epsilon:
                actions.append(random.randint(0, num_actions - 1))
            else:
                actions.append(torch.argmax(q_vals[i]).item())
    return actions

# 🔧 Simplified Training Step
def train_step(buffer, model, target_model, optimizer, loss_fn, gamma, batch_size, device):
    batch_data = buffer.sample(batch_size)
    if batch_data is None:
        return 0.0
        
    s, a, s_, r, d = batch_data
    s, s_, a = s.to(device), s_.to(device), a.to(device).unsqueeze(1)
    r, d = r.to(device), d.to(device)

    with torch.no_grad():
        target_q = target_model(s_)
        max_q = target_q.max(dim=1)[0]
        targets = r + gamma * max_q * (1 - d)

    q_vals = model(s)
    q_action = q_vals.gather(1, a).squeeze()
    loss = loss_fn(q_action, targets)

    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()

# 💾 Memory monitoring
def get_memory_usage(device):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        return allocated, reserved
    return 0, 0

# 🚀 Single GPU Training Function for GTX 1070
def train_single_gpu(params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU (training will be slower)")
        print("   For GTX 1070 training, ensure CUDA drivers are installed")
    
    # Create environments with smaller grid size
    envs = [gym.make("Snake-v0", render_mode=None, grid_size=params["grid_size"]) for _ in range(params["env_count"])]
    seq_len, feature_dim = envs[0].observation_space.shape
    num_actions = envs[0].action_space.n
    
    print(f"Observation space: {seq_len} x {feature_dim}")
    print(f"Action space: {num_actions}")

    # Create model based on type
    if params["model_type"] == "cnn":
        model = SnakeCNNQNet(seq_len, feature_dim, num_actions).to(device)
        target_model = SnakeCNNQNet(seq_len, feature_dim, num_actions).to(device)
    else:  # mlp
        model = SnakeMLPQNet(seq_len, feature_dim, num_actions).to(device)
        target_model = SnakeMLPQNet(seq_len, feature_dim, num_actions).to(device)
    
    target_model.load_state_dict(model.state_dict())
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / 1024 / 1024
    print(f"Model parameters: {total_params:,} ({model_size_mb:.1f} MB)")

    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_fn = nn.HuberLoss()
    buffer = OptimizedReplayBuffer()

    # Training parameters optimized for faster convergence
    eps, eps_min, eps_decay = 1.0, 0.05, 0.999995  # Faster epsilon decay
    gamma, batch_size = params["gamma"], params["batch_size"]
    train_freq, sync_freq, max_steps = 4, 500, 200  # Reduced steps
    
    # Adjust episodes based on device (CPU needs fewer episodes for testing)
    n_episodes = 2000 if device.type == "cuda" else 500
    
    step_count = 0
    rewards_history, all_rewards = deque(maxlen=100), []
    
    # Initialize observations as tensors
    obs_batch = []
    for env in envs:
        obs, _ = env.reset()
        obs_batch.append(torch.tensor(obs.astype(np.float32)))
    
    start_time = time.time()
    print(f"\nStarting training with {params['env_count']} environments for {n_episodes} episodes...")

    for ep in range(n_episodes):
        ep_reward, ep_start = [0.0] * len(envs), time.time()
        
        for step in range(max_steps):
            step_count += 1
            
            # Stack observations for batch processing
            obs_tensor = torch.stack(obs_batch)
            actions = select_action(model, obs_tensor, eps, num_actions, device)
            eps = max(eps * eps_decay, eps_min)

            next_obs_batch, rewards, dones = [], [], []
            for i, env in enumerate(envs):
                obs2, r, term, trunc, _ = env.step(actions[i])
                done = term or trunc
                next_obs_batch.append(torch.tensor(obs2.astype(np.float32)))
                rewards.append(r)
                dones.append(done)
                ep_reward[i] += r
                
                if done:
                    obs2, _ = env.reset()
                    next_obs_batch[i] = torch.tensor(obs2.astype(np.float32))

            # Store transitions in buffer
            buffer.store_batch(obs_batch, actions, next_obs_batch, rewards, dones)
            obs_batch = next_obs_batch

            # Training step
            if step_count % train_freq == 0 and len(buffer) >= batch_size:
                loss = train_step(buffer, model, target_model, optimizer, loss_fn, gamma, batch_size, device)

            # Update target network
            if step_count % sync_freq == 0:
                target_model.load_state_dict(model.state_dict())

        avg_reward = np.mean(ep_reward)
        rewards_history.append(avg_reward)
        all_rewards.append(avg_reward)
        
        # Progress reporting
        if ep % 20 == 0 or ep < 10:
            elapsed = time.time() - start_time
            mem_alloc, mem_reserved = get_memory_usage(device)
            avg_100 = np.mean(list(rewards_history))
            print(f"Ep {ep:4d} | Avg: {avg_reward:6.2f} | Avg100: {avg_100:6.2f} | Eps: {eps:.4f} | Buffer: {len(buffer):5d} | Mem: {mem_alloc:.1f}MB | Time: {elapsed:.1f}s")

        # Early stopping condition (adjusted for smaller grid)
        target_score = 15 if params["grid_size"] <= 15 else 25
        if len(rewards_history) >= 50 and np.mean(rewards_history) > target_score:
            print(f"🎉 Solved at episode {ep}! Average reward: {np.mean(rewards_history):.2f}")
            break

    # Save model and plot
    os.makedirs("saved_models", exist_ok=True)
    tag = f"{params['model_type']}_lr{params['learning_rate']}_bs{params['batch_size']}_env{params['env_count']}_grid{params['grid_size']}"
    model_path = f"saved_models/snake_{tag}.pt"
    plot_path = f"saved_models/curve_{tag}.png"

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # Create training curve
    plt.figure(figsize=(10, 6))
    plt.plot(all_rewards, label="Episode Reward", alpha=0.7)
    if len(all_rewards) >= 50:
        smoothed = np.convolve(all_rewards, np.ones(50)/50, mode="valid")
        plt.plot(range(49, len(all_rewards)), smoothed, label="Moving Avg (50)", color="red", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"Snake Training Progress - {params['model_type'].upper()}\nGrid: {params['grid_size']}x{params['grid_size']}, Envs: {params['env_count']}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {plot_path}")
    
    # Clean up environments
    for env in envs:
        env.close()
    
    return model_path, np.mean(list(rewards_history))
def run_optimized_training():
    """Run training with multiple configurations optimized for GTX 1070"""
    results = []
    
    for params in param_grid_search(param_grid):
        tag = f"{params['model_type']}_lr{params['learning_rate']}_bs{params['batch_size']}_env{params['env_count']}_grid{params['grid_size']}"
        print(f"\n🚀 Training configuration: {tag}")
        print("=" * 60)
        
        try:
            model_path, final_reward = train_single_gpu(params)
            results.append((tag, final_reward, model_path))
            print(f"✅ Completed: {tag} -> Final reward: {final_reward:.2f}")
        except Exception as e:
            print(f"❌ Failed: {tag} -> Error: {str(e)}")
        
        # Small break between runs
        time.sleep(2)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print summary
    print(f"\n{'='*60}")
    print("🏆 TRAINING SUMMARY")
    print(f"{'='*60}")
    if results:
        # Sort by performance
        results.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Rank':<5} {'Configuration':<40} {'Final Reward':<15} {'Model Path'}")
        print("-" * 80)
        for i, (config, reward, path) in enumerate(results, 1):
            print(f"{i:<5} {config:<40} {reward:<15.2f} {os.path.basename(path)}")
    else:
        print("No successful training runs completed.")

if __name__ == "__main__":
    print("🐍 Snake DQN Training - GTX 1070 Optimized")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    run_optimized_training()
