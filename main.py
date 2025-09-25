#!/usr/bin/env python3
"""
Snake Deep Q-Learning Training Script
Simplified version with only essential training functionality
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import gymnasium as gym
import snake_gym_env


# Simple MLP Q-Network
class SnakeQNet(nn.Module):
    def __init__(self, seq_len, feature_dim, num_actions):
        super().__init__()
        input_size = seq_len * feature_dim
        
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


# Simple Replay Buffer
class ReplayBuffer:
    def __init__(self, max_length=50000):
        self.states = deque(maxlen=max_length)
        self.actions = deque(maxlen=max_length)
        self.next_states = deque(maxlen=max_length)
        self.rewards = deque(maxlen=max_length)
        self.dones = deque(maxlen=max_length)

    def store(self, state, action, next_state, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

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


def select_action(model, state, epsilon, num_actions, device):
    """Select action using epsilon-greedy policy"""
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    else:
        with torch.no_grad():
            q_values = model(state.unsqueeze(0).to(device))
            return torch.argmax(q_values).item()


def train_step(buffer, model, target_model, optimizer, loss_fn, gamma, batch_size, device):
    """Single training step"""
    batch_data = buffer.sample(batch_size)
    if batch_data is None:
        return 0.0
        
    states, actions, next_states, rewards, dones = batch_data
    states = states.to(device)
    next_states = next_states.to(device)
    actions = actions.to(device).unsqueeze(1)
    rewards = rewards.to(device)
    dones = dones.to(device)

    # Compute target Q-values
    with torch.no_grad():
        target_q = target_model(next_states)
        max_q = target_q.max(dim=1)[0]
        targets = rewards + gamma * max_q * (1 - dones)

    # Compute current Q-values
    q_values = model(states)
    q_action = q_values.gather(1, actions).squeeze()
    
    # Compute loss
    loss = loss_fn(q_action, targets)

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()


def train_model(grid_size=20, episodes=1000, learning_rate=0.001, batch_size=256):
    """
    Train a Snake AI model
    
    Args:
        grid_size: Size of the game grid
        episodes: Number of training episodes
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = gym.make("Snake-v0", render_mode=None, grid_size=grid_size)
    seq_len, feature_dim = env.observation_space.shape
    num_actions = env.action_space.n
    
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Observation space: {seq_len} x {feature_dim}")
    print(f"Action space: {num_actions}")

    # Create models
    model = SnakeQNet(seq_len, feature_dim, num_actions).to(device)
    target_model = SnakeQNet(seq_len, feature_dim, num_actions).to(device)
    target_model.load_state_dict(model.state_dict())
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.HuberLoss()
    buffer = ReplayBuffer()

    # Training parameters
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    gamma = 0.99
    train_freq = 4
    sync_freq = 500
    max_steps = 200
    
    rewards_history = []
    start_time = time.time()
    
    print(f"\nStarting training for {episodes} episodes...")

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state.astype(np.float32))
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action
            action = select_action(model, state, epsilon, num_actions, device)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.tensor(next_state.astype(np.float32))
            
            # Store transition
            buffer.store(state, action, next_state, reward, done)
            
            episode_reward += reward
            state = next_state
            
            # Train model
            if len(buffer) >= batch_size and step % train_freq == 0:
                loss = train_step(buffer, model, target_model, optimizer, loss_fn, gamma, batch_size, device)
            
            # Update target network
            if step % sync_freq == 0:
                target_model.load_state_dict(model.state_dict())
            
            if done:
                break
        
        # Update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Record reward
        rewards_history.append(episode_reward)
        
        # Print progress
        if episode % 100 == 0 or episode < 10:
            avg_reward = np.mean(rewards_history[-100:])
            elapsed = time.time() - start_time
            print(f"Episode {episode:4d} | Reward: {episode_reward:6.2f} | Avg100: {avg_reward:6.2f} | "
                  f"Epsilon: {epsilon:.4f} | Buffer: {len(buffer):5d} | Time: {elapsed:.1f}s")

        # Early stopping if solved
        if len(rewards_history) >= 100:
            avg_100 = np.mean(rewards_history[-100:])
            target_score = max(10, grid_size)  # Adaptive target based on grid size
            if avg_100 > target_score:
                print(f"\n🎉 Solved at episode {episode}! Average reward: {avg_100:.2f}")
                break

    env.close()
    
    # Save model
    os.makedirs("saved_models", exist_ok=True)
    model_name = f"snake_model_grid{grid_size}_ep{episode+1}.pt"
    model_path = os.path.join("saved_models", model_name)
    
    # Save with metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_params': {
            'seq_len': seq_len,
            'feature_dim': feature_dim,
            'num_actions': num_actions
        },
        'training_info': {
            'grid_size': grid_size,
            'episodes': episode + 1,
            'final_reward': rewards_history[-1],
            'avg_reward': np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
        }
    }, model_path)
    
    print(f"\n✅ Model saved to: {model_path}")
    
    # Create and save learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, alpha=0.7, label='Episode Reward')
    if len(rewards_history) >= 50:
        smoothed = np.convolve(rewards_history, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(rewards_history)), smoothed, label='Moving Average (50)', color='red', linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Snake Training Progress - Grid {grid_size}x{grid_size}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_name = f"snake_model_grid{grid_size}_ep{episode+1}.png"
    plot_path = os.path.join("saved_models", plot_name)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"📊 Learning curve saved to: {plot_path}")
    
    return model_path, np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)


if __name__ == "__main__":
    print("🐍 Snake Deep Q-Learning Training")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Training configuration - modify these as needed
    GRID_SIZE = 20
    EPISODES = 1000
    LEARNING_RATE = 0.001
    BATCH_SIZE = 256
    
    print(f"\nTraining Configuration:")
    print(f"Grid Size: {GRID_SIZE}")
    print(f"Episodes: {EPISODES}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    
    # Train the model
    model_path, final_reward = train_model(
        grid_size=GRID_SIZE,
        episodes=EPISODES,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE
    )
    
    print(f"\n🏆 Training completed!")
    print(f"Final average reward: {final_reward:.2f}")
    print(f"Model saved to: {model_path}")