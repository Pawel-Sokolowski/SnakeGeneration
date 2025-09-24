#!/usr/bin/env python3
"""
Simple test script to verify training and testing functionality
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import snake_gym_env
import os
import time
from collections import deque
import random

# Simple MLP Model
class SnakeMLPQNet(nn.Module):
    def __init__(self, seq_len, feature_dim, num_actions):
        super().__init__()
        self.flatten_size = seq_len * feature_dim
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_actions)
        )
    
    def forward(self, x):
        return self.fc(x.view(x.shape[0], -1))

# Simple Replay Buffer
class OptimizedReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

    def push(self, s, a, s_, r, d):
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
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()

def test_training():
    """Test the training functionality"""
    print("🧪 Testing Snake Training Functionality")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = gym.make("Snake-v0", render_mode=None, grid_size=10)
    seq_len, feature_dim = env.observation_space.shape
    num_actions = env.action_space.n
    
    print(f"Environment: {seq_len}x{feature_dim} obs, {num_actions} actions")
    
    # Create model
    model = SnakeMLPQNet(seq_len, feature_dim, num_actions).to(device)
    target_model = SnakeMLPQNet(seq_len, feature_dim, num_actions).to(device)
    target_model.load_state_dict(model.state_dict())
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.HuberLoss()
    buffer = OptimizedReplayBuffer(capacity=5000)
    
    # Quick training test (just a few episodes)
    print("\n🚀 Starting quick training test...")
    
    eps = 0.5
    rewards_history = []
    
    for episode in range(10):
        obs, _ = env.reset()
        obs_tensor = torch.tensor(obs.astype(np.float32)).unsqueeze(0)
        ep_reward = 0
        
        for step in range(100):  # Max 100 steps per episode
            # Select action
            action = select_action(model, obs_tensor, eps, num_actions, device)[0]
            
            # Take step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs_tensor = torch.tensor(next_obs.astype(np.float32)).unsqueeze(0)
            done = terminated or truncated
            
            # Store transition
            buffer.push([obs_tensor.squeeze(0)], [action], [next_obs_tensor.squeeze(0)], [reward], [float(done)])
            
            ep_reward += reward
            obs_tensor = next_obs_tensor
            
            # Train
            if len(buffer) > 100:
                loss = train_step(buffer, model, target_model, optimizer, loss_fn, 0.99, 32, device)
            
            if done:
                break
        
        rewards_history.append(ep_reward)
        print(f"Episode {episode + 1}: Reward = {ep_reward:.2f}")
    
    print(f"\nAverage reward over {len(rewards_history)} episodes: {np.mean(rewards_history):.2f}")
    
    # Test saving model
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, "test_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_params': {
            'seq_len': seq_len,
            'feature_dim': feature_dim,
            'num_actions': num_actions
        },
        'final_reward': np.mean(rewards_history)
    }, model_path)
    
    print(f"✅ Model saved to: {model_path}")
    
    # Test loading and testing model
    print("\n🎮 Testing model loading and evaluation...")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    test_model = SnakeMLPQNet(checkpoint['model_params']['seq_len'], 
                             checkpoint['model_params']['feature_dim'], 
                             checkpoint['model_params']['num_actions']).to(device)
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model.eval()
    
    # Test the model
    test_scores = []
    for test_ep in range(3):
        obs, _ = env.reset()
        obs = torch.tensor(obs.astype(np.float32)).unsqueeze(0).to(device)
        score = 0
        
        for step in range(200):
            with torch.no_grad():
                q_values = test_model(obs)
                action = torch.argmax(q_values).item()
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = torch.tensor(next_obs.astype(np.float32)).unsqueeze(0).to(device)
            
            if reward > 0:
                score += 1
            
            if terminated or truncated:
                break
        
        test_scores.append(score)
        print(f"Test Episode {test_ep + 1}: Score = {score}, Steps = {step + 1}")
    
    print(f"Average test score: {np.mean(test_scores):.1f}")
    print("\n✅ All tests passed!")
    
    env.close()

if __name__ == "__main__":
    test_training()