#!/usr/bin/env python3
"""
Demo script showing the Snake AI training and testing functionality
This works without a GUI to demonstrate the core features
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

# Import or define the core components
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

def demo_train_model():
    """Demo: Train a Snake AI model"""
    print("🚀 DEMO: Training Snake AI Model")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Training parameters
    params = {
        'learning_rate': 0.001,
        'episodes': 50,  # Short demo training
        'env_count': 4,
        'grid_size': 10,
        'gamma': 0.99,
        'batch_size': 64
    }
    
    print(f"Training parameters: {params}")
    
    # Create environment
    envs = [gym.make("Snake-v0", render_mode=None, grid_size=params["grid_size"]) 
           for _ in range(params["env_count"])]
    seq_len, feature_dim = envs[0].observation_space.shape
    num_actions = envs[0].action_space.n
    
    print(f"Environment: {seq_len}x{feature_dim} obs, {num_actions} actions")
    
    # Create model
    model = SnakeMLPQNet(seq_len, feature_dim, num_actions).to(device)
    target_model = SnakeMLPQNet(seq_len, feature_dim, num_actions).to(device)
    target_model.load_state_dict(model.state_dict())
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_fn = nn.HuberLoss()
    buffer = OptimizedReplayBuffer()
    
    # Training parameters
    eps, eps_min, eps_decay = 1.0, 0.05, 0.999
    gamma, batch_size = params["gamma"], params["batch_size"]
    train_freq, sync_freq, max_steps = 4, 100, 100
    
    step_count = 0
    rewards_history = deque(maxlen=100)
    
    # Initialize observations
    obs_batch = []
    for env in envs:
        obs, _ = env.reset()
        obs_batch.append(torch.tensor(obs.astype(np.float32)))
    
    print(f"\nTraining for {params['episodes']} episodes...")
    start_time = time.time()
    
    # Training loop
    for episode in range(params["episodes"]):
        ep_reward = [0.0] * len(envs)
        
        for step in range(max_steps):
            step_count += 1
            
            # Select actions
            obs_tensor = torch.stack(obs_batch)
            actions = select_action(model, obs_tensor, eps, num_actions, device)
            
            # Take environment steps
            next_obs_batch = []
            rewards = []
            dones = []
            
            for i, (env, action) in enumerate(zip(envs, actions)):
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                next_obs_batch.append(torch.tensor(next_obs.astype(np.float32)))
                rewards.append(reward)
                dones.append(done)
                ep_reward[i] += reward
                
                # Store transition
                buffer.push([obs_batch[i]], [action], [next_obs_batch[i]], [reward], [float(done)])
                
                if done:
                    obs, _ = env.reset()
                    next_obs_batch[i] = torch.tensor(obs.astype(np.float32))
            
            obs_batch = next_obs_batch
            
            # Training
            if step_count % train_freq == 0:
                loss = train_step(buffer, model, target_model, optimizer, loss_fn, gamma, batch_size, device)
            
            # Update target network
            if step_count % sync_freq == 0:
                target_model.load_state_dict(model.state_dict())
            
            # Decay epsilon
            eps = max(eps_min, eps * eps_decay)
        
        # Episode complete
        avg_reward = np.mean(ep_reward)
        rewards_history.append(avg_reward)
        
        if (episode + 1) % 10 == 0:
            avg_100 = np.mean(list(rewards_history)) if rewards_history else 0
            print(f"Episode {episode + 1}: Avg reward = {avg_reward:.2f}, Last 100 avg = {avg_100:.2f}")
    
    # Save model
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    timestamp = int(time.time())
    model_name = f"demo_snake_model_{timestamp}.pt"
    model_path = os.path.join(model_dir, model_name)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_params': {
            'seq_len': seq_len,
            'feature_dim': feature_dim,
            'num_actions': num_actions
        },
        'training_params': params,
        'final_reward': np.mean(list(rewards_history)) if rewards_history else 0,
    }, model_path, pickle_protocol=2)  # Use older protocol for compatibility
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time:.1f} seconds")
    print(f"📁 Model saved: {model_name}")
    print(f"🎯 Final average reward: {np.mean(list(rewards_history)):.2f}")
    
    # Close environments
    for env in envs:
        env.close()
    
    return model_path

def demo_test_model(model_path):
    """Demo: Test a trained Snake AI model"""
    print(f"\n🎮 DEMO: Testing Snake AI Model")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_params = checkpoint['model_params']
    model = SnakeMLPQNet(model_params['seq_len'], model_params['feature_dim'], model_params['num_actions']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded: {os.path.basename(model_path)}")
    print(f"Training final reward: {checkpoint.get('final_reward', 'N/A')}")
    
    # Create test environment
    grid_size = checkpoint.get('training_params', {}).get('grid_size', 15)
    env = gym.make("Snake-v0", render_mode=None, grid_size=grid_size)
    
    # Run test episodes
    num_episodes = 5
    all_scores = []
    all_steps = []
    
    print(f"\nRunning {num_episodes} test episodes...")
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs.astype(np.float32)).unsqueeze(0).to(device)
        score = 0
        steps = 0
        
        for step in range(500):  # Max steps per episode
            with torch.no_grad():
                q_values = model(obs)
                action = torch.argmax(q_values).item()
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = torch.tensor(next_obs.astype(np.float32)).unsqueeze(0).to(device)
            steps += 1
            
            if reward > 0:  # Snake ate food
                score += 1
            
            if terminated or truncated:
                break
        
        all_scores.append(score)
        all_steps.append(steps)
        print(f"Episode {ep + 1}: Score = {score}, Steps = {steps}")
    
    # Display final statistics
    print(f"\n📊 Test Results:")
    print(f"Average Score: {np.mean(all_scores):.1f}")
    print(f"Best Score: {np.max(all_scores)}")
    print(f"Average Steps: {np.mean(all_steps):.1f}")
    
    # Interpret results
    avg_score = np.mean(all_scores)
    if avg_score >= 5:
        print("🎉 Excellent! The AI learned to play Snake well!")
    elif avg_score >= 2:
        print("👍 Good! The AI shows decent Snake-playing ability.")
    elif avg_score >= 1:
        print("🤔 Okay. The AI learned some basic strategies.")
    else:
        print("😐 The AI needs more training to improve.")
    
    env.close()

def main():
    """Run the complete demo"""
    print("🐍 Snake AI Trainer Demo")
    print("This demonstrates the core functionality that the Windows GUI app provides")
    print()
    
    # Demo training
    model_path = demo_train_model()
    
    # Demo testing
    demo_test_model(model_path)
    
    print(f"\n✅ Demo completed!")
    print(f"\n💡 In the full Windows GUI app, you can:")
    print(f"   • Train models with custom parameters using a simple interface")
    print(f"   • Watch the training progress in real-time")
    print(f"   • Test multiple models and compare their performance")
    print(f"   • Watch animated gameplay of your trained AI")
    print(f"   • Save and load models easily")
    print(f"\nTo use the full app, run: python snake_trainer_app.py")

if __name__ == "__main__":
    main()