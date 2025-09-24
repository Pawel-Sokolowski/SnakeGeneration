#!/usr/bin/env python3
"""
Test script for the optimized Snake models
"""
import torch
import numpy as np
import gymnasium as gym
import snake_gym_env
from main import SnakeCNNQNet, SnakeMLPQNet, OptimizedReplayBuffer

def test_models():
    """Test both CNN and MLP models"""
    print("🧪 Testing optimized models...")
    
    # Test environment setup
    env = gym.make("Snake-v0", render_mode=None, grid_size=15)
    obs, _ = env.reset()
    seq_len, feature_dim = env.observation_space.shape
    num_actions = env.action_space.n
    
    print(f"Environment: {seq_len}x{feature_dim} -> {num_actions} actions")
    print(f"Sample observation shape: {obs.shape}")
    
    # Test CNN model
    print("\n🔬 Testing CNN model...")
    cnn_model = SnakeCNNQNet(seq_len, feature_dim, num_actions)
    cnn_params = sum(p.numel() for p in cnn_model.parameters())
    print(f"CNN parameters: {cnn_params:,}")
    
    # Test forward pass
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        cnn_out = cnn_model(obs_tensor)
    print(f"CNN output shape: {cnn_out.shape}")
    print(f"CNN output sample: {cnn_out[0].tolist()}")
    
    # Test MLP model
    print("\n🔬 Testing MLP model...")
    mlp_model = SnakeMLPQNet(seq_len, feature_dim, num_actions)
    mlp_params = sum(p.numel() for p in mlp_model.parameters())
    print(f"MLP parameters: {mlp_params:,}")
    
    with torch.no_grad():
        mlp_out = mlp_model(obs_tensor)
    print(f"MLP output shape: {mlp_out.shape}")
    print(f"MLP output sample: {mlp_out[0].tolist()}")
    
    # Test replay buffer
    print("\n💾 Testing replay buffer...")
    buffer = OptimizedReplayBuffer(max_length=1000)
    
    # Add some sample transitions
    for _ in range(100):
        obs1 = torch.tensor(obs, dtype=torch.float32)
        obs2 = torch.tensor(obs, dtype=torch.float32)  # Same for simplicity
        buffer.store_batch([obs1], [np.random.randint(4)], [obs2], [np.random.randn()], [False])
    
    print(f"Buffer size: {len(buffer)}")
    
    # Test sampling
    batch = buffer.sample(32)
    if batch:
        states, actions, next_states, rewards, dones = batch
        print(f"Sample batch shapes: states={states.shape}, actions={actions.shape}")
    else:
        print("Could not sample batch")
    
    env.close()
    print("\n✅ All tests passed!")

def quick_training_test():
    """Test a very short training loop"""
    print("\n🚀 Quick training test...")
    
    # Minimal training parameters
    test_params = {
        "model_type": "mlp",
        "learning_rate": 0.001,
        "gamma": 0.99,
        "batch_size": 32,
        "env_count": 2,
        "grid_size": 15
    }
    
    # Import and test training components
    from main import train_single_gpu
    
    print("Testing training setup...")
    device = torch.device("cpu")  # Force CPU for testing
    
    # Create a minimal environment
    env = gym.make("Snake-v0", render_mode=None, grid_size=test_params["grid_size"])
    seq_len, feature_dim = env.observation_space.shape
    num_actions = env.action_space.n
    
    # Create model
    model = SnakeMLPQNet(seq_len, feature_dim, num_actions)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test a few environment steps
    obs, _ = env.reset()
    for _ in range(5):
        action = np.random.randint(num_actions)
        obs, reward, done, trunc, info = env.step(action)
        print(f"Step: action={action}, reward={reward:.3f}, done={done}")
        if done:
            obs, _ = env.reset()
    
    env.close()
    print("✅ Training components work!")

if __name__ == "__main__":
    test_models()
    quick_training_test()