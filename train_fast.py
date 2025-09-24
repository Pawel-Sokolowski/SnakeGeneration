#!/usr/bin/env python3
"""
Fast training script for GTX 1070
Runs the most efficient configuration for quick results
"""
from main import train_single_gpu, param_grid_search
import torch

def main():
    """Train the fastest configuration for quick results"""
    print("🚀 Fast Snake Training for GTX 1070")
    print("Using the most efficient configuration for quick results")
    print("=" * 60)
    
    # Best configuration for speed vs performance
    fast_config = {
        "model_type": "mlp",      # Fastest model
        "learning_rate": 0.001,   # Good learning rate
        "gamma": 0.99,           # Standard discount
        "batch_size": 256,       # Efficient batch size
        "env_count": 8,          # Balanced environment count
        "grid_size": 15          # Smaller grid for speed
    }
    
    print("Configuration:")
    for key, value in fast_config.items():
        print(f"  {key}: {value}")
    
    print(f"\nExpected training time: 30-60 minutes on GTX 1070")
    print(f"Target performance: 15+ average score")
    print(f"Memory usage: ~1.4MB VRAM")
    
    try:
        print(f"\n🏁 Starting training...")
        model_path, final_reward = train_single_gpu(fast_config)
        
        print(f"\n🎉 Training completed!")
        print(f"Final performance: {final_reward:.2f}")
        print(f"Model saved: {model_path}")
        
        if final_reward > 10:
            print("✅ Great performance! The agent learned to play Snake well.")
        elif final_reward > 5:
            print("👍 Good performance! The agent shows promising learning.")
        else:
            print("🔄 The agent is still learning. You might want to train longer.")
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("Try running 'python demo.py' first to test the setup")

if __name__ == "__main__":
    main()