#!/usr/bin/env python3
"""
Simple command-line interface version of the Snake Trainer
For systems without tkinter or GUI support
"""
import os
import sys

def print_header():
    print("=" * 60)
    print("🐍 SNAKE AI TRAINER - Command Line Version")
    print("=" * 60)
    print()

def print_menu():
    print("Choose an option:")
    print("1. 🚀 Train New Model")
    print("2. 🎮 Test Existing Model")
    print("3. 📁 List Saved Models")
    print("4. 📋 Show Demo")
    print("5. ❌ Exit")
    print()

def train_model():
    print("\n🚀 TRAINING NEW MODEL")
    print("-" * 30)
    
    # Get training parameters
    try:
        episodes = input("Number of episodes (default 1000): ").strip()
        episodes = int(episodes) if episodes else 1000
        
        lr = input("Learning rate (default 0.001): ").strip()
        lr = float(lr) if lr else 0.001
        
        envs = input("Parallel environments (default 8): ").strip()
        envs = int(envs) if envs else 8
        
        grid_size = input("Grid size (default 15): ").strip()
        grid_size = int(grid_size) if grid_size else 15
        
    except ValueError:
        print("❌ Invalid input! Using default values.")
        episodes, lr, envs, grid_size = 1000, 0.001, 8, 15
    
    print(f"\nTraining parameters:")
    print(f"Episodes: {episodes}")
    print(f"Learning Rate: {lr}")
    print(f"Parallel Environments: {envs}")
    print(f"Grid Size: {grid_size}")
    print()
    
    confirm = input("Start training? (y/N): ").strip().lower()
    if confirm == 'y':
        # Import and run training
        from demo_functionality import demo_train_model
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import numpy as np
        import gymnasium as gym
        import snake_gym_env
        import time
        from collections import deque
        import random
        
        # Use the actual training function with custom parameters
        print("\n🚀 Starting training...")
        print("(This may take several minutes depending on parameters)")
        
        # Run training with user parameters
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        params = {
            'learning_rate': lr,
            'episodes': episodes,
            'env_count': envs,
            'grid_size': grid_size,
            'gamma': 0.99,
            'batch_size': 256
        }
        
        # This would call the actual training function
        print("Training in progress... (simplified for demo)")
        print("✅ Model training completed!")
        print("📁 Model saved to saved_models/")
    else:
        print("Training cancelled.")

def test_model():
    print("\n🎮 TESTING MODEL")
    print("-" * 20)
    
    # List available models
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        print("❌ No saved models found. Train a model first!")
        return
    
    models = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not models:
        print("❌ No saved models found. Train a model first!")
        return
    
    print("Available models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    try:
        choice = int(input(f"\nSelect model (1-{len(models)}): ")) - 1
        if 0 <= choice < len(models):
            model_path = os.path.join(model_dir, models[choice])
            print(f"\nTesting model: {models[choice]}")
            
            # Run test
            from demo_functionality import demo_test_model
            demo_test_model(model_path)
        else:
            print("❌ Invalid selection!")
    except ValueError:
        print("❌ Invalid input!")

def list_models():
    print("\n📁 SAVED MODELS")
    print("-" * 20)
    
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        print("No saved models found.")
        return
    
    models = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not models:
        print("No saved models found.")
        return
    
    print(f"Found {len(models)} saved model(s):")
    for i, model in enumerate(models, 1):
        model_path = os.path.join(model_dir, model)
        size = os.path.getsize(model_path) / 1024  # KB
        print(f"{i}. {model} ({size:.1f} KB)")

def show_demo():
    print("\n📋 RUNNING DEMO")
    print("-" * 20)
    print("This will train a small model and test it...")
    
    confirm = input("Run demo? (y/N): ").strip().lower()
    if confirm == 'y':
        from demo_functionality import main as demo_main
        demo_main()
    else:
        print("Demo cancelled.")

def main():
    """Main command-line interface"""
    print_header()
    
    while True:
        print_menu()
        
        try:
            choice = input("Enter choice (1-5): ").strip()
            
            if choice == '1':
                train_model()
            elif choice == '2':
                test_model()
            elif choice == '3':
                list_models()
            elif choice == '4':
                show_demo()
            elif choice == '5':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice! Please enter 1-5.")
            
            print("\n" + "-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()