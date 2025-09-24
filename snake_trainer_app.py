#!/usr/bin/env python3
"""
Snake Training App - Windows GUI Application
Simple interface to train and test Snake AI models
"""
import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import gymnasium as gym
import snake_gym_env
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plots

# Import training components from main.py
try:
    from main import SnakeMLPQNet, OptimizedReplayBuffer, select_action, train_step
except ImportError:
    # Define simple versions if main.py imports fail
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

MODEL_DIR = "saved_models"

class SnakeTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Snake AI Trainer")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Training state
        self.training_thread = None
        self.is_training = False
        self.training_cancelled = False
        
        # Model state
        self.current_model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize UI
        self.setup_ui()
        self.update_model_list()
        
        # Setup device info
        device_info = f"Device: {self.device}"
        if torch.cuda.is_available():
            device_info += f" ({torch.cuda.get_device_name()})"
        else:
            device_info += " (CPU - Training will be slower)"
        
        self.device_label.config(text=device_info)

    def setup_ui(self):
        # Create main notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Training tab
        self.train_frame = ttk.Frame(notebook)
        notebook.add(self.train_frame, text="Train Model")
        self.setup_training_tab()
        
        # Testing tab
        self.test_frame = ttk.Frame(notebook)
        notebook.add(self.test_frame, text="Test Model")
        self.setup_testing_tab()
        
        # Status bar
        status_frame = tk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        self.device_label = tk.Label(status_frame, text="Device: Loading...", anchor=tk.W)
        self.device_label.pack(side=tk.LEFT)

    def setup_training_tab(self):
        # Training controls
        controls_frame = tk.LabelFrame(self.train_frame, text="Training Configuration", padx=10, pady=10)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Training parameters
        params_frame = tk.Frame(controls_frame)
        params_frame.pack(fill=tk.X)
        
        # Learning rate
        tk.Label(params_frame, text="Learning Rate:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.lr_var = tk.StringVar(value="0.001")
        lr_combo = ttk.Combobox(params_frame, textvariable=self.lr_var, values=["0.0005", "0.001", "0.002"], width=10)
        lr_combo.grid(row=0, column=1, sticky=tk.W)
        
        # Episodes
        tk.Label(params_frame, text="Episodes:").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.episodes_var = tk.StringVar(value="1000")
        episodes_combo = ttk.Combobox(params_frame, textvariable=self.episodes_var, values=["500", "1000", "2000"], width=10)
        episodes_combo.grid(row=0, column=3, sticky=tk.W)
        
        # Environments
        tk.Label(params_frame, text="Parallel Envs:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.envs_var = tk.StringVar(value="8")
        envs_combo = ttk.Combobox(params_frame, textvariable=self.envs_var, values=["4", "8", "16"], width=10)
        envs_combo.grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        # Grid size
        tk.Label(params_frame, text="Grid Size:").grid(row=1, column=2, sticky=tk.W, padx=(20, 10), pady=(10, 0))
        self.grid_var = tk.StringVar(value="15")
        grid_combo = ttk.Combobox(params_frame, textvariable=self.grid_var, values=["10", "15", "20"], width=10)
        grid_combo.grid(row=1, column=3, sticky=tk.W, pady=(10, 0))
        
        # Training buttons
        button_frame = tk.Frame(controls_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.train_button = tk.Button(button_frame, text="🚀 Train Model", command=self.start_training, 
                                     bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), height=2)
        self.train_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = tk.Button(button_frame, text="⏹️ Stop Training", command=self.stop_training, 
                                    bg="#f44336", fg="white", font=("Arial", 12), height=2, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)
        
        # Progress
        progress_frame = tk.LabelFrame(self.train_frame, text="Training Progress", padx=10, pady=10)
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.progress_var = tk.StringVar(value="Ready to train")
        self.progress_label = tk.Label(progress_frame, textvariable=self.progress_var, anchor=tk.W)
        self.progress_label.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Training log
        self.log_text = scrolledtext.ScrolledText(progress_frame, height=15, width=80)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def setup_testing_tab(self):
        # Model selection
        model_frame = tk.LabelFrame(self.test_frame, text="Model Selection", padx=10, pady=10)
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(model_frame, text="Select Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, width=50)
        self.model_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        refresh_btn = tk.Button(model_frame, text="Refresh", command=self.update_model_list)
        refresh_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Test controls
        test_controls_frame = tk.Frame(self.test_frame)
        test_controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.test_button = tk.Button(test_controls_frame, text="🎮 Test Model", command=self.test_model,
                                    bg="#2196F3", fg="white", font=("Arial", 12, "bold"), height=2)
        self.test_button.pack(side=tk.LEFT)
        
        # Results display
        results_frame = tk.LabelFrame(self.test_frame, text="Test Results", padx=10, pady=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Split into two columns: video and stats
        display_frame = tk.Frame(results_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video display
        video_frame = tk.LabelFrame(display_frame, text="Game Playback")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.game_canvas = tk.Label(video_frame, text="Test a model to see gameplay", 
                                   width=50, height=20, bg='#222', fg='white')
        self.game_canvas.pack(pady=10)
        
        # Stats display
        stats_frame = tk.LabelFrame(display_frame, text="Performance Stats")
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, width=40, height=20)
        self.stats_text.pack(padx=10, pady=10)

    def update_model_list(self):
        """Update the list of available models"""
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            models = []
        else:
            models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]
        
        self.model_combo['values'] = models
        if models:
            self.model_var.set(models[0])

    def log_message(self, message):
        """Add a message to the training log"""
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def start_training(self):
        """Start model training in a separate thread"""
        if self.is_training:
            messagebox.showwarning("Training", "Training is already in progress!")
            return
        
        # Get training parameters
        try:
            params = {
                'learning_rate': float(self.lr_var.get()),
                'episodes': int(self.episodes_var.get()),
                'env_count': int(self.envs_var.get()),
                'grid_size': int(self.grid_var.get()),
                'gamma': 0.99,
                'batch_size': 256
            }
        except ValueError:
            messagebox.showerror("Error", "Invalid training parameters!")
            return
        
        # Update UI state
        self.is_training = True
        self.training_cancelled = False
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_bar['value'] = 0
        self.log_text.delete(1.0, tk.END)
        
        # Start training thread
        self.training_thread = threading.Thread(target=self._train_model, args=(params,))
        self.training_thread.daemon = True
        self.training_thread.start()

    def stop_training(self):
        """Stop the current training"""
        self.training_cancelled = True
        self.log_message("Training stop requested...")

    def _train_model(self, params):
        """Main training loop (runs in separate thread)"""
        try:
            self.log_message(f"Starting training with parameters: {params}")
            self.log_message(f"Using device: {self.device}")
            
            # Create environment
            envs = [gym.make("Snake-v0", render_mode=None, grid_size=params["grid_size"]) 
                   for _ in range(params["env_count"])]
            seq_len, feature_dim = envs[0].observation_space.shape
            num_actions = envs[0].action_space.n
            
            self.log_message(f"Environment: {seq_len}x{feature_dim} obs, {num_actions} actions")
            
            # Create model
            model = SnakeMLPQNet(seq_len, feature_dim, num_actions).to(self.device)
            target_model = SnakeMLPQNet(seq_len, feature_dim, num_actions).to(self.device)
            target_model.load_state_dict(model.state_dict())
            
            # Training setup
            optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
            loss_fn = nn.HuberLoss()
            buffer = OptimizedReplayBuffer()
            
            # Training parameters
            eps, eps_min, eps_decay = 1.0, 0.05, 0.999995
            gamma, batch_size = params["gamma"], params["batch_size"]
            train_freq, sync_freq, max_steps = 4, 500, 200
            
            step_count = 0
            rewards_history, all_rewards = deque(maxlen=100), []
            
            # Initialize observations
            obs_batch = []
            for env in envs:
                obs, _ = env.reset()
                obs_batch.append(torch.tensor(obs.astype(np.float32)))
            
            self.log_message("Training started...")
            start_time = time.time()
            
            # Training loop
            for episode in range(params["episodes"]):
                if self.training_cancelled:
                    break
                
                ep_reward = [0.0] * len(envs)
                
                for step in range(max_steps):
                    if self.training_cancelled:
                        break
                    
                    step_count += 1
                    
                    # Select actions
                    obs_tensor = torch.stack(obs_batch)
                    actions = select_action(model, obs_tensor, eps, num_actions, self.device)
                    
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
                        loss = train_step(buffer, model, target_model, optimizer, loss_fn, gamma, batch_size, self.device)
                    
                    # Update target network
                    if step_count % sync_freq == 0:
                        target_model.load_state_dict(model.state_dict())
                    
                    # Decay epsilon
                    eps = max(eps_min, eps * eps_decay)
                
                # Episode complete
                avg_reward = np.mean(ep_reward)
                rewards_history.append(avg_reward)
                all_rewards.append(avg_reward)
                
                # Update progress
                progress = ((episode + 1) / params["episodes"]) * 100
                self.progress_bar['value'] = progress
                self.progress_var.set(f"Episode {episode + 1}/{params['episodes']} - Avg Reward: {avg_reward:.2f}")
                
                # Log progress
                if (episode + 1) % 50 == 0:
                    avg_100 = np.mean(list(rewards_history)) if rewards_history else 0
                    self.log_message(f"Episode {episode + 1}: Avg reward = {avg_reward:.2f}, Last 100 avg = {avg_100:.2f}")
            
            # Training complete
            if not self.training_cancelled:
                # Save model
                timestamp = int(time.time())
                model_name = f"snake_model_{timestamp}.pt"
                model_path = os.path.join(MODEL_DIR, model_name)
                
                # Save both model state and metadata
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_params': {
                        'seq_len': seq_len,
                        'feature_dim': feature_dim,
                        'num_actions': num_actions
                    },
                    'training_params': params,
                    'final_reward': np.mean(list(rewards_history)) if rewards_history else 0,
                    'all_rewards': all_rewards
                }, model_path)
                
                self.log_message(f"Model saved: {model_name}")
                self.log_message(f"Training completed in {time.time() - start_time:.1f} seconds")
                self.progress_var.set("Training completed!")
                
                # Update model list
                self.root.after(0, self.update_model_list)
            else:
                self.log_message("Training cancelled by user")
                self.progress_var.set("Training cancelled")
        
        except Exception as e:
            self.log_message(f"Training error: {str(e)}")
            self.progress_var.set("Training failed")
        
        finally:
            # Reset UI state
            self.is_training = False
            self.root.after(0, lambda: [
                self.train_button.config(state=tk.NORMAL),
                self.stop_button.config(state=tk.DISABLED)
            ])

    def test_model(self):
        """Test the selected model"""
        model_file = self.model_var.get()
        if not model_file:
            messagebox.showwarning("Test", "Please select a model to test!")
            return
        
        model_path = os.path.join(MODEL_DIR, model_file)
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found: {model_file}")
            return
        
        try:
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model_params = checkpoint['model_params']
            model = SnakeMLPQNet(model_params['seq_len'], model_params['feature_dim'], model_params['num_actions']).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Create environment
            env = gym.make("Snake-v0", render_mode="rgb_array", grid_size=checkpoint['training_params'].get('grid_size', 15))
            
            # Run test episodes
            num_episodes = 3
            all_scores = []
            all_frames = []
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, f"Testing model: {model_file}\n")
            self.stats_text.insert(tk.END, f"Running {num_episodes} test episodes...\n\n")
            
            for ep in range(num_episodes):
                obs, _ = env.reset()
                obs = torch.tensor(obs.astype(np.float32)).unsqueeze(0).to(self.device)
                score = 0
                frames = []
                
                for step in range(500):  # Max steps per episode
                    with torch.no_grad():
                        q_values = model(obs)
                        action = torch.argmax(q_values).item()
                    
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    obs = torch.tensor(next_obs.astype(np.float32)).unsqueeze(0).to(self.device)
                    
                    # Capture frame
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                    
                    if reward > 0:  # Snake ate food
                        score += 1
                    
                    if terminated or truncated:
                        break
                
                all_scores.append(score)
                if ep == 0:  # Save frames from first episode for display
                    all_frames = frames
                
                self.stats_text.insert(tk.END, f"Episode {ep + 1}: Score = {score}, Steps = {step + 1}\n")
                self.stats_text.see(tk.END)
                self.root.update_idletasks()
            
            # Display statistics
            avg_score = np.mean(all_scores)
            max_score = np.max(all_scores)
            min_score = np.min(all_scores)
            
            self.stats_text.insert(tk.END, f"\nFinal Results:\n")
            self.stats_text.insert(tk.END, f"Average Score: {avg_score:.1f}\n")
            self.stats_text.insert(tk.END, f"Max Score: {max_score}\n")
            self.stats_text.insert(tk.END, f"Min Score: {min_score}\n")
            
            if 'final_reward' in checkpoint:
                self.stats_text.insert(tk.END, f"Training Final Reward: {checkpoint['final_reward']:.2f}\n")
            
            # Display game frames
            if all_frames:
                self.animate_frames(all_frames)
            
            env.close()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to test model: {str(e)}")

    def animate_frames(self, frames):
        """Animate the game frames"""
        if not frames:
            return
        
        self.current_frames = frames
        self.frame_index = 0
        self.animate_next_frame()

    def animate_next_frame(self):
        """Show the next frame in the animation"""
        if hasattr(self, 'current_frames') and self.frame_index < len(self.current_frames):
            frame = self.current_frames[self.frame_index]
            
            # Convert to PIL Image and resize
            img = Image.fromarray(frame)
            img = img.resize((400, 400), Image.NEAREST)
            photo = ImageTk.PhotoImage(img)
            
            # Update display
            self.game_canvas.config(image=photo, text="")
            self.game_canvas.image = photo  # Keep a reference
            
            self.frame_index += 1
            
            # Schedule next frame
            if self.frame_index < len(self.current_frames):
                self.root.after(100, self.animate_next_frame)  # 10 FPS


def main():
    # Create the tkinter root window
    root = tk.Tk()
    
    # Set up the app
    app = SnakeTrainerApp(root)
    
    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()