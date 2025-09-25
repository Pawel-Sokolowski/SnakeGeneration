#!/usr/bin/env python3
"""
Snake Model Visualizer - Simplified GUI App
Choose trained models and grid size to run the Snake AI
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import gymnasium as gym
import snake_gym_env


# Model directory
MODEL_DIR = "saved_models"


# Simple MLP Q-Network (must match training model)
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


class SnakeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Snake AI - Model Selector")
        self.geometry("800x600")
        self.resizable(True, True)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.setup_ui()
        self.refresh_models()

    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_frame = tk.Frame(self)
        title_frame.pack(fill=tk.X, pady=10)
        
        title_label = tk.Label(title_frame, text="🐍 Snake AI Model Selector", 
                              font=("Arial", 16, "bold"))
        title_label.pack()

        # Controls frame
        controls_frame = tk.Frame(self)
        controls_frame.pack(fill=tk.X, padx=20, pady=10)

        # Model selection
        model_frame = tk.Frame(controls_frame)
        model_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(model_frame, text="Select Model:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                       width=50, state="readonly")
        self.model_combo.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)
        
        # Refresh button
        refresh_btn = tk.Button(model_frame, text="Refresh", command=self.refresh_models)
        refresh_btn.pack(side=tk.RIGHT, padx=(5, 0))

        # Grid size selection
        size_frame = tk.Frame(controls_frame)
        size_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(size_frame, text="Grid Size:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.grid_var = tk.StringVar(value="20")
        self.grid_combo = ttk.Combobox(size_frame, textvariable=self.grid_var, 
                                      values=["10", "15", "20", "25", "30"], 
                                      width=10, state="readonly")
        self.grid_combo.pack(side=tk.LEFT, padx=(10, 0))

        # Action buttons
        button_frame = tk.Frame(controls_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.plot_btn = tk.Button(button_frame, text="📊 Show Learning Curve", 
                                 command=self.show_learning_curve, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.plot_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.play_btn = tk.Button(button_frame, text="🎮 Play Snake", 
                                 command=self.play_snake, bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        self.play_btn.pack(side=tk.LEFT)

        # Learning curve display
        curve_frame = tk.Frame(self)
        curve_frame.pack(fill=tk.BOTH, padx=20, pady=10)
        
        tk.Label(curve_frame, text="Learning Curve:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.plot_label = tk.Label(curve_frame, text="Select a model and click 'Show Learning Curve'", 
                                  bg="white", relief="sunken", height=10)
        self.plot_label.pack(fill=tk.BOTH, expand=True, pady=5)

        # Game display
        game_frame = tk.Frame(self)
        game_frame.pack(fill=tk.BOTH, padx=20, pady=(0, 20), expand=True)
        
        tk.Label(game_frame, text="Game Visualization:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.game_label = tk.Label(game_frame, text="Select a model and click 'Play Snake'", 
                                  bg="black", fg="white", relief="sunken")
        self.game_label.pack(fill=tk.BOTH, expand=True, pady=5)

        # Animation variables
        self.frames = []
        self.frame_idx = 0
        self.animation_speed = 100  # milliseconds

    def refresh_models(self):
        """Refresh the list of available models"""
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            self.model_combo['values'] = []
            self.model_var.set("")
            return
        
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
        model_files.sort()
        
        self.model_combo['values'] = model_files
        if model_files and not self.model_var.get():
            self.model_var.set(model_files[0])
        elif not model_files:
            self.model_var.set("")

    def show_learning_curve(self):
        """Display the learning curve for the selected model"""
        model_file = self.model_var.get()
        if not model_file:
            messagebox.showwarning("Warning", "Please select a model first.")
            return
        
        # Look for corresponding PNG file
        plot_file = model_file.replace(".pt", ".png")
        plot_path = os.path.join(MODEL_DIR, plot_file)
        
        if os.path.exists(plot_path):
            try:
                # Load and resize image
                img = Image.open(plot_path)
                # Resize to fit the display area
                img = img.resize((750, 300), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                self.plot_label.configure(image=photo, text='')
                self.plot_label.image = photo  # Keep a reference
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load learning curve: {str(e)}")
        else:
            self.plot_label.configure(text=f"No learning curve found for {model_file}", image='')
            self.plot_label.image = None

    def play_snake(self):
        """Load model and play Snake game"""
        model_file = self.model_var.get()
        if not model_file:
            messagebox.showwarning("Warning", "Please select a model first.")
            return
        
        grid_size = int(self.grid_var.get())
        model_path = os.path.join(MODEL_DIR, model_file)
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found: {model_file}")
            return
        
        try:
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if 'model_params' in checkpoint:
                # New format with metadata
                model_params = checkpoint['model_params']
                model = SnakeQNet(model_params['seq_len'], model_params['feature_dim'], 
                                 model_params['num_actions']).to(self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Try to infer from environment
                env = gym.make("Snake-v0", render_mode=None, grid_size=grid_size)
                seq_len, feature_dim = env.observation_space.shape
                num_actions = env.action_space.n
                env.close()
                
                model = SnakeQNet(seq_len, feature_dim, num_actions).to(self.device)
                model.load_state_dict(checkpoint)
            
            model.eval()
            
            # Run game and collect frames
            self.run_game_simulation(model, grid_size)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or run model: {str(e)}")

    def run_game_simulation(self, model, grid_size):
        """Run the game simulation and collect frames"""
        env = gym.make("Snake-v0", render_mode="rgb_array", grid_size=grid_size)
        
        state, _ = env.reset()
        state = torch.tensor(state.astype(np.float32))
        
        self.frames = []
        max_steps = 500
        
        for step in range(max_steps):
            # Get action from model
            with torch.no_grad():
                q_values = model(state.unsqueeze(0).to(self.device))
                action = torch.argmax(q_values).item()
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = torch.tensor(next_state.astype(np.float32))
            
            # Capture frame
            frame = env.render()
            if frame is not None:
                # Resize frame for display
                img = Image.fromarray(frame)
                # Calculate size to fit display area while maintaining aspect ratio
                img = img.resize((400, 400), Image.Resampling.NEAREST)
                self.frames.append(ImageTk.PhotoImage(img))
            
            if terminated or truncated:
                break
        
        env.close()
        
        if not self.frames:
            self.game_label.config(text="No frames generated", image='')
            return
        
        # Start animation
        self.frame_idx = 0
        self.game_label.config(text='')
        self.animate_frames()

    def animate_frames(self):
        """Animate the collected frames"""
        if not self.frames or self.frame_idx >= len(self.frames):
            # Animation finished, show completion message
            self.game_label.config(text="Game completed! Click 'Play Snake' to run again.")
            return
        
        # Show current frame
        frame = self.frames[self.frame_idx]
        self.game_label.config(image=frame)
        self.game_label.image = frame  # Keep reference
        
        self.frame_idx += 1
        
        # Schedule next frame
        self.after(self.animation_speed, self.animate_frames)


def main():
    """Main function"""
    # Check if models directory exists, create if not
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created {MODEL_DIR} directory")
    
    # Check for trained models
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
    if not model_files:
        print("No trained models found!")
        print(f"Train a model first by running: python main.py")
        print(f"Models will be saved to: {MODEL_DIR}")
    
    # Start the application
    app = SnakeApp()
    print("🐍 Snake AI Model Selector started")
    print(f"Device: {app.device}")
    app.mainloop()


if __name__ == "__main__":
    main()