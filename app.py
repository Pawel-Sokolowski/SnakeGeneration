import os
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gymnasium as gym
import snake_gym_env

MODEL_DIR = "saved_models"

# Import the model architecture from main.py or define it here
try:
    from main import SnakeMLPQNet
except ImportError:
    # Define simple MLP model if main.py imports fail
    class SnakeMLPQNet(nn.Module):
        def __init__(self, seq_len, feature_dim, num_actions):
            super(SnakeMLPQNet, self).__init__()
            input_size = seq_len * feature_dim
            self.flatten = nn.Flatten()
            self.network = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions)
            )
        
        def forward(self, x):
            x = self.flatten(x)
            return self.network(x)

# Transformer model for the transformer checkpoints
class SimpleTransformerQNet(nn.Module):
    """Simple Transformer Q-Network that matches the saved transformer model structure"""
    def __init__(self, feature_dim=2, embed_dim=128, num_heads=8, num_actions=4):
        super().__init__()
        
        # Embedding layer
        self.embed = nn.Linear(feature_dim, embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, feature_dim)
        x = self.embed(x)
        x = self.transformer(x)
        # Global pooling to get single representation
        x = x.mean(dim=1)
        return self.head(x)

class SnakeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Snake Model Visualizer")
        self.geometry("900x800")
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        controls = tk.Frame(self)
        controls.pack(fill=tk.X, pady=10)

        tk.Label(controls, text="Select Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(controls, textvariable=self.model_var, width=40)
        self.model_files = self.list_models()
        self.model_combo['values'] = self.model_files
        if self.model_files:
            self.model_var.set(self.model_files[0])
        self.model_combo.pack(side=tk.LEFT, padx=5)

        tk.Label(controls, text="Grid Size:").pack(side=tk.LEFT, padx=(20, 5))
        self.grid_var = tk.StringVar(value="20")
        self.grid_combo = ttk.Combobox(controls, textvariable=self.grid_var, values=["10", "15", "20", "25", "30"], width=8)
        self.grid_combo.pack(side=tk.LEFT, padx=5)

        self.plot_btn = tk.Button(controls, text="Show Learning Curve", command=self.show_learning_curve)
        self.plot_btn.pack(side=tk.LEFT, padx=5)
        self.play_btn = tk.Button(controls, text="Play Snake", command=self.play_snake)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.plot_label = tk.Label(self, text="Learning curve will appear here")
        self.plot_label.pack(pady=10)

        self.snake_canvas = tk.Label(self, text="Snake game will appear here", width=420, height=420, bg='#222')
        self.snake_canvas.pack(pady=10)

        self.frames = []
        self.frame_idx = 0

    def list_models(self):
        if not os.path.exists(MODEL_DIR):
            return []
        return [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]

    def show_learning_curve(self):
        model_file = self.model_var.get()
        plot_file = model_file.rsplit('.', 1)[0] + ".png"
        plot_path = os.path.join(MODEL_DIR, plot_file)
        if os.path.exists(plot_path):
            img = Image.open(plot_path)
            img = img.resize((800, 220), Image.BILINEAR)
            photo = ImageTk.PhotoImage(img)
            self.plot_label.configure(image=photo, text='')
            self.plot_label.image = photo
        else:
            self.plot_label.configure(text="No learning curve plot found for this model.", image='')
            self.plot_label.image = None

    def play_snake(self):
        model_file = self.model_var.get()
        if not model_file:
            messagebox.showerror("Error", "Please select a model first.")
            return
            
        model_path = os.path.join(MODEL_DIR, model_file)
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model file not found: {model_file}")
            return
            
        try:
            # Load the checkpoint/state_dict
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Determine model type based on the checkpoint structure
            if isinstance(checkpoint, dict) and 'model_params' in checkpoint:
                # Standard checkpoint with metadata
                model_params = checkpoint['model_params']
                model = SnakeMLPQNet(model_params['seq_len'], model_params['feature_dim'], model_params['num_actions']).to(self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'embed.weight' in checkpoint:
                # Transformer model (state dict only)
                model = SimpleTransformerQNet().to(self.device)
                if isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    # It's a state dict
                    model.load_state_dict(checkpoint)
                else:
                    raise ValueError("Unrecognized checkpoint format")
            else:
                # Try to load as MLP model state dict
                # First, create a test environment to get dimensions
                test_env = gym.make("Snake-v0", render_mode=None, grid_size=20)
                seq_len, feature_dim = test_env.observation_space.shape
                num_actions = test_env.action_space.n
                test_env.close()
                
                model = SnakeMLPQNet(seq_len, feature_dim, num_actions).to(self.device)
                model.load_state_dict(checkpoint)
            
            model.eval()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            return

        grid_size = int(self.grid_var.get())
        env = gym.make("Snake-v0", render_mode="rgb_array", grid_size=grid_size)
        obs, _ = env.reset(seed=None)
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        self.frames = []
        max_steps = 500
        
        for _ in range(max_steps):
            with torch.no_grad():
                q_values = model(obs)
                action = torch.argmax(q_values, dim=1).item()
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            frame = env.render()
            if frame is not None:
                img = Image.fromarray(frame).resize((420, 420), Image.NEAREST)
                self.frames.append(ImageTk.PhotoImage(img))
            if terminated or truncated:
                break
                
        env.close()
        self.frame_idx = 0
        if not self.frames:
            self.snake_canvas.config(text="No frames to display.", image='')
            return
        self.animate_snake()

    def animate_snake(self):
        if not self.frames or self.frame_idx >= len(self.frames):
            return
        frame = self.frames[self.frame_idx]
        self.snake_canvas.config(image=frame, text='')
        self.snake_canvas.image = frame
        self.frame_idx += 1
        if self.frame_idx < len(self.frames):
            self.after(60, self.animate_snake)  # ~16 fps

if __name__ == "__main__":
    app = SnakeApp()
    app.mainloop()