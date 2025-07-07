import os
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import gymnasium as gym
import snake_gym_env

MODEL_DIR = "saved_models"

class SnakeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Snake Model Visualizer")
        self.geometry("900x800")

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
        return [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]

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
        model_path = os.path.join(MODEL_DIR, model_file)
        model = keras.models.load_model(model_path)

        env = gym.make("Snake-v0", render_mode=None)
        obs, _ = env.reset(seed=None)
        obs = np.array(obs, dtype=np.float32)
        self.frames = []
        max_steps = 500
        for _ in range(max_steps):
            action = np.argmax(model(tf.convert_to_tensor(np.expand_dims(obs, axis=0)), training=False).numpy())
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs = np.array(next_obs, dtype=np.float32)
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