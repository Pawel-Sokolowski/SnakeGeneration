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
        self.playing = False  # Flag to control continuous play
        self.model = None  # Store the loaded model

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
        if self.playing:
            self.playing = False  # Stop the current game
            return

        self.playing = True  # Start a new game
        model_file = self.model_var.get()
        model_path = os.path.join(MODEL_DIR, model_file)
        try:
            self.model = keras.models.load_model(model_path)  # Load the model and store it in self.model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.playing = False
            return

        self.env = gym.make("Snake-v0", render_mode='rgb_array')  # Ensure render_mode is 'rgb_array'
        self.obs, _ = self.env.reset(seed=None)
        self.obs = np.array(self.obs, dtype=np.float32)
        self.frames = []
        self.frame_idx = 0
        self.animate_snake()  # Start the animation loop

    def animate_snake(self):
        if not self.playing:
            if hasattr(self, 'env') and self.env:
                self.env.close()
            return

        try:
            # Use self.model to make predictions
            action = np.argmax(self.model(tf.convert_to_tensor(np.expand_dims(self.obs, axis=0)), training=False).numpy())
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.obs = np.array(next_obs, dtype=np.float32)
            frame = self.env.render()

            if frame is not None:
                img = Image.fromarray(frame).resize((420, 420), Image.NEAREST)
                photo = ImageTk.PhotoImage(img)
                self.snake_canvas.config(image=photo, text='')
                self.snake_canvas.image = photo
            else:
                print("Frame is None!")

            if terminated or truncated:
                print("Game Over! Resetting...")
                self.env.close()
                self.env = gym.make("Snake-v0", render_mode='rgb_array')
                self.obs, _ = self.env.reset(seed=None)
                self.obs = np.array(self.obs, dtype=np.float32)

            self.after(60, self.animate_snake)  # Schedule the next frame
        except Exception as e:
            messagebox.showerror("Error", f"Error during gameplay: {e}")
            self.playing = False
            if hasattr(self, 'env') and self.env:
                self.env.close()
            if self.model is not None:
                self.model = None

if __name__ == "__main__":
    app = SnakeApp()
    app.mainloop()