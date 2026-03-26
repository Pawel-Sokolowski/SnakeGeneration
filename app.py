#!/usr/bin/env python3
"""
Snake Viewer GUI — fully refactored and corrected.

Key fixes:
- Model architecture now matches training (64→32→actions)
- Always uses env-provided obs["features"]
- Supports both full checkpoints and raw state_dict files
- Robust metadata inference
- Clean threading and rendering
"""

import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

import snake_gym_env  # registers Snake-v0

MODEL_DIR = "saved_models"
DEFAULT_GRID = 20
GRID_CHOICES = [20, 30, 40]


# ---------------------------------------------------------
# MODEL — must match training architecture EXACTLY
# ---------------------------------------------------------
class SnakeQNet(nn.Module):
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


# ---------------------------------------------------------
# SAFE TORCH LOAD
# ---------------------------------------------------------
def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device)
    except Exception as e:
        msg = str(e)
        if (
            "Weights only load failed" in msg
            or "Unsupported global" in msg
            or "not an allowed global" in msg
        ):
            proceed = messagebox.askyesno(
                "Load fallback",
                "This checkpoint requires a full load.\n"
                "Full load can execute code from the file.\n"
                "Proceed only if you trust this file."
            )
            if proceed:
                return torch.load(path, map_location=device, weights_only=False)
        raise


# ---------------------------------------------------------
# GUI APPLICATION
# ---------------------------------------------------------
class SnakeViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Snake Viewer")
        self.geometry("1000x700")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_meta = None
        self.env = None
        self.play_thread = None
        self.playing = False
        self.frame_image = None

        self.grid_size = DEFAULT_GRID

        self._build_ui()
        self.refresh_models()

    # -----------------------------------------------------
    # UI
    # -----------------------------------------------------
    def _build_ui(self):
        top = tk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(top, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            top, textvariable=self.model_var, width=60, state="readonly"
        )
        self.model_combo.pack(side=tk.LEFT, padx=6)

        tk.Button(top, text="Refresh", command=self.refresh_models).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Load", command=self.load_selected_model).pack(side=tk.LEFT, padx=4)

        # Controls
        right = tk.Frame(top)
        right.pack(side=tk.RIGHT)
        tk.Button(right, text="Run", command=self.start_run).pack(side=tk.LEFT, padx=4)
        tk.Button(right, text="Pause", command=self.pause_run).pack(side=tk.LEFT, padx=4)
        tk.Button(right, text="Step", command=self.step_once).pack(side=tk.LEFT, padx=4)

        # Grid selector
        grid_frame = tk.Frame(self)
        grid_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(grid_frame, text="Grid size:").pack(side=tk.LEFT)
        self.grid_var = tk.StringVar(value=str(DEFAULT_GRID))
        self.grid_combo = ttk.Combobox(
            grid_frame,
            textvariable=self.grid_var,
            values=[str(g) for g in GRID_CHOICES],
            width=10,
            state="readonly",
        )
        self.grid_combo.pack(side=tk.LEFT, padx=6)

        # Speed slider
        speed_frame = tk.Frame(self)
        speed_frame.pack(fill=tk.X, padx=8)
        tk.Label(speed_frame, text="Speed (ms per frame):").pack(side=tk.LEFT)
        self.speed_var = tk.IntVar(value=100)
        tk.Scale(
            speed_frame,
            from_=10, to=500,
            orient=tk.HORIZONTAL,
            variable=self.speed_var
        ).pack(fill=tk.X, padx=6, expand=True)

        # Canvas
        self.canvas = tk.Canvas(self, width=600, height=600, bg="black")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Info panel
        info = tk.Frame(self)
        info.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)

        tk.Label(info, text="Status").pack()
        self.status_label = tk.Label(info, text="No model loaded", anchor=tk.W, justify=tk.LEFT)
        self.status_label.pack(fill=tk.X, pady=4)

        tk.Label(info, text="Model metadata").pack(pady=(10, 0))
        self.meta_text = tk.Text(info, width=40, height=20, state=tk.DISABLED)
        self.meta_text.pack()

    # -----------------------------------------------------
    # MODEL LOADING
    # -----------------------------------------------------
    def refresh_models(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")])
        self.model_combo["values"] = files
        if files:
            self.model_combo.current(0)

    def _infer_meta_from_env(self, grid_size: int):
        env = gym.make("Snake-v0", grid_size=grid_size)
        obs, _ = env.reset()
        feat = np.asarray(obs["features"], dtype=np.float32)
        input_dim = feat.shape[0]
        num_actions = env.action_space.n
        env.close()
        return input_dim, num_actions

    def _normalize_checkpoint(self, ckpt, ui_grid):
        # Full checkpoint
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            input_dim = ckpt.get("input_dim")
            num_actions = ckpt.get("num_actions")
            grid_size = ckpt.get("grid_size", ui_grid)

            if not input_dim or not num_actions:
                input_dim, num_actions = self._infer_meta_from_env(grid_size)

            return {
                "model_state_dict": ckpt["model_state_dict"],
                "input_dim": input_dim,
                "num_actions": num_actions,
                "grid_size": grid_size,
            }

        # Raw state_dict
        grid_size = ui_grid
        input_dim, num_actions = self._infer_meta_from_env(grid_size)
        return {
            "model_state_dict": ckpt,
            "input_dim": input_dim,
            "num_actions": num_actions,
            "grid_size": grid_size,
        }

    def load_selected_model(self):
        sel = self.model_var.get()
        if not sel:
            messagebox.showinfo("No model", "Select a model file first.")
            return

        path = os.path.join(MODEL_DIR, sel)

        try:
            ui_grid = int(self.grid_var.get())
        except ValueError:
            ui_grid = DEFAULT_GRID

        try:
            raw = safe_torch_load(path, self.device)
            ckpt = self._normalize_checkpoint(raw, ui_grid)

            self.model = SnakeQNet(ckpt["input_dim"], ckpt["num_actions"]).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()

            self.model_meta = ckpt | {"file": sel}
            self._show_meta()

            self.status_label.config(text=f"Loaded model: {sel}")

        except Exception as e:
            messagebox.showerror("Load error", f"Failed to load model: {e}")
            self.model = None
            self.model_meta = None

    def _show_meta(self):
        self.meta_text.config(state=tk.NORMAL)
        self.meta_text.delete("1.0", tk.END)
        for k, v in self.model_meta.items():
            self.meta_text.insert(tk.END, f"{k}: {v}\n")
        self.meta_text.config(state=tk.DISABLED)

    # -----------------------------------------------------
    # ENVIRONMENT
    # -----------------------------------------------------
    def _current_grid_size(self):
        try:
            ui = int(self.grid_var.get())
            if ui in GRID_CHOICES:
                return ui
        except ValueError:
            pass

        if self.model_meta:
            return self.model_meta["grid_size"]

        return DEFAULT_GRID

    def _make_env(self):
        try:
            return gym.make("Snake-v0", render_mode="rgb_array", grid_size=self._current_grid_size())
        except Exception as e:
            messagebox.showerror("Env error", f"Failed to create env: {e}")
            return None

    # -----------------------------------------------------
    # CONTROL ACTIONS
    # -----------------------------------------------------
    def start_run(self):
        if not self.model:
            messagebox.showinfo("No model", "Load a model first.")
            return
        if self.playing:
            return

        self.env = self._make_env()
        if not self.env:
            return

        self.playing = True
        self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.play_thread.start()
        self.status_label.config(text="Playing...")

    def pause_run(self):
        self.playing = False
        self.status_label.config(text="Paused")

    def step_once(self):
        if not self.model:
            messagebox.showinfo("No model", "Load a model first.")
            return

        if not self.env:
            self.env = self._make_env()
            if not self.env:
                return
            self.obs, _ = self.env.reset()

        feat = np.asarray(self.obs["features"], dtype=np.float32)

        with torch.no_grad():
            q = self.model(torch.from_numpy(feat).float().to(self.device))
            act = int(torch.argmax(q, dim=1).item())

        self.obs, r, term, trunc, info = self.env.step(act)
        frame = self.env.render()
        self._display_frame(frame)

        self.status_label.config(text=f"Step reward: {r:.2f}")

        if term or trunc:
            self.env.close()
            self.env = None

    # -----------------------------------------------------
    # PLAY LOOP
    # -----------------------------------------------------
    def _play_loop(self):
        try:
            self.obs, _ = self.env.reset()

            for _ in range(2000):
                if not self.playing:
                    break

                feat = np.asarray(self.obs["features"], dtype=np.float32)

                with torch.no_grad():
                    q = self.model(torch.from_numpy(feat).float().to(self.device))
                    act = int(torch.argmax(q, dim=1).item())

                self.obs, r, term, trunc, info = self.env.step(act)
                frame = self.env.render()

                self.after(0, self._display_frame, frame)
                self.after(0, self.status_label.config,
                           {"text": f"Reward: {r:.2f} Score: {info.get('score', '')}"})

                if term or trunc:
                    break

                time.sleep(self.speed_var.get() / 1000.0)

        except Exception as e:
            self.after(0, messagebox.showerror, "Runtime error", f"Error during play: {e}")

        finally:
            self.playing = False
            if self.env:
                self.env.close()
            self.env = None
            self.after(0, self.status_label.config, {"text": "Stopped"})

    # -----------------------------------------------------
    # RENDERING
    # -----------------------------------------------------
    def _display_frame(self, frame):
        if frame is None:
            return
        img = Image.fromarray(frame.astype("uint8"))
        w = self.canvas.winfo_width() or 600
        h = self.canvas.winfo_height() or 600
        img = img.resize((w, h), Image.NEAREST)
        self.frame_image = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.frame_image)


if __name__ == "__main__":
    app = SnakeViewerApp()
    app.mainloop()