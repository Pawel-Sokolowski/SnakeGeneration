# app.py
#!/usr/bin/env python3
"""
Tkinter GUI to load a saved Snake model and show it playing.
- Loads checkpoints saved as dict with 'model_state_dict', 'input_dim', 'num_actions', 'grid_size'
- Runs a deterministic episode (epsilon=0) and displays frames in the GUI
- Background thread for stepping so UI remains responsive
"""
import os
import json
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


# Simple MLP matching training model
class SnakeQNet(nn.Module):
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


def safe_torch_load(path: str, device: torch.device):
    """
    Try normal torch.load; if it fails with weights-only restrictions, prompt user and optionally try full load.
    """
    try:
        ckpt = torch.load(path, map_location=device)
        return ckpt
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" in msg or "Unsupported global" in msg or "not an allowed global" in msg:
            proceed = messagebox.askyesno(
                "Load fallback",
                "This checkpoint requires a full load to restore non-weight objects. "
                "Full load can execute code from the file. Proceed only if you trust the file."
            )
            if proceed:
                try:
                    ckpt = torch.load(path, map_location=device, weights_only=False)
                    return ckpt
                except Exception:
                    pass
        raise


def obs_to_features_for_gui(obs, grid_size):
    """
    Mirror the environment's features extractor. Prefer 'features' key if present.
    """
    if isinstance(obs, dict) and 'features' in obs:
        return np.asarray(obs['features'], dtype=np.float32)

    # fallback to coords parsing
    head = (0, 0)
    fruit = (0, 0)
    body = []
    if isinstance(obs, dict) and 'coords' in obs:
        coords = np.asarray(obs['coords'])
        valid_mask = ~((coords[:, 0] == -1) & (coords[:, 1] == -1))
        valid_coords = coords[valid_mask]
        if valid_coords.shape[0] >= 1:
            head = (int(valid_coords[0, 0]), int(valid_coords[0, 1]))
        if valid_coords.shape[0] >= 2:
            body = [tuple(map(int, p)) for p in valid_coords]
            fruit = (int(valid_coords[-1, 0]), int(valid_coords[-1, 1]))
    else:
        try:
            arr = np.asarray(obs)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                head = (int(arr[0, 0]), int(arr[0, 1]))
                fruit = (int(arr[-1, 0]), int(arr[-1, 1]))
                body = [tuple(map(int, p)) for p in arr[:-1] if not (int(p[0]) == -1 and int(p[1]) == -1)]
        except Exception:
            pass

    if len(body) >= 2:
        neck = body[1]
        dir_x = int(head[0] - neck[0])
        dir_y = int(head[1] - neck[1])
    else:
        dir_x, dir_y = 1, 0

    apple_dx = int(fruit[0] - head[0])
    apple_dy = int(fruit[1] - head[1])

    def is_collision(pos):
        x, y = pos
        if not (0 <= x < grid_size and 0 <= y < grid_size):
            return True
        if (x, y) in body:
            return True
        return False

    hx, hy = dir_x, dir_y
    front = (int(head[0] + hx), int(head[1] + hy))
    left = (int(head[0] - hy), int(head[1] + hx))
    right = (int(head[0] + hy), int(head[1] - hx))

    danger_front = 1.0 if is_collision(front) else 0.0
    danger_left = 1.0 if is_collision(left) else 0.0
    danger_right = 1.0 if is_collision(right) else 0.0

    denom = max(1, grid_size - 1)
    features = np.array([
        head[0] / denom,
        head[1] / denom,
        apple_dx / denom,
        apple_dy / denom,
        hx, hy,
        danger_front, danger_left, danger_right
    ], dtype=np.float32)
    return features


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

        self._build_ui()
        self.refresh_models()

    def _build_ui(self):
        top = tk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(top, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(top, textvariable=self.model_var, width=60, state="readonly")
        self.model_combo.pack(side=tk.LEFT, padx=6)
        tk.Button(top, text="Refresh", command=self.refresh_models).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Load", command=self.load_selected_model).pack(side=tk.LEFT, padx=4)

        right_controls = tk.Frame(top)
        right_controls.pack(side=tk.RIGHT)
        tk.Button(right_controls, text="Run", command=self.start_run).pack(side=tk.LEFT, padx=4)
        tk.Button(right_controls, text="Pause", command=self.pause_run).pack(side=tk.LEFT, padx=4)
        tk.Button(right_controls, text="Step", command=self.step_once).pack(side=tk.LEFT, padx=4)

        speed_frame = tk.Frame(self)
        speed_frame.pack(fill=tk.X, padx=8)
        tk.Label(speed_frame, text="Speed (ms per frame):").pack(side=tk.LEFT)
        self.speed_var = tk.IntVar(value=100)
        self.speed_slider = tk.Scale(speed_frame, from_=10, to=500, orient=tk.HORIZONTAL, variable=self.speed_var)
        self.speed_slider.pack(fill=tk.X, padx=6, expand=True)

        # Canvas for rendering
        self.canvas = tk.Canvas(self, width=600, height=600, bg="black")
        self.canvas.pack(padx=8, pady=8, side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Info panel
        info_frame = tk.Frame(self)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)
        tk.Label(info_frame, text="Status").pack()
        self.status_label = tk.Label(info_frame, text="No model loaded", anchor=tk.W, justify=tk.LEFT)
        self.status_label.pack(fill=tk.X, pady=4)
        tk.Label(info_frame, text="Model metadata").pack(pady=(10,0))
        self.meta_text = tk.Text(info_frame, width=40, height=20, state=tk.DISABLED)
        self.meta_text.pack()

    def refresh_models(self):
        if not os.path.isdir(MODEL_DIR):
            os.makedirs(MODEL_DIR, exist_ok=True)
        files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")])
        self.model_combo['values'] = files
        if files:
            self.model_combo.current(0)
        # show latest learning curve if present
        curve_candidates = [f for f in os.listdir(MODEL_DIR) if f.startswith("learning_curve")]
        if curve_candidates:
            path = os.path.join(MODEL_DIR, curve_candidates[-1])
            try:
                img = Image.open(path)
                img = img.resize((300, 200), Image.LANCZOS)
                self.curve_img = ImageTk.PhotoImage(img)
                # place curve in meta area top
                self.meta_text.config(state=tk.NORMAL)
                self.meta_text.delete("1.0", tk.END)
                self.meta_text.insert(tk.END, f"Found learning curve: {path}\n\n")
                self.meta_text.config(state=tk.DISABLED)
            except Exception:
                pass

    def load_selected_model(self):
        sel = self.model_var.get()
        if not sel:
            messagebox.showinfo("No model", "Select a model file first.")
            return
        path = os.path.join(MODEL_DIR, sel)
        try:
            ckpt = safe_torch_load(path, self.device)
            if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt:
                raise RuntimeError("Checkpoint missing 'model_state_dict'")
            input_dim = int(ckpt.get('input_dim', 0))
            num_actions = int(ckpt.get('num_actions', 0))
            grid_size = int(ckpt.get('grid_size', DEFAULT_GRID))
            if input_dim <= 0 or num_actions <= 0:
                raise RuntimeError("Checkpoint missing valid input_dim or num_actions")
            self.model = SnakeQNet(input_dim, num_actions).to(self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model.eval()
            self.model_meta = {'input_dim': input_dim, 'num_actions': num_actions, 'grid_size': grid_size}
            self.status_label.config(text=f"Loaded model: {sel}")
            self._show_meta()
        except Exception as e:
            messagebox.showerror("Load error", f"Failed to load model: {e}")
            self.model = None
            self.model_meta = None
            self.status_label.config(text="Failed to load model")

    def _show_meta(self):
        self.meta_text.config(state=tk.NORMAL)
        self.meta_text.delete("1.0", tk.END)
        for k, v in (self.model_meta or {}).items():
            self.meta_text.insert(tk.END, f"{k}: {v}\n")
        self.meta_text.config(state=tk.DISABLED)

    def _make_env(self):
        grid_size = int(self.model_meta.get('grid_size', DEFAULT_GRID)) if self.model_meta else DEFAULT_GRID
        try:
            env = gym.make("Snake-v0", render_mode="rgb_array", grid_size=grid_size)
            return env
        except Exception as e:
            messagebox.showerror("Env error", f"Failed to create env: {e}")
            return None

    def start_run(self):
        if self.model is None:
            messagebox.showinfo("No model", "Load a model first.")
            return
        if self.playing:
            return
        # create env for rendering
        self.env = self._make_env()
        if self.env is None:
            return
        self.playing = True
        self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.play_thread.start()
        self.status_label.config(text="Playing...")

    def pause_run(self):
        self.playing = False
        self.status_label.config(text="Paused")

    def step_once(self):
        if self.model is None:
            messagebox.showinfo("No model", "Load a model first.")
            return
        if self.env is None:
            self.env = self._make_env()
            if self.env is None:
                return
            self.obs, _ = self.env.reset()
        # single deterministic step
        feat = obs_to_features_for_gui(self.obs, self.env.unwrapped.grid_size if hasattr(self.env.unwrapped, 'grid_size') else int(self.grid_var.get()))
        with torch.no_grad():
            q = self.model(torch.from_numpy(feat).float().to(self.device))
            act = int(torch.argmax(q, dim=1).item())
        self.obs, r, term, trunc, _ = self.env.step(act)
        frame = self.env.render()
        self._display_frame(frame)
        self.status_label.config(text=f"Step reward: {r:.2f} done:{term or trunc}")
        if term or trunc:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

    def _play_loop(self):
        try:
            self.obs, _ = self.env.reset()
            max_steps = 1000
            for _ in range(max_steps):
                if not self.playing:
                    break
                feat = obs_to_features_for_gui(self.obs, self.env.unwrapped.grid_size if hasattr(self.env.unwrapped, 'grid_size') else int(self.model_meta.get('grid_size', DEFAULT_GRID)))
                with torch.no_grad():
                    q = self.model(torch.from_numpy(feat).float().to(self.device))
                    act = int(torch.argmax(q, dim=1).item())
                self.obs, r, term, trunc, info = self.env.step(act)
                frame = self.env.render()
                # update UI on main thread
                self.after(0, self._display_frame, frame)
                self.after(0, self.status_label.config, {"text": f"Reward: {r:.2f} Score: {info.get('score', '')}"})
                if term or trunc:
                    break
                # sleep according to speed slider
                time.sleep(self.speed_var.get() / 1000.0)
        except Exception as e:
            self.after(0, messagebox.showerror, "Runtime error", f"Error during play: {e}")
        finally:
            self.playing = False
            try:
                if self.env is not None:
                    self.env.close()
            except Exception:
                pass
            self.env = None
            self.after(0, self.status_label.config, {"text": "Stopped"})

    def _display_frame(self, frame):
        if frame is None:
            return
        # frame is HxWx3 numpy array
        try:
            img = Image.fromarray(frame.astype('uint8'))
            # fit canvas size while preserving aspect
            canvas_w = self.canvas.winfo_width() or 600
            canvas_h = self.canvas.winfo_height() or 600
            img = img.resize((canvas_w, canvas_h), Image.NEAREST)
            self.frame_image = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.frame_image)
        except Exception as e:
            # fallback: ignore display errors
            pass


if __name__ == "__main__":
    app = SnakeViewerApp()
    app.mainloop()