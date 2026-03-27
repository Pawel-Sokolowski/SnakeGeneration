import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageTk

import snake_gym_env  # ensures Snake-v0 is registered


MODEL_DIR = "saved_models"
GRAPHICS_DIR = "Graphics"
DEFAULT_GRID = 20
GRID_CHOICES = [10, 15, 20, 25, 30]
CELL_SIZE = 40

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

class SnakeViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Snake Viewer")
        self.geometry("1100x750")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_meta = None
        self.env = None
        self.play_thread = None
        self.playing = False
        self.obs = None
        self.last_score = 0

        self.grid_size = DEFAULT_GRID
        self.canvas_items = {}

        self._load_graphics()
        self._build_ui()
        self.refresh_models()

    def _load_graphics(self):
        def load(name):
            path = os.path.join(GRAPHICS_DIR, name + ".png")
            img = Image.open(path).resize((CELL_SIZE, CELL_SIZE), Image.NEAREST)
            return ImageTk.PhotoImage(img)

        self.apple = load("apple")

        self.head = {
            "up": load("head_up"),
            "down": load("head_down"),
            "left": load("head_left"),
            "right": load("head_right"),
        }

        self.tail = {
            "up": load("tail_up"),
            "down": load("tail_down"),
            "left": load("tail_left"),
            "right": load("tail_right"),
        }

        self.body = {
            "horizontal": load("body_horizontal"),
            "vertical": load("body_vertical"),
            "tl": load("body_tl"),
            "tr": load("body_tr"),
            "bl": load("body_bl"),
            "br": load("body_br"),
        }

    def _build_ui(self):
        top = tk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(top, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(top, textvariable=self.model_var, width=60, state="readonly")
        self.model_combo.pack(side=tk.LEFT, padx=6)

        tk.Button(top, text="Refresh", command=self.refresh_models).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Load", command=self.load_selected_model).pack(side=tk.LEFT, padx=4)

        right = tk.Frame(top)
        right.pack(side=tk.RIGHT)
        tk.Button(right, text="Run", command=self.start_run).pack(side=tk.LEFT, padx=4)
        tk.Button(right, text="Pause", command=self.pause_run).pack(side=tk.LEFT, padx=4)
        tk.Button(right, text="Step", command=self.step_once).pack(side=tk.LEFT, padx=4)

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

        speed_frame = tk.Frame(self)
        speed_frame.pack(fill=tk.X, padx=8)
        tk.Label(speed_frame, text="Speed (ms per frame):").pack(side=tk.LEFT)
        self.speed_var = tk.IntVar(value=100)
        tk.Scale(speed_frame, from_=10, to=500, orient=tk.HORIZONTAL, variable=self.speed_var)\
            .pack(fill=tk.X, padx=6, expand=True)

        # Canvas EXACT size of grid
        self.canvas = tk.Canvas(self, width=DEFAULT_GRID * CELL_SIZE, height=DEFAULT_GRID * CELL_SIZE, bg="black")
        self.canvas.pack(side=tk.LEFT, padx=8, pady=8)

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
            return gym.make("Snake-v0", grid_size=self._current_grid_size())
        except Exception as e:
            messagebox.showerror("Env error", f"Failed to create env: {e}")
            return None

    # -----------------------------------------------------
    # RUN / STEP
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
            self.last_score = 0
            self._draw_background()

        feat = np.asarray(self.obs["features"], dtype=np.float32)

        with torch.no_grad():
            q = self.model(torch.from_numpy(feat).float().to(self.device))
            act = int(torch.argmax(q, dim=1).item())

        self.obs, r, term, trunc, info = self.env.step(act)
        self.last_score = info.get("score", self.last_score)
        self._draw_from_obs(self.obs)

        self.status_label.config(text=f"Step reward: {r:.2f}, Score: {self.last_score}")

        if term or trunc:
            self.env.close()
            self.env = None

    def _play_loop(self):
        try:
            self.obs, _ = self.env.reset()
            self.last_score = 0
            self._draw_background()

            for _ in range(5000):
                if not self.playing:
                    break

                feat = np.asarray(self.obs["features"], dtype=np.float32)
                with torch.no_grad():
                    q = self.model(torch.from_numpy(feat).float().to(self.device))
                    act = int(torch.argmax(q, dim=1).item())

                self.obs, r, term, trunc, info = self.env.step(act)
                self.last_score = info.get("score", self.last_score)
                self._draw_from_obs(self.obs)

                time.sleep(self.speed_var.get() / 1000.0)

                if term or trunc:
                    break

        finally:
            if self.env:
                self.env.close()
                self.env = None
            self.playing = False
            self.status_label.config(text="Finished")

    def _draw_background(self):
        self.canvas.delete("bg")
        grid = self._current_grid_size()

        self.canvas.config(width=grid * CELL_SIZE, height=grid * CELL_SIZE)

        color1 = "#A7D15D"
        color2 = "#9BC653"

        for y in range(grid):
            for x in range(grid):
                color = color1 if (x + y) % 2 == 0 else color2
                self.canvas.create_rectangle(
                    x * CELL_SIZE,
                    y * CELL_SIZE,
                    (x + 1) * CELL_SIZE,
                    (y + 1) * CELL_SIZE,
                    fill=color,
                    outline="",
                    tags="bg"
                )

        self._draw_score(self.last_score)

    def _draw_score(self, score):
        if "score" not in self.canvas_items:
            self.canvas_items["score"] = self.canvas.create_text(
                10, 10,
                anchor="nw",
                text=f"Score: {score}",
                fill="white",
                font=("Arial", 20, "bold"),
                tags="score"
            )
        else:
            self.canvas.itemconfig(self.canvas_items["score"], text=f"Score: {score}")

    def _draw_from_obs(self, obs):
        coords = obs["coords"]
        snake = [(x, y) for (x, y) in coords[:-1] if x >= 0]
        fx, fy = coords[-1]

        if not snake:
            return

        # Fruit
        if "fruit" not in self.canvas_items:
            self.canvas_items["fruit"] = self.canvas.create_image(
                fx * CELL_SIZE, fy * CELL_SIZE,
                anchor=tk.NW, image=self.apple, tags="snake"
            )
        else:
            self.canvas.coords(self.canvas_items["fruit"], fx * CELL_SIZE, fy * CELL_SIZE)

        # Remove old snake pieces
        for key in list(self.canvas_items.keys()):
            if key.startswith("snake_"):
                self.canvas.delete(self.canvas_items[key])
                del self.canvas_items[key]

        # Head
        hx, hy = snake[0]
        head_dir = self._infer_head_direction(snake)
        self.canvas_items["snake_head"] = self.canvas.create_image(
            hx * CELL_SIZE, hy * CELL_SIZE,
            anchor=tk.NW, image=self.head[head_dir], tags="snake"
        )

        # Body
        for i in range(1, len(snake) - 1):
            px, py = snake[i - 1]
            cx, cy = snake[i]
            nx, ny = snake[i + 1]

            tile = self._body_tile((px, py), (cx, cy), (nx, ny))
            self.canvas_items[f"snake_body_{i}"] = self.canvas.create_image(
                cx * CELL_SIZE, cy * CELL_SIZE,
                anchor=tk.NW, image=self.body[tile], tags="snake"
            )

        # Tail
        if len(snake) > 1:
            tx, ty = snake[-1]
            tail_dir = self._infer_tail_direction(snake)
            self.canvas_items["snake_tail"] = self.canvas.create_image(
                tx * CELL_SIZE, ty * CELL_SIZE,
                anchor=tk.NW, image=self.tail[tail_dir], tags="snake"
            )

        self._draw_score(self.last_score)

    def _infer_head_direction(self, snake):
        if len(snake) < 2:
            return "right"
        hx, hy = snake[0]
        nx, ny = snake[1]
        if hy == ny + 1:
            return "down"
        if hy == ny - 1:
            return "up"
        if hx == nx + 1:
            return "right"
        return "left"

    def _infer_tail_direction(self, snake):
        if len(snake) < 2:
            return "right"
        tx, ty = snake[-1]
        px, py = snake[-2]
        if ty == py + 1:
            return "down"
        if ty == py - 1:
            return "up"
        if tx == px + 1:
            return "right"
        return "left"

    def _body_tile(self, prev, cur, nxt):
        px, py = prev
        cx, cy = cur
        nx, ny = nxt

        if px == nx:
            return "vertical"
        if py == ny:
            return "horizontal"

        if (px < cx and ny < cy) or (nx < cx and py < cy):
            return "tl"
        if (px < cx and ny > cy) or (nx < cx and py > cy):
            return "bl"
        if (px > cx and ny < cy) or (nx > cx and py < cy):
            return "tr"
        return "br"

if __name__ == "__main__":
    app = SnakeViewerApp()
    app.mainloop()
