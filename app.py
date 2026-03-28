import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import gymnasium as gym
import torch
import torch.nn as nn
import pygame

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
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


def safe_torch_load(path: str, device: torch.device):
    return torch.load(path, map_location=device)


class SnakeViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Snake Viewer (Tkinter UI + Pygame Game)")
        self.geometry("800x320")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_meta = None

        self.play_thread = None
        self.playing = False

        # Pygame sprite surfaces (set in Pygame thread)
        self.pg_apple = None
        self.pg_head = {}
        self.pg_tail = {}
        self.pg_body = {}

        self._build_ui()
        self.refresh_models()

    # ---------------- UI ----------------
    def _build_ui(self):
        top = tk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(top, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            top, textvariable=self.model_var, width=50, state="readonly"
        )
        self.model_combo.pack(side=tk.LEFT, padx=6)

        tk.Button(top, text="Refresh", command=self.refresh_models).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Load", command=self.load_selected_model).pack(side=tk.LEFT, padx=4)

        right = tk.Frame(top)
        right.pack(side=tk.RIGHT)
        tk.Button(right, text="Run", command=self.start_run).pack(side=tk.LEFT, padx=4)
        tk.Button(right, text="Pause", command=self.pause_run).pack(side=tk.LEFT, padx=4)

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
        tk.Label(speed_frame, text="Speed (ms per step):").pack(side=tk.LEFT)
        self.speed_var = tk.IntVar(value=100)
        tk.Scale(
            speed_frame,
            from_=10,
            to=500,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
        ).pack(fill=tk.X, padx=6, expand=True)

        info = tk.Frame(self)
        info.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        tk.Label(info, text="Status").pack(anchor="w")
        self.status_label = tk.Label(info, text="No model loaded", anchor=tk.W, justify=tk.LEFT)
        self.status_label.pack(fill=tk.X, pady=4)

        tk.Label(info, text="Model metadata").pack(anchor="w", pady=(10, 0))
        self.meta_text = tk.Text(info, width=60, height=8, state=tk.DISABLED)
        self.meta_text.pack(fill=tk.BOTH, expand=True)

    # ------------- MODEL LOADING -------------
    def refresh_models(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")])
        self.model_combo["values"] = files
        if files:
            self.model_combo.current(0)

    def _infer_meta_from_env(self, grid_size: int):
        env = gym.make("Snake-v0", grid_size=grid_size)
        obs, _ = env.reset()
        feat = obs["features"]
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
            messagebox.showinfo("No model", "Select a model first.")
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

    # ------------- ENV / GRID -------------
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

    # ------------- PYGAME SPRITES -------------
    def _load_pygame_sprites(self):
        def load(name):
            path = os.path.join(GRAPHICS_DIR, name + ".png")
            img = pygame.image.load(path).convert_alpha()
            img = pygame.transform.scale(img, (CELL_SIZE, CELL_SIZE))
            return img

        self.pg_apple = load("apple")

        self.pg_head = {
            "up": load("head_up"),
            "down": load("head_down"),
            "left": load("head_left"),
            "right": load("head_right"),
        }

        self.pg_tail = {
            "up": load("tail_up"),
            "down": load("tail_down"),
            "left": load("tail_left"),
            "right": load("tail_right"),
        }

        self.pg_body = {
            "horizontal": load("body_horizontal"),
            "vertical": load("body_vertical"),
            "tl": load("body_tl"),
            "tr": load("body_tr"),
            "bl": load("body_bl"),
            "br": load("body_br"),
        }

    # ------------- RUN / PYGAME -------------
    def start_run(self):
        if not self.model:
            messagebox.showinfo("No model", "Load a model first.")
            return
        if self.playing:
            return

        env = self._make_env()
        if not env:
            return

        self.playing = True
        self.status_label.config(text="Running in Pygame...")
        self.play_thread = threading.Thread(
            target=self._pygame_loop, args=(env,), daemon=True
        )
        self.play_thread.start()

    def pause_run(self):
        self.playing = False
        self.status_label.config(text="Pause requested (Pygame will close)")

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

    def _pygame_loop(self, env):
        pygame.init()

        base_env = env.unwrapped
        grid = base_env.grid_size

        screen = pygame.display.set_mode((grid * CELL_SIZE, grid * CELL_SIZE))
        pygame.display.set_caption("Snake (Pygame)")
        clock = pygame.time.Clock()

        self._load_pygame_sprites()

        color1 = (167, 209, 93)
        color2 = (155, 198, 83)

        obs, _ = env.reset()
        running = True

        while running and self.playing:
            delay_ms = self.speed_var.get()
            delay_ms = max(10, delay_ms)
            clock.tick(1000 // delay_ms)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    self.playing = False

            with torch.no_grad():
                feat = torch.tensor(obs["features"], dtype=torch.float32, device=self.device)
                q = self.model(feat)
                action = int(torch.argmax(q).item())

            obs, reward, terminated, truncated, info = env.step(action)

            coords = obs["coords"]
            snake = [(x, y) for (x, y) in coords[:-1] if x >= 0]
            fx, fy = coords[-1]

            # Background
            for y in range(grid):
                for x in range(grid):
                    color = color1 if (x + y) % 2 == 0 else color2
                    pygame.draw.rect(
                        screen,
                        color,
                        (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    )

            # Fruit
            screen.blit(self.pg_apple, (fx * CELL_SIZE, fy * CELL_SIZE))

            if snake:
                # Head
                hx, hy = snake[0]
                head_dir = self._infer_head_direction(snake)
                screen.blit(self.pg_head[head_dir], (hx * CELL_SIZE, hy * CELL_SIZE))

                # Body
                for i in range(1, len(snake) - 1):
                    px, py = snake[i - 1]
                    cx, cy = snake[i]
                    nx, ny = snake[i + 1]
                    tile = self._body_tile((px, py), (cx, cy), (nx, ny))
                    screen.blit(self.pg_body[tile], (cx * CELL_SIZE, cy * CELL_SIZE))

                # Tail
                if len(snake) > 1:
                    tx, ty = snake[-1]
                    tail_dir = self._infer_tail_direction(snake)
                    screen.blit(self.pg_tail[tail_dir], (tx * CELL_SIZE, ty * CELL_SIZE))

            pygame.display.flip()

            if terminated or truncated:
                obs, _ = env.reset()

        pygame.quit()
        env.close()
        self.playing = False
        self.status_label.after(0, lambda: self.status_label.config(text="Stopped"))

if __name__ == "__main__":
    app = SnakeViewerApp()
    app.mainloop()