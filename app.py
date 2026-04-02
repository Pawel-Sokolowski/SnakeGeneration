# app.py — CNN viewer for 32‑feature Grandmaster Snake
import os, threading, time, tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

import snake_gym_env

MODEL_DIR = "saved_models"
GRID_CHOICES = [10,15,20,25,30]
DEFAULT_GRID = 20
CELL = 30


class SnakeCNN(nn.Module):
    """CNN supporting grid + 32 expert features."""
    def __init__(self, grid, extra_dim, n_actions):
        super().__init__()
        self.grid = grid
        self.extra_dim = extra_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )

        cnn_out = 64 * grid * grid

        self.fc = nn.Sequential(
            nn.Linear(cnn_out + extra_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        grid = x[:, :-self.extra_dim]
        extras = x[:, -self.extra_dim:]

        g = self.grid
        grid = grid.view(-1,1,g,g)

        return self.fc(torch.cat([self.cnn(grid), extras], dim=1))


class SnakeQNet(nn.Module):
    """Legacy fallback MLP."""
    def __init__(self, input_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, act_dim)
        )
    def forward(self,x):
        if x.dim()==1:
            x=x.unsqueeze(0)
        return self.net(x)


def safe_load(path, device):
    try:
        return torch.load(path, map_location=device)
    except:
        return torch.load(path, map_location=device, weights_only=False)


class SnakeApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Snake Viewer (32‑feature CNN)")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.meta = {}
        self.env = None
        self.obs = None
        self.playing = False
        self.items = {}

        self._build_ui()
        self.refresh_models()

    # ============================================================
    # UI
    # ============================================================
    def _build_ui(self):
        top = tk.Frame(self); top.pack(fill=tk.X,pady=4)
        tk.Label(top, text="Model:").pack(side=tk.LEFT)

        self.model_var = tk.StringVar()
        self.model_box = ttk.Combobox(
            top, width=50, textvariable=self.model_var, state="readonly"
        )
        self.model_box.pack(side=tk.LEFT, padx=4)

        tk.Button(top, text="Refresh", command=self.refresh_models).pack(side=tk.LEFT)
        tk.Button(top, text="Load", command=self.load_model).pack(side=tk.LEFT)

        control = tk.Frame(top); control.pack(side=tk.RIGHT)
        tk.Button(control, text="Run", command=self.start_run).pack(side=tk.LEFT)
        tk.Button(control, text="Pause", command=self.pause_run).pack(side=tk.LEFT)
        tk.Button(control, text="Step", command=self.step_once).pack(side=tk.LEFT)

        mid = tk.Frame(self); mid.pack(fill=tk.X)
        tk.Label(mid, text="Grid size:").pack(side=tk.LEFT)
        self.grid_var = tk.StringVar(value=str(DEFAULT_GRID))
        ttk.Combobox(
            mid,
            values=[str(g) for g in GRID_CHOICES],
            textvariable=self.grid_var,
            width=10,
            state="readonly"
        ).pack(side=tk.LEFT, padx=4)

        sp = tk.Frame(self); sp.pack(fill=tk.X)
        tk.Label(sp, text="Speed (ms):").pack(side=tk.LEFT)
        self.speed_var = tk.IntVar(value=100)
        tk.Scale(
            sp, from_=10,to=400,orient=tk.HORIZONTAL,
            variable=self.speed_var
        ).pack(fill=tk.X,padx=4)

        self.canvas = tk.Canvas(
            self, width=DEFAULT_GRID*CELL,
            height=DEFAULT_GRID*CELL, bg="black"
        )
        self.canvas.pack(side=tk.LEFT, padx=6,pady=6)

        info = tk.Frame(self); info.pack(side=tk.RIGHT, fill=tk.Y)

        self.status = tk.Label(info, text="No model loaded")
        self.status.pack(fill=tk.X)

        tk.Label(info, text="Metadata").pack()
        self.meta_box = tk.Text(info, width=40, height=20, state=tk.DISABLED)
        self.meta_box.pack()

    # ============================================================
    # MODEL LOADING
    # ============================================================
    def refresh_models(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        files = [x for x in os.listdir(MODEL_DIR) if x.endswith(".pt")]
        files.sort()
        self.model_box["values"] = files
        if files:
            self.model_box.current(0)

    def load_model(self):
        filename = self.model_var.get()
        if not filename:
            return

        path = os.path.join(MODEL_DIR, filename)
        raw = safe_load(path, self.device)

        grid = int(self.grid_var.get())
        extra_dim = 32
        n_actions = 4

        # New-style dict?
        if isinstance(raw, dict) and "model_state_dict" in raw:
            sd = raw["model_state_dict"]
            if any(k.startswith("cnn.") for k in sd):
                self.model = SnakeCNN(grid, extra_dim, n_actions).to(self.device)
            else:
                self.model = SnakeQNet(grid*grid + extra_dim, n_actions).to(self.device)

            self.model.load_state_dict(sd)
            self.meta = raw | {"file": filename}

        else:
            # bare state dict
            sd = raw
            if any(k.startswith("cnn.") for k in sd):
                self.model = SnakeCNN(grid, extra_dim, n_actions).to(self.device)
            else:
                self.model = SnakeQNet(grid*grid + extra_dim, n_actions).to(self.device)

            self.model.load_state_dict(sd)
            self.meta = {
                "file": filename,
                "grid_size": grid,
                "extra_dim": extra_dim,
                "actions": n_actions
            }

        self.model.eval()
        self._show_meta()
        self.status.config(text=f"Loaded {filename}")

    def _show_meta(self):
        self.meta_box.config(state=tk.NORMAL)
        self.meta_box.delete("1.0", tk.END)
        for k,v in self.meta.items():
            self.meta_box.insert(tk.END, f"{k}: {v}\n")
        self.meta_box.config(state=tk.DISABLED)

    # ============================================================
    # ENV + PLAY
    # ============================================================
    def _grid(self):
        try:
            return int(self.grid_var.get())
        except:
            return DEFAULT_GRID

    def _make_env(self):
        return gym.make("Snake-v0", grid_size=self._grid())

    def start_run(self):
        if not self.model:
            return

        self.env = self._make_env()
        self.obs,_ = self.env.reset()
        self.playing = True

        self._draw_background()
        threading.Thread(target=self._play_loop, daemon=True).start()

    def pause_run(self):
        self.playing = False

    def step_once(self):
        if not self.model:
            return
        if self.env is None:
            self.env = self._make_env()
            self.obs,_ = self.env.reset()
            self._draw_background()
        self._step_logic()

    def _play_loop(self):
        try:
            while self.playing and self.env is not None:
                self._step_logic()
                time.sleep(self.speed_var.get()/1000.0)
        finally:
            self.playing = False
            if self.env:
                self.env.close()
                self.env = None

    def _step_logic(self):
        if self.env is None:
            return

        x = torch.tensor(self.obs.astype(np.float32)).to(self.device)
        with torch.no_grad():
            q = self.model(x)
            act = int(torch.argmax(q).item())

        self.obs, r, term, trunc, info = self.env.step(act)
        self._draw(self.obs)

        self.status.config(text=f"r={r:.3f}  score={info.get('score',0)}")

        if term or trunc:
            self.env.close()
            self.env = None

    # ============================================================
    # DRAWING
    # ============================================================
    def _draw_background(self):
        self.canvas.delete("all")
        g = self._grid()
        self.canvas.config(width=g*CELL, height=g*CELL)

        for y in range(g):
            for x in range(g):
                c = "#a9d864" if (x+y)%2==0 else "#9fce53"
                self.canvas.create_rectangle(
                    x*CELL,y*CELL,(x+1)*CELL,(y+1)*CELL,
                    fill=c, outline=""
                )

    def _draw(self, obs):
        g = self._grid()
        flat = obs[:g*g]
        grid = flat.reshape(g,g)

        for k,v in list(self.items.items()):
            self.canvas.delete(v)
            del self.items[k]

        hx = hy = fx = fy = -1

        for y in range(g):
            for x in range(g):
                v = grid[y,x]
                if v == 1:
                    self.items[f"b{x}_{y}"] = self.canvas.create_rectangle(
                        x*CELL,y*CELL,(x+1)*CELL,(y+1)*CELL,
                        fill="#33aa33", outline="black", width=1
                    )
                elif v == 3:
                    hx,hy = x,y
                elif v == 2:
                    fx,fy = x,y

        if hx>=0:
            self.items["head"] = self.canvas.create_rectangle(
                hx*CELL,hy*CELL,(hx+1)*CELL,(hy+1)*CELL,
                fill="#226622", outline="black", width=1
            )

        if fx>=0:
            self.items["fruit"] = self.canvas.create_oval(
                fx*CELL+4, fy*CELL+4,
                fx*CELL+CELL-4, fy*CELL+CELL-4,
                fill="red", outline="black"
            )


if __name__ == "__main__":
    SnakeApp().mainloop()