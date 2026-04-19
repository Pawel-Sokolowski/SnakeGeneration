# app.py
import os
import pygame
import torch

from env_fast import TorchSnakeEnv
from main import QNet, valid_action_mask
from models import DuelingMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Horizontal layout: 1 row x 4 columns
ROWS = 1
COLS = 4
VIEW_W = 320
VIEW_H = 320
WIN_W = COLS * VIEW_W
WIN_H = ROWS * VIEW_H
FPS = 30

# Models and env configs (only 20x20 variants)
MODEL_CONFIGS = [
    ("MLP 20x20", 20, "mlp", "saved_models/mlp_20x20.pt"),
    ("CNN 20x20", 20, "cnn", "saved_models/cnn_20x20.pt"),
    ("EA 20x20 MLP", 20, "mlp", "saved_models/evo_20x20_from_mlp.pt"),
    ("EA 20x20 CNN", 20, "cnn", "saved_models/evo_20x20_from_cnn.pt"),
]


class Viewer:
    def __init__(self, title, grid, model_type, ckpt_path):
        self.title = title
        self.grid = grid
        self.model_type = model_type
        self.ckpt_path = ckpt_path

        self.env = TorchSnakeEnv(n=1, g=grid, max_steps=grid * grid * 50, device=device)
        obs = self.env.reset().to(device)

        if model_type == "mlp":
            self.stack = [obs.clone() for _ in range(3)]
            self.model = DuelingMLP(34 * 3).to(device)
        else:
            self.stack = None
            self.model = QNet(grid_size=grid).to(device)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if "q" in ckpt:
            self.model.load_state_dict(ckpt["q"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()

    def step(self):
        with torch.no_grad():
            if self.model_type == "mlp":
                x = torch.cat(self.stack, dim=1).to(device).float()
                qvals = self.model(x)
            else:
                grid_obs = self.env.grid_observation().to(device)
                qvals = self.model(grid_obs)

            valid = valid_action_mask(self.env).to(device)
            qvals[~valid] = -1e9
            action = qvals.argmax(1).long()

        obs, _, done = self.env.step(action)

        if self.model_type == "mlp":
            self.stack.pop(0)
            self.stack.append(obs.to(device))
        else:
            # nothing to store; grid is derived from env
            pass

        if bool(done[0].item()):
            obs = self.env.reset().to(device)
            if self.model_type == "mlp":
                self.stack = [obs.clone() for _ in range(3)]

    def render(self, surface, rect, fonts):
        pygame.draw.rect(surface, (10, 10, 10), rect)

        g = self.grid
        cell_w = rect.width / g
        cell_h = rect.height / g

        body = self.env.occupied[0].cpu()
        hx = int(self.env.snake_x[0, self.env.head[0]].item())
        hy = int(self.env.snake_y[0, self.env.head[0]].item())
        fx = int(self.env.fruit_x[0].item())
        fy = int(self.env.fruit_y[0].item())

        for y in range(g):
            for x in range(g):
                px = rect.x + int(x * cell_w)
                py = rect.y + int(y * cell_h)
                pw = int(cell_w)
                ph = int(cell_h)

                if x == fx and y == fy:
                    pygame.draw.rect(surface, (220, 50, 50), (px, py, pw, ph))
                elif body[y, x]:
                    pygame.draw.rect(surface, (50, 200, 50), (px, py, pw, ph))

                if x == hx and y == hy:
                    pygame.draw.rect(surface, (50, 50, 220), (px, py, pw, ph))

        pygame.draw.rect(surface, (80, 80, 80), rect, 1)

        text = fonts["small"].render(self.title, True, (230, 230, 230))
        surface.blit(text, (rect.x + 4, rect.y + 4))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Snake RL - 20x20 Models (Horizontal)")
    clock = pygame.time.Clock()

    fonts = {
        "small": pygame.font.SysFont("consolas", 16),
    }

    viewers = []
    for title, grid, mtype, ckpt in MODEL_CONFIGS:
        viewers.append(Viewer(title, grid, mtype, ckpt))

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        for v in viewers:
            v.step()

        screen.fill((0, 0, 0))

        for i, v in enumerate(viewers):
            row = 0
            col = i
            rect = pygame.Rect(
                col * VIEW_W,
                row * VIEW_H,
                VIEW_W,
                VIEW_H,
            )
            v.render(screen, rect, fonts)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
