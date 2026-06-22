import os
import re
import time
from pathlib import Path

import pygame
import torch
import numpy as np
import imageio

from env_fast import TorchSnakeEnv
from models import DuelingMLP, DuelingCNN, DuelingC51CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# CONFIG
# =========================
VIEW_W = 320
VIEW_H = 390
HEADER_H = 72
FPS = 30
GRID_SIZE = 20
MAX_COLS = 4

# Layout / spacing
PANEL_GAP = 16
OUTER_MARGIN = 16
BG_COLOR = (8, 8, 10)

# Run collection phase for 10 minutes
RUN_SECONDS = 10 * 60

# GIF output
OUTPUT_DIR = "assets"
GIF_PATH = os.path.join(OUTPUT_DIR, "all_models_best_runs.gif")

# Capture every Nth frame while collecting episodes
CAPTURE_EVERY = 2
GIF_FPS = max(1, FPS // CAPTURE_EVERY)

# Final GIF max duration (set to None to use full best episodes)
FINAL_GIF_MAX_SECONDS = 20

# Hold last frame of final GIF for 2 seconds
FINAL_HOLD_FRAMES = GIF_FPS * 2

SPRITES = {}
SCALED_SPRITES = {}


# =========================
# MODEL DETECTION / LISTING
# =========================
def detect_model_kind(ckpt):
    state = ckpt["q"] if "q" in ckpt else ckpt
    is_cnn = any(v.ndim == 4 for v in state.values())
    is_c51 = "support" in state

    if is_cnn and is_c51:
        return "c51_cnn"
    if is_cnn:
        return "cnn"
    return "mlp"


def load_models():
    models = []
    if not os.path.exists("saved_models"):
        return models

    for f in sorted(os.listdir("saved_models")):
        if f.endswith(".pt"):
            path = os.path.join("saved_models", f)
            ckpt = torch.load(path, map_location=device)
            kind = detect_model_kind(ckpt)
            models.append((f, path, kind))
    return models


MODELS = load_models()
COLS = max(1, min(MAX_COLS, len(MODELS) if MODELS else 1))
ROWS = max(1, ((len(MODELS) + COLS - 1) // COLS) if MODELS else 1)

WINDOW_W = OUTER_MARGIN * 2 + COLS * VIEW_W + max(0, COLS - 1) * PANEL_GAP
WINDOW_H = OUTER_MARGIN * 2 + ROWS * VIEW_H + max(0, ROWS - 1) * PANEL_GAP


# =========================
# NAME FORMATTING
# =========================
def pretty_arch(name: str) -> str:
    return {
        "mlp": "MLP",
        "cnn": "CNN",
        "c51_cnn": "C51-CNN",
    }.get(name, name.upper())


def strip_training_suffixes(stem: str) -> str:
    for suf in ("_mlp", "_cnn", "_c51"):
        if stem.endswith(suf):
            return stem[:-len(suf)]
    return stem


def format_model_name(filename: str) -> str:
    stem = strip_training_suffixes(Path(filename).stem)

    m = re.fullmatch(r"ea_g(\d+)_from_(.+)", stem)
    if m:
        gen, source = m.groups()
        return f"EA g{gen} from {format_model_name(source)}"

    m = re.fullmatch(r"(c51_cnn|cnn|mlp)_(\d+x\d+)_seed_(.+)", stem)
    if m:
        arch, size, seed = m.groups()
        return f"{size} {pretty_arch(arch)} (seed: {format_model_name(seed)})"

    m = re.fullmatch(r"(c51_cnn|cnn|mlp)_(\d+x\d+)", stem)
    if m:
        arch, size = m.groups()
        return f"{size} {pretty_arch(arch)}"

    return stem.replace("_", " ")


def draw_wrapped_text(surf, font, text, color, rect, line_spacing=2):
    words = text.split()
    lines = []
    current = ""

    for w in words:
        test = w if not current else current + " " + w
        if font.size(test)[0] <= rect.w:
            current = test
        else:
            if current:
                lines.append(current)
            current = w

    if current:
        lines.append(current)

    y = rect.y
    line_h = font.get_height() + line_spacing
    for line in lines:
        img = font.render(line, True, color)
        surf.blit(img, (rect.x, y))
        y += line_h

    return y


# =========================
# GRAPHICS
# =========================
def resolve_graphics_dir():
    for cand in ("graphics", "Graphics"):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError("Neither 'graphics' nor 'Graphics' directory exists.")


def load_sprites():
    global SPRITES

    base = resolve_graphics_dir()
    names = [
        "apple",
        "body_bl", "body_br", "body_horizontal", "body_tl", "body_tr", "body_vertical",
        "head_down", "head_left", "head_right", "head_up",
        "tail_down", "tail_left", "tail_right", "tail_up",
    ]

    SPRITES = {}
    for name in names:
        path = os.path.join(base, f"{name}.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing sprite: {path}")

        img = pygame.image.load(path)
        img = img.convert_alpha()
        SPRITES[name] = img


def get_scaled_sprites(cell_w, cell_h):
    key = (cell_w, cell_h)
    if key not in SCALED_SPRITES:
        SCALED_SPRITES[key] = {
            name: pygame.transform.smoothscale(img, (cell_w, cell_h))
            for name, img in SPRITES.items()
        }
    return SCALED_SPRITES[key]


def head_sprite_name(head, neck):
    dx = head[0] - neck[0]
    dy = head[1] - neck[1]

    if dx == 1:
        return "head_right"
    if dx == -1:
        return "head_left"
    if dy == 1:
        return "head_down"
    return "head_up"


def tail_sprite_name(pre_tail, tail):
    dx = tail[0] - pre_tail[0]
    dy = tail[1] - pre_tail[1]

    if dx == 1:
        return "tail_right"
    if dx == -1:
        return "tail_left"
    if dy == 1:
        return "tail_down"
    return "tail_up"


def body_sprite_name(prev_seg, cur_seg, next_seg):
    a = (prev_seg[0] - cur_seg[0], prev_seg[1] - cur_seg[1])
    b = (next_seg[0] - cur_seg[0], next_seg[1] - cur_seg[1])
    dirs = {a, b}

    if dirs == {(1, 0), (-1, 0)}:
        return "body_horizontal"
    if dirs == {(0, 1), (0, -1)}:
        return "body_vertical"
    if dirs == {(0, -1), (1, 0)}:
        return "body_tr"
    if dirs == {(0, -1), (-1, 0)}:
        return "body_tl"
    if dirs == {(0, 1), (1, 0)}:
        return "body_br"
    if dirs == {(0, 1), (-1, 0)}:
        return "body_bl"

    return "body_horizontal"


def head_name_from_direction(direction: int):
    return {
        0: "head_right",
        1: "head_down",
        2: "head_left",
        3: "head_up",
    }.get(int(direction), "head_up")


# =========================
# VIEWER / EPISODE TRACKER
# =========================
class Viewer:
    def __init__(self, entry):
        self.name, self.path, self.kind = entry
        self.display_name = format_model_name(self.name)

        self.best_overall = 0
        self.dead = False

        self.current_episode_frames = []
        self.current_episode_score = 0

        self.best_episode_frames = []
        self.best_episode_score = -1

        self.env = TorchSnakeEnv(
            n=1,
            g=GRID_SIZE,
            max_steps=GRID_SIZE ** 2 * 50,
            device=device,
            random_start=True,
        )

        self.load_model()
        self.begin_new_episode()

    def load_model(self):
        ckpt = torch.load(self.path, map_location=device)
        state = ckpt["q"] if "q" in ckpt else ckpt

        if self.kind == "mlp":
            self.model = DuelingMLP(34 * 3).to(device)
            obs = self.env.reset().to(device)
            self.stack = [obs.clone() for _ in range(3)]

        elif self.kind == "cnn":
            self.model = DuelingCNN(37).to(device)
            self.env.reset()

        else:
            n_atoms = state["value.weight"].shape[0]
            self.model = DuelingC51CNN(37, n_atoms).to(device)
            self.env.reset()

        self.model.load_state_dict(state)
        self.model.eval()

    def begin_new_episode(self):
        self.dead = False
        self.current_episode_frames = []
        self.current_episode_score = int(self.env.length[0])
        self.best_overall = max(self.best_overall, self.current_episode_score)

    def restart_episode(self):
        obs = self.env.reset().to(device)
        if self.kind == "mlp":
            self.stack = [obs.clone() for _ in range(3)]
        self.begin_new_episode()

    def step(self):
        if self.dead:
            return False

        with torch.no_grad():
            if self.kind == "mlp":
                x = torch.cat(self.stack, 1).float()
                q = self.model(x)
            else:
                x = self.env.cnn_features().clone().to(device)
                if self.kind == "c51_cnn":
                    probs = self.model(x)
                    q = self.model.expected_q(probs)
                else:
                    q = self.model(x)

            action = q.argmax(1)

        obs, _, done, _ = self.env.step(action, return_obs="vec")

        if self.kind == "mlp":
            self.stack.pop(0)
            self.stack.append(obs.clone())

        score = int(self.env.length[0])
        self.current_episode_score = max(self.current_episode_score, score)
        self.best_overall = max(self.best_overall, score)

        if bool(done[0]):
            self.dead = True
            return True

        return False

    def maybe_replace_best_episode(self):
        cur_score = self.current_episode_score
        cur_len = len(self.current_episode_frames)
        best_len = len(self.best_episode_frames)

        should_replace = False
        if cur_score > self.best_episode_score:
            should_replace = True
        elif cur_score == self.best_episode_score and cur_len > best_len:
            should_replace = True

        if should_replace and cur_len > 0:
            self.best_episode_score = cur_score
            self.best_episode_frames = [f.copy() for f in self.current_episode_frames]

    def capture_panel_frame(self, panel_rgb_array):
        self.current_episode_frames.append(panel_rgb_array.copy())

    def get_snake_points(self):
        length = int(self.env.length[0])
        if length <= 0:
            return []

        head_idx = int(self.env.head[0])

        xs = self.env.snake_x[0].detach().cpu().numpy()
        ys = self.env.snake_y[0].detach().cpu().numpy()
        cap = xs.shape[0]

        # IMPORTANT: your env uses head, head+1, head+2, ...
        idxs = [(head_idx + i) % cap for i in range(length)]
        pts = [(int(xs[i]), int(ys[i])) for i in idxs]
        return pts

    def render_to_surface(self, font, small_font):
        panel = pygame.Surface((VIEW_W, VIEW_H), pygame.SRCALPHA)
        rect = panel.get_rect()

        pygame.draw.rect(panel, (18, 18, 18), rect, border_radius=8)
        pygame.draw.rect(panel, (55, 55, 60), rect, width=1, border_radius=8)

        header = pygame.Rect(0, 0, VIEW_W, HEADER_H)
        pygame.draw.rect(
            panel,
            (35, 35, 35),
            header,
            border_top_left_radius=8,
            border_top_right_radius=8
        )

        score = int(self.env.length[0])

        title_rect = pygame.Rect(8, 6, VIEW_W - 16, 42)
        draw_wrapped_text(panel, font, self.display_name, (240, 240, 240), title_rect)

        badge = "DEAD" if self.dead else "LIVE"
        score_text = f"{badge} | score {score} | best {self.best_overall}"
        score_img = small_font.render(score_text, True, (255, 210, 120))
        panel.blit(score_img, (8, HEADER_H - 22))

        cell_w = VIEW_W // GRID_SIZE
        cell_h = (VIEW_H - HEADER_H) // GRID_SIZE
        y0 = HEADER_H
        sprites = get_scaled_sprites(cell_w, cell_h)

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                r = pygame.Rect(x * cell_w, y0 + y * cell_h, cell_w, cell_h)
                color = (26, 26, 26) if (x + y) % 2 == 0 else (31, 31, 31)
                pygame.draw.rect(panel, color, r)

        fx = int(self.env.fruit_x[0])
        fy = int(self.env.fruit_y[0])
        panel.blit(sprites["apple"], (fx * cell_w, y0 + fy * cell_h))

        pts = self.get_snake_points()
        if not pts:
            return panel

        if len(pts) == 1:
            d = int(self.env.direction[0])
            panel.blit(
                sprites[head_name_from_direction(d)],
                (pts[0][0] * cell_w, y0 + pts[0][1] * cell_h)
            )
            return panel

        hname = head_sprite_name(pts[0], pts[1])
        panel.blit(sprites[hname], (pts[0][0] * cell_w, y0 + pts[0][1] * cell_h))

        for i in range(1, len(pts) - 1):
            bname = body_sprite_name(pts[i - 1], pts[i], pts[i + 1])
            panel.blit(sprites[bname], (pts[i][0] * cell_w, y0 + pts[i][1] * cell_h))

        tname = tail_sprite_name(pts[-2], pts[-1])
        panel.blit(sprites[tname], (pts[-1][0] * cell_w, y0 + pts[-1][1] * cell_h))

        return panel


# =========================
# GRID COMPOSITION
# =========================
def panel_to_rgb_array(panel_surface):
    arr = pygame.surfarray.array3d(panel_surface)
    arr = np.transpose(arr, (1, 0, 2))
    return arr.astype(np.uint8)


def compose_grid_frame(panel_frames):
    canvas_h = OUTER_MARGIN * 2 + ROWS * VIEW_H + max(0, ROWS - 1) * PANEL_GAP
    canvas_w = OUTER_MARGIN * 2 + COLS * VIEW_W + max(0, COLS - 1) * PANEL_GAP

    frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    frame[:, :] = np.array(BG_COLOR, dtype=np.uint8)

    for i, pf in enumerate(panel_frames):
        col = i % COLS
        row = i // COLS

        x = OUTER_MARGIN + col * (VIEW_W + PANEL_GAP)
        y = OUTER_MARGIN + row * (VIEW_H + PANEL_GAP)

        h, w = pf.shape[:2]
        frame[y:y + h, x:x + w] = pf

    return frame


def export_best_runs_gif(viewers, gif_path):
    if not viewers:
        print("No viewers to export.")
        return

    max_len = 0
    for v in viewers:
        if len(v.best_episode_frames) > max_len:
            max_len = len(v.best_episode_frames)

    if max_len == 0:
        print("No best episode frames recorded; nothing to save.")
        return

    if FINAL_GIF_MAX_SECONDS is not None:
        max_export_frames = FINAL_GIF_MAX_SECONDS * GIF_FPS
        max_len = min(max_len, max_export_frames)

    writer = imageio.get_writer(gif_path, mode="I", fps=GIF_FPS)
    try:
        last_full_frame = None

        for t in range(max_len):
            panel_frames = []
            for v in viewers:
                if len(v.best_episode_frames) == 0:
                    blank = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
                    blank[:, :] = np.array((20, 20, 20), dtype=np.uint8)
                    panel_frames.append(blank)
                    continue

                if t < len(v.best_episode_frames):
                    panel_frames.append(v.best_episode_frames[t])
                else:
                    panel_frames.append(v.best_episode_frames[-1])

            full_frame = compose_grid_frame(panel_frames)
            writer.append_data(full_frame)
            last_full_frame = full_frame

        if last_full_frame is not None:
            for _ in range(FINAL_HOLD_FRAMES):
                writer.append_data(last_full_frame)

    finally:
        writer.close()


# =========================
# MAIN
# =========================
def main():
    pygame.init()
    pygame.display.set_caption("Snake Best-Run Collector")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    load_sprites()

    font = pygame.font.SysFont("consolas", 16)
    small_font = pygame.font.SysFont("consolas", 14)
    clock = pygame.time.Clock()

    if not MODELS:
        running = True
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False

            screen.fill(BG_COLOR)
            msg1 = font.render("No models found in saved_models/", True, (240, 240, 240))
            msg2 = small_font.render("Put .pt files there and run again.", True, (200, 200, 200))
            screen.blit(msg1, (20, 20))
            screen.blit(msg2, (20, 52))
            pygame.display.flip()
            clock.tick(FPS)

        pygame.quit()
        return

    viewers = [Viewer(m) for m in MODELS]

    start_time = time.monotonic()
    frame_count = 0
    running = True

    while running:
        now = time.monotonic()
        if now - start_time >= RUN_SECONDS:
            running = False

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        dead_this_frame = []
        for v in viewers:
            just_died = v.step()
            dead_this_frame.append(just_died)

        screen.fill(BG_COLOR)

        panel_surfaces = []
        for i, v in enumerate(viewers):
            panel = v.render_to_surface(font, small_font)
            panel_surfaces.append(panel)

            col = i % COLS
            row = i // COLS
            x = OUTER_MARGIN + col * (VIEW_W + PANEL_GAP)
            y = OUTER_MARGIN + row * (VIEW_H + PANEL_GAP)

            screen.blit(panel, (x, y))

        elapsed = now - start_time
        remaining = max(0, int(RUN_SECONDS - elapsed))
        mins = remaining // 60
        secs = remaining % 60
        progress_text = f"Collecting best runs... {mins:02d}:{secs:02d} remaining"
        prog_img = small_font.render(progress_text, True, (190, 190, 200))
        screen.blit(prog_img, (OUTER_MARGIN, 4))

        pygame.display.flip()
        clock.tick(FPS)

        if frame_count % CAPTURE_EVERY == 0:
            for v, panel in zip(viewers, panel_surfaces):
                rgb = panel_to_rgb_array(panel)
                v.capture_panel_frame(rgb)

        frame_count += 1

        for v, just_died in zip(viewers, dead_this_frame):
            if just_died:
                v.maybe_replace_best_episode()
                v.restart_episode()

    for v in viewers:
        v.maybe_replace_best_episode()

    pygame.quit()

    export_best_runs_gif(viewers, GIF_PATH)
    print("Saved GIF:", GIF_PATH)

    print("\\nBest episode scores:")
    for v in viewers:
        print(f"{v.display_name}: {v.best_episode_score}")


if __name__ == "__main__":
    main()