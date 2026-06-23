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

# Force 5x2 grid for 10 models
PANEL_COLS = 5

# Layout / spacing
PANEL_GAP = 16
OUTER_MARGIN = 16
BG_COLOR = (8, 8, 10)

# Green board colors
BOARD_GREEN_1 = (170, 215, 81)
BOARD_GREEN_2 = (162, 209, 73)

# Panel colors
PANEL_BG = (34, 52, 28)
PANEL_BORDER = (70, 95, 55)
HEADER_BG = (45, 70, 35)

# Timed best-run collection lengths
RUN_SECONDS_10MIN = 10 * 60
RUN_SECONDS_1H = 60 * 60

# Shared seed for fair single-run comparison
FIXED_COMPARE_SEED = 1337

# VIDEO output
OUTPUT_DIR = "assets"
VIDEO_ALL_DIE_PATH = os.path.join(OUTPUT_DIR, "all_models_until_all_dead.mp4")
VIDEO_BEST_10MIN_GRID_PATH = os.path.join(OUTPUT_DIR, "all_models_best_10min_grid.mp4")
VIDEO_BEST_1H_GRID_PATH = os.path.join(OUTPUT_DIR, "all_models_best_1hour_grid.mp4")
VIDEO_FIXED_SEED_COMPARE_PATH = os.path.join(OUTPUT_DIR, "all_models_fixed_seed_compare.mp4")

# Capture every Nth frame
CAPTURE_EVERY = 2
VIDEO_FPS = max(1, FPS // CAPTURE_EVERY)

# Optional clipping of final exported video
FINAL_VIDEO_MAX_SECONDS = None

# Hold last frame for 2 seconds
FINAL_HOLD_FRAMES = VIDEO_FPS * 2

SPRITES = {}
SCALED_SPRITES = {}

# Relative actions used by the trained models / env:
# 0 = straight, 1 = turn left, 2 = turn right
REL_ACTION_NAMES = {
    0: "STRAIGHT",
    1: "LEFT",
    2: "RIGHT",
}

# Absolute directions used by env internals / sprites:
# 0 = right, 1 = down, 2 = left, 3 = up
DIR_NAMES = {
    0: "RIGHT",
    1: "DOWN",
    2: "LEFT",
    3: "UP",
}


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

COLS = PANEL_COLS
ROWS = max(1, (len(MODELS) + COLS - 1) // COLS)

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


def draw_action_arrow(surf, direction_abs, center, size=16, color=(255, 255, 255), outline=(0, 0, 0), width=4):
    """
    Draw an outlined directional arrow using ABSOLUTE direction:
        0 = RIGHT, 1 = DOWN, 2 = LEFT, 3 = UP

    The model predicts RELATIVE actions (straight/left/right);
    Viewer.step() converts that relative action into the absolute move direction.
    """
    cx, cy = center

    if direction_abs == 0:  # RIGHT
        start = (cx - size, cy)
        end = (cx + size, cy)
        head1 = (cx + size - 8, cy - 6)
        head2 = (cx + size - 8, cy + 6)

    elif direction_abs == 1:  # DOWN
        start = (cx, cy - size)
        end = (cx, cy + size)
        head1 = (cx - 6, cy + size - 8)
        head2 = (cx + 6, cy + size - 8)

    elif direction_abs == 2:  # LEFT
        start = (cx + size, cy)
        end = (cx - size, cy)
        head1 = (cx - size + 8, cy - 6)
        head2 = (cx - size + 8, cy + 6)

    elif direction_abs == 3:  # UP
        start = (cx, cy + size)
        end = (cx, cy - size)
        head1 = (cx - 6, cy - size + 8)
        head2 = (cx + 6, cy - size + 8)

    else:
        return

    pygame.draw.line(surf, outline, start, end, width + 2)
    pygame.draw.line(surf, outline, end, head1, width + 2)
    pygame.draw.line(surf, outline, end, head2, width + 2)

    pygame.draw.line(surf, color, start, end, width)
    pygame.draw.line(surf, color, end, head1, width)
    pygame.draw.line(surf, color, end, head2, width)


def relative_action_to_absolute_dir(prev_dir: int, rel_action: int) -> int:
    """
    Env semantics (from TorchSnakeEnv.step):
        0 = straight
        1 = turn left
        2 = turn right

    Absolute directions used by the env/sprites:
        0 = right, 1 = down, 2 = left, 3 = up
    """
    if rel_action == 1:      # left turn
        return (prev_dir - 1) % 4
    if rel_action == 2:      # right turn
        return (prev_dir + 1) % 4
    return prev_dir          # straight


# =========================
# VIEWER / EPISODE TRACKER
# =========================
class Viewer:
    def __init__(self, entry, initial_seed=None):
        self.name, self.path, self.kind = entry
        self.display_name = format_model_name(self.name)

        self.best_overall = 0
        self.dead = False

        self.current_episode_frames = []
        self.current_episode_score = 0

        self.best_episode_frames = []
        self.best_episode_score = -1

        self.last_action_relative = None   # 0=straight, 1=left, 2=right
        self.last_move_direction = None    # 0=right, 1=down, 2=left, 3=up

        self.initial_seed = initial_seed
        self.env = self.make_env(seed=initial_seed)

        self.load_model()
        self.begin_new_episode()

    def make_env(self, seed=None):
        return TorchSnakeEnv(
            n=1,
            g=GRID_SIZE,
            max_steps=GRID_SIZE ** 2 * 50,
            device=device,
            random_start=True,
            seed=seed,
        )

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
        self.current_episode_score = max(0, int(self.env.length[0]) - 3)
        self.best_overall = max(self.best_overall, self.current_episode_score)
        self.last_action_relative = None
        self.last_move_direction = None

    def restart_episode(self):
        # Timed best-run collection remains stochastic on purpose.
        # Recreate env only if an explicit initial seed was requested for single-run compare.
        if self.initial_seed is not None:
            self.env = self.make_env(seed=self.initial_seed)
            obs = self.env.reset().to(device)
        else:
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

            # models output RELATIVE actions:
            #   0 = straight, 1 = left, 2 = right
            self.last_action_relative = int(action.item())

            # Convert to absolute move direction using the CURRENT env direction
            prev_dir = int(self.env.direction[0].item())
            self.last_move_direction = relative_action_to_absolute_dir(prev_dir, self.last_action_relative)

        obs, _, done, _ = self.env.step(action, return_obs="vec")

        if self.kind == "mlp":
            self.stack.pop(0)
            self.stack.append(obs.clone())

        score = max(0, int(self.env.length[0]) - 3)
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

        idxs = [(head_idx + i) % cap for i in range(length)]
        pts = [(int(xs[i]), int(ys[i])) for i in idxs]
        return pts

    def render_to_surface(self, font, small_font):
        panel = pygame.Surface((VIEW_W, VIEW_H), pygame.SRCALPHA)
        rect = panel.get_rect()

        pygame.draw.rect(panel, PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(panel, PANEL_BORDER, rect, width=1, border_radius=8)

        header = pygame.Rect(0, 0, VIEW_W, HEADER_H)
        pygame.draw.rect(
            panel,
            HEADER_BG,
            header,
            border_top_left_radius=8,
            border_top_right_radius=8
        )

        score = max(0, int(self.env.length[0]) - 3)

        title_rect = pygame.Rect(8, 6, VIEW_W - 16, 42)
        draw_wrapped_text(panel, font, self.display_name, (240, 240, 240), title_rect)

        badge = "DEAD" if self.dead else "LIVE"
        score_text = f"{badge} | score {score} | best {self.best_overall}"
        score_img = small_font.render(score_text, True, (255, 240, 180))
        panel.blit(score_img, (8, HEADER_H - 22))

        cell_w = VIEW_W // GRID_SIZE
        cell_h = (VIEW_H - HEADER_H) // GRID_SIZE
        y0 = HEADER_H
        sprites = get_scaled_sprites(cell_w, cell_h)

        # Green checkerboard board
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                r = pygame.Rect(x * cell_w, y0 + y * cell_h, cell_w, cell_h)
                color = BOARD_GREEN_1 if (x + y) % 2 == 0 else BOARD_GREEN_2
                pygame.draw.rect(panel, color, r)

        # Fruit sprite
        fx = int(self.env.fruit_x[0])
        fy = int(self.env.fruit_y[0])
        panel.blit(sprites["apple"], (fx * cell_w, y0 + fy * cell_h))

        # Snake sprite
        pts = self.get_snake_points()
        if not pts:
            return panel

        def draw_head_arrow(head_px_x, head_px_y):
            if self.last_move_direction is None:
                return

            cx = head_px_x + cell_w // 2
            cy = head_px_y + cell_h // 2
            offset = int(min(cell_w, cell_h) * 0.9)

            if self.last_move_direction == 0:      # RIGHT
                arrow_center = (cx + offset, cy)
            elif self.last_move_direction == 1:    # DOWN
                arrow_center = (cx, cy + offset)
            elif self.last_move_direction == 2:    # LEFT
                arrow_center = (cx - offset, cy)
            else:                                  # UP
                arrow_center = (cx, cy - offset)

            arrow_color = (255, 255, 255) if not self.dead else (255, 120, 120)
            draw_action_arrow(
                panel,
                self.last_move_direction,
                center=arrow_center,
                size=max(10, int(cell_w * 0.65)),
                color=arrow_color,
                outline=(0, 0, 0),
                width=4,
            )

        if len(pts) == 1:
            d = self.last_move_direction if self.last_move_direction is not None else int(self.env.direction[0])
            head_x = pts[0][0] * cell_w
            head_y = y0 + pts[0][1] * cell_h

            panel.blit(
                sprites[head_name_from_direction(d)],
                (head_x, head_y)
            )
            draw_head_arrow(head_x, head_y)
            return panel

        head_x = pts[0][0] * cell_w
        head_y = y0 + pts[0][1] * cell_h

        hname = head_sprite_name(pts[0], pts[1])
        panel.blit(sprites[hname], (head_x, head_y))
        draw_head_arrow(head_x, head_y)

        for i in range(1, len(pts) - 1):
            bname = body_sprite_name(pts[i - 1], pts[i], pts[i + 1])
            panel.blit(sprites[bname], (pts[i][0] * cell_w, y0 + pts[i][1] * cell_h))

        tname = tail_sprite_name(pts[-2], pts[-1])
        panel.blit(sprites[tname], (pts[-1][0] * cell_w, y0 + pts[-1][1] * cell_h))

        return panel


# =========================
# GRID / FRAME HELPERS
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


def sanitize_rgb_frame(frame):
    frame = np.array(frame)

    if np.issubdtype(frame.dtype, np.floating):
        frame = np.nan_to_num(frame)

    frame = np.clip(frame, 0, 255).astype(np.uint8)

    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]

    return frame


def make_blank_panel():
    panel = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
    panel[:, :] = np.array(PANEL_BG, dtype=np.uint8)

    cell_w = VIEW_W // GRID_SIZE
    cell_h = (VIEW_H - HEADER_H) // GRID_SIZE

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            yy0 = HEADER_H + y * cell_h
            xx0 = x * cell_w
            color = BOARD_GREEN_1 if (x + y) % 2 == 0 else BOARD_GREEN_2
            panel[yy0:yy0 + cell_h, xx0:xx0 + cell_w] = np.array(color, dtype=np.uint8)

    return panel


def save_video_from_rgb_frames(frames, video_path, hold_frames=FINAL_HOLD_FRAMES):
    """
    Save frames as H.264 MP4 for GitHub-compatible attachment upload.
    Much smaller than GIF in most real-world cases.
    """
    if not frames:
        print(f"No frames to save for: {video_path}")
        return

    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    writer = imageio.get_writer(
        video_path,
        format="FFMPEG",
        mode="I",
        fps=VIDEO_FPS,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=[
            "-crf", "28",
            "-preset", "medium",
            "-movflags", "+faststart",
        ],
    )

    for frame in frames:
        writer.append_data(sanitize_rgb_frame(frame))

    last_frame = sanitize_rgb_frame(frames[-1])
    for _ in range(hold_frames):
        writer.append_data(last_frame)

    writer.close()
    print(f"Saved video: {video_path}")


def build_best_grid_frames(viewers):
    per_viewer_frames = []

    for v in viewers:
        frames = v.best_episode_frames if v.best_episode_frames else v.current_episode_frames

        print(f"{v.display_name}: best_frames={len(frames)} score={v.best_episode_score}")

        if not frames:
            frames = [make_blank_panel()]

        if FINAL_VIDEO_MAX_SECONDS is not None:
            max_frames = int(FINAL_VIDEO_MAX_SECONDS * VIDEO_FPS)
            frames = frames[:max_frames]

        cleaned = [sanitize_rgb_frame(f) for f in frames]
        per_viewer_frames.append(cleaned)

    if not per_viewer_frames:
        return []

    max_len = max(len(frames) for frames in per_viewer_frames)
    grid_frames = []

    for t in range(max_len):
        panel_frames = []
        for frames in per_viewer_frames:
            panel = frames[t] if t < len(frames) else frames[-1]
            panel_frames.append(panel)

        grid = compose_grid_frame(panel_frames)
        grid_frames.append(grid)

    return grid_frames


# =========================
# RENDER / PHASE HELPERS
# =========================
def render_viewers(screen, viewers, font, small_font, status_text):
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

    info_img = small_font.render(status_text, True, (190, 190, 200))
    screen.blit(info_img, (OUTER_MARGIN, 4))

    pygame.display.flip()

    return panel_surfaces


def collect_grid_until_all_dead(screen, viewers, font, small_font, clock, status_text="Phase 1/4: recording until all models die..."):
    """
    Start all models once, never restart, and record the grid until all models are dead.
    """
    grid_frames = []
    frame_count = 0
    running = True

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return grid_frames, False

        for v in viewers:
            v.step()

        panel_surfaces = render_viewers(screen, viewers, font, small_font, status_text)
        clock.tick(FPS)

        if frame_count % CAPTURE_EVERY == 0:
            panel_frames = [panel_to_rgb_array(panel) for panel in panel_surfaces]
            grid = compose_grid_frame(panel_frames)
            grid_frames.append(grid)

        frame_count += 1

        if all(v.dead for v in viewers):
            running = False

    return grid_frames, True


def collect_best_runs_for_duration(screen, viewers, font, small_font, clock, run_seconds, phase_label="best episodes"):
    """
    Timed phase:
    - run for N seconds
    - restart models during timed collection
    - once time expires:
        * stop restarting
        * allow currently running episodes to finish
        * still let those last episodes compete to become the best
    """
    start_time = time.monotonic()
    frame_count = 0

    timed_phase = True
    running = True

    while running:
        now = time.monotonic()
        elapsed = now - start_time

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False

        if timed_phase and elapsed >= run_seconds:
            timed_phase = False
            print(f"Time limit reached for {phase_label}. Waiting for currently running episodes to finish...")

        dead_this_frame = []
        for v in viewers:
            just_died = v.step()
            dead_this_frame.append(just_died)

        if timed_phase:
            remaining = max(0, int(run_seconds - elapsed))
            mins = remaining // 60
            secs = remaining % 60
            status_text = f"{phase_label}: collecting best episodes... {mins:02d}:{secs:02d} remaining"
        else:
            alive_count = sum(not v.dead for v in viewers)
            status_text = f"{phase_label}: finishing active episodes... alive: {alive_count}"

        panel_surfaces = render_viewers(screen, viewers, font, small_font, status_text)
        clock.tick(FPS)

        if frame_count % CAPTURE_EVERY == 0:
            if timed_phase:
                for v, panel in zip(viewers, panel_surfaces):
                    rgb = panel_to_rgb_array(panel)
                    v.capture_panel_frame(rgb)
            else:
                for v, panel, just_died in zip(viewers, panel_surfaces, dead_this_frame):
                    if (not v.dead) or just_died:
                        rgb = panel_to_rgb_array(panel)
                        v.capture_panel_frame(rgb)

        frame_count += 1

        if timed_phase:
            for v, just_died in zip(viewers, dead_this_frame):
                if just_died:
                    v.maybe_replace_best_episode()
                    v.restart_episode()
        else:
            for v, just_died in zip(viewers, dead_this_frame):
                if just_died:
                    v.maybe_replace_best_episode()

            if all(v.dead for v in viewers):
                running = False

    for v in viewers:
        v.maybe_replace_best_episode()

    return True


# =========================
# MAIN
# =========================
def main():
    pygame.init()
    pygame.display.set_caption("Snake Video Collector (5x2 Grid, Green Board, Relative Head Arrow)")

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

    print(f"Loaded {len(MODELS)} models.")
    print(f"Grid layout: {COLS}x{ROWS}")

    # -------------------------
    # PHASE 1: random single run until all die
    # -------------------------
    viewers_phase1 = [Viewer(m) for m in MODELS]
    phase1_frames, continue_after_phase1 = collect_grid_until_all_dead(
        screen,
        viewers_phase1,
        font,
        small_font,
        clock,
        status_text="Phase 1/4: random single run until all models die..."
    )
    save_video_from_rgb_frames(phase1_frames, VIDEO_ALL_DIE_PATH)

    if not continue_after_phase1:
        print("Window closed during phase 1.")
        pygame.quit()
        return

    # -------------------------
    # PHASE 2: 10-minute best runs
    # -------------------------
    viewers_phase2 = [Viewer(m) for m in MODELS]
    completed_phase2 = collect_best_runs_for_duration(
        screen,
        viewers_phase2,
        font,
        small_font,
        clock,
        RUN_SECONDS_10MIN,
        phase_label="Phase 2/4 (10 min best runs)"
    )

    best_grid_10min_frames = build_best_grid_frames(viewers_phase2)
    save_video_from_rgb_frames(best_grid_10min_frames, VIDEO_BEST_10MIN_GRID_PATH)

    print("\nBest episode scores (10 minutes):")
    for v in viewers_phase2:
        print(f"{v.display_name}: {v.best_episode_score}")

    if not completed_phase2:
        print("Window closed during phase 2.")
        pygame.quit()
        return

    # -------------------------
    # PHASE 3: 1-hour best runs
    # -------------------------
    viewers_phase3 = [Viewer(m) for m in MODELS]
    completed_phase3 = collect_best_runs_for_duration(
        screen,
        viewers_phase3,
        font,
        small_font,
        clock,
        RUN_SECONDS_1H,
        phase_label="Phase 3/4 (1 hour best runs)"
    )

    best_grid_1h_frames = build_best_grid_frames(viewers_phase3)
    save_video_from_rgb_frames(best_grid_1h_frames, VIDEO_BEST_1H_GRID_PATH)

    print("\nBest episode scores (1 hour):")
    for v in viewers_phase3:
        print(f"{v.display_name}: {v.best_episode_score}")

    if not completed_phase3:
        print("Window closed during phase 3.")
        pygame.quit()
        return

    # -------------------------
    # PHASE 4: fixed-seed single comparison
    # -------------------------
    viewers_phase4 = [Viewer(m, initial_seed=FIXED_COMPARE_SEED) for m in MODELS]
    phase4_frames, continue_after_phase4 = collect_grid_until_all_dead(
        screen,
        viewers_phase4,
        font,
        small_font,
        clock,
        status_text=f"Phase 4/4: fixed-seed comparison (seed={FIXED_COMPARE_SEED}) until all models die..."
    )
    save_video_from_rgb_frames(phase4_frames, VIDEO_FIXED_SEED_COMPARE_PATH)

    if not continue_after_phase4:
        print("Window closed during phase 4.")

    print("\nDone.")
    print("Generated:")
    print(f"- {VIDEO_ALL_DIE_PATH}")
    print(f"- {VIDEO_BEST_10MIN_GRID_PATH}")
    print(f"- {VIDEO_BEST_1H_GRID_PATH}")
    print(f"- {VIDEO_FIXED_SEED_COMPARE_PATH}")

    pygame.quit()


if __name__ == "__main__":
    main()
