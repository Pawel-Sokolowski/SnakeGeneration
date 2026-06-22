# =====================================================
# TRAINER MODULE (LAZY-TORCH, SKIP-BEFORE-CUDA, 10% CKPT)
# Optimized:
#  - Correct final model naming (strip "final_")
#  - Always-visible logs + TTY-safe progress
#  - MLP stack buffer (no deque/cat)
#  - Curiosity optionally every K steps
#  - CNN/C51: env.step(return_obs="none") avoids wasted _features()
#  - Safe handling of env buffers (clone only where required)
# =====================================================

import os
import sys
import time
import math
import copy
import re
from typing import Optional, Set, Dict, Any, Tuple, List
from collections import deque

# =====================================================
# DIRECTORIES (ABSOLUTE) - avoids cwd surprises
# =====================================================

_BASE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

MODEL_DIR = os.path.join(_BASE, "saved_models")
CHECKPOINT_DIR = os.path.join(_BASE, "checkpoints")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =====================================================
# CONSTANTS (NO TORCH HERE)
# =====================================================

STACK = 3
FEAT_DIM_SINGLE = 34
FEAT_DIM = FEAT_DIM_SINGLE * STACK

GAMMA = 0.99
TARGET_UPDATE = 1000
DEFAULT_LOG_EVERY = 5000

# =====================================================
# LAZY IMPORTS (TORCH/ENV/MODELS ONLY WHEN NEEDED)
# =====================================================

_torch_ready = False

torch = None
np = None
F = None
autocast = None
GradScaler = None

TorchSnakeEnv = None
DuelingMLP = None
DuelingCNN = None
DuelingC51CNN = None

DEVICE = None
_SCALER = None


def _ensure_torch_ready():
    global _torch_ready, torch, np, F, autocast, GradScaler
    global TorchSnakeEnv, DuelingMLP, DuelingCNN, DuelingC51CNN
    global DEVICE, _SCALER

    if _torch_ready:
        return

    import numpy as _np
    import torch as _torch
    import torch.nn.functional as _F
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler

    from env_fast import TorchSnakeEnv as _TorchSnakeEnv
    from models import DuelingMLP as _DuelingMLP, DuelingCNN as _DuelingCNN, DuelingC51CNN as _DuelingC51CNN

    torch = _torch
    np = _np
    F = _F
    autocast = _autocast
    GradScaler = _GradScaler

    TorchSnakeEnv = _TorchSnakeEnv
    DuelingMLP = _DuelingMLP
    DuelingCNN = _DuelingCNN
    DuelingC51CNN = _DuelingC51CNN

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    _SCALER = GradScaler(enabled=(DEVICE.type == "cuda"))
    _torch_ready = True


# =====================================================
# SKIP TRAINING HELPERS (NO TORCH)
# =====================================================

def _model_regex(model_kind: str, grid: int) -> re.Pattern:
    g = f"{grid}x{grid}"
    if model_kind == "mlp":
        core = rf"mlp_{g}"
    elif model_kind == "cnn":
        core = rf"cnn_{g}"
    elif model_kind == "c51":
        core = rf"c51_cnn_{g}"
    else:
        raise ValueError(f"Unknown model_kind={model_kind}")

    pattern = rf"^(?:final_)?{core}(?:_seed_.+?)?(?:_step\d+)?\.pt$"
    return re.compile(pattern)


def find_existing_model(model_kind: str, grid: int) -> Optional[str]:
    rx = _model_regex(model_kind, grid)
    try:
        files = os.listdir(MODEL_DIR)
    except FileNotFoundError:
        return None

    def sort_key(fname: str):
        has_step = ("_step" in fname)
        m = re.search(r"_step(\d+)\.pt$", fname)
        step = int(m.group(1)) if m else -1
        return (has_step, -step)

    for f in sorted(files, key=sort_key):
        if rx.match(f):
            return os.path.join(MODEL_DIR, f)
    return None


def maybe_skip_training(model_kind: str, grid: int, is_benchmark: bool) -> Optional[str]:
    if is_benchmark:
        return None
    found = find_existing_model(model_kind, grid)
    if found:
        print(f"[SKIP] Found existing {model_kind} model for {grid}x{grid}: {found}", flush=True)
        return found
    return None


def final_out_path(model_kind: str, grid: int, ckpt_path: Optional[str]) -> str:
    if ckpt_path:
        base = os.path.basename(ckpt_path)
        base = re.sub(r"^final_", "", base)  # critical: pipeline expects no "final_"
        return os.path.join(MODEL_DIR, base)

    if model_kind == "mlp":
        name = f"mlp_{grid}x{grid}.pt"
    elif model_kind == "cnn":
        name = f"cnn_{grid}x{grid}.pt"
    elif model_kind == "c51":
        name = f"c51_cnn_{grid}x{grid}.pt"
    else:
        raise ValueError(f"Unknown model_kind={model_kind}")

    return os.path.join(MODEL_DIR, name)


# =====================================================
# CHECKPOINT HELPERS
# =====================================================

def _ckpt_prefix_from_path(ckpt_path: Optional[str]) -> str:
    base = os.path.basename(ckpt_path) if ckpt_path else "run"
    return os.path.splitext(base)[0]


def latest_checkpoint(prefix: str) -> Optional[str]:
    try:
        files = [
            f for f in os.listdir(CHECKPOINT_DIR)
            if f.startswith(prefix + "_step") and f.endswith(".pt")
        ]
    except FileNotFoundError:
        return None

    if not files:
        return None

    files.sort(key=lambda x: int(x.split("_step")[-1].split(".pt")[0]))
    return os.path.join(CHECKPOINT_DIR, files[-1])


def save_checkpoint(prefix: str, model, optim, step: int, extra: Optional[Dict[str, Any]] = None) -> str:
    _ensure_torch_ready()
    path = os.path.join(CHECKPOINT_DIR, f"{prefix}_step{step}.pt")
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "step": int(step),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"\n[CKPT] saved {path}", flush=True)
    return path


def checkpoint_steps_10pct(steps: int) -> Set[int]:
    s: Set[int] = set()
    for k in range(1, 10):
        t = int((steps * k) // 10)
        if 1 <= t < steps:
            s.add(t)
    return s


# =====================================================
# ETA PROGRESS BAR (TTY-safe fallback)
# =====================================================

def _fmt_hhmmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _progress_bar(step, total, label="", width=28, min_interval=0.15, _state=None):
    if _state is None:
        _state = {"t0": time.time(), "t_last": 0.0, "step0": max(0, step - 1), "last_line_t": 0.0}

    now = time.time()
    if (now - _state["t_last"] < min_interval) and (step < total):
        return _state
    _state["t_last"] = now

    frac = 0.0 if total <= 0 else step / float(total)
    frac = min(max(frac, 0.0), 1.0)

    filled = int(width * frac)
    bar = "█" * filled + "·" * (width - filled)
    pct = int(frac * 100)

    elapsed = max(1e-6, now - _state["t0"])
    done_steps = max(0, step - _state["step0"])
    rate = done_steps / elapsed

    remaining = max(0, total - step)
    eta_sec = remaining / rate if rate > 1e-9 else float("inf")

    eta_str = _fmt_hhmmss(eta_sec) if math.isfinite(eta_sec) else "--:--:--"
    elapsed_str = _fmt_hhmmss(elapsed)

    is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    msg = (
        f"{label} [{bar}] {pct:3d}% ({step}/{total})  "
        f"{rate:7.1f} it/s  ETA {eta_str}  ELAP {elapsed_str}"
    )

    if is_tty:
        sys.stdout.write("\r" + msg)
        sys.stdout.flush()
        if step >= total:
            sys.stdout.write("\n")
            sys.stdout.flush()
    else:
        if (now - _state["last_line_t"] > 5.0) or (step >= total):
            print(msg, flush=True)
            _state["last_line_t"] = now

    return _state


# =====================================================
# ENV FACTORY + SHAPING
# =====================================================

def make_eval_env(grid: int, n_envs: int = 256, seed: int = 123):
    _ensure_torch_ready()
    return TorchSnakeEnv(
        n=n_envs,
        g=grid,
        max_steps=grid * grid * 20,
        device=DEVICE,
        random_start=True,
        seed=seed,
    )


def shaping_coeffs(grid: int) -> Dict[str, float]:
    scale = grid / 10.0
    return {
        "progress": 0.08 / scale,
        "stall": 0.04 / scale,
        "curiosity": 0.02 * (10.0 / grid),
        "entropy": 0.01 * (10.0 / grid),
    }


def entropy_from_q(q_values):
    probs = F.softmax(q_values, dim=1)
    logp = torch.log(probs + 1e-8)
    ent = -(probs * logp).sum(dim=1)
    return ent.detach()


# =====================================================
# CURIOSITY (as in your original, unchanged)
# =====================================================

class CuriosityTrackerVec:
    def __init__(self, grid: int, decay: float = 0.999, device=None, dtype=None):
        _ensure_torch_ready()
        self.g = int(grid)
        self.decay = float(decay)
        self.device = device if device is not None else DEVICE
        self.counts = torch.zeros((self.g ** 4,), device=self.device, dtype=(dtype if dtype is not None else torch.float32))

    def _encode(self, hx, hy, fx, fy):
        g = self.g
        return (((hx * g + hy) * g + fx) * g + fy).long()

    def bonus(self, hx, hy, fx, fy):
        with torch.no_grad():
            key = self._encode(hx, hy, fx, fy)
            n = key.numel()
            if n == 0:
                return torch.empty((0,), device=self.device, dtype=self.counts.dtype)

            idx = torch.arange(n, device=key.device, dtype=torch.long)
            composite = key * (n + 1) + idx
            order = torch.argsort(composite)
            key_s = key[order]

            change = torch.ones(n, device=key.device, dtype=torch.bool)
            change[1:] = key_s[1:] != key_s[:-1]
            group = torch.cumsum(change.long(), dim=0) - 1

            first_pos = torch.where(change)[0]
            key_unique = key_s[first_pos]
            base = self.counts[key_unique]

            ar = torch.arange(n, device=key.device)
            first_pos_per = first_pos[group]
            rank = ar - first_pos_per

            bonus_sorted = 1.0 / torch.sqrt(base[group] + rank.to(base.dtype) + 1.0)

            occ = torch.bincount(group, minlength=key_unique.numel()).to(self.counts.dtype)
            self.counts[key_unique] += occ

            bonus_out = torch.empty_like(bonus_sorted)
            bonus_out[order] = bonus_sorted
            return bonus_out

    def decay_all(self):
        with torch.no_grad():
            self.counts.mul_(self.decay)
            self.counts.masked_fill_(self.counts < 1e-4, 0.0)


# =====================================================
# EARLY STOPPING
# =====================================================

class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.5):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = -1e18
        self.bad = 0

    def update(self, metric: float):
        improved = metric > (self.best + self.min_delta)
        if improved:
            self.best = float(metric)
            self.bad = 0
        else:
            self.bad += 1
        return (self.bad >= self.patience), improved


def _default_early_stop_interval(steps: int) -> int:
    return max(2_000, steps // 60)


def _default_early_stop_warmup(steps: int) -> int:
    return max(10_000, steps // 10)


def _train_metric_p50(env) -> float:
    with torch.no_grad():
        lengths = env.length.float()
        p50 = torch.quantile(lengths, 0.5).item()
        p90 = torch.quantile(lengths, 0.9).item()
        mean = lengths.mean().item()
        return 0.5 * p50 + 0.3 * p90 + 0.2 * mean


# =====================================================
# DQN TARGET
# =====================================================

def compute_target(tgt, next_obs, rew, done):
    with torch.no_grad():
        q_next = tgt(next_obs).max(1)[0]
        return rew + GAMMA * q_next * (~done)


# =====================================================
# Helper: safe CNN obs handling with env buffers
# =====================================================

def _safe_cnn_obs(env):
    """
    Returns an observation tensor safe to use for forward/backward,
    even if env.cnn_features() uses a reused buffer.
    Preferred: env provides double-buffer; fallback: clone().
    """
    # If env implements double-buffering by exposing a toggle buffers, use it.
    if hasattr(env, "_cnn_out_a") and hasattr(env, "_cnn_out_b") and hasattr(env, "_cnn_toggle"):
        # Optional interface if you add it later
        return env.cnn_features()
    # Our optimized env returns a reused buffer -> clone for correctness
    return env.cnn_features().clone()


# =====================================================
# TRAIN MLP (stack_buf shift, curiosity_every)
# =====================================================

def train_mlp(
    grid: int,
    steps: int,
    ckpt_path: Optional[str],
    batch_size: int,
    n_envs: int,
    lr: float,
    seed: Optional[int] = None,
    load_ckpt: Optional[str] = None,   # compatibility (unused)
    seed_ckpt: Optional[str] = None,
    eval_env=None,                     # compatibility (unused)
    do_early_stop: bool = True,
    early_stop_patience: int = 10,
    early_stop_min_delta: float = 0.5,
    early_stop_interval: Optional[int] = None,
    early_stop_warmup: Optional[int] = None,
    is_benchmark: bool = False,
    progress_label: Optional[str] = None,
    log_every: int = DEFAULT_LOG_EVERY,
    curiosity_every: int = 1,
):
    skipped = maybe_skip_training("mlp", grid, is_benchmark)
    if skipped:
        return skipped

    _ensure_torch_ready()

    t0 = time.time()
    torch.manual_seed(seed or 0)
    np.random.seed(seed or 0)

    out_path = final_out_path("mlp", grid, ckpt_path)
    prefix = _ckpt_prefix_from_path(ckpt_path)

    print(
        f"[TRAIN START MLP g{grid}] steps={steps} n_envs={n_envs} lr={lr} "
        f"device={DEVICE.type} out={out_path} ckpt={ckpt_path}",
        flush=True
    )

    coeff = shaping_coeffs(grid)
    curiosity = CuriosityTrackerVec(grid=grid, decay=0.999, device=DEVICE)

    env = make_eval_env(grid, n_envs=n_envs, seed=seed or 0)

    # env._features returns reusable buffer -> clone snapshot
    obs0 = env._features().clone()
    stack_buf = obs0.repeat(1, STACK)  # [n, 34*STACK]

    q = DuelingMLP(FEAT_DIM).to(DEVICE)
    if seed_ckpt:
        data = torch.load(seed_ckpt, map_location=DEVICE)
        q.load_state_dict(data.get("q", data), strict=False)

    tgt = copy.deepcopy(q).eval()
    opt = torch.optim.Adam(q.parameters(), lr=lr)

    start_step = 0
    if (not is_benchmark) and ckpt_path:
        resume = latest_checkpoint(prefix)
        if resume:
            data = torch.load(resume, map_location=DEVICE)
            q.load_state_dict(data["model"])
            opt.load_state_dict(data["optim"])
            tgt.load_state_dict(q.state_dict())
            start_step = int(data.get("step", 0))
            print(f"[RESUME] {resume} -> step {start_step}", flush=True)

    td_log: List[Tuple[int, float]] = []
    ckpt_steps = checkpoint_steps_10pct(steps)
    bar_state = None

    if early_stop_interval is None:
        early_stop_interval = _default_early_stop_interval(steps)
    if early_stop_warmup is None:
        early_stop_warmup = _default_early_stop_warmup(steps)

    stopper = EarlyStopper(patience=early_stop_patience, min_delta=early_stop_min_delta)

    idx = getattr(env, "_idx", torch.arange(env.n, device=DEVICE))

    for step in range(start_step, steps):
        obs_cat = stack_buf

        with autocast(enabled=(DEVICE.type == "cuda")):
            qvals = q(obs_cat)
            act = qvals.argmax(1)

            next_obs, rew, done, _ = env.step(act, return_obs="vec")
            # snapshot because env._features buffer is reused
            next_obs = next_obs.clone()

            hx = env.snake_x[idx, env.head]
            hy = env.snake_y[idx, env.head]

            dist = (hx - env.fruit_x).abs() + (hy - env.fruit_y).abs()
            rew = rew + coeff["progress"] * (-dist.float() / (2 * grid))

            stall = (env.steps_since_eat.float() / env.no_eat_limit).clamp(0, 1)
            length = (env.length.float() / (grid * grid)).clamp(0, 1)
            rew = rew - coeff["stall"] * stall * (1.0 - length.pow(0.7))

            if curiosity_every <= 1 or ((step + 1) % curiosity_every == 0):
                cur = curiosity.bonus(hx, hy, env.fruit_x, env.fruit_y)
                rew = rew + coeff["curiosity"] * cur
            curiosity.decay_all()

            rew = rew + coeff["entropy"] * entropy_from_q(qvals)

            if done.any():
                reset_obs = env.reset(mask=done).clone()
                next_obs[done] = reset_obs[done]

            # shift stack_buf and append
            stack_buf[:, :-FEAT_DIM_SINGLE] = stack_buf[:, FEAT_DIM_SINGLE:]
            stack_buf[:, -FEAT_DIM_SINGLE:] = next_obs

            next_cat = stack_buf
            target = compute_target(tgt, next_cat, rew, done)
            q_sa = qvals.gather(1, act[:, None]).squeeze(1)
            loss = F.smooth_l1_loss(q_sa, target)

        opt.zero_grad(set_to_none=True)
        _SCALER.scale(loss).backward()
        _SCALER.step(opt)
        _SCALER.update()

        if step % TARGET_UPDATE == 0:
            tgt.load_state_dict(q.state_dict())

        if step % 200 == 0:
            td_log.append((step, float(loss.item())))

        if (step + 1) % max(1, int(log_every)) == 0:
            mean_len = float(env.length.float().mean().item())
            print(f"[TRAIN MLP g{grid}] step={step+1}/{steps} loss={loss.item():.4f} mean_len={mean_len:.2f}", flush=True)

        if (not is_benchmark) and do_early_stop and (step + 1) >= early_stop_warmup and ((step + 1) % early_stop_interval == 0):
            metric = _train_metric_p50(env)
            stop, improved = stopper.update(metric)
            tag = "IMPROVED" if improved else f"pat={stopper.bad}/{stopper.patience}"
            print(f"[EARLYSTOP MLP g{grid}] step={step+1} metric={metric:.2f} {tag}", flush=True)
            if stop:
                print(f"[EARLYSTOP MLP g{grid}] stopping at step {step+1} (best={stopper.best:.2f})", flush=True)
                break

        if is_benchmark:
            label = progress_label or f"BENCH MLP g{grid}"
            bar_state = _progress_bar(step + 1, steps, label=label, _state=bar_state)
        else:
            if ckpt_path and ((step + 1) in ckpt_steps):
                save_checkpoint(prefix, q, opt, step + 1)
                pct = int(100 * (step + 1) / steps)
                print(f"[TRAIN MLP g{grid}] {pct}% mean_len={env.length.float().mean():.1f}", flush=True)

    lengths = env.length.float()
    eval_p50 = torch.quantile(lengths, 0.5).item()
    it_per_s = (steps * n_envs) / max(1e-6, time.time() - t0)

    if is_benchmark:
        return {"td_log": td_log, "eval_p50": eval_p50, "it_per_s": it_per_s}

    torch.save({"q": q.state_dict()}, out_path)
    print(f"[FINAL MLP] saved {out_path}", flush=True)
    return out_path


# =====================================================
# TRAIN CNN (step(return_obs="none"), safe cnn obs)
# =====================================================

def train_cnn(
    grid: int,
    steps: int,
    ckpt_path: Optional[str],
    batch_size: int,
    n_envs: int,
    lr: float,
    seed: Optional[int] = None,
    load_ckpt: Optional[str] = None,   # compatibility (unused)
    seed_ckpt: Optional[str] = None,
    eval_env=None,                     # compatibility (unused)
    do_early_stop: bool = True,
    early_stop_patience: int = 10,
    early_stop_min_delta: float = 0.5,
    early_stop_interval: Optional[int] = None,
    early_stop_warmup: Optional[int] = None,
    is_benchmark: bool = False,
    progress_label: Optional[str] = None,
    log_every: int = DEFAULT_LOG_EVERY,
    curiosity_every: int = 1,
):
    skipped = maybe_skip_training("cnn", grid, is_benchmark)
    if skipped:
        return skipped

    _ensure_torch_ready()

    t0 = time.time()
    torch.manual_seed(seed or 0)
    np.random.seed(seed or 0)

    out_path = final_out_path("cnn", grid, ckpt_path)
    prefix = _ckpt_prefix_from_path(ckpt_path)

    print(
        f"[TRAIN START CNN g{grid}] steps={steps} n_envs={n_envs} lr={lr} "
        f"device={DEVICE.type} out={out_path} ckpt={ckpt_path}",
        flush=True
    )

    coeff = shaping_coeffs(grid)
    curiosity = CuriosityTrackerVec(grid=grid, decay=0.999, device=DEVICE)

    env = make_eval_env(grid, n_envs=n_envs, seed=seed or 0)

    # IMPORTANT: cnn_features may return a reused buffer -> get safe snapshot
    obs = _safe_cnn_obs(env)

    q = DuelingCNN(37).to(DEVICE)
    if seed_ckpt:
        data = torch.load(seed_ckpt, map_location=DEVICE)
        q.load_state_dict(data.get("q", data), strict=False)

    tgt = copy.deepcopy(q).eval()
    opt = torch.optim.Adam(q.parameters(), lr=lr)

    start_step = 0
    if (not is_benchmark) and ckpt_path:
        resume = latest_checkpoint(prefix)
        if resume:
            data = torch.load(resume, map_location=DEVICE)
            q.load_state_dict(data["model"])
            opt.load_state_dict(data["optim"])
            tgt.load_state_dict(q.state_dict())
            start_step = int(data.get("step", 0))
            print(f"[RESUME] {resume} -> step {start_step}", flush=True)

    td_log: List[Tuple[int, float]] = []
    ckpt_steps = checkpoint_steps_10pct(steps)
    bar_state = None

    if early_stop_interval is None:
        early_stop_interval = _default_early_stop_interval(steps)
    if early_stop_warmup is None:
        early_stop_warmup = _default_early_stop_warmup(steps)

    stopper = EarlyStopper(patience=early_stop_patience, min_delta=early_stop_min_delta)

    idx = getattr(env, "_idx", torch.arange(env.n, device=DEVICE))

    for step in range(start_step, steps):
        with autocast(enabled=(DEVICE.type == "cuda")):
            qvals = q(obs)
            act = qvals.argmax(1)

            # Do not compute vec obs in step() (saves time)
            _, rew, done, _ = env.step(act, return_obs="none")

            # next obs snapshot (safe)
            next_obs = _safe_cnn_obs(env)

            hx = env.snake_x[idx, env.head]
            hy = env.snake_y[idx, env.head]

            dist = (hx - env.fruit_x).abs() + (hy - env.fruit_y).abs()
            rew = rew + coeff["progress"] * (-dist.float() / (2 * grid))

            stall = (env.steps_since_eat.float() / env.no_eat_limit).clamp(0, 1)
            length = (env.length.float() / (grid * grid)).clamp(0, 1)
            rew = rew - coeff["stall"] * stall * (1.0 - length.pow(0.7))

            if curiosity_every <= 1 or ((step + 1) % curiosity_every == 0):
                cur = curiosity.bonus(hx, hy, env.fruit_x, env.fruit_y)
                rew = rew + coeff["curiosity"] * cur
            curiosity.decay_all()

            rew = rew + coeff["entropy"] * entropy_from_q(qvals)

            if done.any():
                env.reset(mask=done)  # cnn obs rebuilt next line anyway
                next_obs = _safe_cnn_obs(env)

            target = compute_target(tgt, next_obs, rew, done)
            q_sa = qvals.gather(1, act[:, None]).squeeze(1)
            loss = F.smooth_l1_loss(q_sa, target)

        opt.zero_grad(set_to_none=True)
        _SCALER.scale(loss).backward()
        _SCALER.step(opt)
        _SCALER.update()

        obs = next_obs

        if step % TARGET_UPDATE == 0:
            tgt.load_state_dict(q.state_dict())

        if step % 200 == 0:
            td_log.append((step, float(loss.item())))

        if (step + 1) % max(1, int(log_every)) == 0:
            mean_len = float(env.length.float().mean().item())
            print(f"[TRAIN CNN g{grid}] step={step+1}/{steps} loss={loss.item():.4f} mean_len={mean_len:.2f}", flush=True)

        if (not is_benchmark) and do_early_stop and (step + 1) >= early_stop_warmup and ((step + 1) % early_stop_interval == 0):
            metric = _train_metric_p50(env)
            stop, improved = stopper.update(metric)
            tag = "IMPROVED" if improved else f"pat={stopper.bad}/{stopper.patience}"
            print(f"[EARLYSTOP CNN g{grid}] step={step+1} metric={metric:.2f} {tag}", flush=True)
            if stop:
                print(f"[EARLYSTOP CNN g{grid}] stopping at step {step+1} (best={stopper.best:.2f})", flush=True)
                break

        if is_benchmark:
            label = progress_label or f"BENCH CNN g{grid}"
            bar_state = _progress_bar(step + 1, steps, label=label, _state=bar_state)
        else:
            if ckpt_path and ((step + 1) in ckpt_steps):
                save_checkpoint(prefix, q, opt, step + 1)
                pct = int(100 * (step + 1) / steps)
                print(f"[TRAIN CNN g{grid}] {pct}% mean_len={env.length.float().mean():.1f}", flush=True)

    lengths = env.length.float()
    eval_p50 = torch.quantile(lengths, 0.5).item()
    it_per_s = (steps * n_envs) / max(1e-6, time.time() - t0)

    if is_benchmark:
        return {"td_log": td_log, "eval_p50": eval_p50, "it_per_s": it_per_s}

    torch.save({"q": q.state_dict()}, out_path)
    print(f"[FINAL CNN] saved {out_path}", flush=True)
    return out_path


# =====================================================
# C51 projection
# =====================================================

def c51_project(next_probs, rewards, dones, support, v_min, v_max, gamma):
    with torch.no_grad():
        B, N = next_probs.shape
        delta_z = (v_max - v_min) / (N - 1)

        Tz = rewards[:, None] + gamma * support[None, :] * (~dones[:, None])
        Tz = Tz.clamp(v_min, v_max)

        b = (Tz - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        proj = torch.zeros_like(next_probs)
        offset = (torch.arange(B, device=next_probs.device) * N).unsqueeze(1)

        proj_flat = proj.view(-1)
        next_flat = next_probs.view(-1)

        l_idx = (l + offset).view(-1)
        u_idx = (u + offset).view(-1)

        b_flat = b.view(-1)
        l_flat = l.view(-1).float()
        u_flat = u.view(-1).float()

        eq = (l_idx == u_idx)
        proj_flat.index_add_(0, l_idx[eq], next_flat[eq])

        neq = ~eq
        if neq.any():
            proj_flat.index_add_(0, l_idx[neq], next_flat[neq] * (u_flat[neq] - b_flat[neq]))
            proj_flat.index_add_(0, u_idx[neq], next_flat[neq] * (b_flat[neq] - l_flat[neq]))

        return proj


# =====================================================
# TRAIN C51 CNN (step(return_obs="none"), safe cnn obs)
# =====================================================

def train_c51_cnn(
    grid: int,
    steps: int,
    ckpt_path: Optional[str],
    n_envs: int,
    lr: float,
    n_atoms: int,
    v_min: float,
    v_max: float,
    seed: Optional[int] = None,
    seed_ckpt: Optional[str] = None,
    do_early_stop: bool = True,
    early_stop_patience: int = 10,
    early_stop_min_delta: float = 0.5,
    early_stop_interval: Optional[int] = None,
    early_stop_warmup: Optional[int] = None,
    is_benchmark: bool = False,
    progress_label: Optional[str] = None,
    log_every: int = DEFAULT_LOG_EVERY,
    curiosity_every: int = 1,
):
    skipped = maybe_skip_training("c51", grid, is_benchmark)
    if skipped:
        return skipped

    _ensure_torch_ready()

    t0 = time.time()
    torch.manual_seed(seed or 0)
    np.random.seed(seed or 0)

    out_path = final_out_path("c51", grid, ckpt_path)
    prefix = _ckpt_prefix_from_path(ckpt_path)

    print(
        f"[TRAIN START C51 g{grid}] steps={steps} n_envs={n_envs} lr={lr} atoms={n_atoms} v=({v_min},{v_max}) "
        f"device={DEVICE.type} out={out_path} ckpt={ckpt_path} bench={is_benchmark}",
        flush=True
    )

    curiosity = CuriosityTrackerVec(grid=grid, decay=0.999, device=DEVICE)
    coeff_cur = shaping_coeffs(grid)["curiosity"]

    env = make_eval_env(grid, n_envs=n_envs, seed=seed or 0)
    obs = _safe_cnn_obs(env)

    q = DuelingC51CNN(37, n_atoms=n_atoms, v_min=v_min, v_max=v_max).to(DEVICE)
    if seed_ckpt:
        data = torch.load(seed_ckpt, map_location=DEVICE)
        q.load_state_dict(data.get("q", data), strict=False)

    tgt = copy.deepcopy(q).eval()
    opt = torch.optim.Adam(q.parameters(), lr=lr)

    start_step = 0
    if (not is_benchmark) and ckpt_path:
        resume = latest_checkpoint(prefix)
        if resume:
            data = torch.load(resume, map_location=DEVICE)
            q.load_state_dict(data["model"])
            opt.load_state_dict(data["optim"])
            tgt.load_state_dict(q.state_dict())
            start_step = int(data.get("step", 0))
            print(f"[RESUME] {resume} -> step {start_step}", flush=True)

    td_log: List[Tuple[int, float]] = []
    ckpt_steps = checkpoint_steps_10pct(steps)
    bar_state = None

    if early_stop_interval is None:
        early_stop_interval = _default_early_stop_interval(steps)
    if early_stop_warmup is None:
        early_stop_warmup = _default_early_stop_warmup(steps)

    stopper = EarlyStopper(patience=early_stop_patience, min_delta=early_stop_min_delta)

    support = q.support
    idx = getattr(env, "_idx", torch.arange(env.n, device=DEVICE))

    for step in range(start_step, steps):
        with autocast(enabled=(DEVICE.type == "cuda")):
            probs = q(obs)                # [B,A,N]
            qvals = q.expected_q(probs)   # [B,A]
            act = qvals.argmax(1)         # [B]

            _, rew, done, _ = env.step(act, return_obs="none")
            next_obs = _safe_cnn_obs(env)

            hx = env.snake_x[idx, env.head]
            hy = env.snake_y[idx, env.head]

            if curiosity_every <= 1 or ((step + 1) % curiosity_every == 0):
                cur = curiosity.bonus(hx, hy, env.fruit_x, env.fruit_y)
                rew = rew + coeff_cur * cur
            curiosity.decay_all()

            with torch.no_grad():
                next_probs_all = tgt(next_obs)                      # [B,A,N]
                next_q_all = tgt.expected_q(next_probs_all)         # [B,A]
                next_act = next_q_all.argmax(1)                     # [B]
                next_probs = next_probs_all[idx, next_act]          # [B,N]

                target = c51_project(
                    next_probs=next_probs,
                    rewards=rew,
                    dones=done,
                    support=support,
                    v_min=v_min,
                    v_max=v_max,
                    gamma=GAMMA,
                )

            pred = probs[idx, act]  # [B,N]
            loss = -(target * torch.log(pred + 1e-8)).sum(dim=1).mean()

        opt.zero_grad(set_to_none=True)
        _SCALER.scale(loss).backward()
        _SCALER.step(opt)
        _SCALER.update()

        obs = next_obs
        if done.any():
            env.reset(mask=done)
            obs = _safe_cnn_obs(env)

        if step % TARGET_UPDATE == 0:
            tgt.load_state_dict(q.state_dict())

        if step % 200 == 0:
            td_log.append((step, float(loss.item())))

        if (step + 1) % max(1, int(log_every)) == 0:
            mean_len = float(env.length.float().mean().item())
            print(f"[TRAIN C51 g{grid}] step={step+1}/{steps} loss={loss.item():.4f} mean_len={mean_len:.2f}", flush=True)

        if (not is_benchmark) and do_early_stop and (step + 1) >= early_stop_warmup and ((step + 1) % early_stop_interval == 0):
            metric = _train_metric_p50(env)
            stop, improved = stopper.update(metric)
            tag = "IMPROVED" if improved else f"pat={stopper.bad}/{stopper.patience}"
            print(f"[EARLYSTOP C51 g{grid}] step={step+1} metric={metric:.2f} {tag}", flush=True)
            if stop:
                print(f"[EARLYSTOP C51 g{grid}] stopping at step {step+1} (best={stopper.best:.2f})", flush=True)
                break

        if is_benchmark:
            label = progress_label or f"BENCH C51 g{grid}"
            bar_state = _progress_bar(step + 1, steps, label=label, _state=bar_state)
        else:
            if ckpt_path and ((step + 1) in ckpt_steps):
                save_checkpoint(prefix, q, opt, step + 1, extra={"c51": {"n_atoms": n_atoms, "v_min": v_min, "v_max": v_max}})
                pct = int(100 * (step + 1) / steps)
                print(f"[TRAIN C51 g{grid}] {pct}% mean_len={env.length.float().mean():.1f}", flush=True)

    lengths = env.length.float()
    eval_p50 = torch.quantile(lengths, 0.5).item()
    it_per_s = (steps * n_envs) / max(1e-6, time.time() - t0)

    if is_benchmark:
        return {"td_log": td_log, "eval_p50": eval_p50, "it_per_s": it_per_s}

    torch.save({"q": q.state_dict(), "c51": {"n_atoms": n_atoms, "v_min": v_min, "v_max": v_max}}, out_path)
    print(f"[FINAL C51] saved {out_path}", flush=True)
    return out_path

class EAConfig:
    def __init__(
        self,
        pop_sizes=(32,),
        n_envs_list=(128,),
        base_sigma=0.05,
        sigma_min_factor=0.25,
        sigma_max_factor=2.0,
        sigma_decay=0.9,
        sigma_growth=1.1,
        generations=40,
        best_of_n=3,
        stagnation_patience=10,
        restart_patience=15,
        eval_steps=400,
    ):
        self.pop_sizes = list(pop_sizes)
        self.n_envs_list = list(n_envs_list)
        self.base_sigma = float(base_sigma)
        self.sigma_min = float(base_sigma * sigma_min_factor)
        self.sigma_max = float(base_sigma * sigma_max_factor)
        self.sigma_decay = float(sigma_decay)
        self.sigma_growth = float(sigma_growth)
        self.generations = int(generations)
        self.best_of_n = int(best_of_n)
        self.stagnation_patience = int(stagnation_patience)
        self.restart_patience = int(restart_patience)
        self.eval_steps = int(eval_steps)


class EAEvaluator:
    def __init__(self, model_type, grid, cfg: EAConfig, c51_meta=None):
        self.model_type = model_type
        self.grid = grid
        self.cfg = cfg
        self.c51_meta = c51_meta or {}

    def eval_model(self, model, n_envs: int) -> float:
        _ensure_torch_ready()
        scores_all = []

        with torch.no_grad():
            for _ in range(self.cfg.best_of_n):
                env = TorchSnakeEnv(
                    n=n_envs,
                    g=self.grid,
                    max_steps=self.grid * self.grid * 20,
                    device=DEVICE,
                    random_start=True,
                )
                env.reset()

                if self.model_type == "mlp":
                    obs = env._features()
                    stack = deque([obs.clone()] * STACK, maxlen=STACK)
                    for _ in range(self.cfg.eval_steps):
                        x = torch.cat(list(stack), dim=1)
                        act = model(x).argmax(1)
                        nxt, _, done, _ = env.step(act)
                        if done.any():
                            nxt[done] = env.reset(mask=done)[done]
                        stack.append(nxt.clone())

                elif self.model_type == "cnn":
                    obs = env.cnn_features()
                    for _ in range(self.cfg.eval_steps):
                        act = model(obs).argmax(1)
                        _, _, done, _ = env.step(act)
                        obs = env.cnn_features()
                        if done.any():
                            env.reset(mask=done)

                elif self.model_type == "c51":
                    obs = env.cnn_features()
                    for _ in range(self.cfg.eval_steps):
                        probs = model(obs)
                        qvals = model.expected_q(probs)
                        act = qvals.argmax(1)
                        _, _, done, _ = env.step(act)
                        obs = env.cnn_features()
                        if done.any():
                            env.reset(mask=done)

                idx = getattr(env, "_idx", torch.arange(env.n, device=DEVICE))
                hx = env.snake_x[idx, env.head]
                hy = env.snake_y[idx, env.head]
                dist = (hx - env.fruit_x).abs() + (hy - env.fruit_y).abs()

                progress = (-dist.float()).clamp(min=-50, max=0)
                stagnation = (env.steps_since_eat.float() / env.no_eat_limit).clamp(0, 1)

                score = env.length.float() + 0.3 * progress - 0.5 * stagnation
                scores_all.append(score)

            scores_all = torch.cat(scores_all, dim=0)
            return float(torch.quantile(scores_all, 0.5).item())


class EAMutator:
    def __init__(self, cfg: EAConfig):
        self.cfg = cfg
        self.sigma = cfg.base_sigma

    def adapt_sigma(self, improved: bool):
        if improved:
            self.sigma = max(self.cfg.sigma_min, self.sigma * self.cfg.sigma_decay)
        else:
            self.sigma = min(self.cfg.sigma_max, self.sigma * self.cfg.sigma_growth)

    def mutate(self, model):
        _ensure_torch_ready()
        child = copy.deepcopy(model)
        with torch.no_grad():
            for p in child.parameters():
                p.add_(self.sigma * torch.randn_like(p))
        return child

    def restart_population(self, elite, pop_size: int):
        return [elite] + [self.mutate(elite) for _ in range(pop_size - 1)]


class EARunner:
    def __init__(self, model_type, grid, cfg: EAConfig, c51_meta=None):
        self.model_type = model_type
        self.grid = grid
        self.cfg = cfg
        self.eval = EAEvaluator(model_type, grid, cfg, c51_meta=c51_meta)
        self.mut = EAMutator(cfg)

    def run(self, base_model):
        _ensure_torch_ready()
        best_global = copy.deepcopy(base_model)
        best_global_score = -1e9

        for pop_size in self.cfg.pop_sizes:
            for n_envs in self.cfg.n_envs_list:
                population = [base_model] + [self.mut.mutate(base_model) for _ in range(pop_size - 1)]
                scores = [self.eval.eval_model(m, n_envs) for m in population]

                best_local_score = max(scores)
                best_local = copy.deepcopy(population[int(np.argmax(scores))])

                no_improve = 0
                no_restart = 0

                for gen in range(self.cfg.generations):
                    elite_idx = int(np.argmax(scores))
                    elite = population[elite_idx]
                    elite_score = scores[elite_idx]

                    improved = elite_score > best_local_score
                    if improved:
                        best_local = copy.deepcopy(elite)
                        best_local_score = elite_score
                        no_improve = 0
                    else:
                        no_improve += 1

                    if elite_score > best_global_score:
                        best_global = copy.deepcopy(elite)
                        best_global_score = elite_score

                    print(
                        f"[EA g{self.grid} {self.model_type}] gen={gen:02d} "
                        f"elite={elite_score:.2f} best_global={best_global_score:.2f} "
                        f"sigma={self.mut.sigma:.4f}",
                        flush=True
                    )

                    if no_improve >= self.cfg.stagnation_patience:
                        print("[EA] Early stop (no improvement)", flush=True)
                        break

                    if no_restart >= self.cfg.restart_patience:
                        print("[EA] Restarting from best_local", flush=True)
                        population = self.mut.restart_population(best_local, pop_size)
                        scores = [self.eval.eval_model(m, n_envs) for m in population]
                        no_restart = 0
                        continue

                    self.mut.adapt_sigma(improved)
                    population = [elite] + [self.mut.mutate(elite) for _ in range(pop_size - 1)]
                    scores = [self.eval.eval_model(m, n_envs) for m in population]
                    no_restart += 1

        return best_global, best_global_score


def run_ea_best_observed(model_type: str, grid: int, seed_ckpt_path: str, seed: Optional[int] = None) -> str:
    """
    EA starting from a seed checkpoint.
    SAFE with multiprocessing:
      - load checkpoint on CPU first (no CUDA init during torch.load)
      - then initialize torch/cuda and move model to DEVICE
    """
    seed_tag = os.path.splitext(os.path.basename(seed_ckpt_path))[0]
    out_name = f"ea_g{grid}_from_{seed_tag}_{model_type}.pt"
    out_path = os.path.join(MODEL_DIR, out_name)
    if os.path.exists(out_path):
        print(f"[SKIP] Found existing EA model at {out_path} -> skipping EA", flush=True)
        return out_path

    # Load checkpoint on CPU to avoid CUDA init inside torch.load (important for mp/fork safety)
    import torch as _torch
    import numpy as _np

    if seed is not None:
        _torch.manual_seed(seed)
        _np.random.seed(seed)

    data = _torch.load(seed_ckpt_path, map_location="cpu")
    c51_meta = data.get("c51", None)

    # Now init torch/env/models and choose DEVICE
    _ensure_torch_ready()

    # Build base model on DEVICE and load weights from CPU dict
    if model_type == "mlp":
        base = DuelingMLP(FEAT_DIM).to(DEVICE)
        base.load_state_dict(data["q"], strict=False)
    elif model_type == "cnn":
        base = DuelingCNN(37).to(DEVICE)
        base.load_state_dict(data["q"], strict=False)
    elif model_type == "c51":
        if not c51_meta:
            raise RuntimeError("C51 EA requested but checkpoint has no 'c51' metadata.")
        base = DuelingC51CNN(
            37,
            n_atoms=c51_meta["n_atoms"],
            v_min=c51_meta["v_min"],
            v_max=c51_meta["v_max"],
        ).to(DEVICE)
        base.load_state_dict(data["q"], strict=False)
    else:
        raise ValueError("model_type must be 'mlp', 'cnn', or 'c51'")

    for p in base.parameters():
        p.requires_grad_(False)

    cfg = EAConfig(pop_sizes=(32,), n_envs_list=(128,), base_sigma=0.05)
    runner = EARunner(model_type, grid, cfg, c51_meta=c51_meta)
    best_model, best_score = runner.run(base)

    payload = {"q": best_model.state_dict()}
    if c51_meta:
        payload["c51"] = c51_meta

    _torch.save(payload, out_path)
    print(f"[EA SAVED] {out_path} best_score={best_score:.2f}", flush=True)
    return out_path