import os
import time
import json
import csv
import random
from collections import deque
from typing import Optional, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

import snake_gym_env  # registers Snake-v0

SAVED_DIR = "saved_models"
os.makedirs(SAVED_DIR, exist_ok=True)


# -------------------------
# Model and replay buffer
# -------------------------
class SnakeQNet(nn.Module):
    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, max_length: int = 200_000):
        self.states = deque(maxlen=max_length)
        self.actions = deque(maxlen=max_length)
        self.next_states = deque(maxlen=max_length)
        self.rewards = deque(maxlen=max_length)
        self.dones = deque(maxlen=max_length)

    def store(self, state_np, action, next_state_np, reward, done):
        self.states.append(state_np)
        self.actions.append(int(action))
        self.next_states.append(next_state_np)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def sample(self, batch_size: int):
        if len(self.dones) < batch_size:
            return None
        idx = np.random.choice(len(self.dones), batch_size, replace=False)
        states = np.stack([self.states[i] for i in idx])
        actions = np.array([self.actions[i] for i in idx], dtype=np.int64)
        next_states = np.stack([self.next_states[i] for i in idx])
        rewards = np.array([self.rewards[i] for i in idx], dtype=np.float32)
        dones = np.array([self.dones[i] for i in idx], dtype=np.float32)
        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.dones)


# -------------------------
# Action selection & Double DQN step
# -------------------------
def select_action(
    model: nn.Module,
    state_tensor: torch.Tensor,
    epsilon: float,
    num_actions: int,
    device: torch.device,
) -> int:
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    with torch.no_grad():
        q_values = model(state_tensor.unsqueeze(0).to(device))
        return int(torch.argmax(q_values, dim=1).item())


def train_step_double_dqn(
    buffer: ReplayBuffer,
    model: nn.Module,
    target_model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    gamma: float,
    batch_size: int,
    device: torch.device,
):
    batch = buffer.sample(batch_size)
    if batch is None:
        return None, None, None, None

    states_np, actions_np, next_states_np, rewards_np, dones_np = batch

    states = torch.from_numpy(states_np).float().to(device)
    next_states = torch.from_numpy(next_states_np).float().to(device)
    actions = torch.from_numpy(actions_np).long().to(device).unsqueeze(1)
    rewards = torch.from_numpy(rewards_np).float().to(device)
    dones = torch.from_numpy(dones_np).float().to(device)

    with torch.no_grad():
        next_q_online = model(next_states)
        next_actions = next_q_online.argmax(dim=1, keepdim=True)
        next_q_target = target_model(next_states)
        next_q_selected = next_q_target.gather(1, next_actions).squeeze(1)
        targets = rewards + gamma * next_q_selected * (1.0 - dones)

    q_values = model(states)
    q_action = q_values.gather(1, actions).squeeze(1)

    loss = loss_fn(q_action, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    optimizer.step()

    q_vals_cpu = q_values.detach().cpu().numpy()
    mean_q = float(np.mean(q_vals_cpu))
    max_q = float(np.max(q_vals_cpu))
    min_q = float(np.min(q_vals_cpu))
    std_q = float(np.std(q_vals_cpu))

    return float(loss.item()), mean_q, (max_q, min_q, std_q), float(total_norm)


# -------------------------
# Training loop
# -------------------------
def train_model(
    grid_size=20,
    episodes=2000,
    learning_rate=1e-4,
    batch_size=64,
    seed: Optional[int] = 42,
    print_every=100,
    log_every: Optional[int] = None,
    eval_every=50,
    save_dir=SAVED_DIR,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make("Snake-v0", render_mode=None, grid_size=grid_size)

    sample_obs, _ = env.reset()
    feat = np.asarray(sample_obs["features"], dtype=np.float32)
    input_dim = feat.shape[0]
    num_actions = env.action_space.n

    model = SnakeQNet(input_dim, num_actions).to(device)
    target_model = SnakeQNet(input_dim, num_actions).to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()
    buffer = ReplayBuffer(max_length=200_000)

    gamma = 0.99
    max_steps = 2000

    # epsilon schedule (step-based)
    eps_start = 1.0
    eps_end = 0.05
    eps_decay_steps = 200_000

    rewards_history: List[float] = []
    training_steps = 0
    losses: List[float] = []
    mean_qs: List[float] = []
    diagnostics_rows: List[dict] = []
    best_eval = -float("inf")

    warmup_steps = 5_000
    tau = 0.05  # soft update factor

    start_time = time.time()

    for episode in range(episodes):
        obs, _ = env.reset()
        state_feat = np.asarray(obs["features"], dtype=np.float32)
        episode_reward = 0.0

        for step in range(max_steps):
            # global step increments every env step
            training_steps += 1

            state_tensor = torch.from_numpy(state_feat).float().to(device)

            # epsilon schedule based on total steps
            epsilon = max(eps_end, eps_start - training_steps / eps_decay_steps)

            action = select_action(model, state_tensor, epsilon, num_actions, device)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            next_feat = np.asarray(next_obs["features"], dtype=np.float32)

            # store raw reward (no clipping)
            buffer.store(state_feat, action, next_feat, float(reward), done)
            episode_reward += float(reward)
            state_feat = next_feat

            # train after warmup
            if len(buffer) >= batch_size and training_steps > warmup_steps:
                loss_val, mean_q, q_stats, grad_norm = train_step_double_dqn(
                    buffer,
                    model,
                    target_model,
                    optimizer,
                    loss_fn,
                    gamma,
                    batch_size,
                    device,
                )

                if loss_val is not None:
                    losses.append(loss_val)
                if mean_q is not None:
                    mean_qs.append(mean_q)

                # soft update target network
                for p, tp in zip(model.parameters(), target_model.parameters()):
                    tp.data.mul_(1 - tau).add_(tau * p.data)

                if log_every and log_every > 0 and (training_steps % log_every == 0):
                    avg_loss = float(np.mean(losses[-log_every:])) if losses else 0.0
                    avg_q = float(np.mean(mean_qs[-log_every:])) if mean_qs else 0.0
                    recent_q_stats = q_stats if q_stats is not None else (
                        0.0,
                        0.0,
                        0.0,
                    )
                    recent_grad = grad_norm if grad_norm is not None else 0.0
                    print(
                        f"[TrainStep {training_steps}] avg_loss {avg_loss:.4f} "
                        f"avg_q {avg_q:.4f} q_max {recent_q_stats[0]:.4f} "
                        f"q_min {recent_q_stats[1]:.4f} q_std {recent_q_stats[2]:.4f} "
                        f"grad_norm {recent_grad:.4f}"
                    )
                    diagnostics_rows.append(
                        {
                            "train_step": training_steps,
                            "avg_loss": avg_loss,
                            "avg_q": avg_q,
                            "q_max": recent_q_stats[0],
                            "q_min": recent_q_stats[1],
                            "q_std": recent_q_stats[2],
                            "grad_norm": recent_grad,
                            "episode": episode,
                            "time": time.time() - start_time,
                        }
                    )

            if done:
                break

        rewards_history.append(episode_reward)

        # deterministic evaluation
        if (episode + 1) % eval_every == 0:
            eval_rewards = []
            eval_episodes = 10
            for _ in range(eval_episodes):
                obs_eval, _ = env.reset()
                feat_eval = np.asarray(obs_eval["features"], dtype=np.float32)
                total_r = 0.0
                for _ in range(max_steps):
                    with torch.no_grad():
                        qvals = model(
                            torch.from_numpy(feat_eval)
                            .float()
                            .unsqueeze(0)
                            .to(device)
                        )
                        act = int(torch.argmax(qvals, dim=1).item())
                    next_obs_eval, r_eval, term_eval, trunc_eval, _ = env.step(act)
                    total_r += float(r_eval)
                    feat_eval = np.asarray(
                        next_obs_eval["features"], dtype=np.float32
                    )
                    if term_eval or trunc_eval:
                        break
                eval_rewards.append(total_r)

            eval_mean = float(np.mean(eval_rewards))
            eval_median = float(np.median(eval_rewards))
            print(
                f"[Eval at ep {episode+1}] deterministic_mean_reward "
                f"{eval_mean:.2f} median {eval_median:.2f}"
            )

            if eval_mean > best_eval:
                best_eval = eval_mean
                best_path = os.path.join(save_dir, f"best_snake_grid{grid_size}.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "input_dim": int(input_dim),
                        "num_actions": int(num_actions),
                        "grid_size": int(grid_size),
                        "episodes_trained": int(episode + 1),
                        "best_eval": float(best_eval),
                    },
                    best_path,
                )
                print(
                    f"Saved new best model to {best_path} with eval {best_eval:.2f}"
                )

        if (episode % print_every == 0) or (episode < 10):
            avg100 = (
                float(np.mean(rewards_history[-100:])) if rewards_history else 0.0
            )
            elapsed = time.time() - start_time
            print(
                f"Ep {episode:4d} | R {episode_reward:6.2f} | "
                f"Avg100 {avg100:6.2f} | Eps {epsilon:.3f} | "
                f"Buf {len(buffer):5d} | Steps {training_steps:6d} | "
                f"Time {elapsed:.1f}s"
            )

    env.close()

    # final save
    model_path = os.path.join(
        save_dir, f"snake_model_grid{grid_size}_ep{episodes}.pt"
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(input_dim),
            "num_actions": int(num_actions),
            "grid_size": int(grid_size),
            "episodes_trained": int(episodes),
        },
        model_path,
    )
    print(f"Model saved to {model_path}")

    # learning curve
    curve_path = None
    if rewards_history:
        plt.figure(figsize=(8, 5))
        plt.plot(rewards_history, alpha=0.6)
        if len(rewards_history) >= 50:
            sm = np.convolve(rewards_history, np.ones(50) / 50, mode="valid")
            plt.plot(range(49, len(rewards_history)), sm, color="red", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Snake training grid {grid_size}")
        plt.grid(True)
        plt.tight_layout()
        curve_path = os.path.join(
            save_dir, f"learning_curve_grid{grid_size}.png"
        )
        plt.savefig(curve_path)
        plt.close()

    # diagnostics CSV
    diag_path = os.path.join(save_dir, f"diagnostics_grid{grid_size}.csv")
    if diagnostics_rows:
        keys = diagnostics_rows[0].keys()
        with open(diag_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(keys))
            writer.writeheader()
            for r in diagnostics_rows:
                writer.writerow(r)

    summary = {
        "model_path": model_path,
        "curve_path": curve_path,
        "diag_path": diag_path if diagnostics_rows else None,
        "best_model_path": os.path.join(
            save_dir, f"best_snake_grid{grid_size}.pt"
        ),
        "final_avg100": float(np.mean(rewards_history[-100:]))
        if len(rewards_history) >= 1
        else 0.0,
    }
    with open(
        os.path.join(save_dir, f"summary_grid{grid_size}.json"), "w"
    ) as f:
        json.dump(summary, f)

    return model_path, summary["final_avg100"]


if __name__ == "__main__":
    print("Starting training")
    model_path, final_avg = train_model(
        grid_size=20,
        episodes=200000,
        learning_rate=1e-4,
        batch_size=64,
        seed=42,
        print_every=1000,
        log_every=None,
        eval_every=500,
    )
    print(f"Finished. Final avg reward: {final_avg:.2f}. Model: {model_path}")