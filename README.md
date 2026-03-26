# Snake Deep Q-Learning Agent

A Python implementation of a Snake environment using Gymnasium and a deep Q-learning agent with PyTorch. This repo offers flexible training, a GUI for testing, and both detailed technical settings and practical quick-start instructions.

---

## Quick Start

1. **Create a virtual environment & install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements
   ```
2. **Train a new model (optional):**
   ```bash
   python main.py
   ```
   - Models are saved in `saved_models/`, along with training curves.
3. **Or use the GUI application with the included pretrained model:**
   ```bash
   python app.py
   ```
   - Select your model file and grid size to watch the AI play!

---

## Technical Details

### Environment (`snake_gym_env.py`)
- **Action space:** 4 discrete (0=right, 1=up, 2=left, 3=down)
- **Observation space:**  
  - `coords`: Body positions (head, rest padded with (-1,-1)), and fruit coordinates
  - `features`:  
    - Head x/y (normalized)
    - Fruit relative x/y (normalized)
    - Heading (x, y direction)
    - Danger ahead, left, right (1 if collision, else 0)
- **Reward function:**
  - +1 on eating fruit
  - -1 on collision (terminal)
  - +0.05 for getting closer to fruit
  - -0.05 for getting farther from fruit
  - -0.001 per time step
- **Fruit is never spawned on the snake's body.**


### Sample Feature Extraction Code
```python
def _coords_to_features(self, coords: np.ndarray) -> np.ndarray:
    # ...details...
    return np.array([
        head[0] / denom,
        head[1] / denom,
        dx / denom,
        dy / denom,
        hx, hy,
        danger(*front),
        danger(*left),
        danger(*right)
    ], dtype=np.float32)
```
- Where `danger` returns 1.0 if moving to the tested position would cause a collision.

---

## Training (main.py)

- **Optimizer:** `Adam` (default lr=1e-4)
- **Loss:** `SmoothL1Loss`
- **Replay buffer:** size 50,000
- **Discount factor:** γ = 0.99
- **Batch size:** 256
- **Target network update:** every 1000 steps
- **Epsilon-greedy schedule:**  
  - eps_start = 1, eps_end = 0.05, eps_decay_steps = 200,000

- **Supports multi-environment training:**  
  Default n_envs=8 in parallel (vectorized).

---

## Example Training Call

```python
train_model(
    grid_size=20,
    episodes=2000,
    learning_rate=1e-4,
    batch_size=256,
    seed=42,
    n_envs=8,
    save_dir="saved_models"
)
```

---

## File Structure

- `main.py` — Training loop, model, evaluation
- `app.py` — GUI to visualize or manually play
- `snake_gym_env.py` — Gymnasium environment (core rules & obs)
- `saved_models/` — Pretrained and user-trained weights (e.g., `best_snake_grid20.pt`)
- `Graphics/`, `Sound/`, `Font/` — Art assets

---

## Requirements

- Python 3.8+
- PyTorch >= 1.12.0
- Gymnasium == 1.2.0
- Pygame == 2.6.1
- NumPy, Matplotlib, Pillow

---

## Troubleshooting

- All errors usually result from missing dependencies or missing asset files.
- On Linux, for GUI, install Tkinter:  
  `sudo apt-get install python3-tk`

---

## License

Provided for educational and research purposes.
