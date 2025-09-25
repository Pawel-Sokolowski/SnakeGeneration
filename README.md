# Snake Deep Q-Learning Agent - Simplified

A clean, simplified implementation of a Snake game environment using Gymnasium and deep Q-learning agent trained with PyTorch. This refactored version includes only the essential components for training and running Snake AI models.

## Quick Start

### Training a Model

Run the training script to train a new Snake AI model:

```bash
python main.py
```

This will train a model and save it to the `saved_models/` directory along with a learning curve plot.

### Testing a Trained Model

Launch the GUI application to select and test trained models:

```bash
python app.py
```

Select a model file and grid size, then watch the AI play Snake!

## Core Files

- `main.py` : Complete training script with model architecture and training loop
- `app.py` : GUI application for selecting models and visualizing gameplay  
- `snake_gym_env.py` : Custom Snake environment implementation for Gymnasium
- `Graphics/` : Image assets for the snake and fruit visualization
- `Sound/` : Sound effects (crunch.wav)
- `Font/` : Font file for score display
## Requirements

- Python 3.8+
- PyTorch >= 1.12.0
- Gymnasium == 1.2.0
- Pygame == 2.6.1
- NumPy
- Matplotlib
- Pillow

You can install all dependencies via pip:

```bash
pip install -r requirements
```

## Model Configuration

The simplified training script uses a single MLP model architecture:

### Model Architecture
- Input: Flattened snake position coordinates and fruit position
- Hidden layers: 128 → 64 neurons with ReLU activation
- Output: Q-values for 4 actions (up, down, left, right)
- Parameters: ~111K (efficient and fast to train)

### Training Parameters (configurable in main.py)
- Grid Size: 20x20 (default)
- Episodes: 1000 (default)
- Learning Rate: 0.001
- Batch Size: 256
- Replay Buffer: 50,000 transitions

## Environment Details

### Observation Space
- Snake body coordinates: (max_length + 1) × 2 array
- Each coordinate pair represents (x, y) position
- Padded with (-1, -1) for positions beyond current snake length
- Last coordinate pair is the fruit position

### Action Space
- 0: Move right
- 1: Move up  
- 2: Move left
- 3: Move down

### Rewards
- +10 for eating fruit
- -10 for collision (game over)
- +0.5 for moving closer to fruit
- -0.1 for moving away from fruit
- -0.005 per time step (encourages efficiency)

## Environment Registration

The environment is automatically registered as `Snake-v0` when importing `snake_gym_env`:

```python
import snake_gym_env
import gymnasium as gym

env = gym.make("Snake-v0", render_mode="human", grid_size=20)
```

## Troubleshooting

- If you get import errors, make sure all dependencies are installed
- For GUI issues on Linux, install tkinter: `sudo apt-get install python3-tk`
- Models are saved in `saved_models/` directory - make sure it has write permissions

## License

This project is provided for educational and research purposes.