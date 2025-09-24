# Snake Deep Q-Learning Agent - GTX 1070 Optimized

This repository contains optimized Deep Q-Learning (DQN) implementations for playing Snake, specifically designed to run efficiently on a GTX 1070 with 8GB VRAM while providing fast training times.

## 🚀 GTX 1070 Optimizations

### Model Architectures
- **SnakeCNNQNet**: Lightweight CNN with 1D convolutions (~139K parameters, 0.5MB)
- **SnakeMLPQNet**: Simple MLP baseline (~31-82K parameters, 0.1-0.3MB depending on grid size)

### Memory Optimizations
- **Grid Size Options**: 15x15, 20x20 (vs original 40x40) for faster training
- **Batch Sizes**: 256, 512 (vs original 4096) to fit in 8GB VRAM
- **Environment Count**: 8, 16 (vs original 64) for balanced performance
- **Buffer Size**: 50K (vs 100K) for reduced memory usage
- **Single GPU Training**: No DDP overhead, optimized for single GPU

### Training Optimizations
- **Enhanced Reward System**: +10 for fruit, -10 for collision, +0.5 for moving closer
- **Faster Convergence**: Improved reward shaping and epsilon decay
- **Early Stopping**: Automatic stopping when target performance reached
- **Memory Monitoring**: Real-time VRAM usage tracking
- **Gradient Clipping**: Stable training with max_norm=1.0

## 📊 Performance Comparison

| Configuration | Parameters | VRAM Usage | Training Time | Target Score |
|--------------|------------|------------|---------------|--------------|
| CNN 15x15    | 139K       | ~2.4MB     | ~1-2 hours    | 15+ points   |
| CNN 20x20    | 139K       | ~3.4MB     | ~2-3 hours    | 25+ points   |
| MLP 15x15    | 31K        | ~1.4MB     | ~30-60 min    | 15+ points   |
| MLP 20x20    | 54K        | ~2.5MB     | ~1-2 hours    | 25+ points   |

*Times estimated for GTX 1070, actual performance may vary*

## Features

- **Custom Snake Environment**: Built using `gymnasium` and `pygame`, featuring graphical rendering and configurable grid size.
- **Deep Q-Learning Agent**: Employs a fully-connected neural network to learn optimal policies for the Snake game.
- **Replay Memory**: Uses experience replay with a configurable memory buffer and batch training.
- **Target Network**: Implements a target network for stable Q-value updates.
- **Reward Shaping**: Encourages the snake to move toward the fruit and penalizes collisions and unnecessary wandering.
- **Rendering**: After training, you can watch the trained agent play the game live.

## Files

- `main.py` : Contains the agent, training loop, and the function to render the trained agent.
- `snake_gym_env.py` : Implements the custom `SnakeEnv` environment for gymnasium.
- `Graphics/` : Folder containing image assets for the snake and fruit.
- `Sound/` : Folder containing sound assets (e.g., crunch.wav).
- `Font/` : Folder containing the font used for score display.

## Requirements

- Python 3.8+
- [TensorFlow](https://www.tensorflow.org/) (tested with 2.x)
- [gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [pygame](https://www.pygame.org/)
- [numpy](https://numpy.org/)
- [tqdm](https://tqdm.github.io/)

You can install the dependencies via pip:

```bash
pip install tensorflow gymnasium pygame numpy tqdm
```

## How It Works

### Snake Environment

The environment is based on the OpenAI Gymnasium interface. The observation is a 4-dimensional vector:

- Snake's head x position
- Snake's head y position
- Fruit x position
- Fruit y position

Actions are discrete:
- `0`: Move right
- `1`: Move up
- `2`: Move left
- `3`: Move down

Rewards:
- +1 for eating fruit
- -1 for collision (game over)
- +0.3 for moving closer to the fruit
- -0.01 per step (encourages efficiency)

### Deep Q-Network Agent

- Uses a neural network with 3 hidden layers (sigmoid activations), outputting a softmax over possible actions.
- Trains using experience replay, Huber loss, and Adam optimizer.
- Implements epsilon-greedy policy for exploration.
- Maintains a target network updated every fixed number of steps for stable learning.

### Training and Rendering

The agent will train for up to 100,000 episodes or until a running average reward threshold is reached. Once training is complete, you can watch the trained agent play the game visually.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements
```

### 2. Run Model Demo
```bash
python demo.py  # See model comparison and quick test
```

### 3. Start Training
```bash
python main.py  # Train multiple model configurations
```

### 4. Monitor Training
The script will automatically:
- Train CNN and MLP models with different configurations
- Save best models to `saved_models/` directory
- Generate training curves as PNG files
- Display memory usage and performance metrics

## 🎮 Model Selection Guide

**For fastest training (recommended for quick experiments):**
- Model: MLP
- Grid size: 15x15
- Batch size: 256
- Training time: ~30-60 minutes

**For best performance:**
- Model: CNN
- Grid size: 20x20  
- Batch size: 512
- Training time: ~2-3 hours

**For balanced approach:**
- Model: CNN
- Grid size: 15x15
- Batch size: 256
- Training time: ~1-2 hours

## Customization

- **Grid Size**: Change the `grid_size` parameter in `main.py` or the environment constructor.
- **Reward Strategy**: Adjust the reward logic in `SnakeEnv.step()`.
- **Neural Network Architecture**: Modify `create_dense_q_model` in `main.py`.

## Registration

The environment is registered as `Snake-v0` via:

```python
register(
    id="Snake-v0",
    entry_point="snake_gym_env:SnakeEnv",
    max_episode_steps=1000,
)
```

## License

This project is provided for educational and research purposes.

## Acknowledgments

- Inspired by OpenAI Gym and classic Snake game implementations.
- Uses assets (snake, apple, sounds, font) assumed to be present in the specified directories.

---

**Note:** Ensure that all asset paths are correct and the required files are available. If you encounter errors related to missing files, please add the missing graphics, sounds, or fonts as appropriate.