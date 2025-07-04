# Snake Deep Q-Learning Agent

This repository contains a Deep Q-Learning (DQN) implementation for playing the classic Snake game, using TensorFlow, Keras, and a custom Gymnasium environment built with Pygame.

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

## Usage

1. **Prepare Assets**: Ensure you have the required image, sound, and font files in the appropriate `Graphics/`, `Sound/`, and `Font/` directories.
   
2. **Run Training**:

   ```bash
   python main.py
   ```

   The agent will train and periodically print progress. After training, the agent will play the game using the trained model, rendering it in a window.

3. **Stop Rendering**: Press `Ctrl+C` to stop the rendering loop.

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