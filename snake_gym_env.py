import random
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from pygame.math import Vector2  # only for convenience math, no rendering
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register


class SnakeEnv(gym.Env):
    metadata = {}

    def __init__(self, grid_size: int = 20, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        self.grid_size = int(grid_size)
        self._rng = random.Random(seed)

        self.max_snake_length = self.grid_size * self.grid_size

        coords_shape = (self.max_snake_length + 1, 2)
        coords_space = spaces.Box(
            low=-1,
            high=self.grid_size - 1,
            shape=coords_shape,
            dtype=np.int32,
        )

        features_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict({
            "coords": coords_space,
            "features": features_space,
        })

        self.action_space = spaces.Discrete(4)

        self.snake: List[Vector2] = []
        self.fruit: Vector2 = Vector2(-1, -1)
        self.direction = Vector2(1, 0)
        self.terminated = False
        self.truncated = False
        self.score = 0
        self.steps = 0
        self.max_steps = self.grid_size * self.grid_size * 10

    def seed(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)

        mid = self.grid_size // 2
        self.direction = Vector2(1, 0)
        self.snake = [
            Vector2(mid + 1, mid),
            Vector2(mid, mid),
            Vector2(mid - 1, mid),
        ]

        self._randomize_fruit()
        self.terminated = False
        self.truncated = False
        self.score = 0
        self.steps = 0

        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        if self.terminated or self.truncated:
            return self._get_obs(), 0.0, True, False, {}

        self.steps += 1

        action_map = [
            Vector2(1, 0),
            Vector2(0, -1),
            Vector2(-1, 0),
            Vector2(0, 1),
        ]
        new_dir = action_map[int(action)]
        if new_dir + self.direction != Vector2(0, 0):
            self.direction = new_dir

        head = self.snake[0]
        new_head = head + self.direction

        fruit = self.fruit
        prev_dist = abs(head.x - fruit.x) + abs(head.y - fruit.y)
        new_dist = abs(new_head.x - fruit.x) + abs(new_head.y - fruit.y)

        reward = 0.0

        if self._collision(new_head):
            self.terminated = True
            return self._get_obs(), -1.0, True, False, {"collision": True, "score": self.score}

        self.snake.insert(0, new_head)

        if new_head == self.fruit:
            self.score += 1
            reward += 1.0
            self._randomize_fruit()
        else:
            self.snake.pop()

        if new_dist < prev_dist:
            reward += 0.2
        elif new_dist > prev_dist:
            reward -= 0.2

        move_vec = np.array([self.direction.x, self.direction.y], dtype=np.float32)
        fruit_vec = np.array([fruit.x - new_head.x, fruit.y - new_head.y], dtype=np.float32)
        norm = np.linalg.norm(fruit_vec)
        if norm > 0:
            fruit_dir = fruit_vec / norm
            reward += 0.05 * float(np.dot(move_vec, fruit_dir))

        reward += 0.01
        reward -= 0.001

        if len(self.snake) >= self.max_snake_length or self.steps >= self.max_steps:
            self.truncated = True

        return self._get_obs(), float(reward), self.terminated, self.truncated, {"score": self.score}

    def _collision(self, pos: Vector2) -> bool:
        if not (0 <= pos.x < self.grid_size and 0 <= pos.y < self.grid_size):
            return True
        for b in self.snake[1:]:
            if pos == b:
                return True
        return False

    def _randomize_fruit(self):
        free = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}
        for b in self.snake:
            free.discard((int(b.x), int(b.y)))
        if not free:
            self.fruit = Vector2(-1, -1)
            return
        fx, fy = self._rng.choice(list(free))
        self.fruit = Vector2(fx, fy)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        body_coords = [(int(b.x), int(b.y)) for b in self.snake]
        padding = self.max_snake_length - len(body_coords)
        padded = body_coords + [(-1, -1)] * padding
        fruit_coord = [(int(self.fruit.x), int(self.fruit.y))]
        coords = np.array(padded + fruit_coord, dtype=np.int32)
        features = self._coords_to_features(coords)
        return {"coords": coords, "features": features}

    def _coords_to_features(self, coords: np.ndarray) -> np.ndarray:
        head = coords[0]
        fruit = coords[-1]

        body = []
        for (x, y) in coords[1:-1]:
            if x == -1:
                break
            body.append((int(x), int(y)))

        if len(coords) >= 2 and coords[1][0] != -1:
            neck = coords[1]
            dir_x = head[0] - neck[0]
            dir_y = head[1] - neck[1]
        else:
            dir_x, dir_y = 1, 0

        dx = fruit[0] - head[0]
        dy = fruit[1] - head[1]

        def danger(x, y):
            if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                return 1.0
            if (x, y) in body:
                return 1.0
            return 0.0

        hx, hy = dir_x, dir_y
        front = (head[0] + hx, head[1] + hy)
        left = (head[0] - hy, head[1] + hx)
        right = (head[0] + hy, head[1] - hx)

        denom = max(1, self.grid_size - 1)

        return np.array([
            head[0] / denom,
            head[1] / denom,
            dx / denom,
            dy / denom,
            hx, hy,
            danger(*front),
            danger(*left),
            danger(*right),
        ], dtype=np.float32)

    def render(self):
        return None

    def close(self):
        pass


register(
    id="Snake-v0",
    entry_point="snake_gym_env:SnakeEnv",
    max_episode_steps=2000,
)
