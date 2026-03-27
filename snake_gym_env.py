import random
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pygame
from pygame.math import Vector2
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, grid_size: int = 20, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        self.grid_size = int(grid_size)
        self.cell_size = 40
        self.window_size = self.grid_size * self.cell_size
        self.render_mode = render_mode
        self._rng = random.Random(seed)

        # Max snake length = all cells
        self.max_snake_length = self.grid_size * self.grid_size

        # coords: snake body (padded) + fruit
        coords_shape = (self.max_snake_length + 1, 2)
        coords_space = spaces.Box(
            low=-1,
            high=self.grid_size - 1,
            shape=coords_shape,
            dtype=np.int32
        )

        # features: [head_x, head_y, apple_dx, apple_dy, dir_x, dir_y, danger_front, danger_left, danger_right]
        features_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "coords": coords_space,
            "features": features_space
        })

        self.action_space = spaces.Discrete(4)  # right, up, left, down

        # pygame
        self._pygame_initialized = False
        self._assets_loaded = False

        # game state
        self.snake: List[Vector2] = []
        self.fruit: Vector2 = Vector2(-1, -1)
        self.direction = Vector2(1, 0)
        self.terminated = False
        self.truncated = False
        self.score = 0
        self.steps = 0
        self.max_steps = self.grid_size * self.grid_size * 10  # safety cap


    def seed(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)
        self._init_pygame()

        mid = self.grid_size // 2
        self.direction = Vector2(1, 0)
        self.snake = [
            Vector2(mid + 1, mid),
            Vector2(mid, mid),
            Vector2(mid - 1, mid)
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

        # Map action to direction
        action_map = [
            Vector2(1, 0),   # right
            Vector2(0, -1),  # up
            Vector2(-1, 0),  # left
            Vector2(0, 1)    # down
        ]
        new_dir = action_map[int(action)]

        # Prevent reversing
        if new_dir + self.direction != Vector2(0, 0):
            self.direction = new_dir

        head = self.snake[0]
        new_head = head + self.direction

        # Distance to fruit before and after move (Manhattan)
        fruit = self.fruit
        prev_dist = abs(head.x - fruit.x) + abs(head.y - fruit.y)
        new_dist = abs(new_head.x - fruit.x) + abs(new_head.y - fruit.y)

        # Base reward
        reward = 0.0

        # Collision check
        if self._collision(new_head):
            self.terminated = True
            reward -= 1.0  # strong penalty for dying
            return self._get_obs(), float(reward), True, False, {"collision": True, "score": self.score}

        # Move snake
        self.snake.insert(0, new_head)

        # Fruit eaten
        if new_head == self.fruit:
            self.score += 1
            reward += 1.0  # main positive reward
            self._randomize_fruit()
        else:
            self.snake.pop()

        if new_dist < prev_dist:
            reward += 0.2
        elif new_dist > prev_dist:
            reward -= 0.2

        move_vec = np.array([self.direction.x, self.direction.y], dtype=np.float32)
        fruit_vec = np.array([fruit.x - new_head.x, fruit.y - new_head.y], dtype=np.float32)
        fruit_norm = np.linalg.norm(fruit_vec)
        if fruit_norm > 0:
            fruit_dir = fruit_vec / fruit_norm
            alignment = float(np.dot(move_vec, fruit_dir))  # in [-1, 1]
            reward += 0.05 * alignment

        #survival
        reward += 0.01
        #circling
        reward -= 0.001

        # Truncation if snake fills board or too many steps
        if len(self.snake) >= self.max_snake_length or self.steps >= self.max_steps:
            self.truncated = True

        return self._get_obs(), float(reward), self.terminated, self.truncated, {"score": self.score}

    def _collision(self, pos: Vector2) -> bool:
        # wall
        if not (0 <= pos.x < self.grid_size and 0 <= pos.y < self.grid_size):
            return True
        # body (exclude head)
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
        # coords
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

        # body excluding head
        body = []
        for (x, y) in coords[1:-1]:
            if x == -1:
                break
            body.append((int(x), int(y)))

        # direction from neck
        if len(coords) >= 2 and coords[1][0] != -1:
            neck = coords[1]
            dir_x = head[0] - neck[0]
            dir_y = head[1] - neck[1]
        else:
            dir_x, dir_y = 1, 0

        # apple delta
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
            danger(*right)
        ], dtype=np.float32)

    def _init_pygame(self):
        if self.render_mode is None or self._pygame_initialized:
            return

        pygame.init()
        if self.render_mode == "rgb_array":
            self.screen = pygame.Surface((self.window_size, self.window_size))
        else:
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 25)
        self._pygame_initialized = True

    def render(self):
        if self.render_mode is None:
            return None

        self._init_pygame()
        surface = self.screen
        surface.fill((175, 215, 70))

        grass_color = (167, 209, 61)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row + col) % 2 == 0:
                    rect = pygame.Rect(
                        col * self.cell_size,
                        row * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )
                    pygame.draw.rect(surface, grass_color, rect)

        # fruit
        if self.fruit.x >= 0:
            fx = int(self.fruit.x * self.cell_size)
            fy = int(self.fruit.y * self.cell_size)
            rect = pygame.Rect(fx, fy, self.cell_size, self.cell_size)
            pygame.draw.rect(surface, (255, 0, 0), rect)

        # snake
        for i, block in enumerate(self.snake):
            x_pos = int(block.x * self.cell_size)
            y_pos = int(block.y * self.cell_size)
            rect = pygame.Rect(x_pos, y_pos, self.cell_size, self.cell_size)
            color = (0, 100, 0) if i == 0 else (0, 150, 0)
            pygame.draw.rect(surface, color, rect)

        # score
        try:
            score_surface = self.font.render(str(self.score), True, (56, 74, 12))
            surface.blit(score_surface, (5, 5))
        except Exception:
            pass

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None
        elif self.render_mode == "rgb_array":
            arr = pygame.surfarray.array3d(surface)
            return np.transpose(arr, (1, 0, 2))
        else:
            return None

    def close(self):
        if self._pygame_initialized:
            try:
                pygame.quit()
            except Exception:
                pass
            self._pygame_initialized = False
            self._assets_loaded = False


# Register environment
register(
    id="Snake-v0",
    entry_point="snake_gym_env:SnakeEnv",
    max_episode_steps=2000
)
