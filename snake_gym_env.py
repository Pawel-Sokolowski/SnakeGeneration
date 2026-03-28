import random
from typing import Optional, Tuple, List, Dict

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from pygame.math import Vector2


# ===================================================================
#                            SNAKE ENV (48 features)
# ===================================================================
class SnakeEnv(gym.Env):
    metadata = {}

    def __init__(self, grid_size=20, render_mode=None, seed=None):
        super().__init__()
        self.grid_size = grid_size
        self._rng = random.Random(seed)

        self.max_len = grid_size * grid_size

        self.observation_space = spaces.Dict({
            "coords": spaces.Box(
                low=-1, high=grid_size,
                shape=(self.max_len + 1, 2),
                dtype=np.int32
            ),
            "features": spaces.Box(
                low=-2.0, high=2.0,
                shape=(48,),
                dtype=np.float32
            )
        })

        self.action_space = spaces.Discrete(4)

        self.snake: List[Vector2] = []
        self.direction = Vector2(1, 0)
        self.fruit = Vector2(-1, -1)
        self.score = 0
        self.steps = 0
        self.terminated = False
        self.truncated = False

        self.max_steps = grid_size * grid_size * 10

    # ===================================================================
    # RESET
    # ===================================================================
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = random.Random(seed)

        mid = self.grid_size // 2
        self.direction = Vector2(1, 0)

        self.snake = [
            Vector2(mid + 1, mid),
            Vector2(mid, mid),
            Vector2(mid - 1, mid),
        ]

        self._place_fruit()
        self.score = 0
        self.steps = 0
        self.terminated = False
        self.truncated = False

        return self._obs(), {}

    # ===================================================================
    # STEP
    # ===================================================================
    def step(self, action):
        if self.terminated or self.truncated:
            return self._obs(), 0.0, True, False, {}

        self.steps += 1

        dirs = [
            Vector2(1, 0),
            Vector2(0, -1),
            Vector2(-1, 0),
            Vector2(0, 1)
        ]

        new_dir = dirs[action]
        if new_dir + self.direction != Vector2(0, 0):
            self.direction = new_dir

        head = self.snake[0]
        new_head = head + self.direction

        reward = 0.0

        # distance change to fruit
        prev_d = abs(head.x - self.fruit.x) + abs(head.y - self.fruit.y)
        new_d = abs(new_head.x - self.fruit.x) + abs(new_head.y - self.fruit.y)

        # collision
        if self._hit(new_head):
            self.terminated = True
            return self._obs(), -1.0, True, False, {"score": self.score}

        # move
        self.snake.insert(0, new_head)

        # eat fruit?
        if new_head == self.fruit:
            self.score += 1
            reward += 1.0
            self._place_fruit()
        else:
            self.snake.pop()

        # shaping
        if new_d < prev_d:
            reward += 0.2
        elif new_d > prev_d:
            reward -= 0.2

        if not self.truncated:
            reward += 0.01

        if self.steps >= self.max_steps:
            self.truncated = True

        return self._obs(), reward, self.terminated, self.truncated, {"score": self.score}

    # ===================================================================
    # UTILS
    # ===================================================================
    def _hit(self, p):
        if not (0 <= p.x < self.grid_size and 0 <= p.y < self.grid_size):
            return True
        for b in self.snake[1:]:
            if p == b:
                return True
        return False

    def _place_fruit(self):
        free = set((x, y) for x in range(self.grid_size) for y in range(self.grid_size))
        for b in self.snake:
            free.discard((int(b.x), int(b.y)))

        if not free:
            self.fruit = Vector2(-1, -1)
            return

        fx, fy = self._rng.choice(list(free))
        self.fruit = Vector2(fx, fy)

    # ===================================================================
    # OBSERVATION
    # ===================================================================
    def _obs(self):
        coords = np.full((self.max_len + 1, 2), -1, dtype=np.int32)
        arr = [(int(b.x), int(b.y)) for b in self.snake]
        arr.append((int(self.fruit.x), int(self.fruit.y)))
        arr = np.asarray(arr, dtype=np.int32)
        coords[:len(arr)] = arr

        return {
            "coords": coords,
            "features": self._features()
        }

    # ===================================================================
    # STABLE 48-DIM FEATURES (ALWAYS VALID SHAPE)
    # ===================================================================
    def _features(self):
        head = self.snake[0]
        tail = self.snake[-1]
        fruit = self.fruit
        body = {(int(b.x), int(b.y)) for b in self.snake[1:]}

        N = max(1, self.grid_size - 1)

        # normalized fruit vector
        fdx = (fruit.x - head.x) / N
        fdy = (fruit.y - head.y) / N

        # head movement vector
        if len(self.snake) >= 2:
            neck = self.snake[1]
            hx = (head.x - neck.x)
            hy = (head.y - neck.y)
        else:
            hx, hy = 1, 0

        # tail direction
        if len(self.snake) >= 2:
            if len(self.snake) >= 3:
                t2 = self.snake[-2]
                tx = tail.x - t2.x
                ty = tail.y - t2.y
            else:
                tx, ty = -hx, -hy
        else:
            tx, ty = -hx, -hy

        # length info
        length = len(self.snake)
        length_norm = length / (self.grid_size * self.grid_size)

        # danger 4 directions
        def danger(x, y):
            if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                return 1.0
            return 1.0 if (x, y) in body else 0.0

        front = (int(head.x + hx), int(head.y + hy))
        left  = (int(head.x - hy), int(head.y + hx))
        right = (int(head.x + hy), int(head.y - hx))
        back  = (int(head.x - hx), int(head.y - hy))

        # wall distance normalized
        wl = head.x / N
        wr = (self.grid_size - 1 - head.x) / N
        wu = head.y / N
        wd = (self.grid_size - 1 - head.y) / N

        # tail distance normalized
        tdx = (tail.x - head.x) / N
        tdy = (tail.y - head.y) / N
        tman = (abs(tail.x - head.x) + abs(tail.y - head.y)) / N

        # ----------------------------
        # Local 3x3 occupancy (exact 9 values)
        # ----------------------------
        grid9 = []
        for yy in [-1, 0, 1]:
            for xx in [-1, 0, 1]:
                cx, cy = int(head.x + xx), int(head.y + yy)
                if not (0 <= cx < self.grid_size and 0 <= cy < self.grid_size):
                    grid9.append(1.0)
                elif (cx, cy) in body:
                    grid9.append(1.0)
                else:
                    grid9.append(0.0)

        # ----------------------------
        # Raycasts (4 values)
        # ----------------------------
        def ray(dx, dy):
            x, y = int(head.x), int(head.y)
            d = 0
            while True:
                x += dx; y += dy
                if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                    break
                if (x, y) in body:
                    break
                d += 1
            return d / self.grid_size

        ray_u = ray(0, -1)
        ray_d = ray(0, 1)
        ray_l = ray(-1, 0)
        ray_r = ray(1, 0)

        # ----------------------------
        # Flood fill (safe area)
        # ----------------------------
        def flood():
            sx, sy = int(head.x), int(head.y)
            st = [(sx, sy)]
            vis = {(sx, sy)}
            c = 0
            while st:
                cx, cy = st.pop()
                c += 1
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        if (nx, ny) not in vis and (nx, ny) not in body:
                            vis.add((nx, ny))
                            st.append((nx, ny))
            return c / (self.grid_size * self.grid_size)

        area = flood()
        if np.isnan(area):
            area = 0.0

        dead = 1.0 if area < 0.10 else 0.0
        encl = 1.0 if area < length_norm else 0.0
        fcor = 1.0 if (abs(fdx) + abs(fdy) < 0.20 and area < 0.15) else 0.0

        # ----------------------------
        # BUILD FIXED SHAPE VECTOR
        # ----------------------------
        vec = [
            fdx, fdy,
            hx, hy,
            length_norm, float(length),
            tx, ty,

            danger(*front),
            danger(*left),
            danger(*right),
            danger(*back),

            wl, wr, wu, wd,

            tdx, tdy, tman,

            *grid9,

            ray_u, ray_d, ray_l, ray_r,

            area, dead, encl, fcor
        ]

        vec = np.array(vec, dtype=np.float32)

        # SAFETY GUARANTEE
        if vec.shape[0] != 48:
            padded = np.zeros(48, dtype=np.float32)
            n = min(len(vec), 48)
            padded[:n] = vec[:n]
            vec = padded

        return vec.reshape((48,)).astype(np.float32)

    # ===================================================================
    def render(self): pass
    def close(self): pass


# REGISTER
register(
    id="Snake-v0",
    entry_point="snake_gym_env:SnakeEnv",
    max_episode_steps=8000
)
