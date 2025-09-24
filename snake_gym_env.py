import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from pygame.math import Vector2
import random
from gymnasium.envs.registration import register

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, grid_size=20, render_mode='human'):
        super().__init__()
        self.grid_size = grid_size
        self.cell_size = 40
        self.window_size = self.grid_size * self.cell_size
        self.render_mode = render_mode

        self.max_snake_length = self.grid_size * self.grid_size
        obs_shape = (self.max_snake_length + 1, 2)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1, high=self.grid_size-1, shape=obs_shape, dtype=np.int32
        )

        self._pygame_initialized = False
        self.snake = None
        self.fruit = None
        self.direction = None
        self.done = False
        self.score = 0

    def _init_pygame(self):
        if self.render_mode == "human" and not self._pygame_initialized:
            pygame.init()
            pygame.mixer.pre_init(44100, -16, 2, 512)
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font('Font/PoetsenOne-Regular.ttf', 25)
            self.head_up = pygame.image.load('Graphics/head_up.png').convert_alpha()
            self.head_down = pygame.image.load('Graphics/head_down.png').convert_alpha()
            self.head_right = pygame.image.load('Graphics/head_right.png').convert_alpha()
            self.head_left = pygame.image.load('Graphics/head_left.png').convert_alpha()
            self.tail_up = pygame.image.load('Graphics/tail_up.png').convert_alpha()
            self.tail_down = pygame.image.load('Graphics/tail_down.png').convert_alpha()
            self.tail_right = pygame.image.load('Graphics/tail_right.png').convert_alpha()
            self.tail_left = pygame.image.load('Graphics/tail_left.png').convert_alpha()
            self.body_vertical = pygame.image.load('Graphics/body_vertical.png').convert_alpha()
            self.body_horizontal = pygame.image.load('Graphics/body_horizontal.png').convert_alpha()
            self.body_tr = pygame.image.load('Graphics/body_tr.png').convert_alpha()
            self.body_tl = pygame.image.load('Graphics/body_tl.png').convert_alpha()
            self.body_br = pygame.image.load('Graphics/body_br.png').convert_alpha()
            self.body_bl = pygame.image.load('Graphics/body_bl.png').convert_alpha()
            self.apple = pygame.image.load('Graphics/apple.png').convert_alpha()
            self.crunch_sound = pygame.mixer.Sound('Sound/crunch.wav')
            self._pygame_initialized = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.render_mode == "human":
            self._init_pygame()
        self.direction = Vector2(1, 0)
        self.snake = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10)]
        self._randomize_fruit()
        self.done = False
        self.score = 0
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        head_pos = self.snake[0]
        fruit_pos = self.fruit
        prev_dist = abs(head_pos.x - fruit_pos.x) + abs(head_pos.y - fruit_pos.y)

        action_map = [Vector2(1, 0), Vector2(0, -1), Vector2(-1, 0), Vector2(0, 1)]
        new_direction = action_map[action]
        if not (new_direction + self.direction == Vector2(0, 0)):
            self.direction = new_direction

        new_head = self.snake[0] + self.direction
        if self._collision(new_head):
            self.done = True
            return self._get_obs(), -10, True, False, {}  # Increased collision penalty

        self.snake.insert(0, new_head)
        reward = -0.005  # Reduced time penalty

        new_dist = abs(new_head.x - fruit_pos.x) + abs(new_head.y - fruit_pos.y)
        if new_dist < prev_dist:
            reward += 0.5  # Increased reward for getting closer
        else:
            reward -= 0.1  # Small penalty for moving away

        if new_head == self.fruit:
            self.score += 1
            reward += 10  # Increased fruit reward
            if self.render_mode == "human" and self._pygame_initialized:
                self.crunch_sound.play()
            self._randomize_fruit()
        else:
            self.snake.pop()

        return self._get_obs(), reward, self.done, False, {}

    def _collision(self, pos):
        if not (0 <= pos.x < self.grid_size and 0 <= pos.y < self.grid_size):
            return True
        if pos in self.snake[1:]:
            return True
        return False

    def _randomize_fruit(self):
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            fruit = Vector2(x, y)
            if fruit not in self.snake:
                self.fruit = fruit
                break

    def _get_obs(self):
        body_coords = [(int(block.x), int(block.y)) for block in self.snake]
        padding_length = self.max_snake_length - len(body_coords)
        padded_coords = body_coords + [(-1, -1)] * padding_length
        fruit_coord = [(int(self.fruit.x), int(self.fruit.y))]
        obs = np.array(padded_coords + fruit_coord, dtype=np.int32)
        return obs

    def render(self):
        if self.render_mode != "human":
            return
        if not self._pygame_initialized:
            self._init_pygame()
        self.screen.fill((175, 215, 70))
        self._draw_grass()
        self._draw_snake()
        self._draw_fruit()
        self._draw_score()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def _draw_snake(self):
        self._update_head_graphics()
        self._update_tail_graphics()
        for index, block in enumerate(self.snake):
            x_pos = int(block.x * self.cell_size)
            y_pos = int(block.y * self.cell_size)
            block_rect = pygame.Rect(x_pos, y_pos, self.cell_size, self.cell_size)
            if index == 0:
                self.screen.blit(self.head, block_rect)
            elif index == len(self.snake) - 1:
                self.screen.blit(self.tail, block_rect)
            else:
                prev = self.snake[index+1] - block
                next = self.snake[index-1] - block
                if prev.x == next.x:
                    self.screen.blit(self.body_vertical, block_rect)
                elif prev.y == next.y:
                    self.screen.blit(self.body_horizontal, block_rect)
                else:
                    if (prev.x == -1 and next.y == -1) or (prev.y == -1 and next.x == -1):
                        self.screen.blit(self.body_tl, block_rect)
                    elif (prev.x == -1 and next.y == 1) or (prev.y == 1 and next.x == -1):
                        self.screen.blit(self.body_bl, block_rect)
                    elif (prev.x == 1 and next.y == -1) or (prev.y == -1 and next.x == 1):
                        self.screen.blit(self.body_tr, block_rect)
                    elif (prev.x == 1 and next.y == 1) or (prev.y == 1 and next.x == 1):
                        self.screen.blit(self.body_br, block_rect)

    def _update_head_graphics(self):
        if len(self.snake) < 2:
            self.head = self.head_right
            return
        head_relation = self.snake[1] - self.snake[0]
        if head_relation == Vector2(1,0): self.head = self.head_left
        elif head_relation == Vector2(-1,0): self.head = self.head_right
        elif head_relation == Vector2(0,1): self.head = self.head_up
        elif head_relation == Vector2(0,-1): self.head = self.head_down
        else: self.head = self.head_right

    def _update_tail_graphics(self):
        if len(self.snake) < 2:
            self.tail = self.tail_right
            return
        tail_relation = self.snake[-2] - self.snake[-1]
        if tail_relation == Vector2(1,0): self.tail = self.tail_left
        elif tail_relation == Vector2(-1,0): self.tail = self.tail_right
        elif tail_relation == Vector2(0,1): self.tail = self.tail_up
        elif tail_relation == Vector2(0,-1): self.tail = self.tail_down
        else: self.tail = self.tail_right

    def _draw_fruit(self):
        fruit_rect = pygame.Rect(int(self.fruit.x * self.cell_size), int(self.fruit.y * self.cell_size), self.cell_size, self.cell_size)
        self.screen.blit(self.apple, fruit_rect)

    def _draw_grass(self):
        grass_color = (167, 209, 61)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row + col) % 2 == 0:
                    grass_rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, grass_color, grass_rect)

    def _draw_score(self):
        score_text = str(self.score)
        score_surface = self.font.render(score_text, True, (56,74,12))
        score_x = int(self.cell_size * self.grid_size - 60)
        score_y = int(self.cell_size * self.grid_size - 40)
        score_rect = score_surface.get_rect(center=(score_x, score_y))
        apple_rect = self.apple.get_rect(midright=(score_rect.left, score_rect.centery))
        bg_rect = pygame.Rect(apple_rect.left, apple_rect.top, apple_rect.width + score_rect.width + 6, apple_rect.height)
        pygame.draw.rect(self.screen, (167,209,61), bg_rect)
        self.screen.blit(score_surface, score_rect)
        self.screen.blit(self.apple, apple_rect)
        pygame.draw.rect(self.screen, (56,74,12), bg_rect, 2)

    def close(self):
        if self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False

register(
    id="Snake-v0",
    entry_point="snake_gym_env:SnakeEnv",
    max_episode_steps=1000
)
