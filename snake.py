import pygame, sys, random
from pygame.math import Vector2
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import itertools

param_grid = {
	'alpha':	 [0.001, 0.01, 0.1, 1],
	'gamma':	 [0.90, 0.95, 0.99],
	'epsilon':   [1.0],
	'eps_decay': [0.995],
	'eps_min':   [0.01]
}

param_sets = list(itertools.product(
	param_grid['alpha'],
	param_grid['gamma'],
	param_grid['epsilon'],
	param_grid['eps_decay'],
	param_grid['eps_min']
))


class SARSAAgent:
	def __init__(self, alpha, gamma, epsilon, eps_decay, eps_min):
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_decay = eps_decay
		self.eps_min = eps_min
		self.Q = defaultdict(lambda: [0.0] * 4)

	def choose_action(self, state):
		if random.random() < self.epsilon:
			return random.randrange(4)
		qs = self.Q[state]
		max_q = max(qs)
		# break ties randomly
		return random.choice([i for i, q in enumerate(qs) if q == max_q])

	def update(self, s, a, r, s2, a2):
		td_target = r + self.gamma * self.Q[s2][a2]
		td_error = td_target - self.Q[s][a]
		self.Q[s][a] += self.alpha * td_error
		self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

ACTIONS = {
	0: Vector2(0, -1),
	1: Vector2(1, 0),
	2: Vector2(0, 1),
	3: Vector2(-1, 0)
}

def get_state(snake, fruit, cell_number):
	head = snake.body[0]
	dx = (fruit.pos.x - head.x) / cell_number
	dy = (fruit.pos.y - head.y) / cell_number

	dir_list = [
		Vector2(0, -1), Vector2(1, 0),
		Vector2(0, 1), Vector2(-1, 0)
	]
	if snake.direction not in dir_list:
		dir_idx = 0
	else:
		dir_idx = dir_list.index(snake.direction)

	def is_danger(a):
		np = head + ACTIONS[a]
		if not (0 <= np.x < cell_number and 0 <= np.y < cell_number):
			return 1
		return 1 if any(bp == np for bp in snake.body[1:]) else 0

	danger_straight = is_danger(dir_idx)
	danger_right = is_danger((dir_idx + 1) % 4)
	danger_left = is_danger((dir_idx - 1) % 4)

	fruit_left = int(dx < 0)
	fruit_right = int(dx > 0)
	fruit_up = int(dy < 0)
	fruit_down = int(dy > 0)

	return (
		round(dx, 2), round(dy, 2), dir_idx,
		danger_straight, danger_right, danger_left,
		fruit_left, fruit_right, fruit_up, fruit_down
	)

class SNAKE:
	def __init__(self):
		self.body = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10)]
		self.direction = Vector2(0, 0)
		self.new_block = False

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

		self.crunch_sound = pygame.mixer.Sound('Sound/crunch.wav')

	def draw_snake(self):
		self.update_head_graphics()
		self.update_tail_graphics()
		for idx, block in enumerate(self.body):
			x = int(block.x * cell_size)
			y = int(block.y * cell_size)
			rect = pygame.Rect(x, y, cell_size, cell_size)
			if idx == 0:
				screen.blit(self.head, rect)
			elif idx == len(self.body) - 1:
				screen.blit(self.tail, rect)
			else:
				prev = self.body[idx + 1] - block
				nxt = self.body[idx - 1] - block
				if prev.x == nxt.x:
					screen.blit(self.body_vertical, rect)
				elif prev.y == nxt.y:
					screen.blit(self.body_horizontal, rect)
				else:
					if (prev.x, prev.y, nxt.x, nxt.y) in [(-1, 0, 0, -1), (0, -1, -1, 0)]:
						screen.blit(self.body_tl, rect)
					elif (prev.x, prev.y, nxt.x, nxt.y) in [(-1, 0, 0, 1), (0, 1, -1, 0)]:
						screen.blit(self.body_bl, rect)
					elif (prev.x, prev.y, nxt.x, nxt.y) in [(1, 0, 0, -1), (0, -1, 1, 0)]:
						screen.blit(self.body_tr, rect)
					else:
						screen.blit(self.body_br, rect)

	def update_head_graphics(self):
		rel = self.body[1] - self.body[0]
		if rel == Vector2(1, 0):
			self.head = self.head_left
		elif rel == Vector2(-1, 0):
			self.head = self.head_right
		elif rel == Vector2(0, 1):
			self.head = self.head_up
		elif rel == Vector2(0, -1):
			self.head = self.head_down

	def update_tail_graphics(self):
		rel = self.body[-2] - self.body[-1]
		if rel == Vector2(1, 0):
			self.tail = self.tail_left
		elif rel == Vector2(-1, 0):
			self.tail = self.tail_right
		elif rel == Vector2(0, 1):
			self.tail = self.tail_up
		elif rel == Vector2(0, -1):
			self.tail = self.tail_down

	def move_snake(self):
		if self.new_block:
			body_copy = self.body[:]
			body_copy.insert(0, body_copy[0] + self.direction)
			self.body = body_copy[:]
			self.new_block = False
		else:
			body_copy = self.body[:-1]
			body_copy.insert(0, body_copy[0] + self.direction)
			self.body = body_copy[:]

	def add_block(self):
		self.new_block = True

	def play_crunch_sound(self):
		self.crunch_sound.play()

	def reset(self):
		self.body = [Vector2(5, 10), Vector2(4, 10), Vector2(3, 10)]
		self.direction = Vector2(0, 0)
		self.new_block = False


class FRUIT:
	def __init__(self):
		self.randomize()

	def draw_fruit(self):
		rect = pygame.Rect(int(self.pos.x * cell_size),
						   int(self.pos.y * cell_size),
						   cell_size, cell_size)
		screen.blit(apple, rect)

	def randomize(self):
		self.x = random.randint(0, cell_number - 1)
		self.y = random.randint(0, cell_number - 1)
		self.pos = Vector2(self.x, self.y)

class SCORE:
	def __init__(self):
		self.score = 0

class MAIN:
	def __init__(self, agent_params):
		self.snake = SNAKE()
		self.fruit = FRUIT()
		self.score = SCORE()
		self.agent = SARSAAgent(**agent_params)
		self.s = get_state(self.snake, self.fruit, cell_number)
		self.a = self.agent.choose_action(self.s)
		self.prev_head = Vector2(self.snake.body[0])
		self.episode_rewards = []
		self.current_reward = self.score.score
		self.episode_count = 0
		self.agent_params = agent_params

	def plot_rewards(self):
		plt.clf()
		plt.title("SARSA Agent Learning Curve")
		plt.xlabel("Episode")
		plt.ylabel("Total Reward")
		plt.plot(self.episode_rewards, label="Reward")
		plt.legend()
		plt.pause(0.01)

	def manhattan_distance(self, a, b):
		return abs(a.x - b.x) + abs(a.y - b.y)

	def update(self):
		self.snake.direction = ACTIONS[self.a]
		prev_dist = self.manhattan_distance(self.prev_head, self.fruit.pos)
		self.snake.move_snake()
		ate = self.check_collision()
		dead = self.check_fail()
		head = self.snake.body[0]
		new_dist = self.manhattan_distance(head, self.fruit.pos)
		r = 0
		if new_dist < prev_dist:
			r += 1
		if new_dist > prev_dist:
			r -= 1
		if ate:
			r += 10
		if dead:
			r -= 10

		s2 = get_state(self.snake, self.fruit, cell_number)
		a2 = self.agent.choose_action(s2)
		self.agent.update(self.s, self.a, r, s2, a2)
		self.s, self.a = s2, a2
		self.prev_head = Vector2(head)
		self.current_reward += r

	def draw_elements(self):
		self.draw_grass()
		self.fruit.draw_fruit()
		self.snake.draw_snake()
		self.draw_score()
	def check_collision(self):
		ate = False
		if self.fruit.pos == self.snake.body[0]:
			ate = True
			self.fruit.randomize()
			self.snake.add_block()
			self.snake.play_crunch_sound()
			self.score.score += 10
		for b in self.snake.body[1:]:
			if b == self.fruit.pos:
				self.fruit.randomize()
		return ate


	def check_fail(self):
		head = self.snake.body[0]
		if not 0 <= head.x < cell_number or not 0 <= head.y < cell_number:
			self.game_over()
			return True
		for b in self.snake.body[1:]:
			if b == head:
				self.game_over()
				return True  # Make sure this is inside the `if` block
		return False

	def game_over(self):
		self.episode_rewards.append(self.current_reward)
		print(f"Episode {self.episode_count} - Reward: {self.current_reward}")
		self.current_reward = 0
		self.episode_count += 1
		if self.episode_count % 10 == 0:
			self.plot_rewards()
			self.snake.reset()
			self.score.score = 0

	def draw_grass(self):
		grass_color = (167, 209, 61)
		for row in range(cell_number):
			for col in range(cell_number):
				if (row + col) % 2 == 0:
					r = pygame.Rect(col * cell_size, row * cell_size,
									cell_size, cell_size)
					pygame.draw.rect(screen, grass_color, r)

	def draw_score(self):
		txt = str(self.score.score)
		surf = game_font.render(txt, True, (56, 74, 12))
		x = cell_size * cell_number - 60
		y = cell_size * cell_number - 40
		rect = surf.get_rect(center=(x, y))
		apple_rect = apple.get_rect(midright=(rect.left, rect.centery))
		bg = pygame.Rect(apple_rect.left, apple_rect.top,
							 apple_rect.width + rect.width + 6,
							 apple_rect.height)
		pygame.draw.rect(screen, (167, 209, 61), bg)
		screen.blit(surf, rect)
		screen.blit(apple, apple_rect)
		pygame.draw.rect(screen, (56, 74, 12), bg, 2)


pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()
cell_size = 40
cell_number = 20
screen = pygame.display.set_mode((cell_number * cell_size,
								  cell_number * cell_size))
clock = pygame.time.Clock()

apple = pygame.image.load('Graphics/apple.png').convert_alpha()
game_font = pygame.font.Font('Font/PoetsenOne-Regular.ttf', 25)

SCREEN_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SCREEN_UPDATE, 150)
plt.ion()

for i, (alpha, gamma, epsilon, eps_decay, eps_min) in enumerate(param_sets):
	agent_params = {
		'alpha': alpha,
		'gamma': gamma,
		'epsilon': epsilon,
		'eps_decay': eps_decay,
		'eps_min': eps_min
	}
	main_game = MAIN(agent_params=agent_params)
	print(f"\n=== Training config {i + 1}/{len(param_sets)} ===")
	print(f"α={alpha}, γ={gamma}, ε={epsilon}, decay={eps_decay}, min_ε={eps_min}\n")

	while main_game.episode_count < 500:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit()
			if event.type == SCREEN_UPDATE:
				main_game.update()
		screen.fill((175, 215, 70))
		main_game.draw_elements()
		pygame.display.update()
		clock.tick(60)
