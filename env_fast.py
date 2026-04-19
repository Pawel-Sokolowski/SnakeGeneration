# env_fast.py
import torch
import torch.nn.functional as F   # required for one_hot


class TorchSnakeEnv:
    """
    Vector-only Snake environment with optional randomized starts.

    State features (dim = 34):
      0:  fwd_apple
      1:  rgt_apple
      2:  mdist_norm
      3:  length_norm
      4:  steps_eat_norm
      5-8: dir_onehot[0..3]
      9-33: danger_5x5 (25 cells)
    """

    def __init__(
        self,
        n,
        g,
        max_steps,
        device,
        step_penalty=-0.001,
        eat_reward=1.0,
        crash_penalty=-1.0,
        timeout_penalty=-0.5,
        length_reward_scale=0.05,
        shaping_scale=0.05,
        no_eat_limit=100,
        no_eat_penalty=-0.5,
        cycle_window=16,
        cycle_unique_thr=4,
        success_length_threshold=None,
        random_start=True,
        seed=None,
    ):
        self.n = n
        self.g = g
        self.max_steps = int(max_steps)
        self.device = device

        self.step_penalty = float(step_penalty)
        self.eat_reward = float(eat_reward)
        self.crash_penalty = float(crash_penalty)
        self.timeout_penalty = float(timeout_penalty)
        self.length_reward_scale = float(length_reward_scale)
        self.shaping_scale = float(shaping_scale)

        self.no_eat_limit = int(no_eat_limit)
        self.no_eat_penalty = float(no_eat_penalty)

        self.cycle_window = int(cycle_window)
        self.cycle_unique_thr = int(cycle_unique_thr)

        self.random_start = bool(random_start)

        if success_length_threshold is None:
            self.success_length_threshold = max(10, int(0.35 * (g * g)))
        else:
            self.success_length_threshold = int(success_length_threshold)

        # stats
        self.stat_crash = torch.zeros(n, dtype=torch.long, device=device)
        self.stat_timeout = torch.zeros(n, dtype=torch.long, device=device)
        self.stat_noeat = torch.zeros(n, dtype=torch.long, device=device)
        self.stat_cycle = torch.zeros(n, dtype=torch.long, device=device)

        # occupancy
        self.occupied = torch.zeros((n, g, g), dtype=torch.bool, device=device)

        # snake body circular buffer
        self.snake_x = torch.zeros((n, g * g), dtype=torch.long, device=device)
        self.snake_y = torch.zeros_like(self.snake_x)

        self.head = torch.zeros(n, dtype=torch.long, device=device)
        self.tail = torch.zeros_like(self.head)
        self.length = torch.zeros_like(self.head)
        self.steps = torch.zeros(n, dtype=torch.long, device=device)
        self.done = torch.zeros(n, dtype=torch.bool, device=device)

        self.direction = torch.zeros(n, dtype=torch.long, device=device)
        self.prev_direction = torch.zeros_like(self.direction)

        # steps since last apple
        self.steps_since_eat = torch.zeros(n, dtype=torch.long, device=device)

        # movement vectors (right, down, left, up)
        self.dirs = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=device)

        # fruit
        self.fruit_x = torch.zeros(n, dtype=torch.long, device=device)
        self.fruit_y = torch.zeros_like(self.fruit_x)

        # cycle detection buffer
        self.head_hist_x = torch.zeros((n, self.cycle_window), dtype=torch.long, device=device)
        self.head_hist_y = torch.zeros_like(self.head_hist_x)
        self.hist_ptr = torch.zeros(n, dtype=torch.long, device=device)

        # reproducible RNG on device if seed provided
        self.rng = None
        if seed is not None:
            try:
                self.rng = torch.Generator(device=device)
                self.rng.manual_seed(int(seed))
            except Exception:
                self.rng = torch.Generator()
                self.rng.manual_seed(int(seed))

        self.reset()

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.n, dtype=torch.bool, device=self.device)

        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            return self._features()

        # clear and init
        self.occupied[idx] = False
        self.done[idx] = False
        self.steps[idx] = 0
        self.steps_since_eat[idx] = 0

        self.length[idx] = 3
        self.head[idx] = 0
        self.tail[idx] = 2
        self.prev_direction[idx] = 0

        if self.random_start:
            k = idx.numel()

            # choose head positions safely away from edges so initial 3-cell body fits
            if self.g >= 5:
                # head in [2, g-3] so ±1, ±2 stay in [0, g-1]
                low = 2
                high = self.g - 2  # torch.randint is [low, high)
            else:
                # tiny grids: fall back to full range; body logic still works for g>=3
                low = 0
                high = self.g

            if self.rng is not None:
                hx = torch.randint(low, high, (k,), generator=self.rng, device=self.device)
                hy = torch.randint(low, high, (k,), generator=self.rng, device=self.device)
                dir_rand = torch.randint(0, 4, (k,), generator=self.rng, device=self.device)
            else:
                hx = torch.randint(low, high, (k,), device=self.device)
                hy = torch.randint(low, high, (k,), device=self.device)
                dir_rand = torch.randint(0, 4, (k,), device=self.device)

            # compute body coordinates for each chosen direction
            xs = torch.zeros((k, 3), dtype=torch.long, device=self.device)
            ys = torch.zeros((k, 3), dtype=torch.long, device=self.device)

            # direction 0: right  -> body extends left
            m0 = dir_rand == 0
            if m0.any():
                i = torch.where(m0)[0]
                xs[i, 0] = hx[i]; ys[i, 0] = hy[i]
                xs[i, 1] = hx[i] - 1; ys[i, 1] = hy[i]
                xs[i, 2] = hx[i] - 2; ys[i, 2] = hy[i]

            # direction 1: down -> body extends up
            m1 = dir_rand == 1
            if m1.any():
                i = torch.where(m1)[0]
                xs[i, 0] = hx[i]; ys[i, 0] = hy[i]
                xs[i, 1] = hx[i]; ys[i, 1] = hy[i] - 1
                xs[i, 2] = hx[i]; ys[i, 2] = hy[i] - 2

            # direction 2: left -> body extends right
            m2 = dir_rand == 2
            if m2.any():
                i = torch.where(m2)[0]
                xs[i, 0] = hx[i]; ys[i, 0] = hy[i]
                xs[i, 1] = hx[i] + 1; ys[i, 1] = hy[i]
                xs[i, 2] = hx[i] + 2; ys[i, 2] = hy[i]

            # direction 3: up -> body extends down
            m3 = dir_rand == 3
            if m3.any():
                i = torch.where(m3)[0]
                xs[i, 0] = hx[i]; ys[i, 0] = hy[i]
                xs[i, 1] = hx[i]; ys[i, 1] = hy[i] + 1
                xs[i, 2] = hx[i]; ys[i, 2] = hy[i] + 2

            # write into circular buffers: assign first 3 positions for each env
            self.snake_x[idx, :3] = xs
            self.snake_y[idx, :3] = ys

            # set direction for those indices
            self.direction[idx] = dir_rand

            # mark occupancy: flatten per-env coords and assign
            flat_idx = idx.repeat_interleave(3)        # shape (k*3,)
            flat_x = xs.view(-1)                       # shape (k*3,)
            flat_y = ys.view(-1)                       # shape (k*3,)
            self.occupied[flat_idx, flat_y, flat_x] = True

            # reset head history pointers
            self.head_hist_x[idx] = 0
            self.head_hist_y[idx] = 0
            self.hist_ptr[idx] = 0

        else:
            # deterministic center placement (legacy behavior)
            m = self.g // 2
            self.snake_x[idx, :3] = torch.tensor([m, m - 1, m - 2], device=self.device)
            self.snake_y[idx, :3] = m
            self.direction[idx] = 0
            self.occupied[idx, m, m] = True
            self.occupied[idx, m, m - 1] = True
            self.occupied[idx, m, m - 2] = True
            self.head_hist_x[idx] = 0
            self.head_hist_y[idx] = 0
            self.hist_ptr[idx] = 0

        # place fruit avoiding occupied cells
        self._place_fruit(idx)

        return self._features()

    def _record_head(self, idx, hx, hy):
        if idx.numel() == 0:
            return
        ptr = self.hist_ptr[idx]
        self.head_hist_x[idx, ptr] = hx
        self.head_hist_y[idx, ptr] = hy
        self.hist_ptr[idx] = (ptr + 1) % self.cycle_window

    def _detect_cycle_mask(self, move_mask):
        pos = self.head_hist_x * self.g + self.head_hist_y
        sorted_pos, _ = pos.sort(dim=1)
        if sorted_pos.size(1) > 1:
            diffs = sorted_pos[:, 1:] != sorted_pos[:, :-1]
            uniq_counts = 1 + diffs.sum(dim=1)
        else:
            uniq_counts = torch.ones(self.n, dtype=torch.long, device=self.device)

        ss_cond = self.steps_since_eat > (self.no_eat_limit // 2)
        uniq_cond = uniq_counts <= self.cycle_unique_thr
        cycle = uniq_cond & ss_cond & move_mask
        return cycle

    def step(self, action):
        alive = ~self.done
        idx = torch.arange(self.n, device=self.device)

        self.steps[alive] += 1
        self.steps_since_eat[alive] += 1

        hx = self.snake_x[idx, self.head]
        hy = self.snake_y[idx, self.head]

        alive_idx = torch.where(alive)[0]
        if alive_idx.numel():
            self._record_head(alive_idx, hx[alive_idx], hy[alive_idx])

        dist_before = (hx - self.fruit_x).abs() + (hy - self.fruit_y).abs()

        self.prev_direction[:] = self.direction
        left_mask = action == 1
        right_mask = action == 2
        if left_mask.any():
            self.direction[left_mask] = (self.direction[left_mask] - 1) % 4
        if right_mask.any():
            self.direction[right_mask] = (self.direction[right_mask] + 1) % 4

        dvec = self.dirs[self.direction]
        nx = hx + dvec[:, 0]
        ny = hy + dvec[:, 1]

        reward = torch.zeros(self.n, device=self.device)
        reward[alive] = self.step_penalty

        out = (nx < 0) | (ny < 0) | (nx >= self.g) | (ny >= self.g)

        tx = self.snake_x[idx, self.tail]
        ty = self.snake_y[idx, self.tail]
        moving_tail = (nx == tx) & (ny == ty)

        occ = self.occupied[idx, ny.clamp(0, self.g - 1), nx.clamp(0, self.g - 1)]
        eat = alive & (nx == self.fruit_x) & (ny == self.fruit_y)
        collision = alive & (out | (occ & ~(moving_tail & ~eat)))

        reward[collision] = self.crash_penalty
        self.stat_crash[collision] += 1
        self.done |= collision

        nx_c = nx.clamp(0, self.g - 1)
        ny_c = ny.clamp(0, self.g - 1)
        dist_after = (nx_c - self.fruit_x).abs() + (ny_c - self.fruit_y).abs()

        max_md = max(1, 2 * (self.g - 1))
        delta = (dist_before - dist_after).float() / max_md
        shaping = self.shaping_scale * delta.clamp(min=0.0)

        move = alive & ~collision
        reward += shaping * move.float()

        if move.any():
            self.head[move] = (self.head[move] - 1) % (self.g * self.g)
            h = self.head[move]
            self.snake_x[move, h] = nx[move]
            self.snake_y[move, h] = ny[move]
            self.occupied[move, ny[move], nx[move]] = True

        cycle_mask = self._detect_cycle_mask(move & ~self.done)
        if cycle_mask.any():
            reward[cycle_mask] = self.crash_penalty
            self.stat_cycle[cycle_mask] += 1
            self.done |= cycle_mask

        if eat.any():
            reward[eat] = self.eat_reward
            self.length[eat] += 1
            reward[eat] += self.length_reward_scale * (self.length[eat].float() ** 1.2)
            self.steps_since_eat[eat] = 0
            eat_idx = torch.where(eat)[0]
            if eat_idx.numel():
                self._place_fruit(eat_idx)

        success_mask = (self.length >= self.success_length_threshold) & (~self.done)
        if success_mask.any():
            self.stat_timeout[success_mask] += 1
            self.done |= success_mask

        shrink = move & ~eat & ~self.done
        if shrink.any():
            t = self.tail[shrink]
            self.occupied[shrink, self.snake_y[shrink, t], self.snake_x[shrink, t]] = False
            self.tail[shrink] = (self.tail[shrink] - 1) % (self.g * self.g)

        timeout = (self.steps >= self.max_steps) & alive & ~self.done
        if timeout.any():
            reward[timeout] = self.timeout_penalty
            self.stat_timeout[timeout] += 1
            self.done |= timeout

        no_eat = (self.steps_since_eat >= self.no_eat_limit) & alive & ~self.done
        if no_eat.any():
            reward[no_eat] = self.no_eat_penalty
            self.stat_noeat[no_eat] += 1
            self.done |= no_eat

        return self._features(), reward, self.done

    def _place_fruit(self, idx):
        if idx.numel() == 0:
            return
        remaining = idx
        while remaining.numel():
            k = remaining.numel()
            if self.rng is not None:
                x = torch.randint(0, self.g, (k,), generator=self.rng, device=self.device)
                y = torch.randint(0, self.g, (k,), generator=self.rng, device=self.device)
            else:
                x = torch.randint(0, self.g, (k,), device=self.device)
                y = torch.randint(0, self.g, (k,), device=self.device)
            ok = ~self.occupied[remaining, y, x]
            if ok.any():
                good = remaining[ok]
                self.fruit_x[good] = x[ok]
                self.fruit_y[good] = y[ok]
            remaining = remaining[~ok]

    def _features(self):
        idx = torch.arange(self.n, device=self.device)
        hx = self.snake_x[idx, self.head]
        hy = self.snake_y[idx, self.head]

        dx = (self.fruit_x - hx).float()
        dy = (self.fruit_y - hy).float()

        d = self.direction
        fwd = torch.empty_like(dx)
        rgt = torch.empty_like(dy)

        m = d == 0
        fwd[m], rgt[m] = dx[m], dy[m]
        m = d == 1
        fwd[m], rgt[m] = dy[m], -dx[m]
        m = d == 2
        fwd[m], rgt[m] = -dx[m], -dy[m]
        m = d == 3
        fwd[m], rgt[m] = -dy[m], dx[m]

        norm = max(1.0, float(self.g - 1))
        fwd = (fwd / norm).clamp(-1, 1)
        rgt = (rgt / norm).clamp(-1, 1)

        offsets = torch.tensor(
            [[dxo, dyo] for dyo in range(-2, 3) for dxo in range(-2, 3)],
            device=self.device,
        )

        nx = hx[:, None] + offsets[None, :, 0]
        ny = hy[:, None] + offsets[None, :, 1]

        out = (nx < 0) | (ny < 0) | (nx >= self.g) | (ny >= self.g)
        occ = self.occupied[
            idx[:, None],
            ny.clamp(0, self.g - 1),
            nx.clamp(0, self.g - 1),
        ]
        danger = (out | occ).float()

        mdist = (hx - self.fruit_x).abs() + (hy - self.fruit_y).abs()
        mdist_norm = (mdist.float() / max(1, 2 * (self.g - 1))).clamp(0, 1)

        length_norm = (self.length.float() / (self.g * self.g)).clamp(0, 1)
        steps_norm = (self.steps_since_eat.float() / self.no_eat_limit).clamp(0, 1)

        dir_onehot = F.one_hot(self.direction, num_classes=4).float()

        return torch.cat(
            [
                fwd[:, None],
                rgt[:, None],
                mdist_norm[:, None],
                length_norm[:, None],
                steps_norm[:, None],
                dir_onehot,
                danger,
            ],
            dim=1,
        )

    def grid_observation(self):
        n = self.n
        g = self.g
        device = self.device
        body = self.occupied.float()
        idx = torch.arange(n, device=device)
        hx = self.snake_x[idx, self.head]
        hy = self.snake_y[idx, self.head]
        head_mask = torch.zeros((n, g, g), dtype=torch.float32, device=device)
        head_mask[idx, hy, hx] = 1.0
        fruit_mask = torch.zeros((n, g, g), dtype=torch.float32, device=device)
        fruit_mask[idx, self.fruit_y, self.fruit_x] = 1.0
        grid = torch.stack([body, head_mask, fruit_mask], dim=1)
        return grid
