import torch
import torch.nn.functional as F  # still used elsewhere if needed


def _precompute_rot_lin(g: int, device):
    """
    Precompute linear indices mapping for 4 rotations.
    We replicate the original sampling logic:

    k==0: in(x,y) = (x, y)
    k==1: in(x,y) = (y, g-1-x)
    k==2: in(x,y) = (g-1-x, g-1-y)
    k==3: in(x,y) = (g-1-y, x)

    Where (x,y) are output coords and we sample input at (xin, yin).
    Return shape [4, g*g] with dtype long.
    """
    xs = torch.arange(g, device=device)
    ys = torch.arange(g, device=device)
    y, x = torch.meshgrid(ys, xs, indexing="ij")  # [g,g]

    # k=0
    xin0, yin0 = x, y
    # k=1
    xin1, yin1 = y, (g - 1 - x)
    # k=2
    xin2, yin2 = (g - 1 - x), (g - 1 - y)
    # k=3
    xin3, yin3 = (g - 1 - y), x

    lin0 = (yin0 * g + xin0).reshape(-1)
    lin1 = (yin1 * g + xin1).reshape(-1)
    lin2 = (yin2 * g + xin2).reshape(-1)
    lin3 = (yin3 * g + xin3).reshape(-1)

    rot_lin = torch.stack([lin0, lin1, lin2, lin3], dim=0).long()  # [4, g*g]
    return rot_lin


class TorchSnakeEnv:
    """Vector-only Snake environment with optional randomized starts.

    Vector state features (dim = 34):
      0:  fwd_apple
      1:  rgt_apple
      2:  mdist_norm
      3:  length_norm
      4:  steps_eat_norm
      5-8: dir_onehot[0..3]
      9-33: danger_5x5 (25 cells)

    CNN state features (dim = 37 channels):
      0:  body mask
      1:  head mask
      2:  fruit mask
      3-36: broadcasted vector features (34 planes)

    Performance upgrades included:
      - Precompute 4 rotation gather maps (2a)
      - Cache CNN output tensor (2b)
      - Keep float occupancy updated (2c)
      - Fix + speed up cycle detection (3)
      - Replace one_hot with lookup table (4a)
      - Preallocate feature buffer and fill (4b)
      - Optional step(return_obs="none") to avoid wasted _features() (big practical win)
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
        self.n = int(n)
        self.g = int(g)
        self.max_steps = int(max_steps)
        self.device = device

        # base rewards / penalties
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
            self.success_length_threshold = max(10, int(0.35 * (self.g * self.g)))
        else:
            self.success_length_threshold = int(success_length_threshold)

        # stats
        self.stat_crash = torch.zeros(self.n, dtype=torch.long, device=device)
        self.stat_timeout = torch.zeros(self.n, dtype=torch.long, device=device)
        self.stat_noeat = torch.zeros(self.n, dtype=torch.long, device=device)
        self.stat_cycle = torch.zeros(self.n, dtype=torch.long, device=device)

        # occupancy (bool + float32 cached)  (2c)
        self.occupied = torch.zeros((self.n, self.g, self.g), dtype=torch.bool, device=device)
        self.occupied_f = torch.zeros((self.n, self.g, self.g), dtype=torch.float32, device=device)

        # snake body circular buffer
        self.snake_x = torch.zeros((self.n, self.g * self.g), dtype=torch.long, device=device)
        self.snake_y = torch.zeros_like(self.snake_x)

        self.head = torch.zeros(self.n, dtype=torch.long, device=device)
        self.tail = torch.zeros_like(self.head)
        self.length = torch.zeros_like(self.head)
        self.steps = torch.zeros(self.n, dtype=torch.long, device=device)
        self.done = torch.zeros(self.n, dtype=torch.bool, device=device)

        self.direction = torch.zeros(self.n, dtype=torch.long, device=device)
        self.prev_direction = torch.zeros_like(self.direction)

        self.steps_since_eat = torch.zeros(self.n, dtype=torch.long, device=device)

        # movement vectors (right, down, left, up)
        self.dirs = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=device)

        # fruit
        self.fruit_x = torch.zeros(self.n, dtype=torch.long, device=device)
        self.fruit_y = torch.zeros_like(self.fruit_x)

        # cycle detection buffer
        self.head_hist_x = torch.zeros((self.n, self.cycle_window), dtype=torch.long, device=device)
        self.head_hist_y = torch.zeros_like(self.head_hist_x)
        self.hist_ptr = torch.zeros(self.n, dtype=torch.long, device=device)

        # Cached constants
        self._idx = torch.arange(self.n, device=device)

        self._offsets_5x5 = torch.tensor(
            [[dxo, dyo] for dyo in range(-2, 3) for dxo in range(-2, 3)],
            device=device,
        )

        self._max_md = max(1, 2 * (self.g - 1))

        # (4a) direction onehot lookup
        self._dir_onehot = torch.eye(4, device=device, dtype=torch.float32)

        # (4b) feature buffer
        self._feat_buf = torch.empty((self.n, 34), device=device, dtype=torch.float32)

        # (2a) rotation gather maps
        self._rot_lin = _precompute_rot_lin(self.g, device=device)  # [4, g*g]

        # (2b) CNN cached buffers (allocated lazily)
        self._head_mask = torch.zeros((self.n, self.g, self.g), dtype=torch.float32, device=device)
        self._fruit_mask = torch.zeros((self.n, self.g, self.g), dtype=torch.float32, device=device)
        self._cnn_out = torch.empty((self.n, 37, self.g, self.g), dtype=torch.float32, device=device)

        # RNG
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
        self.occupied_f[idx] = 0.0

        self.done[idx] = False
        self.steps[idx] = 0
        self.steps_since_eat[idx] = 0

        self.length[idx] = 3
        self.head[idx] = 0
        self.tail[idx] = 2
        self.prev_direction[idx] = 0

        if self.random_start:
            k = idx.numel()

            if self.g >= 5:
                low = 2
                high = self.g - 2
            else:
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

            xs = torch.zeros((k, 3), dtype=torch.long, device=self.device)
            ys = torch.zeros((k, 3), dtype=torch.long, device=self.device)

            m0 = dir_rand == 0
            if m0.any():
                i = torch.where(m0)[0]
                xs[i, 0] = hx[i]; ys[i, 0] = hy[i]
                xs[i, 1] = hx[i] - 1; ys[i, 1] = hy[i]
                xs[i, 2] = hx[i] - 2; ys[i, 2] = hy[i]

            m1 = dir_rand == 1
            if m1.any():
                i = torch.where(m1)[0]
                xs[i, 0] = hx[i]; ys[i, 0] = hy[i]
                xs[i, 1] = hx[i]; ys[i, 1] = hy[i] - 1
                xs[i, 2] = hx[i]; ys[i, 2] = hy[i] - 2

            m2 = dir_rand == 2
            if m2.any():
                i = torch.where(m2)[0]
                xs[i, 0] = hx[i]; ys[i, 0] = hy[i]
                xs[i, 1] = hx[i] + 1; ys[i, 1] = hy[i]
                xs[i, 2] = hx[i] + 2; ys[i, 2] = hy[i]

            m3 = dir_rand == 3
            if m3.any():
                i = torch.where(m3)[0]
                xs[i, 0] = hx[i]; ys[i, 0] = hy[i]
                xs[i, 1] = hx[i]; ys[i, 1] = hy[i] + 1
                xs[i, 2] = hx[i]; ys[i, 2] = hy[i] + 2

            self.snake_x[idx, :3] = xs
            self.snake_y[idx, :3] = ys
            self.direction[idx] = dir_rand

            flat_idx = idx.repeat_interleave(3)
            flat_x = xs.reshape(-1)
            flat_y = ys.reshape(-1)

            self.occupied[flat_idx, flat_y, flat_x] = True
            self.occupied_f[flat_idx, flat_y, flat_x] = 1.0

            self.head_hist_x[idx] = 0
            self.head_hist_y[idx] = 0
            self.hist_ptr[idx] = 0

        else:
            m = self.g // 2
            self.snake_x[idx, :3] = torch.tensor([m, m - 1, m - 2], device=self.device)
            self.snake_y[idx, :3] = m
            self.direction[idx] = 0

            self.occupied[idx, m, m] = True
            self.occupied[idx, m, m - 1] = True
            self.occupied[idx, m, m - 2] = True

            self.occupied_f[idx, m, m] = 1.0
            self.occupied_f[idx, m, m - 1] = 1.0
            self.occupied_f[idx, m, m - 2] = 1.0

            self.head_hist_x[idx] = 0
            self.head_hist_y[idx] = 0
            self.hist_ptr[idx] = 0

        self._place_fruit(idx)
        return self._features()

    def _record_head(self, idx, hx, hy):
        if idx.numel() == 0:
            return
        ptr = self.hist_ptr[idx]
        self.head_hist_x[idx, ptr] = hx
        self.head_hist_y[idx, ptr] = hy
        self.hist_ptr[idx] = (ptr + 1) % self.cycle_window

    # (3) Faster + correct per-row unique count
    def _detect_cycle_mask(self, move_mask):
        if not move_mask.any():
            return torch.zeros_like(move_mask)

        idx = torch.where(move_mask)[0]
        hx = self.head_hist_x[idx]  # [k,W]
        hy = self.head_hist_y[idx]
        h = hx * self.g + hy        # [k,W]

        h_sorted, _ = h.sort(dim=1)
        uniq = 1 + (h_sorted[:, 1:] != h_sorted[:, :-1]).sum(dim=1)

        cycle = (uniq <= self.cycle_unique_thr) & (self.steps_since_eat[idx] > (self.no_eat_limit // 2))

        out = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        out[idx] = cycle
        return out

    def step(self, action, return_obs: str = "vec"):
        """
        return_obs:
          - "vec": return vector obs (34)  (default, backward compatible)
          - "cnn": return cnn obs (37,g,g)
          - "none": return None (fastest; caller can call _features/cnn_features explicitly once)
        """
        alive = ~self.done
        idx = self._idx

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

        delta = (dist_before - dist_after).float() / float(self._max_md)
        shaping_raw = delta.clamp(min=0.0)

        move = alive & ~collision
        if move.any():
            self.head[move] = (self.head[move] - 1) % (self.g * self.g)
            h = self.head[move]
            self.snake_x[move, h] = nx[move]
            self.snake_y[move, h] = ny[move]
            self.occupied[move, ny[move], nx[move]] = True
            self.occupied_f[move, ny[move], nx[move]] = 1.0

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
            y_t = self.snake_y[shrink, t]
            x_t = self.snake_x[shrink, t]
            self.occupied[shrink, y_t, x_t] = False
            self.occupied_f[shrink, y_t, x_t] = 0.0
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

        info = {
            "shaping_raw": shaping_raw,
            "dist_before": dist_before,
            "dist_after": dist_after,
            "cycle_mask": cycle_mask,
        }

        if return_obs == "vec":
            obs = self._features()
        elif return_obs == "cnn":
            obs = self.cnn_features()
        elif return_obs == "none":
            obs = None
        else:
            raise ValueError(f"return_obs must be vec/cnn/none, got {return_obs}")

        return obs, reward, self.done, info

    def _place_fruit(self, idx):
        if idx.numel() == 0:
            return

        k = idx.numel()
        g = self.g

        B = k * 8

        if self.rng is not None:
            xs = torch.randint(0, g, (B,), generator=self.rng, device=self.device)
            ys = torch.randint(0, g, (B,), generator=self.rng, device=self.device)
        else:
            xs = torch.randint(0, g, (B,), device=self.device)
            ys = torch.randint(0, g, (B,), device=self.device)

        occ = self.occupied[idx.repeat_interleave(8), ys, xs]
        occ = occ.view(k, 8)
        free_mask = ~occ

        if not free_mask.any():
            remaining = idx
            while remaining.numel():
                x = torch.randint(0, g, (remaining.numel(),), device=self.device)
                y = torch.randint(0, g, (remaining.numel(),), device=self.device)
                ok = ~self.occupied[remaining, y, x]
                if ok.any():
                    good = remaining[ok]
                    self.fruit_x[good] = x[ok]
                    self.fruit_y[good] = y[ok]
                remaining = remaining[~ok]
            return

        first_free = free_mask.float().argmax(dim=1)

        chosen_x = xs.view(k, 8)[torch.arange(k, device=self.device), first_free]
        chosen_y = ys.view(k, 8)[torch.arange(k, device=self.device), first_free]

        self.fruit_x[idx] = chosen_x
        self.fruit_y[idx] = chosen_y

    # ===== Vector features for MLP (34-dim) =====
    def _features(self):
        """
        Fills and returns a reusable buffer self._feat_buf.
        If you need persistence across env calls, clone() it in caller.
        """
        idx = self._idx
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

        offsets = self._offsets_5x5
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
        mdist_norm = (mdist.float() / float(self._max_md)).clamp(0, 1)

        length_norm = (self.length.float() / float(self.g * self.g)).clamp(0, 1)
        steps_norm = (self.steps_since_eat.float() / float(self.no_eat_limit)).clamp(0, 1)

        # (4a) onehot lookup
        dir_onehot = self._dir_onehot[self.direction]  # [n,4]

        # (4b) fill preallocated buffer instead of cat
        outb = self._feat_buf
        outb[:, 0] = fwd
        outb[:, 1] = rgt
        outb[:, 2] = mdist_norm
        outb[:, 3] = length_norm
        outb[:, 4] = steps_norm
        outb[:, 5:9] = dir_onehot
        outb[:, 9:34] = danger  # 25 cols

        return outb

    def cnn_features(self):
        """
        Returns cached tensor self._cnn_out (reused).
        Caller should clone() if needs persistence.
        Optimizations:
          - use occupied_f (2c)
          - rotation via gather (2a)
          - reuse out tensor (2b)
        """
        n, g, device = self.n, self.g, self.device

        body = self.occupied_f  # [n,g,g], already float32

        idx = self._idx
        hx = self.snake_x[idx, self.head]
        hy = self.snake_y[idx, self.head]

        head_mask = self._head_mask
        fruit_mask = self._fruit_mask
        head_mask.zero_()
        fruit_mask.zero_()

        head_mask[idx, hy, hx] = 1.0
        fruit_mask[idx, self.fruit_y, self.fruit_x] = 1.0

        # rotation so snake faces RIGHT
        rot_k = (4 - self.direction) % 4  # [n]
        lin = self._rot_lin[rot_k]        # [n, g*g]

        body_r = body.view(n, -1).gather(1, lin).view(n, g, g)
        head_r = head_mask.view(n, -1).gather(1, lin).view(n, g, g)
        fruit_r = fruit_mask.view(n, -1).gather(1, lin).view(n, g, g)

        # vector features
        vec = self._features()  # [n,34] (reused buffer)

        # broadcast vector features
        vec_exp = vec[:, :, None, None].expand(-1, -1, g, g)

        out = self._cnn_out
        out[:, 0] = body_r
        out[:, 1] = head_r
        out[:, 2] = fruit_r
        out[:, 3:3 + 34] = vec_exp

        return out