"""
PCB Trace-Growth Environment (round-robin, 1mm steps).

A fundamentally different framing from TPPlacementEnv. Instead of choosing a
test-point ENDPOINT for each trace in one shot, the agent GROWS every trace
1mm at a time, choosing a direction at each step. The key properties:

  * One "round" = one 1mm extension for EACH trace (round-robin over traces).
    After R rounds every trace is exactly R mm long, so LENGTH EQUALITY IS
    GUARANTEED BY CONSTRUCTION -- no length-matching reward term is needed.

  * Within a round the active trace is chosen in a RANDOM order (re-shuffled
    each round) so the policy can't overfit to a fixed extension order and the
    learned policy generalizes across "whose turn is it".

  * Movement is 1mm in one of 8 directions. Diagonal moves are unit-normalized
    so EVERY move travels exactly 1mm regardless of direction (a NE move is
    1mm, not sqrt(2)mm). This keeps all traces exactly equal in length.

  * Traces are continuous polylines (no candidate grid). The test-point is
    simply wherever a trace ends after the final round.

  * A short fixed BREAKOUT segment is drawn straight out from the connector
    for every trace before the agent takes control, so adjacent traces (which
    start ~1.3mm apart) have separated enough to have real directional freedom
    instead of being almost fully masked for the first several steps.

Reward (deliberately minimal, per design discussion):
  * DENSE per-step: current minimum pairwise distance between trace TIPS,
    normalized -- gives signal every step instead of only at the end.
  * TERMINAL: minimum pairwise distance between trace ENDPOINTS (the quantity
    the portfolio actually ranks on), plus a validity gate.
  There is intentionally NO length-spread term (equality is structural), NO
  fanout bonus (curling to consume length is desired), and NO soft constraint
  penalty (hard masking handles obstacles; being close-but-not-touching is OK).

Hard masking (not reward) enforces:
  * board-boundary / edge clearance
  * obstacle + connector clearance
  * trace-to-trace clearance along the whole path (a move that would bring the
    active tip too close to ANY segment of ANY other trace is masked)

Observation (dict):
  * image:    HxHx3 uint8 render (default 256x256 -- 1mm steps are sub-pixel at
              64x64 on a 180x120mm board, so high resolution is mandatory).
                Red   = obstacles / clearance / connector / edge
                Green = all grown traces so far (active trace brighter)
                Blue  = active trace tip + its valid next-step directions
  * trace_id: one-hot vector (length num_traces) marking the active trace, so
              the model unambiguously knows which trace it controls even when
              traces are visually similar.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, List, Tuple

from envs.board import (
    BoardSpec, load_te_example,
    TRACE_TO_EDGE_MIN, TRACE_TO_TRACE_MIN, TRACE_WIDTH,
    TP_TO_TP_MIN, TP_TO_EDGE_MIN, TP_TO_CONNECTOR_MIN,
)

# 8 movement directions (unit vectors, all normalized to length 1mm).
_SQRT2_INV = 1.0 / np.sqrt(2.0)
DIRECTIONS = np.array([
    (1.0, 0.0),                 # E
    (_SQRT2_INV, _SQRT2_INV),   # NE
    (0.0, 1.0),                 # N
    (-_SQRT2_INV, _SQRT2_INV),  # NW
    (-1.0, 0.0),                # W
    (-_SQRT2_INV, -_SQRT2_INV), # SW
    (0.0, -1.0),                # S
    (_SQRT2_INV, -_SQRT2_INV),  # SE
], dtype=np.float64)
NUM_DIRECTIONS = len(DIRECTIONS)

STEP_MM = 1.0                 # each move travels exactly this far
BREAKOUT_MM = 10.0           # fixed fanned breakout before agent control
DEFAULT_IMG_SIZE = 256

# Minimum center-to-center distance between two trace paths. Trace width plus
# the edge-to-edge clearance, same basis as TRACE_MIN_CENTER_TO_CENTER but kept
# explicit here for the path-clearance mask.
TRACE_PATH_CLEARANCE = TRACE_TO_TRACE_MIN + TRACE_WIDTH  # ~1.3286 mm


def _densify(points, step):
    """Resample a polyline into points spaced ~`step` mm apart (keeps the
    first and last vertex). Used so the fixed breakout polyline becomes a
    dense point list comparable to the agent's 1mm growth steps."""
    out = [tuple(points[0])]
    for a, b in zip(points[:-1], points[1:]):
        ax, ay = a
        bx, by = b
        seg = np.hypot(bx - ax, by - ay)
        if seg < 1e-9:
            continue
        n = max(1, int(round(seg / step)))
        for k in range(1, n + 1):
            t = k / n
            out.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    return out


class TraceGrowEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        board: Optional[BoardSpec] = None,
        num_traces: int = 6,
        max_length_mm: float = 60.0,
        img_size: int = DEFAULT_IMG_SIZE,
        seed: int = 0,
        board_width: float = 180.0,
        board_height: float = 120.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self._board_seed = seed
        self.img_size = img_size
        self.max_length_mm = max_length_mm

        if board is None:
            # seed=None -> fixed connector position (single-board search).
            board = load_te_example(num_traces=num_traces, seed=None,
                                    board_width=board_width,
                                    board_height=board_height)
        self.board = board
        # The growth env does not depend on board.traces beyond their start
        # points; it can use as many traces as the board provides.
        self.num_traces = min(num_traces, len(self.board.traces))
        self.board.traces = self.board.traces[:self.num_traces]

        # Rounds = how many 1mm extensions each trace gets. After all rounds
        # every trace is exactly (BREAKOUT_MM + rounds*STEP_MM) long.
        self.num_rounds = int(round(max_length_mm / STEP_MM))
        # Total agent steps if no trace is ever blocked.
        self.ideal_steps = self.num_rounds * self.num_traces
        # Hard cap with slack so blocked traces get extra turns to free up
        # without letting the episode run unbounded. TimeLimit should match
        # episode_steps. 1.5x gives headroom for occasional blocking.
        self.episode_steps = int(self.ideal_steps * 1.5)

        self.action_space = spaces.Discrete(NUM_DIRECTIONS)
        self.observation_space = spaces.Box(
            0, 255, (img_size, img_size, 3), dtype=np.uint8
        )

        # Coordinate transform (world mm -> pixel).
        self._x_scale = (img_size - 1) / max(self.board.width, 1e-6)
        self._y_scale = (img_size - 1) / max(self.board.height, 1e-6)

        # Per-trace polylines (list of (x, y) points); index 0 is the start.
        self.paths: List[List[Tuple[float, float]]] = []
        self.tips: np.ndarray = np.zeros((self.num_traces, 2))
        self.last_dir = np.zeros(self.num_traces, dtype=int)
        # grown[ti] = how many successful 1mm extensions trace ti has made.
        # Length equality is enforced on this count: the episode ends only when
        # EVERY trace has grown num_rounds times (or the hard step cap is hit).
        # A trace that is temporarily boxed in simply doesn't advance its count
        # that turn; the round-robin keeps returning to it. Any trace that
        # reaches num_rounds is exactly the same length as every other completed
        # trace. The terminal metric reports whether all traces completed.
        self.grown = np.zeros(self.num_traces, dtype=int)
        self.steps_taken = 0
        self.order: List[int] = []
        self.order_pos = 0
        self.active = 0
        self.current_mask = np.ones(NUM_DIRECTIONS, dtype=bool)

        self._episode_invalid_actions = 0
        self._terminal_metrics = {}

    # ------------------------------------------------------------------
    # coordinate / drawing helpers
    # ------------------------------------------------------------------

    def _w2p(self, x: float, y: float) -> Tuple[int, int]:
        px = int((x - self.board.x_min) * self._x_scale)
        py = int((y - self.board.y_min) * self._y_scale)
        return (np.clip(px, 0, self.img_size - 1),
                np.clip(py, 0, self.img_size - 1))

    def _draw_circle(self, img, cx, cy, r_mm, ch, val=255):
        pcx, pcy = self._w2p(cx, cy)
        pr = max(1, int(r_mm * self._x_scale))
        for dy in range(-pr, pr + 1):
            for dx in range(-pr, pr + 1):
                if dx * dx + dy * dy <= pr * pr:
                    py, px = pcy + dy, pcx + dx
                    if 0 <= py < self.img_size and 0 <= px < self.img_size:
                        img[py, px, ch] = min(255, int(img[py, px, ch]) + val)

    def _draw_rect(self, img, xmin, ymin, xmax, ymax, ch, val=255):
        px0, py0 = self._w2p(xmin, ymin)
        px1, py1 = self._w2p(xmax, ymax)
        py0, py1 = max(0, min(py0, py1)), min(self.img_size, max(py0, py1) + 1)
        px0, px1 = max(0, min(px0, px1)), min(self.img_size, max(px0, px1) + 1)
        img[py0:py1, px0:px1, ch] = np.minimum(
            255, img[py0:py1, px0:px1, ch].astype(np.int16) + val
        ).astype(np.uint8)

    def _draw_segment(self, img, p0, p1, ch, val=255):
        """Rasterize a line segment (mm coords) into channel `ch`."""
        x0, y0 = self._w2p(*p0)
        x1, y1 = self._w2p(*p1)
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            if 0 <= y0 < self.img_size and 0 <= x0 < self.img_size:
                img[y0, x0, ch] = min(255, int(img[y0, x0, ch]) + val)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------

    def _render_obs(self) -> np.ndarray:
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # RED: obstacles + clearance + connector + edge
        for obs in self.board.rect_obstacles:
            xn, yn, xx, yx = obs.bounds
            self._draw_rect(img, xn - obs.clearance, yn - obs.clearance,
                            xx + obs.clearance, yx + obs.clearance, 0, 150)
            self._draw_rect(img, xn, yn, xx, yx, 0, 255)
        for obs in self.board.circ_obstacles:
            self._draw_circle(img, obs.cx, obs.cy,
                              obs.radius + obs.clearance, 0, 150)
            self._draw_circle(img, obs.cx, obs.cy, obs.radius, 0, 255)
        edge_px = max(1, int(TP_TO_EDGE_MIN * self._x_scale))
        img[:edge_px, :, 0] = 100
        img[-edge_px:, :, 0] = 100
        img[:, :edge_px, 0] = 100
        img[:, -edge_px:, 0] = 100
        if self.board.connector_w > 0:
            self._draw_rect(img, self.board.connector_x, self.board.connector_y,
                            self.board.connector_x + self.board.connector_w,
                            self.board.connector_y + self.board.connector_h,
                            0, 180)

        # GREEN: all grown traces. Active trace is brighter so the CNN has an
        # extra cue (in addition to the one-hot trace_id) for which it controls.
        for ti, path in enumerate(self.paths):
            val = 255 if ti == self.active else 130
            for k in range(len(path) - 1):
                self._draw_segment(img, path[k], path[k + 1], 1, val)

        # BLUE: active tip + its currently-valid next directions
        if self.num_traces:
            tx, ty = self.tips[self.active]
            self._draw_circle(img, tx, ty, 1.5, 2, 255)
            for d in range(NUM_DIRECTIONS):
                if self.current_mask[d]:
                    nx = tx + DIRECTIONS[d, 0] * STEP_MM * 2
                    ny = ty + DIRECTIONS[d, 1] * STEP_MM * 2
                    px, py = self._w2p(nx, ny)
                    if 0 <= py < self.img_size and 0 <= px < self.img_size:
                        img[py, px, 2] = min(255, int(img[py, px, 2]) + 120)

        return img

    def _trace_id_onehot(self) -> np.ndarray:
        v = np.zeros(self.num_traces, dtype=np.float32)
        if self.num_traces:
            v[self.active] = 1.0
        return v

    # ------------------------------------------------------------------
    # validity / masking
    # ------------------------------------------------------------------

    def _point_clear_of_static(self, x: float, y: float) -> bool:
        """Board edge + obstacle + connector clearance for a path point."""
        # Edge clearance (use trace-to-edge, the routing clearance, not TP).
        if (x - self.board.x_min < TRACE_TO_EDGE_MIN or
                self.board.x_max - x < TRACE_TO_EDGE_MIN or
                y - self.board.y_min < TRACE_TO_EDGE_MIN or
                self.board.y_max - y < TRACE_TO_EDGE_MIN):
            return False

        # Connector outline clearance
        cxmin = self.board.connector_x
        cxmax = self.board.connector_x + self.board.connector_w
        cymin = self.board.connector_y
        cymax = self.board.connector_y + self.board.connector_h
        if self.board.connector_w > 0:
            if cxmin <= x <= cxmax and cymin <= y <= cymax:
                return False
            dx = max(cxmin - x, 0.0, x - cxmax)
            dy = max(cymin - y, 0.0, y - cymax)
            if np.hypot(dx, dy) < TP_TO_CONNECTOR_MIN:
                return False

        # Rectangular obstacles (with clearance)
        for obs in self.board.rect_obstacles:
            xmin, ymin, xmax, ymax = obs.bounds
            buf = obs.clearance
            if xmin - buf < x < xmax + buf and ymin - buf < y < ymax + buf:
                return False

        # Circular obstacles (with clearance)
        for obs in self.board.circ_obstacles:
            if np.hypot(x - obs.cx, y - obs.cy) < obs.radius + obs.clearance:
                return False

        return True

    def _point_clear_of_other_traces(self, x: float, y: float,
                                     active: int) -> bool:
        """Path-level clearance: (x,y) must stay TRACE_PATH_CLEARANCE away from
        every segment of every OTHER trace, and from earlier (non-adjacent)
        points of its own path."""
        for ti, path in enumerate(self.paths):
            if len(path) < 2:
                # Still guard against the lone start point of another trace.
                if ti != active and path:
                    if np.hypot(x - path[0][0], y - path[0][1]) < TRACE_PATH_CLEARANCE:
                        return False
                continue
            # For the active trace, skip the last couple of points (the tip and
            # its immediate predecessor) so we don't self-collide trivially.
            seg_end = len(path) - 1
            if ti == active:
                seg_end = max(0, len(path) - 3)
            for k in range(seg_end):
                if self._point_seg_dist(x, y, path[k], path[k + 1]) < TRACE_PATH_CLEARANCE:
                    return False
        return True

    @staticmethod
    def _point_seg_dist(px, py, a, b) -> float:
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-12:
            return np.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
        t = max(0.0, min(1.0, t))
        cx, cy = ax + t * dx, ay + t * dy
        return np.hypot(px - cx, py - cy)

    def _compute_mask(self) -> np.ndarray:
        """Mask of valid next directions for the active trace's tip."""
        tx, ty = self.tips[self.active]
        mask = np.zeros(NUM_DIRECTIONS, dtype=bool)
        for d in range(NUM_DIRECTIONS):
            nx = tx + DIRECTIONS[d, 0] * STEP_MM
            ny = ty + DIRECTIONS[d, 1] * STEP_MM
            if not self._point_clear_of_static(nx, ny):
                continue
            if not self._point_clear_of_other_traces(nx, ny, self.active):
                continue
            mask[d] = True
        return mask

    # ------------------------------------------------------------------
    # reward
    # ------------------------------------------------------------------

    def _min_pairwise(self, pts: np.ndarray) -> float:
        if len(pts) < 2:
            return 0.0
        m = float("inf")
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                if d < m:
                    m = d
        return m

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.RandomState(
            seed if seed is not None else self._board_seed)

        # Initialize each path with its start point + a fixed straight breakout
        # so adjacent traces separate before the agent takes over.
        self.paths = []
        self.tips = np.zeros((self.num_traces, 2))
        self.last_dir = np.zeros(self.num_traces, dtype=int)

        conn_cx = self.board.connector_x + self.board.connector_w / 2
        conn_cy = self.board.connector_y + self.board.connector_h / 2
        conn_top = self.board.connector_y + self.board.connector_h
        conn_bot = self.board.connector_y

        # Breakout. The trace starts sit INSIDE the connector outline, and the
        # board is asymmetric (here the connector hugs the bottom edge, so the
        # only roomy exit is "up", i.e. toward whichever board edge is far).
        # Every trace breaks out in two phases of EQUAL TOTAL LENGTH so length
        # equality is preserved:
        #   Phase 1: straight along the roomy axis until clear of the connector.
        #   Phase 2: fan laterally, spreading traces toward target TP spacing.
        # The two-segment polyline is then resampled to a fixed total length so
        # all traces' breakouts are identical length regardless of fan amount.
        space_up = self.board.y_max - conn_top
        space_down = conn_bot - self.board.y_min
        if space_up >= space_down:
            dir_y, exit_y = 1.0, conn_top + (TP_TO_CONNECTOR_MIN + 1.0)
        else:
            dir_y, exit_y = -1.0, conn_bot - (TP_TO_CONNECTOR_MIN + 1.0)

        order = sorted(range(self.num_traces),
                       key=lambda i: self.board.traces[i].start_x)
        n = self.num_traces

        # Equal-arc fan from a virtual pivot above the connector. Every trace
        # is routed start -> exit (clear of connector) -> a fan endpoint placed
        # on an arc of radius R about the connector center, at equal angular
        # spacing across a wide arc. Equal radius => equal-distance endpoints
        # from the pivot, and the start->exit->endpoint polylines are then
        # length-normalized so all breakouts are exactly equal length. The arc
        # placement guarantees the endpoints are well separated regardless of
        # how tightly the starts are packed.
        R = BREAKOUT_MM + abs(exit_y - conn_cy)
        arc_span = np.deg2rad(160.0)
        base_ang = (np.pi / 2 if dir_y > 0 else -np.pi / 2)
        raw_paths = {}
        for rank, ti in enumerate(order):
            t = self.board.traces[ti]
            sx, sy = t.start_x, t.start_y
            frac = 0.0 if n == 1 else (rank / (n - 1) - 0.5)  # [-0.5, 0.5]
            ang = base_ang + frac * arc_span
            ex = conn_cx + R * np.cos(ang)
            ey = conn_cy + R * np.sin(ang)
            # via point: straight out of the connector first, then to the arc
            p_exit = (sx, exit_y)
            raw_paths[ti] = [(sx, sy), p_exit, (ex, ey)]

        def _polylen(pts):
            return sum(np.hypot(pts[k + 1][0] - pts[k][0],
                                pts[k + 1][1] - pts[k][1])
                       for k in range(len(pts) - 1))

        max_blen = max(_polylen(p) for p in raw_paths.values())
        indexed_paths = []
        for ti, raw in raw_paths.items():
            blen = _polylen(raw)
            pad = max_blen - blen
            if pad > 1e-6:
                ex, ey = raw[-1]
                # extend along the outward radial so padding keeps the fan shape
                rad = np.array([ex - conn_cx, ey - conn_cy])
                rad = rad / (np.linalg.norm(rad) + 1e-9)
                raw = raw + [(ex + rad[0] * pad, ey + rad[1] * pad)]
            path = _densify(raw, STEP_MM)
            indexed_paths.append((ti, path))
            self.tips[ti] = path[-1]
            vx, vy = path[-1][0] - path[-2][0], path[-1][1] - path[-2][1]
            nrm = np.hypot(vx, vy) + 1e-9
            self.last_dir[ti] = int(np.argmax(DIRECTIONS @ np.array([vx / nrm, vy / nrm])))

        indexed_paths.sort(key=lambda x: x[0])
        self.paths = [p for _, p in indexed_paths]

        self.grown = np.zeros(self.num_traces, dtype=int)
        self.steps_taken = 0
        # Round-robin order over INCOMPLETE traces, reshuffled each pass.
        self.order = list(range(self.num_traces))
        self._rng.shuffle(self.order)
        self.order_pos = 0
        self.active = self._next_active(reset=True)
        self.current_mask = self._compute_mask()

        self._episode_invalid_actions = 0
        self._terminal_metrics = {}

        return self._render_obs(), self._get_info()

    def _next_active(self, reset=False) -> int:
        """Pick the next trace that still has growth left, round-robin with a
        fresh shuffle once every incomplete trace has had a turn this pass."""
        incomplete = [t for t in range(self.num_traces)
                      if self.grown[t] < self.num_rounds]
        if not incomplete:
            return self.active  # all done; caller will terminate
        if reset:
            self.order = [t for t in self.order if t in incomplete]
            if not self.order:
                self.order = incomplete[:]
                self._rng.shuffle(self.order)
            self.order_pos = 0
            return self.order[0]
        # advance within the current pass; rebuild when exhausted
        self.order_pos += 1
        if self.order_pos >= len(self.order):
            self.order = incomplete[:]
            self._rng.shuffle(self.order)
            self.order_pos = 0
        # current entry may have completed since the pass started; skip ahead
        while self.order[self.order_pos] not in incomplete:
            self.order_pos += 1
            if self.order_pos >= len(self.order):
                self.order = incomplete[:]
                self._rng.shuffle(self.order)
                self.order_pos = 0
                break
        return self.order[self.order_pos]

    def step(self, action: int):
        action = int(action)
        reward = 0.0
        invalid_this_step = False
        moved = False

        # Apply move (with redirect-to-nearest-valid fallback if the policy
        # picks a masked direction; we never terminate on collision).
        if not self.current_mask[action]:
            invalid_this_step = True
            self._episode_invalid_actions += 1
            valid = np.where(self.current_mask)[0]
            if len(valid) > 0:
                target = DIRECTIONS[action]
                dots = DIRECTIONS[valid] @ target
                action = int(valid[int(np.argmax(dots))])
            else:
                action = None  # fully blocked this turn

        if action is not None:
            tx, ty = self.tips[self.active]
            nx = tx + DIRECTIONS[action, 0] * STEP_MM
            ny = ty + DIRECTIONS[action, 1] * STEP_MM
            self.paths[self.active].append((nx, ny))
            self.tips[self.active] = (nx, ny)
            self.last_dir[self.active] = action
            self.grown[self.active] += 1
            moved = True

        # DENSE reward: current min pairwise distance between tips, normalized
        # by the target TP spacing. Capped at 1 so it stays well-scaled.
        tip_min = self._min_pairwise(self.tips)
        reward += 0.05 * min(tip_min / TP_TO_TP_MIN, 1.0)

        self.steps_taken += 1

        # Termination: every trace has grown its full length, OR the hard step
        # cap is reached (covers the pathological case of a permanently boxed-in
        # trace). Length equality holds for all traces that reached num_rounds.
        all_complete = bool(np.all(self.grown >= self.num_rounds))
        cap_reached = self.steps_taken >= self.episode_steps
        terminated = all_complete or cap_reached

        if terminated:
            reward += self._terminal_reward(all_complete)
        else:
            self.active = self._next_active()
            self.current_mask = self._compute_mask()

        return (self._render_obs(), np.float32(reward), terminated, False,
                self._get_info(invalid_this_step))

    def _terminal_reward(self, all_complete: bool) -> float:
        endpoints = self.tips.copy()
        ep_min = self._min_pairwise(endpoints)

        # Endpoint validity: every endpoint must be a legal test-point location
        # (edge + connector + obstacle clearance), and TP-to-TP spacing met.
        valid_endpoints = all(
            self._point_clear_of_static(x, y) for x, y in endpoints
        )
        spacing_ok = ep_min >= TP_TO_TP_MIN if self.num_traces > 1 else True

        # Length equality holds by construction only if every trace completed.
        # If the step cap was hit with some trace boxed in, lengths differ and
        # the solution is invalid.
        completed = all_complete and bool(np.all(self.grown >= self.num_rounds))

        if completed and valid_endpoints and spacing_ok:
            reward_spacing = 10.0 * min(ep_min / (2 * TP_TO_TP_MIN), 1.0)
            gate = 10.0
        else:
            reward_spacing = 0.0
            # graded penalty: pull toward valid spacing + reward completion
            # progress, but always below the worst valid solution.
            frac_complete = float(np.mean(self.grown / max(self.num_rounds, 1)))
            gate = -8.0 + 4.0 * frac_complete + 4.0 * min(ep_min / TP_TO_TP_MIN, 1.0)

        # length spread across COMPLETED traces (should be ~0 by construction)
        comp_lengths = [
            sum(np.hypot(p[k + 1][0] - p[k][0], p[k + 1][1] - p[k][1])
                for k in range(len(p) - 1))
            for ti, p in enumerate(self.paths)
            if self.grown[ti] >= self.num_rounds
        ]
        length_spread = (max(comp_lengths) - min(comp_lengths)) if len(comp_lengths) > 1 else 0.0
        total_len = sum(
            sum(np.hypot(p[k + 1][0] - p[k][0], p[k + 1][1] - p[k][1])
                for k in range(len(p) - 1))
            for p in self.paths
        )

        self._terminal_metrics = {
            "min_tp_spacing": ep_min,
            "endpoints_valid": 1.0 if valid_endpoints else 0.0,
            "spacing_ok": 1.0 if spacing_ok else 0.0,
            "all_complete": 1.0 if completed else 0.0,
            "routable": 1.0 if (completed and valid_endpoints and spacing_ok) else 0.0,
            "total_length": total_len,
            "length_spread": length_spread,
            "reward_spacing": reward_spacing,
            "reward_gate": gate,
        }
        return reward_spacing + gate

    def _get_info(self, invalid_this_step: bool = False):
        info = {
            "min_grown": int(self.grown.min()) if self.num_traces else 0,
            "active_trace": self.active,
            "invalid_this_step": invalid_this_step,
        }
        info.update(self._terminal_metrics)
        return info

    def render(self):
        return self._render_obs()

    # Convenience for trackers / visualization.
    def get_endpoints(self) -> np.ndarray:
        return self.tips.copy()

    def get_paths(self):
        return [list(p) for p in self.paths]
