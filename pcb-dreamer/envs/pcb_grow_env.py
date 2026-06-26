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
    BoardSpec, load_te_example, load_actual_te_board,
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
        num_traces: int = 8,
        max_length_mm: float = 60.0,
        img_size: int = DEFAULT_IMG_SIZE,
        seed: int = 0,
        board_width: float = 135.0,
        board_height: float = 90.0,
        step_mm: float = 2.0,
        trace_indices: Optional[List[int]] = None,
        dense_reward_weight: float = 0.005,
        spacing_threshold: float = 5.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self._board_seed = seed
        self.img_size = img_size
        self.max_length_mm = max_length_mm
        self.step_mm = step_mm
        self.dense_reward_weight = dense_reward_weight
        self.spacing_threshold = spacing_threshold  # mm, min pairwise endpoint spacing for a valid solution

        if board is None:
            # Use the exact TE board by default; trace_indices selects which
            # physical pins to route (1-based, default [1,2,3,4,11,12,13,14]).
            board = load_actual_te_board(
                trace_indices=trace_indices or [1, 2, 3, 4, 11, 12, 13, 14]
            )
        self.board = board
        # num_traces is the count of traces we actually route, capped by
        # how many the board defines.
        self.num_traces = min(num_traces, len(self.board.traces))
        self.board.traces = self.board.traces[:self.num_traces]

        # Rounds = how many step_mm extensions each trace gets.
        self.num_rounds = int(round(max_length_mm / step_mm))
        # Total agent steps if no trace is ever blocked.
        self.ideal_steps = self.num_rounds * self.num_traces
        # Hard cap with slack so blocked traces get extra turns to free up.
        # 3x headroom so temporarily-blocked traces can recover.
        self.episode_steps = int(self.ideal_steps * 3)

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

        # WHITE: pin origin crosses (sub-pixel at board scale but visible close up)
        for ti, path in enumerate(self.paths):
            sx, sy = path[0]
            px, py = self._w2p(sx, sy)
            for delta in range(-2, 3):
                for ppx, ppy in [(px+delta, py), (px, py+delta)]:
                    if 0 <= ppy < self.img_size and 0 <= ppx < self.img_size:
                        img[ppy, ppx, 0] = 255
                        img[ppy, ppx, 1] = 255
                        img[ppy, ppx, 2] = 255

        # BLUE: active tip + its currently-valid next directions
        if self.num_traces:
            tx, ty = self.tips[self.active]
            self._draw_circle(img, tx, ty, 1.5, 2, 255)
            for d in range(NUM_DIRECTIONS):
                if self.current_mask[d]:
                    nx = tx + DIRECTIONS[d, 0] * self.step_mm * 2
                    ny = ty + DIRECTIONS[d, 1] * self.step_mm * 2
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

    def _point_clear_of_static(self, x: float, y: float,
                               for_endpoint: bool = False) -> bool:
        """Board edge + obstacle + connector clearance for a path point.

        During growth (for_endpoint=False) we only enforce TRACE_TO_EDGE_MIN
        (0.26mm) — traces are free to travel anywhere on the board, including
        the lower region near the connector, to allow curling behaviour like
        the reference image shows.

        The terminal endpoint validity check uses TP_TO_EDGE_MIN (14mm) and is
        handled separately in _terminal_reward via for_endpoint=True. The agent
        learns this constraint through the terminal reward signal — deliberately
        not hard-masking it so the agent can route through the lower region on
        the way to a valid endpoint elsewhere.
        """
        # Use the full 14mm spec clearance for endpoints. With 2mm steps
        # (the recommended step_mm), the grid quantization error is ≤2mm,
        # so valid endpoint positions are always reachable.
        edge_clear = TP_TO_EDGE_MIN if for_endpoint else TRACE_TO_EDGE_MIN
        if (x - self.board.x_min < edge_clear or
                self.board.x_max - x < edge_clear or
                y - self.board.y_min < edge_clear or
                self.board.y_max - y < edge_clear):
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
        """Path-level clearance: (x,y) must stay TRACE_PATH_CLEARANCE away
        from recent segments of every other trace and early segments of own
        path. Only checks the last LOOKBACK segments for efficiency."""
        LOOKBACK = 8  # check this many recent segments per trace
        for ti, path in enumerate(self.paths):
            if len(path) < 2:
                if ti != active and path:
                    if np.hypot(x - path[0][0], y - path[0][1]) < TRACE_PATH_CLEARANCE:
                        return False
                continue
            if ti == active:
                # For own trace: skip the very tip (last 2 points) to avoid
                # trivial self-collision, check a few earlier segments
                seg_end = max(0, len(path) - 3)
                seg_start = max(0, seg_end - LOOKBACK)
            else:
                # For other traces: check the most recent LOOKBACK segments
                seg_end = len(path) - 1
                seg_start = max(0, seg_end - LOOKBACK)
            for k in range(seg_start, seg_end):
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

    def _physical_obstacle_hit(self, x: float, y: float) -> bool:
        """True if (x,y) is strictly inside a physical copper obstacle.
        Uses strict inequality so points exactly on the boundary (like pin
        start positions) are NOT considered inside -- they can move freely.
        """
        for obs in self.board.rect_obstacles:
            xmin, ymin, xmax, ymax = obs.bounds
            if xmin < x < xmax and ymin < y < ymax:
                return True
        for obs in self.board.circ_obstacles:
            if np.hypot(x - obs.cx, y - obs.cy) < obs.radius:
                return True
        return False

    def _obstacle_penetration(self, x: float, y: float) -> float:
        """Return depth of penetration into any obstacle/connector (0 if clear)."""
        depth = 0.0
        # Connector
        cxmin = self.board.connector_x
        cxmax = self.board.connector_x + self.board.connector_w
        cymin = self.board.connector_y
        cymax = self.board.connector_y + self.board.connector_h
        if self.board.connector_w > 0:
            if cxmin <= x <= cxmax and cymin <= y <= cymax:
                depth = max(depth, 1.0)
            else:
                dx = max(cxmin - x, 0.0, x - cxmax)
                dy = max(cymin - y, 0.0, y - cymax)
                d = np.hypot(dx, dy)
                if d < TP_TO_CONNECTOR_MIN:
                    depth = max(depth, TP_TO_CONNECTOR_MIN - d)
        # Rectangular obstacles
        for obs in self.board.rect_obstacles:
            xmin, ymin, xmax, ymax = obs.bounds
            buf = obs.clearance
            if xmin - buf < x < xmax + buf and ymin - buf < y < ymax + buf:
                pen = min(x - (xmin - buf), (xmax + buf) - x,
                          y - (ymin - buf), (ymax + buf) - y)
                depth = max(depth, max(0.0, pen))
        # Circular obstacles
        for obs in self.board.circ_obstacles:
            d = np.hypot(x - obs.cx, y - obs.cy)
            if d < obs.radius + obs.clearance:
                depth = max(depth, obs.radius + obs.clearance - d)
        return depth

    def _crossing_penalty(self, x: float, y: float, active: int) -> float:
        """Return a penalty (0-1 scale) for proximity to other trace segments."""
        LOOKBACK = 8
        total = 0.0
        for ti, path in enumerate(self.paths):
            if ti == active or len(path) < 2:
                continue
            seg_end = len(path) - 1
            seg_start = max(0, seg_end - LOOKBACK)
            for k in range(seg_start, seg_end):
                d = self._point_seg_dist(x, y, path[k], path[k + 1])
                if d < TRACE_PATH_CLEARANCE:
                    total += (TRACE_PATH_CLEARANCE - d) / TRACE_PATH_CLEARANCE
        return total

    def _compute_mask(self) -> np.ndarray:
        """Mask of valid next directions for the active trace's tip.

        Hard masks (physical impossibilities):
          - Off-board
          - Through the connector body or PCB obstacles
          - Trace-to-trace crossing

        Edge clearance remains soft (reward only) so traces can curl near
        edges and through the lower board region.
        """
        tx, ty = self.tips[self.active]
        mask = np.zeros(NUM_DIRECTIONS, dtype=bool)
        for d in range(NUM_DIRECTIONS):
            nx = tx + DIRECTIONS[d, 0] * self.step_mm
            ny = ty + DIRECTIONS[d, 1] * self.step_mm
            # Hard: off-board
            if (nx < self.board.x_min or nx > self.board.x_max or
                    ny < self.board.y_min or ny > self.board.y_max):
                continue
            # Hard: through actual PCB obstacles (NRZ rectangle, UPTH holes,
            # tab pads). The connector OUTLINE is a clearance zone for endpoints
            # only -- traces can route near it. Only physical copper obstacles
            # are hard-masked during growth.
            if self._physical_obstacle_hit(nx, ny):
                continue
            # Hard: trace-to-trace crossing
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

    def _mean_pairwise(self, pts: np.ndarray) -> float:
        """Mean distance over all pairs. Better training signal than min:
        gives gradient on all pairs simultaneously, not just the closest."""
        if len(pts) < 2:
            return 0.0
        total, count = 0.0, 0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                total += np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                count += 1
        return total / count if count > 0 else 0.0

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.RandomState(
            seed if seed is not None else self._board_seed)

        # No pre-computed breakout. The agent routes from the exact connector
        # pin positions. The trace-to-trace crossing mask naturally forces
        # adjacent pins to diverge since they cannot step into each other,
        # producing organic fanout without pre-determined geometry. This also
        # means no trace geometry passes through the connector obstacle zone.
        self.paths = []
        self.tips = np.zeros((self.num_traces, 2))
        self.last_dir = np.zeros(self.num_traces, dtype=int)

        for ti, t in enumerate(self.board.traces):
            sx, sy = t.start_x, t.start_y
            self.paths.append([(sx, sy)])
            self.tips[ti] = (sx, sy)
            # Default initial direction: upward (N), away from connector body
            self.last_dir[ti] = 2

        self.grown = np.zeros(self.num_traces, dtype=int)
        self.steps_taken = 0
        # Round-robin order over INCOMPLETE traces, reshuffled each pass.
        self.order = list(range(self.num_traces))
        self._rng.shuffle(self.order)
        self.order_pos = 0
        self.rounds_completed = 0  # full round-robin cycles completed
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
            nx = tx + DIRECTIONS[action, 0] * self.step_mm
            ny = ty + DIRECTIONS[action, 1] * self.step_mm
            self.paths[self.active].append((nx, ny))
            self.tips[self.active] = (nx, ny)
            self.last_dir[self.active] = action
            self.grown[self.active] += 1
            moved = True

        # DENSE reward: soft signals, no hard gates.
        # All constraints are expressed as graded penalties or bonuses so the
        # world model always receives a meaningful reward signal and the random
        # policy is never blocked from completing an episode.

        # 1. Tip spacing bonus: mean pairwise distance in mm.
        #    Rewards spreading all pairs, not just the closest.
        tip_mean = self._mean_pairwise(self.tips)
        reward += self.dense_reward_weight * tip_mean * 0.1

        tx, ty = self.tips[self.active]

        # 2. Edge proximity soft penalty
        edge_margin = min(
            tx - self.board.x_min, self.board.x_max - tx,
            ty - self.board.y_min, self.board.y_max - ty,
        )
        if edge_margin < TP_TO_EDGE_MIN:
            reward += self.dense_reward_weight * (edge_margin - TP_TO_EDGE_MIN) / TP_TO_EDGE_MIN

        # 3. Obstacle / connector soft penalty
        obs_pen = self._obstacle_penetration(tx, ty)
        if obs_pen > 0:
            reward -= self.dense_reward_weight * min(obs_pen / 5.0, 1.0)

        self.steps_taken += 1

        # Per-round intermediate reward: fires whenever the round-robin wraps.
        # A round wraps when order_pos is about to roll over to 0 on the next
        # call to _next_active. We detect this by checking if the CURRENT
        # order_pos is at the last position in the current order.
        # This gives the model a graded signal every ~8 steps instead of
        # only at episode termination (every 320 steps).
        if moved and (self.order_pos + 1 >= len(self.order)):
            self.rounds_completed += 1
            reward += self._per_round_reward()

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

    def _per_round_reward(self) -> float:
        """Intermediate reward at end of each complete round-robin cycle (~8 steps).

        Evaluates current tip positions as proxy endpoints, scaled by progress
        through the episode (progress^2 so early rounds don't dominate).
        This puts meaningful constraint signals within the 15-step imagination
        horizon rather than 305 steps away at termination.
        """
        if self.num_rounds == 0:
            return 0.0
        progress = min(self.rounds_completed / self.num_rounds, 1.0)

        # Spacing signal: mean pairwise distance, scaled down for intermediate reward.
        spacing_signal = self._mean_pairwise(self.tips) * 0.05

        # Endpoint proximity penalty (lighter weight than terminal)
        ep_pen = 0.0
        for x, y in self.tips:
            edge = min(x - self.board.x_min, self.board.x_max - x,
                       y - self.board.y_min, self.board.y_max - y)
            if edge < TP_TO_EDGE_MIN:
                ep_pen += 0.1 * (TP_TO_EDGE_MIN - edge) / TP_TO_EDGE_MIN
            obs = self._obstacle_penetration(x, y)
            if obs > 0:
                ep_pen += 0.1 * min(obs / 5.0, 1.0)

        # Scale by progress^2: near-zero early (tips still near connector),
        # meaningful in the final third of the episode
        return (progress ** 2) * (spacing_signal - ep_pen)

    def _terminal_reward(self, all_complete: bool) -> float:
        endpoints = self.tips.copy()
        ep_min = self._min_pairwise(endpoints)
        frac_complete = float(np.mean(self.grown / max(self.num_rounds, 1)))
        completed = all_complete and bool(np.all(self.grown >= self.num_rounds))

        # ── Endpoint soft penalties ───────────────────────────────────────────
        endpoint_penalty = 0.0
        for x, y in endpoints:
            edge_margin = min(x - self.board.x_min, self.board.x_max - x,
                              y - self.board.y_min, self.board.y_max - y)
            if edge_margin < TP_TO_EDGE_MIN:
                endpoint_penalty += (TP_TO_EDGE_MIN - edge_margin) / TP_TO_EDGE_MIN
            obs_pen = self._obstacle_penetration(x, y)
            if obs_pen > 0:
                endpoint_penalty += min(obs_pen / 5.0, 1.0)

        # ── Combined terminal reward ──────────────────────────────────────────
        # No completion bonus: it was constant (+10) so it just shifted all
        # returns up without providing any gradient signal, and the policy
        # exploited it by clustering near walls (small penalty but still
        # Spacing reward: mean pairwise distance in mm, no threshold.
        # Directly maximizes distance between all endpoint pairs.
        # Scale: 0.5 * mean_mm so 20mm mean -> +10, 40mm -> +20.
        ep_mean = self._mean_pairwise(endpoints)
        reward_spacing = ep_mean * 0.5

        valid_endpoints = all(
            self._point_clear_of_static(x, y, for_endpoint=True)
            for x, y in endpoints
        )
        spacing_ok = ep_min >= self.spacing_threshold if self.num_traces > 1 else True

        # Completion bonus: mean distance × endpoint cleanliness.
        # No threshold. Larger mean distance + cleaner endpoints = larger bonus.
        # Scale: 1.0 * mean_mm so 20mm mean + clean endpoints -> +20.
        clean_fraction = max(0.0, 1.0 - endpoint_penalty / max(self.num_traces, 1))
        completion_bonus = ep_mean * 1.0 * clean_fraction if completed else 0.0

        # ── Logging metrics ───────────────────────────────────────────────────
        comp_lengths = [
            sum(np.hypot(p[k+1][0]-p[k][0], p[k+1][1]-p[k][1])
                for k in range(len(p)-1))
            for ti, p in enumerate(self.paths) if self.grown[ti] >= self.num_rounds
        ]
        length_spread = (max(comp_lengths) - min(comp_lengths)) if len(comp_lengths) > 1 else 0.0
        total_len = sum(
            sum(np.hypot(p[k+1][0]-p[k][0], p[k+1][1]-p[k][1])
                for k in range(len(p)-1))
            for p in self.paths
        )
        self._terminal_metrics = {
            "min_tp_spacing": ep_min,
            "mean_tp_spacing": ep_mean,
            "endpoints_valid": 1.0 if valid_endpoints else 0.0,
            "spacing_ok": 1.0 if spacing_ok else 0.0,
            "all_complete": 1.0 if completed else 0.0,
            "routable": 1.0 if (completed and valid_endpoints and spacing_ok) else 0.0,
            "total_length": total_len,
            "length_spread": length_spread,
            "reward_spacing": reward_spacing,
            "reward_gate": completion_bonus,
            "endpoint_penalty": endpoint_penalty,
        }
        return reward_spacing + completion_bonus - endpoint_penalty

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
