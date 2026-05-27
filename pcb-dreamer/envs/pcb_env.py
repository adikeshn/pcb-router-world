"""
PCB Test Point Placement Environment.

The agent places test points one at a time, one per trace.
Trace i gets TP i — placement order is the assignment.
After all TPs are placed, a validation loop routes all traces
and computes the reward.

Two routing modes:
  - A* (default): fast cell-based router for training (~100ms)
  - FreeRouting: industry autorouter for evaluation (~3s)

Observation: 64x64x3 uint8 image.
  Red:   obstacles + clearance zones + board edge
  Green: placed test points + exclusion zones + routed traces
  Blue:  current trace starting point + valid remaining candidates
Action: discrete index into candidate grid.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, List, Tuple

from envs.board import (
    BoardSpec, load_te_example, generate_candidate_grid,
    check_tp_spacing, TP_TO_TP_MIN, TP_TO_EDGE_MIN,
)
from envs.routing import route_all_traces as route_astar

IMG_SIZE = 64


class TPPlacementEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        board: Optional[BoardSpec] = None,
        num_traces: int = 10,
        candidate_resolution: float = 6.5,
        use_freerouting: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.use_freerouting = use_freerouting

        if board is None:
            board = load_te_example(num_traces=num_traces)
        self.board = board
        self.num_traces = min(num_traces, len(self.board.traces))
        self.board.traces = self.board.traces[:self.num_traces]

        self.candidates = generate_candidate_grid(self.board, candidate_resolution)
        self.num_candidates = len(self.candidates)
        assert self.num_candidates > 0, "No valid candidates!"

        self.action_space = spaces.Discrete(self.num_candidates)
        self.observation_space = spaces.Box(
            0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8
        )

        # Coordinate transform
        self._x_scale = (IMG_SIZE - 1) / max(self.board.width, 1e-6)
        self._y_scale = (IMG_SIZE - 1) / max(self.board.height, 1e-6)

        self.placed_tps: List[Tuple[float, float]] = []
        self.current_trace: int = 0
        self.candidate_mask = np.ones(self.num_candidates, dtype=bool)

        # Filled after validation
        self.routed_paths = None
        self.routed_lengths = None

    # ---- coordinate / drawing helpers ----

    def _w2p(self, x: float, y: float) -> Tuple[int, int]:
        px = int((x - self.board.x_min) * self._x_scale)
        py = int((y - self.board.y_min) * self._y_scale)
        return np.clip(px, 0, IMG_SIZE - 1), np.clip(py, 0, IMG_SIZE - 1)

    def _draw_circle(self, img, cx, cy, r_mm, ch, val=255):
        pcx, pcy = self._w2p(cx, cy)
        pr = max(1, int(r_mm * self._x_scale))
        for dy in range(-pr, pr + 1):
            for dx in range(-pr, pr + 1):
                if dx * dx + dy * dy <= pr * pr:
                    py, px = pcy + dy, pcx + dx
                    if 0 <= py < IMG_SIZE and 0 <= px < IMG_SIZE:
                        img[py, px, ch] = min(255, int(img[py, px, ch]) + val)

    def _draw_rect(self, img, xmin, ymin, xmax, ymax, ch, val=255):
        px0, py0 = self._w2p(xmin, ymin)
        px1, py1 = self._w2p(xmax, ymax)
        py0, py1 = max(0, min(py0, py1)), min(IMG_SIZE, max(py0, py1) + 1)
        px0, px1 = max(0, min(px0, px1)), min(IMG_SIZE, max(px0, px1) + 1)
        img[py0:py1, px0:px1, ch] = np.minimum(
            255, img[py0:py1, px0:px1, ch].astype(np.int16) + val
        ).astype(np.uint8)

    # ---- observation ----

    def _render_obs(self) -> np.ndarray:
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        # RED: obstacles + edge clearance + connector
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

        # GREEN: placed TPs + exclusion zones
        for tx, ty in self.placed_tps:
            self._draw_circle(img, tx, ty, TP_TO_TP_MIN / 2, 1, 60)
            self._draw_circle(img, tx, ty, 1.5, 1, 255)

        # BLUE: current trace start + valid candidates
        if self.current_trace < self.num_traces:
            t = self.board.traces[self.current_trace]
            self._draw_circle(img, t.start_x, t.start_y, 3.0, 2, 255)
        for i in range(self.num_candidates):
            if self.candidate_mask[i]:
                cx, cy = self.candidates[i]
                px, py = self._w2p(cx, cy)
                if 0 <= py < IMG_SIZE and 0 <= px < IMG_SIZE:
                    img[py, px, 2] = 150

        # Dim starting points (all traces) in red
        for t in self.board.traces:
            px, py = self._w2p(t.start_x, t.start_y)
            if 0 <= py < IMG_SIZE and 0 <= px < IMG_SIZE:
                img[py, px, 0] = min(255, int(img[py, px, 0]) + 80)

        return img

    # ---- candidate mask ----

    def _update_candidate_mask(self):
        for i in range(self.num_candidates):
            if self.candidate_mask[i]:
                cx, cy = self.candidates[i]
                if not check_tp_spacing(self.placed_tps, cx, cy):
                    self.candidate_mask[i] = False

    # ---- validation (runs after all TPs placed) ----

    def _validate(self) -> float:
        """Route all traces and compute reward."""
        if self.use_freerouting:
            try:
                from envs.freerouting import route_with_freerouting
                paths, lengths, failures = route_with_freerouting(
                    self.board, self.placed_tps
                )
            except FileNotFoundError:
                import warnings
                warnings.warn(
                    "FreeRouting not found, falling back to A*. "
                    "Set FREEROUTING_JAR env var or place freerouting.jar "
                    "in project root.",
                    stacklevel=2,
                )
                self.use_freerouting = False
                paths, lengths, failures = route_astar(
                    self.board, self.placed_tps
                )
        else:
            paths, lengths, failures = route_astar(
                self.board, self.placed_tps
            )

        self.routed_paths = paths
        self.routed_lengths = lengths

        reward = 0.0

        # Routability
        if failures == 0:
            reward += 15.0
        else:
            reward -= 10.0 * failures

        # Trace lengths
        finite = [l for l in lengths if l < float('inf')]
        if finite:
            diag = np.hypot(self.board.width, self.board.height)
            # Minimize total length
            reward -= 3.0 * sum(finite) / (len(finite) * diag)
            # Length matching
            if len(finite) > 1:
                spread = (max(finite) - min(finite)) / max(np.mean(finite), 1e-6)
                reward -= 5.0 * spread

        # TP spacing quality
        if len(self.placed_tps) > 1:
            min_sp = min(
                np.hypot(a[0] - b[0], a[1] - b[1])
                for i, a in enumerate(self.placed_tps)
                for b in self.placed_tps[i + 1:]
            )
            reward += 2.0 * min(min_sp / TP_TO_TP_MIN, 2.0)

        return reward

    # ---- gym interface ----

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.placed_tps = []
        self.current_trace = 0
        self.candidate_mask = np.ones(self.num_candidates, dtype=bool)
        self.routed_paths = None
        self.routed_lengths = None
        return self._render_obs(), self._get_info()

    def step(self, action: int):
        tp_x, tp_y = self.candidates[action]
        reward = 0.0

        # Per-step reward
        if not self.candidate_mask[action]:
            reward -= 2.0
        elif check_tp_spacing(self.placed_tps, tp_x, tp_y):
            reward += 1.0
        else:
            reward -= 2.0

        self.placed_tps.append((tp_x, tp_y))
        self.current_trace += 1
        self._update_candidate_mask()

        # Preserve future options
        valid_frac = self.candidate_mask.sum() / max(self.num_candidates, 1)
        reward += 0.3 * valid_frac

        terminated = self.current_trace >= self.num_traces

        if terminated:
            reward += self._validate()

        return self._render_obs(), reward, terminated, False, self._get_info()

    def _get_info(self):
        info = {
            "current_trace": self.current_trace,
            "traces_placed": len(self.placed_tps),
        }
        if self.routed_lengths is not None:
            info["trace_lengths"] = self.routed_lengths
            info["failures"] = sum(
                1 for l in self.routed_lengths if l == float('inf')
            )
        return info

    def render(self):
        return self._render_obs()