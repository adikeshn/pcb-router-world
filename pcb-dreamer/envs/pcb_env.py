"""
PCB Test Point Placement Environment with image observations.

Observation: 64x64x3 uint8 image of the board state.
  Red: obstacles + clearance zones
  Green: placed test points + exclusion zones + routed traces
  Blue: current trace start + valid candidate positions
Action: discrete index into candidate grid
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple, List

from envs.board import (
    BoardSpec, load_te_example, generate_toy_board, generate_candidate_grid,
    check_tp_spacing, TP_TO_TP_MIN,
)
from envs.routing import route_all_traces

IMG_SIZE = 64


class TPPlacementEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
            self,
            board: Optional[BoardSpec] = None,
            num_traces: int = 8,
            candidate_resolution: float = 6.5,
            routing_resolution: float = 0.5,
            render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.routing_resolution = routing_resolution

        # Use toy board by default (properly spaced starts)
        if board is None:
            board = generate_toy_board(num_traces=num_traces)
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

        # Coordinate transform: world → pixel
        self._x_min = self.board.x_min
        self._y_min = self.board.y_min
        self._x_scale = (IMG_SIZE - 1) / max(self.board.width, 1e-6)
        self._y_scale = (IMG_SIZE - 1) / max(self.board.height, 1e-6)

        self.placed_tps: List[Tuple[float, float]] = []
        self.placed_lengths: List[float] = []
        self.current_trace_idx = 0
        self.candidate_mask = np.ones(self.num_candidates, dtype=bool)

    def _w2p(self, x: float, y: float) -> Tuple[int, int]:
        """World coords to pixel coords."""
        px = int((x - self._x_min) * self._x_scale)
        py = int((y - self._y_min) * self._y_scale)
        return (np.clip(px, 0, IMG_SIZE - 1), np.clip(py, 0, IMG_SIZE - 1))

    def _draw_circle(self, img: np.ndarray, cx: float, cy: float,
                     radius_mm: float, channel: int, value: int = 255):
        """Draw filled circle on image channel."""
        pcx, pcy = self._w2p(cx, cy)
        pr = max(1, int(radius_mm * self._x_scale))
        for dy in range(-pr, pr + 1):
            for dx in range(-pr, pr + 1):
                if dx * dx + dy * dy <= pr * pr:
                    py, px = pcy + dy, pcx + dx
                    if 0 <= py < IMG_SIZE and 0 <= px < IMG_SIZE:
                        img[py, px, channel] = min(255, int(img[py, px, channel]) + value)

    def _draw_rect(self, img: np.ndarray, xmin: float, ymin: float,
                   xmax: float, ymax: float, channel: int, value: int = 255):
        """Draw filled rectangle on image channel."""
        px0, py0 = self._w2p(xmin, ymin)
        px1, py1 = self._w2p(xmax, ymax)
        py0, py1 = min(py0, py1), max(py0, py1)
        px0, px1 = min(px0, px1), max(px0, px1)
        py0, py1 = max(0, py0), min(IMG_SIZE, py1 + 1)
        px0, px1 = max(0, px0), min(IMG_SIZE, px1 + 1)
        img[py0:py1, px0:px1, channel] = np.minimum(
            255, img[py0:py1, px0:px1, channel] + value
        )

    def _render_obs(self) -> np.ndarray:
        """Render 64x64x3 RGB observation."""
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        # === RED: obstacles + clearance ===
        for obs in self.board.rect_obstacles:
            xmin, ymin, xmax, ymax = obs.bounds
            buf = obs.clearance
            self._draw_rect(img, xmin - buf, ymin - buf, xmax + buf, ymax + buf, 0, 150)
            self._draw_rect(img, xmin, ymin, xmax, ymax, 0, 255)

        for obs in self.board.circ_obstacles:
            self._draw_circle(img, obs.cx, obs.cy, obs.radius + obs.clearance, 0, 150)
            self._draw_circle(img, obs.cx, obs.cy, obs.radius, 0, 255)

        # Board edge clearance (red border)
        from envs.board import TP_TO_EDGE_MIN
        edge_px = max(1, int(TP_TO_EDGE_MIN * self._x_scale))
        img[:edge_px, :, 0] = 100
        img[-edge_px:, :, 0] = 100
        img[:, :edge_px, 0] = 100
        img[:, -edge_px:, 0] = 100

        # Connector outline
        if self.board.connector_w > 0:
            self._draw_rect(img, self.board.connector_x, self.board.connector_y,
                            self.board.connector_x + self.board.connector_w,
                            self.board.connector_y + self.board.connector_h, 0, 180)

        # === GREEN: placed test points + exclusion zones ===
        for i, (tx, ty) in enumerate(self.placed_tps):
            self._draw_circle(img, tx, ty, TP_TO_TP_MIN / 2, 1, 60)  # exclusion zone
            self._draw_circle(img, tx, ty, 1.5, 1, 255)  # test point dot

        # Placed trace paths (estimated as straight lines for speed)
        for i, (tx, ty) in enumerate(self.placed_tps):
            if i < len(self.board.traces):
                t = self.board.traces[i]
                self._draw_line(img, t.start_x, t.start_y, tx, ty, 1, 120)

        # === BLUE: current trace + valid candidates ===
        if self.current_trace_idx < self.num_traces:
            t = self.board.traces[self.current_trace_idx]
            self._draw_circle(img, t.start_x, t.start_y, 2.0, 2, 255)  # start point

            # Valid candidates
            for i in range(self.num_candidates):
                if self.candidate_mask[i]:
                    cx, cy = self.candidates[i]
                    px, py = self._w2p(cx, cy)
                    if 0 <= py < IMG_SIZE and 0 <= px < IMG_SIZE:
                        img[py, px, 2] = 150

        return img

    def _draw_line(self, img: np.ndarray, x1: float, y1: float,
                   x2: float, y2: float, channel: int, value: int):
        """Draw line using Bresenham."""
        px1, py1 = self._w2p(x1, y1)
        px2, py2 = self._w2p(x2, y2)
        steps = max(abs(px2 - px1), abs(py2 - py1), 1)
        for t in range(steps + 1):
            frac = t / steps
            px = int(px1 + frac * (px2 - px1))
            py = int(py1 + frac * (py2 - py1))
            if 0 <= py < IMG_SIZE and 0 <= px < IMG_SIZE:
                img[py, px, channel] = min(255, int(img[py, px, channel]) + value)

    def _update_candidate_mask(self):
        for i in range(self.num_candidates):
            if not self.candidate_mask[i]:
                continue
            cx, cy = self.candidates[i]
            if not check_tp_spacing(self.placed_tps, cx, cy):
                self.candidate_mask[i] = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.placed_tps = []
        self.placed_lengths = []
        self.current_trace_idx = 0
        self.candidate_mask = np.ones(self.num_candidates, dtype=bool)
        return self._render_obs(), self._get_info()

    def step(self, action: int):
        tp_x, tp_y = self.candidates[action]
        reward = 0.0

        if not self.candidate_mask[action]:
            reward -= 2.0
        elif check_tp_spacing(self.placed_tps, tp_x, tp_y):
            reward += 1.0
        else:
            reward -= 2.0

        self.placed_tps.append((tp_x, tp_y))

        trace = self.board.traces[self.current_trace_idx]
        est_len = np.sqrt((tp_x - trace.start_x) ** 2 + (tp_y - trace.start_y) ** 2) + trace.breakout_length
        self.placed_lengths.append(est_len)

        self._update_candidate_mask()
        valid_frac = self.candidate_mask.sum() / max(self.num_candidates, 1)
        reward += 0.3 * valid_frac

        self.current_trace_idx += 1
        terminated = self.current_trace_idx >= self.num_traces

        if terminated:
            reward += self._episode_reward()

        return self._render_obs(), reward, terminated, False, self._get_info()

    def _episode_reward(self) -> float:
        paths, lengths, failures = route_all_traces(
            self.board, self.placed_tps, self.routing_resolution
        )
        self.placed_lengths = lengths
        reward = 0.0

        if failures > 0:
            reward -= 10.0 * failures
        else:
            reward += 10.0

        finite = [l for l in lengths if l < float('inf')]
        if finite:
            diag = np.sqrt(self.board.width ** 2 + self.board.height ** 2)
            reward -= 5.0 * sum(finite) / (len(finite) * diag)

        if len(self.placed_tps) > 1:
            min_sp = min(
                np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
                for i, a in enumerate(self.placed_tps)
                for b in self.placed_tps[i + 1:]
            )
            reward += 2.0 * (min_sp / TP_TO_TP_MIN)

        if len(finite) > 1:
            spread = (max(finite) - min(finite)) / max(np.mean(finite), 1e-6)
            reward -= 3.0 * spread

        return reward

    def _get_info(self):
        info = {"current_trace": self.current_trace_idx, "traces_placed": len(self.placed_tps)}
        if self.current_trace_idx >= self.num_traces:
            info["trace_lengths"] = self.placed_lengths
        return info

    def render(self):
        return self._render_obs()