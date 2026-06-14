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
Action: discrete index into candidate grid (fixed size MAX_CANDIDATES).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, List, Tuple

from envs.board import (
    BoardSpec, load_te_example, generate_candidate_grid,
    check_tp_spacing, TP_TO_TP_MIN, TP_TO_EDGE_MIN, MAX_CANDIDATES,
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
        seed: int = 0,
        reward_version: str = "v1",
        board_width: float = 135.0,
        board_height: float = 90.0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.use_freerouting = use_freerouting
        self._num_traces_requested = num_traces
        self._candidate_resolution = candidate_resolution
        self._board_seed = seed
        self.reward_version = reward_version

        if board is None:
            board = load_te_example(num_traces=num_traces, seed=seed,
                                    board_width=board_width,
                                    board_height=board_height)
        self.board = board
        self.num_traces = min(num_traces, len(self.board.traces))
        self.board.traces = self.board.traces[:self.num_traces]

        self.candidates, self._real_count = generate_candidate_grid(
            self.board, candidate_resolution, MAX_CANDIDATES
        )
        self.num_candidates = MAX_CANDIDATES

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
        # Mask out padding entries
        self.candidate_mask[self._real_count:] = False

        # Episode-level diagnostics
        self._episode_invalid_actions = 0

        # Filled after validation
        self.routed_paths = None
        self.routed_lengths = None
        self._terminal_metrics = {}

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
        for i in range(self._real_count):  # only draw real candidates
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
        for i in range(self._real_count):  # only check real candidates
            if self.candidate_mask[i]:
                cx, cy = self.candidates[i]
                if not check_tp_spacing(self.placed_tps, cx, cy):
                    self.candidate_mask[i] = False

    # ---- validation (runs after all TPs placed) ----

    def _validate(self) -> float:
        """Route all traces, compute reward, and record diagnostic metrics."""
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

        n = self.num_traces
        finite = [l for l in lengths if l < float('inf')]
        total_length = sum(finite) if finite else 0.0
        spread = 0.0
        if len(finite) > 1:
            spread = (max(finite) - min(finite)) / max(np.mean(finite), 1e-6)
        min_sp = 0.0
        if len(self.placed_tps) > 1:
            min_sp = min(
                np.hypot(a[0] - b[0], a[1] - b[1])
                for i, a in enumerate(self.placed_tps)
                for b in self.placed_tps[i + 1:]
            )
        diag = np.hypot(self.board.width, self.board.height)

        # Edge-clearance check: how close routed traces come to the board edge,
        # and how many traces violate the minimum edge clearance. Used by v3 to
        # penalize routes that hug / push past the board boundary.
        edge_violations = 0
        edge_min = float('inf')
        try:
            from envs.routing import validate_routing_constraints
            vinfo = validate_routing_constraints(self.board, paths)
            edge_min = vinfo.get("trace_to_edge_min", float('inf'))
            edge_violations = sum(
                1 for v in vinfo.get("violations", []) if v[0] == "trace_to_edge"
            )
        except Exception:
            pass

        if self.reward_version == "v3":
            (reward_routability, reward_length,
             reward_spread, reward_spacing) = self._reward_v3(
                failures, n, finite, total_length, spread, min_sp, diag,
                edge_violations)
        elif self.reward_version == "v2":
            (reward_routability, reward_length,
             reward_spread, reward_spacing) = self._reward_v2(
                failures, n, finite, total_length, spread, min_sp, diag)
        else:
            (reward_routability, reward_length,
             reward_spread, reward_spacing) = self._reward_v1(
                failures, finite, total_length, spread, min_sp, diag)

        self._terminal_metrics = {
            "failures": failures,
            "routable": 1.0 if failures == 0 else 0.0,
            "total_length": total_length,
            "length_spread": spread,
            "min_tp_spacing": min_sp,
            "edge_violations": edge_violations,
            "min_edge_clearance": edge_min if edge_min != float('inf') else 0.0,
            "reward_routability": reward_routability,
            "reward_length": reward_length,
            "reward_spread": reward_spread,
            "reward_spacing": reward_spacing,
        }

        return reward_routability + reward_length + reward_spread + reward_spacing

    def _reward_v1(self, failures, finite, total_length, spread, min_sp, diag):
        """Original reward (unchanged)."""
        reward_routability = 15.0 if failures == 0 else -10.0 * failures
        reward_length = 0.0
        reward_spread = 0.0
        if finite:
            reward_length = -5.0 * sum(finite) / (len(finite) * diag)
            if len(finite) > 1:
                reward_spread = -20.0 * spread
        reward_spacing = 0.0
        if len(self.placed_tps) > 1:
            reward_spacing = 2.0 * min(min_sp / TP_TO_TP_MIN, 2.0)
        return reward_routability, reward_length, reward_spread, reward_spacing

    def _reward_v2(self, failures, n, finite, total_length, spread, min_sp, diag):
        """Revised reward.

        Changes vs v1, all aimed at making the terminal signal smoother and
        better-scaled for the world model's reward head:

        1. Routability is graded by the FRACTION of traces routed (gives the
           world model a "getting closer" gradient) plus a completion bonus
           for full routability, instead of a +15 / -10*failures cliff.
           Range: roughly [0, 10].
        2. Length-spread penalty is CAPPED and down-weighted, so a single
           outlier trace can't dominate the whole reward. Length matching is
           largely handled post-hoc by meandering, so this is a soft steer.
           Range: [-6, 0].
        3. Length penalty is bounded to a comparable scale. Range: [-4, 0].
        4. Spacing bonus unchanged in spirit but slightly rescaled.
           Range: [0, 3].

        Net terminal reward roughly in [-6, ~16], comparable in magnitude to
        the per-step rewards accumulated over an episode (~±5), rather than
        the v1 terminal block that could swing to -200+.
        """
        # 1. Graded routability
        routed = n - failures
        frac_routed = routed / max(n, 1)
        reward_routability = 6.0 * frac_routed
        if failures == 0:
            reward_routability += 4.0  # completion bonus -> max 10

        # 2/3. Length terms only meaningful when something routed
        reward_length = 0.0
        reward_spread = 0.0
        if finite:
            avg_frac = (sum(finite) / len(finite)) / diag
            reward_length = -4.0 * min(avg_frac, 1.0)  # bounded [-4, 0]
            if len(finite) > 1:
                reward_spread = -6.0 * min(spread, 1.0)  # capped [-6, 0]

        # 4. Spacing quality
        reward_spacing = 0.0
        if len(self.placed_tps) > 1:
            reward_spacing = 3.0 * min(min_sp / TP_TO_TP_MIN, 1.0)  # [0, 3]

        return reward_routability, reward_length, reward_spread, reward_spacing

    def _reward_v3(self, failures, n, finite, total_length, spread, min_sp, diag,
                   edge_violations=0):
        """Length-matching-first reward, with the dropped-trace exploit closed.

        IMPORTANT FIX: in the original v3, spread/length/spacing were computed
        over only the *successfully routed* traces. That created a perverse
        incentive to let traces FAIL -- dropping a trace shrinks the finite-
        length set to a trivially length-matched subset, scoring well on the
        (dominant) spread term while dodging routing cost. The policy collapsed
        onto a 2-of-4-failing placement because it out-scored fully-routed ones.

        This version makes full routability STRICTLY dominate, with a
        guaranteed margin: the WORST possible fully-routed reward is held
        above the BEST possible partial reward, so no quality penalty can ever
        make completing the board look worse than failing a trace.

          full route:  +20 base, minus a quality penalty capped at 12 total
                       -> full-route reward in [+8, +20]
          partial:     4*frac - 8*failures  (best case, 1 failure: < 0)
                       -> always far below +8, with zero quality reward

        Within the 12-point full-route quality budget, length SPREAD is the
        dominant differentiator (up to -10), with length and spacing as small
        tiebreakers, preserving the 'length-matching first' intent for ranking
        among complete solutions.
        """
        if failures == 0:
            base = 20.0
            # Quality penalties (deducted from base); spread dominant.
            pen_spread = 0.0
            pen_length = 0.0
            if finite:
                avg_frac = (sum(finite) / len(finite)) / diag
                pen_length = 1.0 * min(avg_frac, 1.0)        # up to -1
                if len(finite) > 1:
                    pen_spread = 10.0 * min(spread, 1.0)     # up to -10 (dominant)
            bonus_spacing = 0.0
            if len(self.placed_tps) > 1:
                bonus_spacing = 1.0 * min(min_sp / TP_TO_TP_MIN, 1.0)  # up to +1

            # Edge-clearance penalty: discourage routes that hug or cross the
            # board boundary. Folded into the length term's reported slot.
            # Capped at -3 so the total quality penalty (spread<=10, length<=1,
            # edge<=3 -> <=14) still leaves full-route reward >= 20-14 = +6,
            # which stays above the best partial (1 failure ~ -4.67).
            pen_edge = 1.5 * min(edge_violations, 2)         # 0, -1.5, or -3

            reward_routability = base
            reward_length = -(pen_length + pen_edge)
            reward_spread = -pen_spread
            reward_spacing = bonus_spacing
            return reward_routability, reward_length, reward_spread, reward_spacing

        # Partial routing: graded "getting closer" signal only, no quality
        # reward. Best possible partial (1 failure) is 4*((n-1)/n) - 8 < 0,
        # which is far below the +8 worst-case full route -- so completing the
        # board always wins regardless of how poorly length-matched it is.
        routed = n - failures
        reward_routability = 4.0 * (routed / max(n, 1)) - 8.0 * failures
        return reward_routability, 0.0, 0.0, 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.placed_tps = []
        self.current_trace = 0
        self.candidate_mask = np.ones(self.num_candidates, dtype=bool)
        self.candidate_mask[self._real_count:] = False  # mask padding
        self._episode_invalid_actions = 0
        self.routed_paths = None
        self.routed_lengths = None
        self._terminal_metrics = {}
        return self._render_obs(), self._get_info()

    def step(self, action: int):
        tp_x, tp_y = self.candidates[action]
        reward = 0.0

        # Per-step reward
        invalid_this_step = False
        if not self.candidate_mask[action]:
            reward -= 2.0
            invalid_this_step = True
        elif check_tp_spacing(self.placed_tps, tp_x, tp_y):
            reward += 1.0
        else:
            reward -= 2.0
            invalid_this_step = True

        if invalid_this_step:
            self._episode_invalid_actions += 1

        self.placed_tps.append((tp_x, tp_y))
        self.current_trace += 1
        self._update_candidate_mask()

        # Preserve future options (count only real candidates).
        # Only awarded on valid placements -- previously this was added
        # unconditionally, partially offsetting the -2 penalty for an
        # invalid placement.
        if not invalid_this_step:
            valid_frac = self.candidate_mask[:self._real_count].sum() / max(self._real_count, 1)
            reward += 0.3 * valid_frac

        terminated = self.current_trace >= self.num_traces

        if terminated:
            reward += self._validate()

        return self._render_obs(), reward, terminated, False, self._get_info(invalid_this_step)

    def _get_info(self, invalid_this_step: bool = False):
        info = {
            "current_trace": self.current_trace,
            "traces_placed": len(self.placed_tps),
            "invalid_this_step": invalid_this_step,
            "episode_invalid_actions": self._episode_invalid_actions,
        }
        if self.routed_lengths is not None:
            info["trace_lengths"] = self.routed_lengths
        info.update(self._terminal_metrics)
        return info

    def render(self):
        return self._render_obs()