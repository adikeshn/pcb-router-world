"""Round-robin trace-growth environment (gymnasium API).

Key properties, carried over from the design discussion:

* 8 compass directions, every step extends the active trace by exactly
  ``step_mm`` (diagonals unit-normalised) — equal length by construction.
* Round-robin activation; turn order randomised ONCE PER EPISODE and
  exposed in the observation (robustness without per-step dynamics noise).
* Per-step, freshly recomputed clearance-inflated action masks (no
  caching across a round; the "residual crossings" of v1 are designed out).
* Per-episode sampled growth budget in [budget_min_mm, budget_max_mm],
  visible in the observation, restoring total length as a searchable axis.
* Boxed-in trace  ->  truncation (portfolio-gate failure), no reward cliff.
* Terminal reward strictly dominates cumulative dense reward
  (verify with validate.reward_scale_check before training).
* Brute-force final audit, independent of the spatial hash, feeds the
  portfolio gate.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from .board import Board
from .breakout import breakout_end_directions, build_breakout
from .config import Config
from .geometry import audit_paths

# 8 compass directions: N, NE, E, SE, S, SW, W, NW (matches v1 convention)
_SQ = 1.0 / math.sqrt(2.0)
DIRS = np.asarray([
    (0.0, 1.0), (_SQ, _SQ), (1.0, 0.0), (_SQ, -_SQ),
    (0.0, -1.0), (-_SQ, -_SQ), (-1.0, 0.0), (-_SQ, _SQ),
], dtype=np.float64)
DIR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def nearest_dir_index(v: np.ndarray) -> int:
    return int(np.argmax(DIRS @ (v / (np.linalg.norm(v) + 1e-12))))


class RoundRobinTraceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: Config, seed: Optional[int] = None):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.board = Board.from_config(cfg)
        self.n = self.board.n_traces
        self.checker = self.board.make_checker(cfg)
        if cfg.use_breakout:
            self.breakout = build_breakout(cfg, self.board)
            self._breakout_dirs = breakout_end_directions(self.breakout)
        else:
            self.breakout = None
            self._breakout_dirs = None
        self.rng = np.random.default_rng(seed)

        self.action_space = gym.spaces.Discrete(8)
        self._obs_dim = (2 * self.n) + (2 * self.n) + self.n + 2 + self.n \
            + cfg.n_rays + 8 + (self.n - 1) + 1
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)

        self._ray_dirs = np.stack([
            (math.cos(a), math.sin(a))
            for a in np.linspace(0, 2 * math.pi, cfg.n_rays, endpoint=False)
        ])

        # episode state initialised in reset()
        self.paths: List[List[np.ndarray]] = []
        self._episode_count = 0

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        opts = options or {}

        self.checker.reset()
        self.paths = []
        if self.cfg.use_breakout:
            for tid, poly in enumerate(self.breakout):
                self.paths.append([p.copy() for p in poly])
                self.checker.add_polyline(tid, [tuple(p) for p in poly])
            self._breakout_pts = [len(p) for p in self.breakout]
            self.last_dir_idx = np.asarray(
                [nearest_dir_index(d) for d in self._breakout_dirs],
                dtype=np.int64)
        else:
            for tid in range(self.n):
                pin = self.board.pins[tid].copy()
                self.paths.append([pin])
                # degenerate zero-length segment: the pin is copper other
                # traces must clear from step one, even before it moves
                self.checker.add_segment(tid, (pin[0], pin[1]),
                                         (pin[0], pin[1]))
            self._breakout_pts = [1] * self.n
            # -1 sentinel: no heading yet, so no reverse ban on first step
            self.last_dir_idx = np.full(self.n, -1, dtype=np.int64)

        budget = opts.get("budget_mm")
        if budget is None:
            budget = self.rng.uniform(self.cfg.budget_min_mm, self.cfg.budget_max_mm)
        self.budget_rounds = max(1, int(round(float(budget) / self.cfg.step_mm)))
        self.budget_mm = self.budget_rounds * self.cfg.step_mm

        self.order = self.rng.permutation(self.n)      # fixed for the episode
        self.ptr = 0
        self.rounds_done = 0
        self.redirects = 0
        self.boxed_in = False

        # running stats
        self._path_clearance_samples: List[float] = []
        self._terms = {"spacing_dense": 0.0, "edge_penalty": 0.0,
                       "path_penalty": 0.0, "terminal": 0.0}

        self._mask_cache: Optional[Tuple[int, np.ndarray]] = None
        self._episode_count += 1
        return self._obs(), {}

    # ------------------------------------------------------------------ #
    # Masking
    # ------------------------------------------------------------------ #
    def _active(self) -> int:
        return int(self.order[self.ptr])

    def _compute_mask(self, tid: int) -> np.ndarray:
        tip = self.paths[tid][-1]
        mask = np.zeros(8, dtype=bool)
        step = self.cfg.step_mm
        banned = ((self.last_dir_idx[tid] + 4) % 8
                  if (self.cfg.ban_reverse and self.last_dir_idx[tid] >= 0)
                  else -1)
        for k in range(8):
            if k == banned:
                continue
            b = (tip[0] + DIRS[k, 0] * step, tip[1] + DIRS[k, 1] * step)
            ok, _ = self.checker.segment_valid(tid, (tip[0], tip[1]), b)
            mask[k] = ok
        return mask

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO hook — always freshly recomputed for the
        currently active trace (never cached across steps)."""
        tid = self._active()
        key = (tid, self.rounds_done, self.ptr)
        if self._mask_cache is not None and self._mask_cache[0] == hash(key):
            return self._mask_cache[1]
        mask = self._compute_mask(tid)
        self._mask_cache = (hash(key), mask)
        return mask

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, action: int):
        cfg = self.cfg
        tid = self._active()
        mask = self.action_masks()
        self._mask_cache = None

        if not mask.any():
            # boxed in: nothing this trace can do — episode fails the gate
            self.boxed_in = True
            info = self._finish(complete=False)
            return self._obs(), 0.0, False, True, info

        action = int(action)
        if not mask[action]:
            # Safety net only: MaskablePPO and the ForcedExplorer both sample
            # from the mask, so this fires ~never; redirect to the nearest
            # valid direction by angular distance rather than wasting a step.
            offsets = np.argsort([min((abs(action - k)) % 8, (abs(k - action)) % 8)
                                  for k in range(8)])
            action = int(next(k for k in offsets if mask[k]))
            self.redirects += 1

        tip = self.paths[tid][-1]
        new_tip = tip + DIRS[action] * cfg.step_mm
        ok, d_other = self.checker.segment_valid(tid, (tip[0], tip[1]),
                                                 (new_tip[0], new_tip[1]))
        assert ok, "mask/step disagreement — geometry bug"
        self.checker.add_segment(tid, (tip[0], tip[1]), (new_tip[0], new_tip[1]))
        self.paths[tid].append(new_tip)
        self.last_dir_idx[tid] = action

        reward = 0.0

        # dense path-clearance penalty (only when another trace is nearby)
        if d_other < cfg.path_soft_mm:
            margin = (cfg.path_soft_mm - d_other) / max(cfg.path_soft_mm, 1e-9)
            pen = cfg.w_path_penalty * margin
            reward -= pen
            self._terms["path_penalty"] -= pen
        self._path_clearance_samples.append(min(d_other, cfg.path_soft_mm))

        # dense edge-proximity penalty
        e = self._edge_distance(new_tip)
        if e < cfg.edge_soft_mm:
            pen = cfg.w_edge_penalty * (cfg.edge_soft_mm - e) / cfg.edge_soft_mm
            reward -= pen
            self._terms["edge_penalty"] -= pen

        # advance the round-robin pointer
        self.ptr += 1
        terminated = False
        if self.ptr == self.n:
            self.ptr = 0
            self.rounds_done += 1
            # dense spacing reward once per completed round: hinge on the
            # MINIMUM pairwise tip distance (mean is gameable by outliers)
            min_pair = self._min_pairwise_tip_distance()
            r = cfg.w_spacing_dense * min(min_pair, cfg.spacing_target_mm) \
                / cfg.spacing_target_mm
            reward += r
            self._terms["spacing_dense"] += r
            if self.rounds_done >= self.budget_rounds:
                terminated = True

        info: Dict[str, Any] = {}
        if terminated:
            info = self._finish(complete=True)
            reward += info["episode_data"]["reward_terminal"]

        return self._obs(), float(reward), terminated, False, info

    # ------------------------------------------------------------------ #
    # Episode finish: audit, terminal reward, info payload
    # ------------------------------------------------------------------ #
    def _finish(self, complete: bool) -> Dict[str, Any]:
        cfg = self.cfg
        paths_np = [np.asarray(p) for p in self.paths]
        violations, min_inter = audit_paths(
            paths_np, cfg.trace_clearance_mm, cfg.self_clearance_mm,
            cfg.self_skip_mm)

        endpoints = np.asarray([p[-1] for p in paths_np])
        min_ep = self._min_pairwise(endpoints)
        meets_spec = bool(min_ep >= cfg.endpoint_spec_mm)

        mean_clr = (float(np.mean(self._path_clearance_samples))
                    if self._path_clearance_samples else cfg.path_soft_mm)

        terminal = 0.0
        gate = complete and violations == 0
        if gate:
            q_spacing = min(min_ep, cfg.spacing_target_mm) / cfg.spacing_target_mm
            q_clear = mean_clr / cfg.path_soft_mm
            span = max(cfg.budget_max_mm - cfg.budget_min_mm, 1e-9)
            q_short = 1.0 - (self.budget_mm - cfg.budget_min_mm) / span
            quality = (cfg.q_endpoint_spacing * q_spacing
                       + cfg.q_path_clearance * q_clear
                       + cfg.q_short_budget * q_short)
            terminal = cfg.w_terminal_base + cfg.w_terminal_quality * quality
        self._terms["terminal"] = terminal

        lengths = [float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
                   for p in paths_np]

        episode_data = {
            "paths": [p.tolist() for p in paths_np],
            "breakout_points": list(self._breakout_pts),
            "endpoints": endpoints.tolist(),
            "budget_mm": self.budget_mm,
            "complete": bool(complete),
            "boxed_in": bool(self.boxed_in),
            "violations": int(violations),
            "gate_pass": bool(gate),
            "min_endpoint_spacing_mm": float(min_ep),
            "meets_spec": meets_spec,
            "min_inter_trace_mm": float(min_inter) if math.isfinite(min_inter) else -1.0,
            "mean_path_clearance_mm": mean_clr,
            "trace_lengths_mm": lengths,
            "length_spread_mm": float(max(lengths) - min(lengths)),
            "reward_terminal": float(terminal),
            "reward_terms": dict(self._terms),
            "redirects": int(self.redirects),
            "rounds_done": int(self.rounds_done),
        }
        return {"episode_data": episode_data}

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def _obs(self) -> np.ndarray:
        cfg = self.cfg
        w, h, diag = self.board.width, self.board.height, cfg.board_diag_mm
        tips = np.asarray([p[-1] for p in self.paths])
        tid = self._active()
        tip = tips[tid]

        parts = []
        norm_tips = (tips / np.asarray([w, h])) * 2.0 - 1.0
        parts.append(norm_tips.ravel())

        headings = np.asarray([
            DIRS[d] if d >= 0 else (0.0, 0.0) for d in self.last_dir_idx])
        parts.append(headings.ravel())

        one_hot = np.zeros(self.n)
        one_hot[tid] = 1.0
        parts.append(one_hot)

        remaining = 1.0 - self.rounds_done / max(self.budget_rounds, 1)
        span = max(cfg.budget_max_mm - cfg.budget_min_mm, 1e-9)
        budget_norm = (self.budget_mm - cfg.budget_min_mm) / span
        parts.append(np.asarray([remaining * 2 - 1, budget_norm * 2 - 1]))

        pos_in_order = np.empty(self.n)
        for slot, t in enumerate(self.order):
            pos_in_order[t] = slot / max(self.n - 1, 1)
        parts.append(pos_in_order * 2 - 1)

        rays = np.asarray([
            self.checker.ray_distance(tid, (tip[0], tip[1]),
                                      (d[0], d[1]), cfg.ray_max_mm)
            for d in self._ray_dirs]) / cfg.ray_max_mm
        parts.append(rays * 2 - 1)

        parts.append(self.action_masks().astype(np.float64) * 2 - 1)

        others = np.linalg.norm(tips - tip, axis=1)
        others = np.delete(others, tid) / diag
        parts.append(np.clip(others, 0, 1) * 2 - 1)
        parts.append(np.asarray([
            np.clip(self._min_pairwise_tip_distance() / diag, 0, 1) * 2 - 1]))

        obs = np.concatenate(parts).astype(np.float32)
        assert obs.shape == (self._obs_dim,), (obs.shape, self._obs_dim)
        return np.clip(obs, -1.0, 1.0)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _edge_distance(self, p: np.ndarray) -> float:
        return float(min(p[0], self.board.width - p[0],
                         p[1], self.board.height - p[1]))

    def _min_pairwise_tip_distance(self) -> float:
        tips = np.asarray([p[-1] for p in self.paths])
        return self._min_pairwise(tips)

    @staticmethod
    def _min_pairwise(pts: np.ndarray) -> float:
        n = len(pts)
        best = math.inf
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(pts[i] - pts[j]))
                best = min(best, d)
        return best


def mask_fn(env) -> np.ndarray:
    """ActionMasker hook."""
    return env.unwrapped.action_masks()
