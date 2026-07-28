"""Round-robin trace-growth environment (gymnasium API).

See environment_spec.md for the full mechanism description.
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


def make_dirs(n: int) -> np.ndarray:
    """n compass headings, index 0 = North, increasing clockwise."""
    return np.asarray([(math.sin(2 * math.pi * k / n), math.cos(2 * math.pi * k / n))
                       for k in range(n)], dtype=np.float64)


def dir_names(n: int) -> List[str]:
    base = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    if n == 16:
        return base
    if n == 8:
        return base[::2]
    return [f"d{k}" for k in range(n)]


def turn_units(a: int, b: int, n: int) -> int:
    """Turn size between two direction indices, in units of 360/n degrees."""
    d = abs(a - b) % n
    return min(d, n - d)


class RoundRobinTraceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: Config, seed: Optional[int] = None):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.board = Board.from_config(cfg)
        self.n = self.board.n_traces
        self.nd = cfg.n_dirs
        self.DIRS = make_dirs(self.nd)
        self.DIR_NAMES = dir_names(self.nd)
        self.checker = self.board.make_checker(cfg)
        if cfg.use_breakout:
            self.breakout = build_breakout(cfg, self.board)
            self._breakout_dirs = breakout_end_directions(self.breakout)
        else:
            self.breakout = None
            self._breakout_dirs = None
        self.rng = np.random.default_rng(seed)

        self.action_space = gym.spaces.Discrete(self.nd)
        # tips(2n) headings(2n) active(n) remaining(1) order(n)
        # rays(n_rays) mask(nd) other-tip-dists(n-1) min-pair(1) freedom(n)
        self._obs_dim = (2 * self.n) + (2 * self.n) + self.n + 1 + self.n \
            + cfg.n_rays + self.nd + (self.n - 1) + 1 + self.n
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (self._obs_dim,), np.float32)
        self._ray_dirs = np.stack([(math.cos(a), math.sin(a)) for a in
                                   np.linspace(0, 2 * math.pi, cfg.n_rays, endpoint=False)])
        self.paths: List[List[np.ndarray]] = []

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg = self.cfg
        opts = options or {}

        self.checker.reset()
        self.paths = []
        if cfg.use_breakout:
            for tid, poly in enumerate(self.breakout):
                self.paths.append([p.copy() for p in poly])
                self.checker.add_polyline(tid, [tuple(p) for p in poly])
            self._breakout_pts = [len(p) for p in self.breakout]
            self.last_dir_idx = np.asarray(
                [int(np.argmax(self.DIRS @ d)) for d in self._breakout_dirs],
                dtype=np.int64)
        else:
            for tid in range(self.n):
                pin = self.board.pins[tid].copy()
                self.paths.append([pin])
                # zero-length segment: the pin is copper others must clear
                self.checker.add_segment(tid, (pin[0], pin[1]), (pin[0], pin[1]))
            self._breakout_pts = [1] * self.n
            self.last_dir_idx = np.full(self.n, -1, dtype=np.int64)

        # single fixed budget (override only for eval / inference)
        budget = float(opts.get("budget_mm", cfg.budget_mm))
        self.budget_rounds = max(1, int(round(budget / cfg.step_mm)))
        self.budget_mm = self.budget_rounds * cfg.step_mm
        self.total_steps = self.budget_rounds * self.n

        # length-invariant dense weights (totals divided by event counts)
        self.w_spacing = cfg.spacing_dense_total / self.budget_rounds
        self.w_constrict = cfg.constriction_penalty_total / self.budget_rounds
        self.w_path = cfg.path_penalty_total / self.total_steps
        self.w_self = cfg.self_penalty_total / self.total_steps
        self.w_turn = cfg.turn_penalty_total / self.total_steps
        self.w_edge = cfg.edge_penalty_total / self.total_steps

        self.order = self.rng.permutation(self.n)
        self.ptr = 0
        self.rounds_done = 0
        self.redirects = 0
        self.boxed_in = False
        self.boxed_trace = -1
        self.turn_units_total = 0

        self._clearance_samples: List[float] = []
        self._self_samples: List[float] = []
        self._freedom_hist: List[float] = []
        self._min_freedom_seen = self.nd
        self._terms = {"spacing_dense": 0.0, "constriction": 0.0,
                       "path_penalty": 0.0, "self_penalty": 0.0,
                       "turn_penalty": 0.0, "edge_penalty": 0.0, "terminal": 0.0}
        # cached per-trace masks for the freedom observation (<=n-1 steps stale)
        self._cached_masks = np.ones((self.n, self.nd), dtype=bool)
        for tid in range(self.n):
            self._cached_masks[tid] = self._compute_mask(tid)
        self._mask_cache: Optional[Tuple[int, np.ndarray]] = None
        return self._obs(), {}

    # ------------------------------------------------------------------ #
    def _active(self) -> int:
        return int(self.order[self.ptr])

    def _compute_mask(self, tid: int) -> np.ndarray:
        tip = self.paths[tid][-1]
        mask = np.zeros(self.nd, dtype=bool)
        step = self.cfg.step_mm
        banned = ((self.last_dir_idx[tid] + self.nd // 2) % self.nd
                  if (self.cfg.ban_reverse and self.last_dir_idx[tid] >= 0) else -1)
        for k in range(self.nd):
            if k == banned:
                continue
            b = (tip[0] + self.DIRS[k, 0] * step, tip[1] + self.DIRS[k, 1] * step)
            ok, _, _ = self.checker.segment_valid(tid, (tip[0], tip[1]), b)
            mask[k] = ok
        return mask

    def action_masks(self) -> np.ndarray:
        """Fresh every step for the active trace; never cached across steps."""
        tid = self._active()
        key = hash((tid, self.rounds_done, self.ptr))
        if self._mask_cache is not None and self._mask_cache[0] == key:
            return self._mask_cache[1]
        mask = self._compute_mask(tid)
        self._cached_masks[tid] = mask
        self._mask_cache = (key, mask)
        return mask

    # ------------------------------------------------------------------ #
    def step(self, action: int):
        cfg = self.cfg
        tid = self._active()
        mask = self.action_masks()
        self._mask_cache = None

        if not mask.any():
            self.boxed_in = True
            self.boxed_trace = tid
            info = self._finish(complete=False)
            return self._obs(), 0.0, False, True, info

        action = int(action)
        if not mask[action]:
            order = np.argsort([turn_units(action, k, self.nd) for k in range(self.nd)])
            action = int(next(k for k in order if mask[k]))
            self.redirects += 1

        tip = self.paths[tid][-1]
        new_tip = tip + self.DIRS[action] * cfg.step_mm
        ok, d_other, d_self_far = self.checker.segment_valid(
            tid, (tip[0], tip[1]), (new_tip[0], new_tip[1]))
        assert ok, "mask/step disagreement — geometry bug"
        self.checker.add_segment(tid, (tip[0], tip[1]), (new_tip[0], new_tip[1]))
        self.paths[tid].append(new_tip)

        prev_dir = self.last_dir_idx[tid]
        self.last_dir_idx[tid] = action
        reward = 0.0

        # --- other-trace crowding --------------------------------------- #
        if math.isfinite(d_other) and d_other < cfg.path_soft_mm:
            pen = self.w_path * (cfg.path_soft_mm - d_other) / cfg.path_soft_mm
            reward -= pen; self._terms["path_penalty"] -= pen
        self._clearance_samples.append(
            min(d_other, cfg.terminal_clearance_target_mm)
            if math.isfinite(d_other) else cfg.terminal_clearance_target_mm)

        # --- SELF crowding: only against path far behind (see geometry) -- #
        if math.isfinite(d_self_far) and d_self_far < cfg.self_soft_mm:
            pen = self.w_self * (cfg.self_soft_mm - d_self_far) / cfg.self_soft_mm
            reward -= pen; self._terms["self_penalty"] -= pen
        self._self_samples.append(
            min(d_self_far, cfg.self_soft_mm * 2)
            if math.isfinite(d_self_far) else cfg.self_soft_mm * 2)

        # --- turning, with a FREE BAND for quantisation ------------------ #
        if prev_dir >= 0:
            t = turn_units(int(prev_dir), action, self.nd)
            self.turn_units_total += t
            charged = max(0, t - cfg.turn_free_units)
            if charged > 0:
                denom = max(1, self.nd // 2 - cfg.turn_free_units)
                pen = self.w_turn * (charged / denom)
                reward -= pen; self._terms["turn_penalty"] -= pen

        # --- edge proximity (weight 0 by default) ----------------------- #
        if self.w_edge > 0:
            e = self._edge_distance(new_tip)
            if e < cfg.edge_soft_mm:
                pen = self.w_edge * (cfg.edge_soft_mm - e) / cfg.edge_soft_mm
                reward -= pen; self._terms["edge_penalty"] -= pen

        # --- advance round-robin ---------------------------------------- #
        self.ptr += 1
        terminated = False
        if self.ptr == self.n:
            self.ptr = 0
            self.rounds_done += 1

            # spacing: hinge on the MINIMUM pairwise tip distance
            min_pair = self._min_pairwise_tip_distance()
            r = self.w_spacing * min(min_pair, cfg.dense_spacing_target_mm) \
                / cfg.dense_spacing_target_mm
            reward += r; self._terms["spacing_dense"] += r

            # constriction: recompute ALL masks fresh so the signal is
            # accurate, and charge at the end of the round in which the
            # constriction appeared -- within n steps of the culprit move
            # rather than hundreds.
            for t2 in range(self.n):
                self._cached_masks[t2] = self._compute_mask(t2)
            freedom = self._cached_masks.sum(axis=1)
            f_min = int(freedom.min())
            self._freedom_hist.append(float(f_min))
            self._min_freedom_seen = min(self._min_freedom_seen, f_min)
            if f_min < cfg.constriction_comfort:
                pen = self.w_constrict * (cfg.constriction_comfort - f_min) \
                    / cfg.constriction_comfort
                reward -= pen; self._terms["constriction"] -= pen

            if self.rounds_done >= self.budget_rounds:
                terminated = True

        info: Dict[str, Any] = {}
        if terminated:
            info = self._finish(complete=True)
            reward += info["episode_data"]["reward_terminal"]
        return self._obs(), float(reward), terminated, False, info

    # ------------------------------------------------------------------ #
    def _finish(self, complete: bool) -> Dict[str, Any]:
        cfg = self.cfg
        paths_np = [np.asarray(p) for p in self.paths]
        violations, min_inter, min_self = audit_paths(
            paths_np, cfg.trace_clearance_mm, cfg.self_clearance_mm)

        endpoints = np.asarray([p[-1] for p in paths_np])
        min_ep = self._min_pairwise(endpoints)
        meets_spec = bool(min_ep >= cfg.endpoint_spec_mm)
        min_ep_edge = float(min(self._edge_distance(p) for p in endpoints))

        mean_clr = float(np.mean(self._clearance_samples)) if self._clearance_samples \
            else cfg.terminal_clearance_target_mm
        mean_self = float(np.mean(self._self_samples)) if self._self_samples else -1.0

        terminal = 0.0
        gate = complete and violations == 0
        r_spacing = q_clear = q_edge = 0.0
        if gate:
            r_spacing = cfg.spacing_reward_coeff * math.sqrt(max(min_ep, 0.0))
            q_clear = min(mean_clr, cfg.terminal_clearance_target_mm) / cfg.terminal_clearance_target_mm
            q_edge = min(min_ep_edge, cfg.terminal_edge_target_mm) / cfg.terminal_edge_target_mm
            terminal = (cfg.w_terminal_base + r_spacing
                        + cfg.w_path_clearance_bonus * q_clear
                        + cfg.w_endpoint_edge_bonus * q_edge)
        self._terms["terminal"] = terminal

        lengths = [float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
                   for p in paths_np]
        n_segments = max(sum(len(p) - 1 for p in paths_np), 1)

        return {"episode_data": {
            "paths": [p.tolist() for p in paths_np],
            "breakout_points": list(self._breakout_pts),
            "endpoints": endpoints.tolist(),
            "budget_mm": self.budget_mm,
            "complete": bool(complete),
            "boxed_in": bool(self.boxed_in),
            "boxed_trace": int(self.boxed_trace),
            "rounds_done": int(self.rounds_done),
            "frac_survived": self.rounds_done / max(self.budget_rounds, 1),
            "violations": int(violations),
            "gate_pass": bool(gate),
            "min_endpoint_spacing_mm": float(min_ep),
            "meets_spec": meets_spec,
            "min_endpoint_edge_mm": min_ep_edge,
            "min_inter_trace_mm": float(min_inter) if math.isfinite(min_inter) else -1.0,
            "min_self_distance_mm": float(min_self) if math.isfinite(min_self) else -1.0,
            "mean_path_clearance_mm": mean_clr,
            "mean_self_far_mm": mean_self,
            "min_freedom": int(self._min_freedom_seen),
            "mean_freedom": float(np.mean(self._freedom_hist)) if self._freedom_hist else float(self.nd),
            "turn_rate": self.turn_units_total / n_segments,
            "trace_lengths_mm": lengths,
            "length_spread_mm": float(max(lengths) - min(lengths)),
            "reward_terminal": float(terminal),
            "r_spacing": float(r_spacing), "q_clear": float(q_clear),
            "q_edge": float(q_edge),
            "reward_terms": dict(self._terms),
            "redirects": int(self.redirects),
        }}

    # ------------------------------------------------------------------ #
    def _obs(self) -> np.ndarray:
        cfg = self.cfg
        w, h, diag = self.board.width, self.board.height, cfg.board_diag_mm
        tips = np.asarray([p[-1] for p in self.paths])
        tid = self._active(); tip = tips[tid]
        parts = [((tips / np.asarray([w, h])) * 2.0 - 1.0).ravel()]
        parts.append(np.asarray([self.DIRS[d] if d >= 0 else (0.0, 0.0)
                                 for d in self.last_dir_idx]).ravel())
        oh = np.zeros(self.n); oh[tid] = 1.0
        parts.append(oh)
        remaining = 1.0 - self.rounds_done / max(self.budget_rounds, 1)
        parts.append(np.asarray([remaining * 2 - 1]))
        pos = np.empty(self.n)
        for slot, t in enumerate(self.order):
            pos[t] = slot / max(self.n - 1, 1)
        parts.append(pos * 2 - 1)
        rays = np.asarray([self.checker.ray_distance(tid, (tip[0], tip[1]),
                                                     (d[0], d[1]), cfg.ray_max_mm)
                           for d in self._ray_dirs]) / cfg.ray_max_mm
        parts.append(rays * 2 - 1)
        parts.append(self.action_masks().astype(np.float64) * 2 - 1)
        others = np.delete(np.linalg.norm(tips - tip, axis=1), tid) / diag
        if cfg.sort_tip_distances:
            # nearest other trace always occupies the same slot, so the
            # representation does not depend on arbitrary trace numbering
            others = np.sort(others)
        parts.append(np.clip(others, 0, 1) * 2 - 1)
        parts.append(np.asarray([np.clip(self._min_pairwise_tip_distance() / diag, 0, 1) * 2 - 1]))
        # per-trace freedom: how many legal directions each trace has left.
        # Without this the policy cannot see that it is walling another
        # trace in -- positions alone do not convey it.
        freedom = self._cached_masks.sum(axis=1) / self.nd
        parts.append(freedom * 2 - 1)
        obs = np.concatenate(parts).astype(np.float32)
        assert obs.shape == (self._obs_dim,), (obs.shape, self._obs_dim)
        return np.clip(obs, -1.0, 1.0)

    # ------------------------------------------------------------------ #
    def _edge_distance(self, p) -> float:
        return float(min(p[0], self.board.width - p[0], p[1], self.board.height - p[1]))

    def _min_pairwise_tip_distance(self) -> float:
        return self._min_pairwise(np.asarray([p[-1] for p in self.paths]))

    @staticmethod
    def _min_pairwise(pts: np.ndarray) -> float:
        n = len(pts); best = math.inf
        for i in range(n):
            for j in range(i + 1, n):
                best = min(best, float(np.linalg.norm(pts[i] - pts[j])))
        return best


def mask_fn(env) -> np.ndarray:
    return env.unwrapped.action_masks()
