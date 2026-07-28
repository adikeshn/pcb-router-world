"""Optional deterministic side-aware breakout (disabled by default).

With use_breakout=False (the default) the agent grows directly from the pins
and this module is never invoked.  It is retained for boards where a fixed
escape fan is preferred.  See git history for the full derivation of the
rate-capped gradual fan.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from .board import Board
from .config import Config
from .geometry import ClearanceChecker, seg_intersects_rect, seg_seg_dist


def _polyline_length(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _simplify(pts: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    for k in range(1, len(pts) - 1):
        d1 = pts[k] - out[-1]; d2 = pts[k + 1] - pts[k]
        if abs(d1[0] * d2[1] - d1[1] * d2[0]) > tol or np.dot(d1, d2) < 0:
            out.append(pts[k])
    out.append(pts[-1])
    return np.asarray(out)


def _resolve_mode(cfg: Config, board: Board) -> str:
    mode = cfg.breakout_mode
    if mode not in ("auto", "split", "up", "down"):
        raise ValueError(f"Unknown breakout_mode: {mode!r}")
    if mode != "auto":
        return mode
    below = board.connector_rect[1]
    above = board.height - board.connector_rect[3]
    if min(above, below) >= cfg.breakout_split_min_room_mm:
        return "split"
    return "up" if above > below else "down"


def _group_polylines(cfg, board, idxs, sign, allow_jogs):
    pins = board.pins
    cx0, cy0, cx1, cy1 = board.connector_rect
    conn_cx = 0.5 * (cx0 + cx1)
    escape_edge = cy1 if sign > 0 else cy0
    clear_y = escape_edge + sign * (cfg.obstacle_clearance_mm + cfg.breakout_clear_margin_mm)

    lane_x = {i: float(pins[i, 0]) for i in idxs}
    if allow_jogs:
        jog = cfg.breakout_lane_jog_mm
        for i in idxs:
            xi, yi = pins[i]
            if any(j != i and abs(pins[j, 0] - xi) < cfg.trace_clearance_mm
                   and (pins[j, 1] - yi) * sign > 0 for j in idxs):
                lane_x[i] = xi - jog if xi <= conn_cx else xi + jog

    ordered = sorted(idxs, key=lambda i: lane_x[i])
    gaps = np.diff(np.asarray([lane_x[i] for i in ordered]))
    if len(gaps) and gaps.min() < cfg.trace_clearance_mm:
        raise ValueError(f"Breakout lanes too close ({gaps.min():.2f} mm < "
                         f"clearance {cfg.trace_clearance_mm} mm).")

    n = len(idxs); pitch = cfg.breakout_fan_pitch_mm; span = pitch * (n - 1)
    margin = board.edge_clearance + 2.0
    fan_left = min(max(conn_cx - span / 2.0, margin), board.width - margin - span)
    fan_x = {tr: fan_left + pitch * k for k, tr in enumerate(ordered)}

    clr_safe = cfg.trace_clearance_mm + cfg.breakout_fan_safety_mm
    drop = cfg.breakout_fan_step_mm
    pos = {i: lane_x[i] for i in idxs}
    y = clear_y
    fan_pts: Dict[int, List[List[float]]] = {i: [] for i in idxs}
    it = 0
    while max(abs(pos[i] - fan_x[i]) for i in idxs) > 1e-6:
        it += 1
        if it > 500:
            raise ValueError("Gradual fan failed to converge.")
        cur = np.asarray([pos[i] for i in ordered])
        if n > 1:
            min_gap = float(np.min(np.diff(cur)))
            if min_gap <= clr_safe:
                raise ValueError(f"Fan gap collapsed to {min_gap:.3f} mm")
            rate = math.sqrt((min_gap / clr_safe) ** 2 - 1.0)
        else:
            rate = math.inf
        max_dx = rate * drop
        y += sign * drop
        for i in idxs:
            pos[i] += float(np.clip(fan_x[i] - pos[i], -max_dx, max_dx))
            fan_pts[i].append([pos[i], y])

    polys = {}
    for i in idxs:
        pts = [pins[i].tolist()]
        if abs(lane_x[i] - pins[i, 0]) > 1e-9:
            pts.append([lane_x[i], pins[i, 1] + sign * abs(lane_x[i] - pins[i, 0])])
        pts.append([lane_x[i], clear_y])
        pts.extend(fan_pts[i])
        pts.append([fan_x[i], pts[-1][1] + sign * cfg.breakout_runout_mm])
        polys[i] = pts
    return polys


def build_breakout(cfg: Config, board: Board) -> List[np.ndarray]:
    pins = board.pins; n = board.n_traces
    mode = _resolve_mode(cfg, board)
    if mode == "split":
        ys = pins[:, 1]; med = float(np.median(ys))
        top = [i for i in range(n) if ys[i] >= med]
        bot = [i for i in range(n) if ys[i] < med]
        if not top or not bot:
            raise ValueError("breakout_mode='split' needs pins on both sides.")
        raw = {}
        raw.update(_group_polylines(cfg, board, top, +1, allow_jogs=False))
        raw.update(_group_polylines(cfg, board, bot, -1, allow_jogs=False))
    else:
        raw = _group_polylines(cfg, board, list(range(n)),
                               +1 if mode == "up" else -1, allow_jogs=True)

    polys = [np.asarray(raw[i], dtype=np.float64) for i in range(n)]
    lengths = [_polyline_length(p) for p in polys]
    target = max(lengths)
    for i in range(n):
        deficit = target - lengths[i]
        if deficit > 1e-9:
            d = polys[i][-1] - polys[i][-2]
            d = d / (np.linalg.norm(d) + 1e-12)
            polys[i] = np.vstack([polys[i], polys[i][-1] + d * deficit])
    polys = [_simplify(p) for p in polys]
    lengths = [_polyline_length(p) for p in polys]
    assert max(lengths) - min(lengths) < 1e-6, "breakout normalisation failed"
    _validate_breakout(cfg, board, polys)
    return polys


def _validate_breakout(cfg: Config, board: Board, polys: List[np.ndarray]) -> None:
    checker: ClearanceChecker = board.make_checker(cfg)
    eps = 1e-6
    inner = [(r[0] + eps, r[1] + eps, r[2] - eps, r[3] - eps)
             for r in board.keepout_rects
             if r[2] - r[0] > 2 * eps and r[3] - r[1] > 2 * eps]
    for i, pts in enumerate(polys):
        for k in range(len(pts) - 1):
            a, b = tuple(pts[k]), tuple(pts[k + 1])
            if k > 0 and not (checker.point_in_board(a) and checker.point_in_board(b)):
                raise ValueError(f"Breakout trace {i} seg {k} leaves board area")
            for r in inner:
                if seg_intersects_rect(a, b, r):
                    raise ValueError(
                        f"Breakout trace {i} seg {k} crosses a keep-out interior.")
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            for a0, a1 in zip(polys[i][:-1], polys[i][1:]):
                for b0, b1 in zip(polys[j][:-1], polys[j][1:]):
                    d = seg_seg_dist(tuple(a0), tuple(a1), tuple(b0), tuple(b1))
                    if d < cfg.trace_clearance_mm - 1e-6:
                        raise ValueError(
                            f"Breakout traces {i} and {j} violate clearance ({d:.3f} mm).")


def breakout_end_directions(polys: List[np.ndarray]) -> np.ndarray:
    return np.asarray([(p[-1] - p[-2]) / (np.linalg.norm(p[-1] - p[-2]) + 1e-12)
                       for p in polys])
