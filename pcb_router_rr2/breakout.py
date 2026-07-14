"""Deterministic two-phase breakout, length-normalised.

Phase 1 (clear the connector): each trace descends from its pin toward the
open side of the board.  Top-row traces first take a short 45-degree "jog"
into their own lane so they do not collide with the bottom-row pins that
sit directly beneath them (a failure the naive "straight out" description
in the v1 doc would have hit).

Phase 2 (fan out): a GRADUAL fan.  A single straight diagonal per trace is
not clearance-safe: the perpendicular distance from a neighbour's fan
start to a tilted fan segment is gap * D / L, which drops below clearance
for any meaningful lateral movement while gaps are still at lane pitch
(build-time validation caught exactly this at 1.115 mm < 1.33 mm).  So the
fan descends in small vertical sub-steps, and at each sub-step every trace
moves laterally toward its target at a rate capped by
    rate <= sqrt((min_gap / (clearance + safety))^2 - 1)
which guarantees the perpendicular-distance bound stays above clearance.
As outer traces spread, gaps grow, the cap relaxes, and the fan finishes.
Lanes and fan targets are matched in x-order so ordering (and hence
non-crossing) is preserved throughout.

Length normalisation: every polyline is extended straight along its final
direction until all breakouts share the exact length of the longest one,
preserving the equal-length-by-construction guarantee end to end.

The full breakout set is validated with the same ClearanceChecker used at
runtime; if your geometry config makes the breakout invalid you find out
at env construction, not after an hour of training.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

from .board import Board
from .config import Config
from .geometry import ClearanceChecker, seg_seg_dist


def _polyline_length(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _simplify(pts: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Merge consecutive collinear segments (keeps validation and the
    spatial hash from carrying hundreds of redundant 1 mm fan slivers)."""
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    for k in range(1, len(pts) - 1):
        d1 = pts[k] - out[-1]
        d2 = pts[k + 1] - pts[k]
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) > tol or np.dot(d1, d2) < 0:
            out.append(pts[k])
    out.append(pts[-1])
    return np.asarray(out)


def build_breakout(cfg: Config, board: Board) -> List[np.ndarray]:
    """Returns one polyline (float64 array of points) per trace, all of
    identical length, ending fanned out below the connector."""
    pins = board.pins
    n = board.n_traces
    cx0, cy0, cx1, cy1 = board.connector_rect
    conn_cx = 0.5 * (cx0 + cx1)

    # The open side: this board's connector hugs the top edge, so we grow
    # downward (-y).  Generalising to other sides is a straight substitution.
    clear_y = cy0 - cfg.obstacle_clearance_mm - cfg.breakout_clear_margin_mm

    # ---- lane assignment ------------------------------------------------ #
    # Bottom row keeps its pin x (its column is free below).  Top row jogs
    # laterally by breakout_lane_jog_mm to an intermediate lane.  Jog
    # direction: outer pins jog outward, inner pins jog toward the wider gap.
    ys = pins[:, 1]
    top_row = ys >= np.median(ys)
    lane_x = pins[:, 0].copy()
    jog = cfg.breakout_lane_jog_mm
    top_idx = np.where(top_row)[0]
    for i in top_idx:
        x = pins[i, 0]
        lane_x[i] = x - jog if x <= conn_cx else x + jog

    # sanity: all lanes distinct and above trace clearance apart
    order = np.argsort(lane_x)
    gaps = np.diff(lane_x[order])
    if np.any(gaps < cfg.trace_clearance_mm):
        raise ValueError(
            f"Breakout lanes too close ({gaps.min():.2f} mm < clearance "
            f"{cfg.trace_clearance_mm} mm). Increase breakout_lane_jog_mm "
            f"or pin pitch."
        )

    # ---- fan targets ------------------------------------------------------ #
    pitch = cfg.breakout_fan_pitch_mm
    span = pitch * (n - 1)
    fan_left = conn_cx - span / 2.0
    # keep fan inside the board with margin
    margin = board.edge_clearance + 2.0
    fan_left = min(max(fan_left, margin), board.width - margin - span)
    fan_xs_sorted = fan_left + pitch * np.arange(n)
    fan_x = np.empty(n)
    fan_x[order] = fan_xs_sorted          # x-order matched -> no crossings

    # ---- gradual fan schedule ------------------------------------------- #
    # Descend in sub-steps; per sub-step, lateral movement is capped so the
    # perpendicular distance between adjacent traces' sub-segments can never
    # fall below (trace clearance + safety).  Rate cap derivation:
    #   perp >= gap / sqrt(rate^2 + 1)  ->  rate <= sqrt((gap/clr)^2 - 1)
    clr_safe = cfg.trace_clearance_mm + cfg.breakout_fan_safety_mm
    drop = cfg.breakout_fan_step_mm
    pos = lane_x.copy()
    y = clear_y
    fan_pts: List[List[List[float]]] = [[] for _ in range(n)]
    max_iters = 500
    it = 0
    while np.max(np.abs(pos - fan_x)) > 1e-6:
        it += 1
        if it > max_iters:
            raise ValueError(
                "Gradual fan failed to converge; lane gaps too close to "
                "clearance. Increase pin pitch, lane jog, or fan safety.")
        min_gap = float(np.min(np.diff(pos[order])))
        if min_gap <= clr_safe:
            raise ValueError(
                f"Fan gap collapsed to {min_gap:.3f} mm <= {clr_safe:.3f} mm")
        rate = math.sqrt((min_gap / clr_safe) ** 2 - 1.0)
        max_dx = rate * drop
        y -= drop
        step_dx = np.clip(fan_x - pos, -max_dx, max_dx)
        pos = pos + step_dx
        for i in range(n):
            fan_pts[i].append([float(pos[i]), float(y)])

    # ---- assemble polylines ------------------------------------------------ #
    polys: List[np.ndarray] = []
    for i in range(n):
        pts = [pins[i].tolist()]
        if abs(lane_x[i] - pins[i, 0]) > 1e-9:
            # 45-degree jog: drop by the same amount we shift laterally
            pts.append([lane_x[i], pins[i, 1] - abs(lane_x[i] - pins[i, 0])])
        pts.append([lane_x[i], clear_y])            # phase 1: descend
        pts.extend(fan_pts[i])                      # phase 2: gradual fan
        # straight run-out: hand the agent a tip whose recent own path is
        # a clean vertical line, not the curled fan tail (which otherwise
        # sits within self-clearance of the tip and traps the first steps)
        last_y = pts[-1][1]
        pts.append([fan_x[i], last_y - cfg.breakout_runout_mm])
        polys.append(_simplify(np.asarray(pts, dtype=np.float64)))

    # ---- length normalisation ---------------------------------------------- #
    lengths = [_polyline_length(p) for p in polys]
    target = max(lengths)
    for i in range(n):
        deficit = target - lengths[i]
        if deficit > 1e-9:
            d = polys[i][-1] - polys[i][-2]
            d = d / (np.linalg.norm(d) + 1e-12)
            polys[i] = np.vstack([polys[i], polys[i][-1] + d * deficit])

    lengths = [_polyline_length(p) for p in polys]
    assert max(lengths) - min(lengths) < 1e-6, "breakout normalisation failed"

    _validate_breakout(cfg, board, polys)
    return polys


def _validate_breakout(cfg: Config, board: Board, polys: List[np.ndarray]) -> None:
    """Breakout must obey the same rules the agent will: inside board,
    clear of keep-outs, and inter-trace clearance respected everywhere.
    (Pins themselves live inside the connector rect, so the first segment
    of each polyline is exempt from the keep-out test — it is *escaping*
    the connector.)"""
    checker: ClearanceChecker = board.make_checker(cfg)

    for i, pts in enumerate(polys):
        for k in range(len(pts) - 1):
            a, b = tuple(pts[k]), tuple(pts[k + 1])
            if not (checker.point_in_board(a) and checker.point_in_board(b)):
                # pins may legitimately sit inside edge clearance at the
                # connector edge; only enforce once clear of the connector
                if k > 0:
                    raise ValueError(f"Breakout trace {i} seg {k} leaves board area")

    # inter-trace clearance, brute force (few segments, one-time cost)
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            for a0, a1 in zip(polys[i][:-1], polys[i][1:]):
                for b0, b1 in zip(polys[j][:-1], polys[j][1:]):
                    d = seg_seg_dist(tuple(a0), tuple(a1), tuple(b0), tuple(b1))
                    if d < cfg.trace_clearance_mm - 1e-6:
                        raise ValueError(
                            f"Breakout traces {i} and {j} violate clearance "
                            f"({d:.3f} mm < {cfg.trace_clearance_mm} mm). "
                            f"Adjust lane jog / fan pitch."
                        )


def breakout_end_directions(polys: List[np.ndarray]) -> np.ndarray:
    """Unit direction of each breakout's final segment (initial heading)."""
    dirs = []
    for p in polys:
        d = p[-1] - p[-2]
        dirs.append(d / (np.linalg.norm(d) + 1e-12))
    return np.asarray(dirs)
