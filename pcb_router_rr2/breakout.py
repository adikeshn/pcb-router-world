"""Deterministic side-aware breakout, length-normalised.

Modes (cfg.breakout_mode):
* "split" — pin rows exit on OPPOSITE sides of the connector: the top row
  breaks out upward, the bottom row downward.  Each row has a clear column
  on its own side, so no lane jogs are needed.  This is the natural mode
  for a connector in the middle of the board.
* "down" / "up" — every trace exits on one side (the original behaviour
  for a connector hugging a board edge).  Rows whose escape column is
  blocked by the other row's pins take a short 45-degree jog into an
  interleaved lane first.
* "auto" — "split" when both sides of the connector have at least
  breakout_split_min_room_mm of board; otherwise exit toward the roomier
  side.

Per group, the phases are: (jog) -> straight escape past the connector ->
GRADUAL fan -> straight run-out.  The gradual fan descends in small
vertical sub-steps with per-sub-step lateral movement capped by
    rate <= sqrt((min_gap / (clearance + safety))^2 - 1)
which keeps the perpendicular distance between adjacent traces' fan
sub-segments above clearance at every point (a single straight diagonal
fan violates this near the fan start — caught by build-time validation).
The straight run-out hands the agent a tip whose recent own path is a
clean line, not a curled fan tail (which otherwise sits within
self-clearance of the tip and traps the first steps).

Finally every polyline is extended along its last direction until all
share the exact length of the longest one, preserving equal length by
construction end to end; the whole set is then brute-force validated with
the same clearance rules the agent lives under.
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


def _resolve_mode(cfg: Config, board: Board) -> str:
    mode = cfg.breakout_mode
    if mode not in ("auto", "split", "up", "down"):
        raise ValueError(f"Unknown breakout_mode: {mode!r}")
    if mode != "auto":
        return mode
    room_below = board.connector_rect[1]
    room_above = board.height - board.connector_rect[3]
    if min(room_above, room_below) >= cfg.breakout_split_min_room_mm:
        return "split"
    return "up" if room_above > room_below else "down"


def _group_polylines(cfg: Config, board: Board, idxs: List[int],
                     sign: int, allow_jogs: bool) -> Dict[int, List[List[float]]]:
    """Build (un-normalised) polylines for one group of traces that all
    exit on the same side.  sign = +1 exits upward (+y), -1 downward."""
    pins = board.pins
    cx0, cy0, cx1, cy1 = board.connector_rect
    conn_cx = 0.5 * (cx0 + cx1)
    escape_edge = cy1 if sign > 0 else cy0
    clear_y = escape_edge + sign * (cfg.obstacle_clearance_mm
                                    + cfg.breakout_clear_margin_mm)

    # ---- lanes: jog only if another pin in the group blocks the column ---
    lane_x = {i: float(pins[i, 0]) for i in idxs}
    if allow_jogs:
        jog = cfg.breakout_lane_jog_mm
        for i in idxs:
            xi, yi = pins[i]
            blocked = any(
                j != i
                and abs(pins[j, 0] - xi) < cfg.trace_clearance_mm
                and (pins[j, 1] - yi) * sign > 0        # pin ahead on escape path
                for j in idxs)
            if blocked:
                lane_x[i] = xi - jog if xi <= conn_cx else xi + jog

    ordered = sorted(idxs, key=lambda i: lane_x[i])
    lanes_sorted = np.asarray([lane_x[i] for i in ordered])
    gaps = np.diff(lanes_sorted)
    if len(gaps) and gaps.min() < cfg.trace_clearance_mm:
        raise ValueError(
            f"Breakout lanes too close ({gaps.min():.2f} mm < clearance "
            f"{cfg.trace_clearance_mm} mm). Increase breakout_lane_jog_mm "
            f"or pin pitch.")

    # ---- fan targets: pitch-spread, x-order matched (no crossings) -------
    n = len(idxs)
    pitch = cfg.breakout_fan_pitch_mm
    span = pitch * (n - 1)
    margin = board.edge_clearance + 2.0
    fan_left = conn_cx - span / 2.0
    fan_left = min(max(fan_left, margin), board.width - margin - span)
    fan_x = {tr: fan_left + pitch * k for k, tr in enumerate(ordered)}

    # ---- gradual fan schedule (rate-capped, see module docstring) --------
    clr_safe = cfg.trace_clearance_mm + cfg.breakout_fan_safety_mm
    drop = cfg.breakout_fan_step_mm
    pos = {i: lane_x[i] for i in idxs}
    targets = np.asarray([fan_x[i] for i in ordered])
    y = clear_y
    fan_pts: Dict[int, List[List[float]]] = {i: [] for i in idxs}
    it = 0
    while max(abs(pos[i] - fan_x[i]) for i in idxs) > 1e-6:
        it += 1
        if it > 500:
            raise ValueError(
                "Gradual fan failed to converge; lane gaps too close to "
                "clearance. Increase pin pitch, lane jog, or fan safety.")
        cur = np.asarray([pos[i] for i in ordered])
        if n > 1:
            min_gap = float(np.min(np.diff(cur)))
            if min_gap <= clr_safe:
                raise ValueError(
                    f"Fan gap collapsed to {min_gap:.3f} mm <= {clr_safe:.3f} mm")
            rate = math.sqrt((min_gap / clr_safe) ** 2 - 1.0)
        else:
            rate = math.inf
        max_dx = rate * drop
        y += sign * drop
        for i in idxs:
            dx = float(np.clip(fan_x[i] - pos[i], -max_dx, max_dx))
            pos[i] += dx
            fan_pts[i].append([pos[i], y])

    # ---- assemble: pin -> (jog) -> escape -> fan -> run-out --------------
    polys: Dict[int, List[List[float]]] = {}
    for i in idxs:
        pts = [pins[i].tolist()]
        if abs(lane_x[i] - pins[i, 0]) > 1e-9:
            # 45-degree jog: advance along escape axis by the lateral shift
            pts.append([lane_x[i],
                        pins[i, 1] + sign * abs(lane_x[i] - pins[i, 0])])
        pts.append([lane_x[i], clear_y])
        pts.extend(fan_pts[i])
        last_y = pts[-1][1]
        pts.append([fan_x[i], last_y + sign * cfg.breakout_runout_mm])
        polys[i] = pts
    return polys


def build_breakout(cfg: Config, board: Board) -> List[np.ndarray]:
    """Returns one polyline (float64 array) per trace, all of identical
    length, fanned out on the appropriate side(s) of the connector."""
    pins = board.pins
    n = board.n_traces
    mode = _resolve_mode(cfg, board)

    if mode == "split":
        ys = pins[:, 1]
        med = float(np.median(ys))
        top = [i for i in range(n) if ys[i] >= med]
        bot = [i for i in range(n) if ys[i] < med]
        if not top or not bot:
            raise ValueError(
                "breakout_mode='split' needs pins on both sides of the "
                "median row; use 'up' or 'down' for a single-row connector.")
        raw: Dict[int, List[List[float]]] = {}
        raw.update(_group_polylines(cfg, board, top, +1, allow_jogs=False))
        raw.update(_group_polylines(cfg, board, bot, -1, allow_jogs=False))
    else:
        sign = +1 if mode == "up" else -1
        raw = _group_polylines(cfg, board, list(range(n)), sign,
                               allow_jogs=True)

    polys = [np.asarray(raw[i], dtype=np.float64) for i in range(n)]

    # ---- length normalisation across ALL traces (both sides) ------------ #
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
    """Breakout must obey the same rules the agent will: inside board,
    inter-trace clearance respected everywhere, and NO segment may cross
    the interior of the connector or any obstacle.  Pins sit ON the
    connector edge (enforced by Config.validate), so a correct escape
    segment only touches the footprint boundary at its start point; the
    keep-out rects are deflated by a hair so boundary contact passes but
    any genuine interior crossing fails loudly."""
    checker: ClearanceChecker = board.make_checker(cfg)

    eps = 1e-6
    inner_rects = []
    for r in board.keepout_rects:
        if r[2] - r[0] > 2 * eps and r[3] - r[1] > 2 * eps:
            inner_rects.append((r[0] + eps, r[1] + eps, r[2] - eps, r[3] - eps))

    for i, pts in enumerate(polys):
        for k in range(len(pts) - 1):
            a, b = tuple(pts[k]), tuple(pts[k + 1])
            if not (checker.point_in_board(a) and checker.point_in_board(b)):
                # pins may legitimately sit inside edge clearance at the
                # connector edge; only enforce once clear of the connector
                if k > 0:
                    raise ValueError(f"Breakout trace {i} seg {k} leaves board area")
            for r in inner_rects:
                if seg_intersects_rect(a, b, r):
                    raise ValueError(
                        f"Breakout trace {i} seg {k} crosses a keep-out "
                        f"interior (connector/obstacle). Pins must sit on "
                        f"the connector edge FACING their escape direction "
                        f"(top edge for upward escape, bottom edge for "
                        f"downward); check pins / breakout_mode.")

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
                            f"Adjust lane jog / fan pitch.")


def breakout_end_directions(polys: List[np.ndarray]) -> np.ndarray:
    """Unit direction of each breakout's final segment (initial heading)."""
    dirs = []
    for p in polys:
        d = p[-1] - p[-2]
        dirs.append(d / (np.linalg.norm(d) + 1e-12))
    return np.asarray(dirs)
