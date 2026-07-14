"""Clearance-inflated collision geometry + spatial hash.

Every validity decision in the environment funnels through this module so
that masking, ray-casting and the end-of-episode audit are guaranteed to
agree with each other (a mask/audit mismatch was the root cause of the
"residual crossings" in v1).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Rect = Tuple[float, float, float, float]  # x0, y0, x1, y1


# --------------------------------------------------------------------------- #
# Primitive distances
# --------------------------------------------------------------------------- #
def seg_point_dist(a: Point, b: Point, p: Point) -> float:
    """Minimum distance from point p to segment ab."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    ll = dx * dx + dy * dy
    if ll <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / ll
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def _seg_seg_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    return False


def seg_seg_dist(p1: Point, p2: Point, p3: Point, p4: Point) -> float:
    """Minimum distance between segments p1p2 and p3p4 (0 if they cross)."""
    if _seg_seg_intersect(p1, p2, p3, p4):
        return 0.0
    return min(
        seg_point_dist(p1, p2, p3),
        seg_point_dist(p1, p2, p4),
        seg_point_dist(p3, p4, p1),
        seg_point_dist(p3, p4, p2),
    )


def rect_inflate(r: Rect, m: float) -> Rect:
    return (r[0] - m, r[1] - m, r[2] + m, r[3] + m)


def point_in_rect(p: Point, r: Rect) -> bool:
    return r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]


def seg_intersects_rect(a: Point, b: Point, r: Rect) -> bool:
    """True if segment ab touches rectangle r (used with pre-inflated rects)."""
    if point_in_rect(a, r) or point_in_rect(b, r):
        return True
    x0, y0, x1, y1 = r
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    return any(_seg_seg_intersect(a, b, e0, e1) for e0, e1 in edges)


# --------------------------------------------------------------------------- #
# Spatial hash over segments
# --------------------------------------------------------------------------- #
class SegmentHash:
    """Uniform-grid spatial hash storing (trace_id, seg_idx, ax, ay, bx, by).

    Cell size should be >= the largest query radius so a 1-cell halo query
    is sufficient; we use a halo computed from the radius to stay correct
    for any cell size.
    """

    def __init__(self, cell_mm: float = 4.0):
        self.cell = float(cell_mm)
        self.grid: Dict[Tuple[int, int], List[Tuple[int, int, float, float, float, float]]] = {}

    def clear(self) -> None:
        self.grid.clear()

    def _cells_for_aabb(self, x0: float, y0: float, x1: float, y1: float) -> Iterable[Tuple[int, int]]:
        c = self.cell
        i0, i1 = int(math.floor(x0 / c)), int(math.floor(x1 / c))
        j0, j1 = int(math.floor(y0 / c)), int(math.floor(y1 / c))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                yield (i, j)

    def add_segment(self, trace_id: int, seg_idx: int, a: Point, b: Point) -> None:
        entry = (trace_id, seg_idx, a[0], a[1], b[0], b[1])
        x0, x1 = min(a[0], b[0]), max(a[0], b[0])
        y0, y1 = min(a[1], b[1]), max(a[1], b[1])
        for cell in self._cells_for_aabb(x0, y0, x1, y1):
            self.grid.setdefault(cell, []).append(entry)

    def _candidates(self, x0, y0, x1, y1, radius) -> List[Tuple[int, int, float, float, float, float]]:
        seen = set()
        out = []
        for cell in self._cells_for_aabb(x0 - radius, y0 - radius, x1 + radius, y1 + radius):
            for e in self.grid.get(cell, ()):  # type: ignore[arg-type]
                key = (e[0], e[1])
                if key not in seen:
                    seen.add(key)
                    out.append(e)
        return out

    def near_segment(self, a: Point, b: Point, radius: float):
        x0, x1 = min(a[0], b[0]), max(a[0], b[0])
        y0, y1 = min(a[1], b[1]), max(a[1], b[1])
        return self._candidates(x0, y0, x1, y1, radius)

    def near_point(self, p: Point, radius: float):
        return self._candidates(p[0], p[1], p[0], p[1], radius)


# --------------------------------------------------------------------------- #
# Full-board clearance checker
# --------------------------------------------------------------------------- #
class ClearanceChecker:
    """Wraps the hash + board keep-outs into the validity queries the env,
    the ray-caster, the breakout builder and the audit all share."""

    def __init__(
        self,
        board_w: float,
        board_h: float,
        edge_clearance: float,
        keepout_rects: Sequence[Rect],
        obstacle_clearance: float,
        trace_clearance: float,
        self_clearance: float,
        self_skip_mm: float,
        cell_mm: float = 4.0,
    ):
        self.w, self.h = board_w, board_h
        self.edge = edge_clearance
        self.trace_clr = trace_clearance
        self.self_clr = self_clearance
        self.self_skip_mm = self_skip_mm
        self.inflated_keepouts: List[Rect] = [rect_inflate(r, obstacle_clearance) for r in keepout_rects]
        self.hash = SegmentHash(cell_mm)
        # per-trace segment count + cumulative arc length at each segment's
        # END, so "recent own path" is defined by DISTANCE along the path
        # (a fixed segment-count window breaks when segment lengths vary,
        # e.g. the simplified breakout fan).
        self.seg_count: Dict[int, int] = {}
        self.seg_end_len: Dict[int, List[float]] = {}

    # -- mutation -------------------------------------------------------- #
    def reset(self) -> None:
        self.hash.clear()
        self.seg_count.clear()
        self.seg_end_len.clear()

    def add_polyline(self, trace_id: int, pts: Sequence[Point]) -> None:
        for k in range(len(pts) - 1):
            self.add_segment(trace_id, pts[k], pts[k + 1])

    def add_segment(self, trace_id: int, a: Point, b: Point) -> None:
        idx = self.seg_count.get(trace_id, 0)
        self.hash.add_segment(trace_id, idx, a, b)
        self.seg_count[trace_id] = idx + 1
        lens = self.seg_end_len.setdefault(trace_id, [])
        prev = lens[-1] if lens else 0.0
        lens.append(prev + math.hypot(b[0] - a[0], b[1] - a[1]))

    def _own_exempt(self, trace_id: int, sidx: int) -> bool:
        lens = self.seg_end_len.get(trace_id)
        if not lens:
            return False
        return (lens[-1] - lens[sidx]) < self.self_skip_mm

    # -- queries ---------------------------------------------------------- #
    def point_in_board(self, p: Point) -> bool:
        e = self.edge
        return e <= p[0] <= self.w - e and e <= p[1] <= self.h - e

    def point_in_keepout(self, p: Point) -> bool:
        return any(point_in_rect(p, r) for r in self.inflated_keepouts)

    def seg_hits_keepout(self, a: Point, b: Point) -> bool:
        return any(seg_intersects_rect(a, b, r) for r in self.inflated_keepouts)

    def seg_min_dists(self, trace_id: int, a: Point, b: Point) -> Tuple[float, float]:
        """(min distance to OTHER traces' segments,
            min distance to OWN non-recent segments) for proposed segment ab.

        "Recent" own segments (the last `self_skip` already placed) are
        exempt because the new segment legitimately connects to them.
        """
        radius = max(self.trace_clr, self.self_clr) + 0.5
        min_other = math.inf
        min_own = math.inf
        for (tid, sidx, ax, ay, bx, by) in self.hash.near_segment(a, b, radius):
            d = seg_seg_dist(a, b, (ax, ay), (bx, by))
            if tid != trace_id:
                if d < min_other:
                    min_other = d
            else:
                if not self._own_exempt(trace_id, sidx) and d < min_own:
                    min_own = d
        return min_other, min_own

    def segment_valid(self, trace_id: int, a: Point, b: Point) -> Tuple[bool, float]:
        """Full validity for a proposed growth segment.

        Returns (valid, min_other_trace_distance) — the distance is reused
        for the dense path-clearance penalty and terminal metric.
        """
        if not self.point_in_board(b):
            return False, math.inf
        if self.seg_hits_keepout(a, b):
            return False, math.inf
        d_other, d_own = self.seg_min_dists(trace_id, a, b)
        if d_other < self.trace_clr:
            return False, d_other
        if d_own < self.self_clr:
            return False, d_other
        return True, d_other

    def point_free(self, trace_id: int, p: Point) -> bool:
        """Point-level check used by the ray-caster (cheap approximation)."""
        if not self.point_in_board(p):
            return False
        if self.point_in_keepout(p):
            return False
        radius = max(self.trace_clr, self.self_clr) + 0.5
        for (tid, sidx, ax, ay, bx, by) in self.hash.near_point(p, radius):
            d = seg_point_dist((ax, ay), (bx, by), p)
            if tid != trace_id and d < self.trace_clr:
                return False
            if (tid == trace_id and not self._own_exempt(trace_id, sidx)
                    and d < self.self_clr):
                return False
        return True

    def ray_distance(self, trace_id: int, origin: Point, direction: Point,
                     max_mm: float, step_mm: float = 1.0) -> float:
        """Distance along `direction` until the first blocked point."""
        d = step_mm
        while d <= max_mm + 1e-9:
            p = (origin[0] + direction[0] * d, origin[1] + direction[1] * d)
            if not self.point_free(trace_id, p):
                return d - step_mm
            d += step_mm
        return max_mm


# --------------------------------------------------------------------------- #
# End-of-episode audit (independent of the hash — brute force, trustworthy)
# --------------------------------------------------------------------------- #
def audit_paths(
    paths: Sequence[np.ndarray],
    trace_clearance: float,
    self_clearance: float,
    self_skip_mm: float,
) -> Tuple[int, float]:
    """Brute-force check of the final layout.

    Returns (violation_count, min_inter_trace_distance).  Runs once per
    episode; O(total_segments^2) is fine at this scale and it deliberately
    does NOT share code paths with the spatial hash so it can catch hash
    bugs as well as mask bugs.  Own-path pairs are exempt when the
    arc-length gap between them is below self_skip_mm — the exact rule the
    live checker uses.
    """
    segs = []       # (trace_id, seg_idx, a, b)
    starts = []     # arc length at segment start
    ends = []       # arc length at segment end
    for tid, pts in enumerate(paths):
        acc = 0.0
        for k in range(len(pts) - 1):
            a = (float(pts[k][0]), float(pts[k][1]))
            b = (float(pts[k + 1][0]), float(pts[k + 1][1]))
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            segs.append((tid, k, a, b))
            starts.append(acc)
            ends.append(acc + L)
            acc += L
    violations = 0
    min_inter = math.inf
    n = len(segs)
    for i in range(n):
        ti, si, a1, b1 = segs[i]
        for j in range(i + 1, n):
            tj, sj, a2, b2 = segs[j]
            if ti == tj:
                if (starts[j] - ends[i]) < self_skip_mm:
                    continue
                d = seg_seg_dist(a1, b1, a2, b2)
                if d < self_clearance - 1e-6:
                    violations += 1
            else:
                d = seg_seg_dist(a1, b1, a2, b2)
                if d < min_inter:
                    min_inter = d
                if d < trace_clearance - 1e-6:
                    violations += 1
    return violations, (min_inter if min_inter is not math.inf else float("inf"))
