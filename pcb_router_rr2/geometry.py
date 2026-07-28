"""Clearance-inflated collision geometry + spatial hash.

Every validity decision funnels through this module so masking, ray-casting
and the end-of-episode audit are guaranteed to agree with each other.

SELF-EXEMPTION RULE
-------------------
Two segments of the SAME trace are exempt from the self-clearance check only
when they are topologically ADJACENT (|i - j| <= 1), i.e. they share an
endpoint and are therefore trivially at distance zero.

This replaces an earlier arc-length window, which had a proven blind spot:
a 3-step sequence N, SE, W is a legal action sequence (no 180 reversal) whose
third segment genuinely CROSSES the first, yet the two sat within the
arc-length window and were never checked -- so neither the mask nor the audit
reported a violation.  Adjacency has no such hole and is robust to variable
segment lengths (adjacent segments share an endpoint whatever their length).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Rect = Tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
def seg_point_dist(a: Point, b: Point, p: Point) -> float:
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    ll = dx * dx + dy * dy
    if ll <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / ll))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _seg_seg_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = orient(p3, p4, p1); d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3); d4 = orient(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def seg_seg_dist(p1: Point, p2: Point, p3: Point, p4: Point) -> float:
    if _seg_seg_intersect(p1, p2, p3, p4):
        return 0.0
    return min(seg_point_dist(p1, p2, p3), seg_point_dist(p1, p2, p4),
               seg_point_dist(p3, p4, p1), seg_point_dist(p3, p4, p2))


def rect_inflate(r: Rect, m: float) -> Rect:
    return (r[0] - m, r[1] - m, r[2] + m, r[3] + m)


def rect_distance(p: Point, r: Rect) -> float:
    dx = max(r[0] - p[0], 0.0, p[0] - r[2])
    dy = max(r[1] - p[1], 0.0, p[1] - r[3])
    return math.hypot(dx, dy)


def point_in_rect(p: Point, r: Rect) -> bool:
    return r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]


def seg_intersects_rect(a: Point, b: Point, r: Rect) -> bool:
    if point_in_rect(a, r) or point_in_rect(b, r):
        return True
    x0, y0, x1, y1 = r
    c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return any(_seg_seg_intersect(a, b, c[i], c[(i + 1) % 4]) for i in range(4))


# --------------------------------------------------------------------------- #
class SegmentHash:
    """Uniform-grid spatial hash over segments."""

    def __init__(self, cell_mm: float = 4.0):
        self.cell = float(cell_mm)
        self.grid: Dict[Tuple[int, int], List[Tuple]] = {}

    def clear(self) -> None:
        self.grid.clear()

    def _cells(self, x0, y0, x1, y1) -> Iterable[Tuple[int, int]]:
        c = self.cell
        for i in range(int(math.floor(x0 / c)), int(math.floor(x1 / c)) + 1):
            for j in range(int(math.floor(y0 / c)), int(math.floor(y1 / c)) + 1):
                yield (i, j)

    def add_segment(self, trace_id: int, seg_idx: int, a: Point, b: Point) -> None:
        e = (trace_id, seg_idx, a[0], a[1], b[0], b[1])
        for cell in self._cells(min(a[0], b[0]), min(a[1], b[1]),
                                max(a[0], b[0]), max(a[1], b[1])):
            self.grid.setdefault(cell, []).append(e)

    def _candidates(self, x0, y0, x1, y1, radius) -> List[Tuple]:
        seen, out = set(), []
        for cell in self._cells(x0 - radius, y0 - radius, x1 + radius, y1 + radius):
            for e in self.grid.get(cell, ()):
                k = (e[0], e[1])
                if k not in seen:
                    seen.add(k); out.append(e)
        return out

    def near_segment(self, a: Point, b: Point, radius: float):
        return self._candidates(min(a[0], b[0]), min(a[1], b[1]),
                                max(a[0], b[0]), max(a[1], b[1]), radius)

    def near_point(self, p: Point, radius: float):
        return self._candidates(p[0], p[1], p[0], p[1], radius)


# --------------------------------------------------------------------------- #
class ClearanceChecker:
    def __init__(self, board_w, board_h, edge_clearance, keepout_rects,
                 obstacle_clearance, trace_clearance, self_clearance,
                 cell_mm: float = 4.0):
        self.w, self.h = board_w, board_h
        self.edge = edge_clearance
        self.trace_clr = trace_clearance
        self.self_clr = self_clearance
        self.raw_keepouts: List[Rect] = list(keepout_rects)
        self.inflated_keepouts: List[Rect] = [
            rect_inflate(r, obstacle_clearance) for r in keepout_rects]
        eps = 1e-6
        self.inner_keepouts: List[Rect] = [
            (r[0] + eps, r[1] + eps, r[2] - eps, r[3] - eps)
            if (r[2] - r[0] > 2 * eps and r[3] - r[1] > 2 * eps) else r
            for r in keepout_rects]
        self.escape_progress_mm = 0.25
        self.hash = SegmentHash(cell_mm)
        self.seg_count: Dict[int, int] = {}

    # -- mutation --------------------------------------------------------- #
    def reset(self) -> None:
        self.hash.clear()
        self.seg_count.clear()

    def add_polyline(self, trace_id: int, pts: Sequence[Point]) -> None:
        for k in range(len(pts) - 1):
            self.add_segment(trace_id, pts[k], pts[k + 1])

    def add_segment(self, trace_id: int, a: Point, b: Point) -> None:
        idx = self.seg_count.get(trace_id, 0)
        self.hash.add_segment(trace_id, idx, a, b)
        self.seg_count[trace_id] = idx + 1

    def _own_exempt(self, trace_id: int, sidx: int) -> bool:
        """Adjacency-only exemption: the segment about to be placed has index
        n = seg_count, so only segment n-1 shares an endpoint with it."""
        return sidx >= self.seg_count.get(trace_id, 0) - 1

    # -- queries ---------------------------------------------------------- #
    def point_in_board(self, p: Point) -> bool:
        e = self.edge
        return e <= p[0] <= self.w - e and e <= p[1] <= self.h - e

    def point_in_keepout(self, p: Point) -> bool:
        return any(point_in_rect(p, r) for r in self.inflated_keepouts)

    def keepout_segment_ok(self, a: Point, b: Point) -> bool:
        """Keep-out rule with an ESCAPE MODE for pins on the footprint edge.

        If the tip `a` is inside a rect's clearance halo (true for a pin on the
        footprint edge), the move must never touch the rect body and must
        strictly increase distance from it.  Once outside the halo the standard
        rule applies, which also makes re-entering impossible."""
        for raw, infl, inner in zip(self.raw_keepouts, self.inflated_keepouts,
                                    self.inner_keepouts):
            if point_in_rect(a, infl):
                if seg_intersects_rect(a, b, inner):
                    return False
                if rect_distance(b, raw) < rect_distance(a, raw) + self.escape_progress_mm:
                    return False
            else:
                if seg_intersects_rect(a, b, infl):
                    return False
        return True

    def seg_min_dists(self, trace_id: int, a: Point, b: Point) -> Tuple[float, float]:
        """(min distance to OTHER traces, min distance to OWN non-adjacent
        segments) for the proposed segment ab."""
        radius = max(self.trace_clr, self.self_clr) + self.escape_progress_mm + 4.0
        min_other = math.inf
        min_own = math.inf
        for (tid, sidx, ax, ay, bx, by) in self.hash.near_segment(a, b, radius):
            d = seg_seg_dist(a, b, (ax, ay), (bx, by))
            if tid != trace_id:
                if d < min_other:
                    min_other = d
            elif not self._own_exempt(trace_id, sidx):
                if d < min_own:
                    min_own = d
        return min_other, min_own

    def segment_valid(self, trace_id: int, a: Point, b: Point) -> Tuple[bool, float, float]:
        """Returns (valid, min_other_trace_distance, min_self_distance).
        Both distances feed the dense reward terms."""
        if not self.point_in_board(b):
            return False, math.inf, math.inf
        if not self.keepout_segment_ok(a, b):
            return False, math.inf, math.inf
        d_other, d_own = self.seg_min_dists(trace_id, a, b)
        if d_other < self.trace_clr:
            return False, d_other, d_own
        if d_own < self.self_clr:
            return False, d_other, d_own
        return True, d_other, d_own

    def point_free(self, trace_id: int, p: Point) -> bool:
        if not self.point_in_board(p) or self.point_in_keepout(p):
            return False
        radius = max(self.trace_clr, self.self_clr) + 0.5
        for (tid, sidx, ax, ay, bx, by) in self.hash.near_point(p, radius):
            d = seg_point_dist((ax, ay), (bx, by), p)
            if tid != trace_id and d < self.trace_clr:
                return False
            if tid == trace_id and not self._own_exempt(trace_id, sidx) and d < self.self_clr:
                return False
        return True

    def ray_distance(self, trace_id, origin, direction, max_mm, step_mm=1.0) -> float:
        d = step_mm
        while d <= max_mm + 1e-9:
            p = (origin[0] + direction[0] * d, origin[1] + direction[1] * d)
            if not self.point_free(trace_id, p):
                return d - step_mm
            d += step_mm
        return max_mm


# --------------------------------------------------------------------------- #
def audit_paths(paths: Sequence[np.ndarray], trace_clearance: float,
                self_clearance: float) -> Tuple[int, float, float]:
    """Brute-force final check, deliberately not sharing code with the spatial
    hash so it can catch hash bugs as well as mask bugs.

    Uses the SAME adjacency-only self-exemption as the live checker.
    Returns (violation_count, min_inter_trace_distance, min_self_distance)."""
    segs = []
    for tid, pts in enumerate(paths):
        for k in range(len(pts) - 1):
            segs.append((tid, k,
                         (float(pts[k][0]), float(pts[k][1])),
                         (float(pts[k + 1][0]), float(pts[k + 1][1]))))
    violations = 0
    min_inter = math.inf
    min_self = math.inf
    n = len(segs)
    for i in range(n):
        ti, si, a1, b1 = segs[i]
        for j in range(i + 1, n):
            tj, sj, a2, b2 = segs[j]
            if ti == tj:
                if abs(si - sj) <= 1:
                    continue
                d = seg_seg_dist(a1, b1, a2, b2)
                if d < min_self:
                    min_self = d
                if d < self_clearance - 1e-6:
                    violations += 1
            else:
                d = seg_seg_dist(a1, b1, a2, b2)
                if d < min_inter:
                    min_inter = d
                if d < trace_clearance - 1e-6:
                    violations += 1
    inf = float("inf")
    return violations, (min_inter if math.isfinite(min_inter) else inf), \
           (min_self if math.isfinite(min_self) else inf)
