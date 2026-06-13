"""
Diverse top-K solution tracking for single-board test-point placement.

When training on a single fixed board, the whole run is a search. Rather than
keeping only the single best placement, this keeps the best K *meaningfully
distinct* fully-routable placements found across all episodes -- a portfolio
of alternative layouts an engineer can choose between.

Priority (length matching first, per the problem's requirements):
  solutions are ranked by   (length_spread asc, total_length asc,
                              min_tp_spacing desc)
so the kept set is, first and foremost, the most length-matched layouts.

Diversity: a new solution is only added if it is sufficiently DIFFERENT from
every solution already kept. "Different" is deliberately strict so the top-K
aren't near-duplicates with one point nudged:

  Two placements are considered the SAME layout unless BOTH hold:
    (a) the closest matched pair of test points between them is at least
        `min_point_shift` apart  -- i.e. no test point is essentially shared;
    (b) at least `min_moved_frac` of the test points each moved by at least
        `min_point_shift`        -- i.e. it's not just one point that moved.

  Test points are matched as an unordered set via nearest-neighbor assignment
  (a placement is a spatial configuration, not an ordered list).

When a new diverse solution is better than the worst kept one and the set is
full, it replaces the worst (subject to staying mutually diverse).

Outputs (under <logdir>/solutions/):
  solution_0.json ... solution_{K-1}.json   (rank 0 = best)
  solution_0.png  ... solution_{K-1}.png
  summary.json     (ranked overview of the portfolio)
"""

import json
import math
import pathlib
from typing import List, Optional

from envs.visualize import plot_board


def _rank_key(s: dict):
    # length_spread asc, total_length asc, min_tp_spacing desc
    return (s["length_spread"], s["total_length"], -s["min_tp_spacing"])


def _is_better(a: dict, b: dict) -> bool:
    return _rank_key(a) < _rank_key(b)


def _match_distances(tps_a, tps_b):
    """Greedy nearest-neighbor matching between two equal-size point sets.

    Returns the list of matched-pair distances. Greedy (not optimal Hungarian)
    to avoid a scipy dependency; for small K this is more than adequate for a
    similarity heuristic.
    """
    a = [tuple(p) for p in tps_a]
    b = [tuple(p) for p in tps_b]
    if len(a) != len(b) or not a:
        return None
    remaining = list(range(len(b)))
    dists = []
    for pa in a:
        best_j = None
        best_d = float("inf")
        for j in remaining:
            d = math.hypot(pa[0] - b[j][0], pa[1] - b[j][1])
            if d < best_d:
                best_d = d
                best_j = j
        dists.append(best_d)
        remaining.remove(best_j)
    return dists


class DiverseSolutionTracker:
    def __init__(
        self,
        logdir,
        board,
        k: int = 5,
        min_point_shift: float = 13.0,
        min_moved_frac: float = 0.5,
    ):
        self.logdir = pathlib.Path(logdir)
        self.board = board
        self.k = k
        self.min_point_shift = min_point_shift
        self.min_moved_frac = min_moved_frac
        self.solutions: List[dict] = []  # kept sorted best-first
        self.outdir = self.logdir / "solutions"
        self.outdir.mkdir(parents=True, exist_ok=True)
        self._resume()

    def _resume(self):
        summary = self.outdir / "summary.json"
        if summary.exists():
            try:
                self.solutions = json.loads(summary.read_text()).get("solutions", [])
            except Exception:
                self.solutions = []

    def _is_diverse_from(self, candidate: dict, other: dict) -> bool:
        """True if candidate is a meaningfully different layout from `other`."""
        dists = _match_distances(candidate["placed_tps"], other["placed_tps"])
        if dists is None:
            return True
        # (a) no test point essentially shared
        if min(dists) < self.min_point_shift:
            return False
        # (b) enough points each moved substantially
        moved = sum(1 for d in dists if d >= self.min_point_shift)
        if moved < math.ceil(self.min_moved_frac * len(dists)):
            return False
        return True

    def update(self, inner_env, step: int, source: str = "train") -> Optional[str]:
        """Inspect a finished episode. Returns a short status string if the
        portfolio changed (for logging), else None."""
        m = getattr(inner_env, "_terminal_metrics", None)
        if not m or m.get("failures", 1) != 0:
            return None  # only fully-routable solutions qualify

        candidate = {
            "step": int(step),
            "source": source,
            "failures": int(m["failures"]),
            "total_length": float(m["total_length"]),
            "length_spread": float(m["length_spread"]),
            "min_tp_spacing": float(m["min_tp_spacing"]),
            "reward_terminal": float(
                m["reward_routability"] + m["reward_length"]
                + m["reward_spread"] + m["reward_spacing"]
            ),
            "placed_tps": [list(map(float, tp)) for tp in inner_env.placed_tps],
        }

        # Which kept solutions does this candidate share a layout niche with?
        similar_idx = [
            i for i, s in enumerate(self.solutions)
            if not self._is_diverse_from(candidate, s)
        ]

        changed = False
        if similar_idx:
            best_similar = min(similar_idx, key=lambda i: _rank_key(self.solutions[i]))
            if _is_better(candidate, self.solutions[best_similar]):
                for i in sorted(similar_idx, reverse=True):
                    self.solutions.pop(i)
                self.solutions.append(candidate)
                changed = True
        else:
            if len(self.solutions) < self.k:
                self.solutions.append(candidate)
                changed = True
            else:
                worst = max(range(len(self.solutions)),
                            key=lambda i: _rank_key(self.solutions[i]))
                if _is_better(candidate, self.solutions[worst]):
                    self.solutions.pop(worst)
                    self.solutions.append(candidate)
                    changed = True

        if not changed:
            return None

        self.solutions.sort(key=_rank_key)
        self.solutions = self.solutions[: self.k]
        self._persist(inner_env)
        ranks = ", ".join(f"{s['length_spread']:.2f}" for s in self.solutions)
        return (f"portfolio now {len(self.solutions)} layouts, "
                f"spreads=[{ranks}]")

    def _persist(self, inner_env):
        summary = {
            "k": self.k,
            "min_point_shift": self.min_point_shift,
            "min_moved_frac": self.min_moved_frac,
            "solutions": self.solutions,
        }
        (self.outdir / "summary.json").write_text(json.dumps(summary, indent=2))

        # Clear stale solution files beyond current count.
        for stale in self.outdir.glob("solution_*.json"):
            stale.unlink()
        for stale in self.outdir.glob("solution_*.png"):
            stale.unlink()

        for rank, s in enumerate(self.solutions):
            (self.outdir / f"solution_{rank}.json").write_text(json.dumps(s, indent=2))
            try:
                paths, _, _ = _route_for_render(inner_env, s["placed_tps"])
                plot_board(
                    self.board,
                    test_points=[tuple(p) for p in s["placed_tps"]],
                    paths=paths,
                    candidates=inner_env.candidates[:inner_env._real_count],
                    title=(f"Solution {rank} (0=best): "
                           f"spread={s['length_spread']:.2f}, "
                           f"len={s['total_length']:.0f}mm, "
                           f"spacing={s['min_tp_spacing']:.1f}mm"),
                    filename=str(self.outdir / f"solution_{rank}.png"),
                )
            except Exception as e:
                print(f"[tracker] render of solution {rank} failed: {e}")


def _route_for_render(inner_env, placed_tps):
    """Re-run the env's router on a stored placement to recover paths for
    rendering (placements are stored as coordinates, not paths)."""
    from envs.routing import route_all_traces
    tps = [tuple(p) for p in placed_tps]
    return route_all_traces(inner_env.board, tps)
