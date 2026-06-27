"""Diverse top-K tracker for the trace-growth env.

Parallels DiverseSolutionTracker but adapted to TraceGrowEnv:

  * Solutions are full trace POLYLINES + endpoints, not endpoint-only
    placements routed post-hoc. The env already produced the geometry, so no
    A* re-routing is needed for rendering.
  * Ranking is by endpoint spacing DESCENDING only (length equality is
    structural, so spread is not a ranking term). Total length is a tiebreaker
    (shorter preferred).
  * Diversity is measured on ENDPOINT positions, matching what an engineer
    chooses between (per the design discussion -- endpoint diversity, not
    path-shape diversity).
  * Only fully-complete, valid solutions (all traces grown, endpoints legal,
    spacing >= TP_TO_TP_MIN) qualify for the portfolio.
"""

import json
import math
import pathlib
from typing import List, Optional

import numpy as np


def _match_distances(a_pts, b_pts):
    """Greedy nearest-neighbor match between two endpoint sets; returns the
    matched distances (or None if counts differ)."""
    if a_pts is None or b_pts is None or len(a_pts) != len(b_pts):
        return None
    a = np.array(a_pts, dtype=float)
    b = np.array(b_pts, dtype=float)
    used = set()
    dists = []
    for p in a:
        best_j, best_d = None, float("inf")
        for j in range(len(b)):
            if j in used:
                continue
            d = np.hypot(p[0] - b[j][0], p[1] - b[j][1])
            if d < best_d:
                best_d, best_j = d, j
        used.add(best_j)
        dists.append(best_d)
    return dists


def _rank_key(s):
    # spacing DESC (so negate), then total_length ASC as tiebreaker
    return (-s["min_tp_spacing"], s["total_length"])


def _is_better(cand, other):
    return _rank_key(cand) < _rank_key(other)


class DiverseGrowTracker:
    def __init__(self, logdir, board, k=5,
                 min_point_shift=13.0, min_moved_frac=0.5):
        self.logdir = pathlib.Path(logdir)
        self.board = board
        self.k = k
        self.min_point_shift = min_point_shift
        self.min_moved_frac = min_moved_frac
        self.solutions: List[dict] = []
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

    def _is_diverse_from(self, candidate, other) -> bool:
        dists = _match_distances(candidate["endpoints"], other["endpoints"])
        if dists is None:
            return True
        if min(dists) < self.min_point_shift:
            return False
        moved = sum(1 for d in dists if d >= self.min_point_shift)
        if moved < math.ceil(self.min_moved_frac * len(dists)):
            return False
        return True

    def update(self, inner_env, step: int, source: str = "train") -> Optional[str]:
        m = getattr(inner_env, "_terminal_metrics", None)
        if not m:
            return None

        # Gate: require completion (all traces fully grown -- an incomplete
        # layout isn't a usable solution) and no path crossings. We do NOT
        # require the 13mm spacing / 14mm edge thresholds here -- instead we
        # record the actual spacing and a 'meets_spec' label. The portfolio
        # holds the model's best work ranked by spacing; the threshold is a
        # label you can read off, not a filter that hides near-misses.
        if m.get("all_complete", 0.0) != 1.0:
            return None  # incomplete layout, not usable

        # Reject solutions with path crossings -- the mask prevents most but
        # sequential routing can produce rare violations where trace A lays a
        # segment that trace B later crosses. Don't put these in the portfolio.
        from envs.pcb_grow_env import TRACE_PATH_CLEARANCE
        paths = inner_env.get_paths()
        for ti, pi in enumerate(paths):
            for tj, pj in enumerate(paths):
                if tj <= ti:
                    continue
                for k in range(len(pi) - 1):
                    for m2 in range(len(pj) - 1):
                        d = inner_env._point_seg_dist(
                            pi[k][0], pi[k][1], pj[m2], pj[m2 + 1]
                        )
                        if d < TRACE_PATH_CLEARANCE:
                            return None  # has a path crossing, reject

        endpoints = inner_env.get_endpoints().tolist()
        candidate = {
            "step": int(step),
            "source": source,
            "min_tp_spacing": float(m["min_tp_spacing"]),
            "mean_tp_spacing": float(m.get("mean_tp_spacing", 0.0)),
            "total_length": float(m["total_length"]),
            "length_spread": float(m["length_spread"]),
            "reward_terminal": float(m["reward_spacing"] + m["reward_gate"]),
            # meets_spec is a LABEL, not a filter: True if this layout clears
            # the 13mm spacing + 14mm edge thresholds. Lets you see at a glance
            # which portfolio entries are manufacturable as-is.
            "meets_spec": bool(m.get("routable", 0.0) == 1.0),
            "endpoints": [list(map(float, p)) for p in endpoints],
            "paths": [[list(map(float, pt)) for pt in path]
                      for path in inner_env.get_paths()],
        }

        similar_idx = [i for i, s in enumerate(self.solutions)
                       if not self._is_diverse_from(candidate, s)]

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
        self._persist()
        spc = ", ".join(f"{s['min_tp_spacing']:.1f}" for s in self.solutions)
        return f"portfolio now {len(self.solutions)} layouts, spacings=[{spc}]"

    def _persist(self):
        summary = {
            "k": self.k,
            "min_point_shift": self.min_point_shift,
            "min_moved_frac": self.min_moved_frac,
            "solutions": self.solutions,
        }
        (self.outdir / "summary.json").write_text(json.dumps(summary, indent=2))
        for stale in self.outdir.glob("solution_*.json"):
            stale.unlink()
        for stale in self.outdir.glob("solution_*.png"):
            stale.unlink()
        for rank, s in enumerate(self.solutions):
            (self.outdir / f"solution_{rank}.json").write_text(json.dumps(s, indent=2))
            try:
                self._render(rank, s)
            except Exception as e:
                print(f"[grow-tracker] render of solution {rank} failed: {e}")

    def _render(self, rank, s):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 7))
        b = self.board
        # board outline
        ax.plot([b.x_min, b.x_max, b.x_max, b.x_min, b.x_min],
                [b.y_min, b.y_min, b.y_max, b.y_max, b.y_min],
                "k-", lw=2)
        # connector
        if b.connector_w > 0:
            ax.add_patch(plt.Rectangle((b.connector_x, b.connector_y),
                                       b.connector_w, b.connector_h,
                                       fill=True, color="violet", alpha=0.3,
                                       zorder=2))
            # outline so it's distinct from traces
            ax.add_patch(plt.Rectangle((b.connector_x, b.connector_y),
                                       b.connector_w, b.connector_h,
                                       fill=False, edgecolor="purple",
                                       linewidth=1.5, linestyle="--", zorder=3))
        # obstacles
        for obs in b.rect_obstacles:
            xn, yn, xx, yx = obs.bounds
            ax.add_patch(plt.Rectangle((xn, yn), xx - xn, yx - yn,
                                       fill=True, color="red", alpha=0.3))
        for obs in b.circ_obstacles:
            ax.add_patch(plt.Circle((obs.cx, obs.cy), obs.radius,
                                    fill=True, color="green", alpha=0.5))
        # traces
        cmap = plt.get_cmap("tab10")
        for ti, path in enumerate(s["paths"]):
            arr = np.array(path)
            ax.plot(arr[:, 0], arr[:, 1], "-", color=cmap(ti % 10), lw=1.5, zorder=4)
            # endpoint (test-point) — large filled circle
            ax.plot(arr[-1, 0], arr[-1, 1], "o", color=cmap(ti % 10),
                    markersize=9, markeredgecolor="k", zorder=6)
            # pin origin inside connector — tiny dot (may overlap others)
            ax.plot(arr[0, 0], arr[0, 1], ".", color=cmap(ti % 10),
                    markersize=4, zorder=5)

        # Mark connector pin cluster with a single label rather than per-trace
        # markers (all pins are within ~3mm of each other, sub-pixel at board scale)
        conn_cx = b.connector_x + b.connector_w / 2
        conn_cy = b.connector_y + b.connector_h / 2
        ax.annotate("pins", xy=(conn_cx, conn_cy),
                    fontsize=7, color="purple", ha="center", va="center",
                    fontweight="bold", zorder=7)
        ax.set_xlim(b.x_min - 5, b.x_max + 5)
        ax.set_ylim(b.y_min - 5, b.y_max + 5)
        ax.set_aspect("equal")
        spec_tag = "MEETS SPEC" if s.get("meets_spec", False) else "below spec"
        ax.set_title(f"Solution {rank} (0=best): min={s['min_tp_spacing']:.1f}mm "
                     f"mean={s.get('mean_tp_spacing',0):.1f}mm  [{spec_tag}]\n"
                     f"len={s['total_length']:.0f}mm, spread={s['length_spread']:.2f}")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        fig.savefig(str(self.outdir / f"solution_{rank}.png"),
                    dpi=110, bbox_inches="tight")
        plt.close(fig)
