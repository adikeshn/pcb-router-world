"""
Best-solution tracking for single-board test-point placement.

When training on a single fixed board, the whole training run is effectively
a search over placements. The policy's final state isn't necessarily its best
discovery -- it can drift between fixed-points of different quality. This
tracker watches every completed episode (train and eval) and persists the
best VALID placement found so far as the run's "clean final answer",
independent of wherever the policy happens to end up.

Ranking (only valid solutions, failures == 0, are ever recorded):
  1. fewest failures   (must be 0 to qualify at all)
  2. lowest length_spread   (primary objective: length matching)
  3. lowest total_length    (secondary: shorter routing)
  4. highest min_tp_spacing (tiebreak: more robust spacing)

Outputs (under <logdir>/):
  best_solution.json  - metrics + placed test points + step found
  best_solution.png   - rendered board layout of the best placement
"""

import json
import pathlib
from typing import Optional

from envs.visualize import plot_board


def _is_better(candidate: dict, current: Optional[dict]) -> bool:
    """Lexicographic comparison; candidate must already be valid (failures==0)."""
    if current is None:
        return True
    # spread asc, length asc, spacing desc
    c = (candidate["length_spread"], candidate["total_length"], -candidate["min_tp_spacing"])
    b = (current["length_spread"], current["total_length"], -current["min_tp_spacing"])
    return c < b


class BestSolutionTracker:
    def __init__(self, logdir, board):
        self.logdir = pathlib.Path(logdir)
        self.board = board
        self.best: Optional[dict] = None
        self._json_path = self.logdir / "best_solution.json"
        self._png_path = self.logdir / "best_solution.png"
        # Resume an existing best across restarts.
        if self._json_path.exists():
            try:
                self.best = json.loads(self._json_path.read_text())
            except Exception:
                self.best = None

    def update(self, inner_env, step: int, source: str = "train") -> bool:
        """Inspect a just-finished episode's inner env. Returns True if improved.

        `inner_env` is the TPPlacementEnv (post-terminal), exposing
        `_terminal_metrics`, `placed_tps`, and `routed_paths`.
        """
        m = getattr(inner_env, "_terminal_metrics", None)
        if not m:
            return False
        if m.get("failures", 1) != 0:
            return False  # only track fully-routable solutions

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

        if not _is_better(candidate, self.best):
            return False

        self.best = candidate
        self._json_path.write_text(json.dumps(candidate, indent=2))
        try:
            plot_board(
                self.board,
                test_points=inner_env.placed_tps,
                paths=inner_env.routed_paths,
                candidates=inner_env.candidates[:inner_env._real_count],
                title=(f"Best @ step {step} ({source}): "
                       f"len={candidate['total_length']:.0f}mm, "
                       f"spread={candidate['length_spread']:.2f}, "
                       f"spacing={candidate['min_tp_spacing']:.1f}mm"),
                filename=str(self._png_path),
            )
        except Exception as e:
            print(f"[best_solution] render failed: {e}")
        return True
