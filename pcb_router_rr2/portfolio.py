"""Diverse top-K solution portfolio.

Gate      : all traces complete AND zero audited clearance violations.
Ranking   : lexicographic — (meets_spec DESC, reward_terminal DESC).
            13 mm spec stays out of the gate/reward, but spec-passing
            layouts always outrank spec-missing ones inside the portfolio.
Diversity : a candidate is a duplicate of an entry unless >= min_moved_frac
            of its endpoints each moved >= min_point_shift_mm vs that entry.
            Duplicates replace the entry they collide with only if better.

The portfolio is a passive observer: every finished episode (training,
eval, forced-explorer) is offered to it.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .rendering import episode_figure, fig_to_png_path


def _score(ed: Dict[str, Any]) -> Tuple[int, float]:
    return (1 if ed["meets_spec"] else 0, float(ed["reward_terminal"]))


class Portfolio:
    def __init__(self, cfg: Config, out_dir: str):
        self.cfg = cfg
        self.k = cfg.portfolio_k
        self.dir = out_dir
        os.makedirs(self.dir, exist_ok=True)
        self.entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.updates = 0          # bumped whenever contents change
        self.considered = 0
        self.gate_passed = 0

    # ------------------------------------------------------------------ #
    def _is_distinct(self, cand_eps: np.ndarray, entry: Dict[str, Any]) -> bool:
        eps = np.asarray(entry["endpoints"])
        shifts = np.linalg.norm(cand_eps - eps, axis=1)
        moved = int(np.sum(shifts >= self.cfg.min_point_shift_mm))
        need = int(np.ceil(self.cfg.min_moved_frac * len(cand_eps)))
        return moved >= need

    def consider(self, ed: Dict[str, Any]) -> bool:
        """Offer an episode; returns True if the portfolio changed."""
        with self._lock:
            self.considered += 1
            if not ed.get("gate_pass"):
                return False
            self.gate_passed += 1
            cand_eps = np.asarray(ed["endpoints"])
            cand_score = _score(ed)

            # duplicate handling: collide with the first non-distinct entry
            for i, entry in enumerate(self.entries):
                if not self._is_distinct(cand_eps, entry):
                    if cand_score > _score(entry):
                        self.entries[i] = ed
                        self._resort_and_save()
                        return True
                    return False

            if len(self.entries) < self.k:
                self.entries.append(ed)
                self._resort_and_save()
                return True

            worst = min(range(len(self.entries)),
                        key=lambda i: _score(self.entries[i]))
            if cand_score > _score(self.entries[worst]):
                self.entries[worst] = ed
                self._resort_and_save()
                return True
            return False

    # ------------------------------------------------------------------ #
    def _resort_and_save(self) -> None:
        self.entries.sort(key=_score, reverse=True)
        self.updates += 1
        self._save()

    def _save(self) -> None:
        index = []
        for rank, ed in enumerate(self.entries):
            png = os.path.join(self.dir, f"rank_{rank}.png")
            fig = episode_figure(self.cfg, ed, title=f"Portfolio rank {rank}")
            fig_to_png_path(fig, png)
            entry_json = os.path.join(self.dir, f"rank_{rank}.json")
            with open(entry_json, "w") as f:
                json.dump(ed, f)
            index.append({
                "rank": rank,
                "meets_spec": ed["meets_spec"],
                "reward_terminal": ed["reward_terminal"],
                "min_endpoint_spacing_mm": ed["min_endpoint_spacing_mm"],
                "budget_mm": ed["budget_mm"],
                "length_spread_mm": ed["length_spread_mm"],
                "png": png, "json": entry_json,
            })
        with open(os.path.join(self.dir, "portfolio.json"), "w") as f:
            json.dump(index, f, indent=2)

    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        with self._lock:
            best = _score(self.entries[0]) if self.entries else (0, 0.0)
            return {
                "size": len(self.entries),
                "considered": self.considered,
                "gate_passed": self.gate_passed,
                "best_meets_spec": best[0],
                "best_reward_terminal": best[1],
                "spec_pass_count": sum(1 for e in self.entries if e["meets_spec"]),
            }

    def image_paths(self) -> List[str]:
        return [os.path.join(self.dir, f"rank_{i}.png")
                for i in range(len(self.entries))]
