"""Diverse top-K solution portfolio (single fixed budget -> no bands).

Gate    : all traces complete AND zero audited clearance violations.
Ranking : lexicographic (meets_spec DESC, reward_terminal DESC).
Diversity: a candidate must move >= min_moved_frac of its endpoints by
           >= min_point_shift_mm versus every existing entry.

With one fixed budget every board is the same length, so terminal rewards are
directly comparable and a single ranking is honest. The diversity filter now
carries ALL the variety (budget bands previously provided most of it), which
is why the thresholds are tighter than before -- an earlier run filled four
portfolio slots with topologically near-identical layouts.

The portfolio is the product: when the policy collapsed in a previous run it
still held the good boards found near the peak. Every episode from every
source (training, eval, forced exploration) is offered to it.

PNG rendering is decoupled from consider(): call render() on a logging
cadence so re-rendering does not cost wall-clock inside the training loop.
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
        self.updates = 0
        self.considered = 0
        self.gate_passed = 0
        self._rendered_at = -1

    def _is_distinct(self, cand_eps: np.ndarray, entry: Dict[str, Any]) -> bool:
        eps = np.asarray(entry["endpoints"])
        shifts = np.linalg.norm(cand_eps - eps, axis=1)
        need = int(np.ceil(self.cfg.min_moved_frac * len(cand_eps)))
        return int(np.sum(shifts >= self.cfg.min_point_shift_mm)) >= need

    def consider(self, ed: Dict[str, Any], episode: Optional[int] = None,
                 steps: Optional[int] = None) -> bool:
        with self._lock:
            self.considered += 1
            if not ed.get("gate_pass"):
                return False
            self.gate_passed += 1
            ed = dict(ed)
            ed["found_at_episode"] = episode
            ed["found_at_steps"] = steps
            cand_eps = np.asarray(ed["endpoints"])
            cand_score = _score(ed)
            for i, entry in enumerate(self.entries):
                if not self._is_distinct(cand_eps, entry):
                    if cand_score > _score(entry):
                        self.entries[i] = ed
                        self._commit(); return True
                    return False
            if len(self.entries) < self.k:
                self.entries.append(ed); self._commit(); return True
            worst = min(range(len(self.entries)), key=lambda i: _score(self.entries[i]))
            if cand_score > _score(self.entries[worst]):
                self.entries[worst] = ed; self._commit(); return True
            return False

    def _commit(self) -> None:
        self.entries.sort(key=_score, reverse=True)
        self.updates += 1
        self._save_index()

    def _save_index(self) -> None:
        index = []
        for rank, ed in enumerate(self.entries):
            js = os.path.join(self.dir, f"rank_{rank}.json")
            with open(js, "w") as f:
                json.dump(ed, f)
            index.append({
                "rank": rank, "meets_spec": ed["meets_spec"],
                "reward_terminal": ed["reward_terminal"],
                "min_endpoint_spacing_mm": ed["min_endpoint_spacing_mm"],
                "mean_path_clearance_mm": ed["mean_path_clearance_mm"],
                "min_self_distance_mm": ed["min_self_distance_mm"],
                "min_endpoint_edge_mm": ed["min_endpoint_edge_mm"],
                "turn_rate": ed["turn_rate"],
                "turn_reversal_rate": ed["turn_reversal_rate"],
                "p5_path_clearance_mm": ed["p5_path_clearance_mm"],
                "min_freedom": ed["min_freedom"],
                "length_spread_mm": ed["length_spread_mm"],
                "budget_mm": ed["budget_mm"],
                "found_at_episode": ed.get("found_at_episode"),
                "found_at_steps": ed.get("found_at_steps"),
                "png": os.path.join(self.dir, f"rank_{rank}.png"),
                "json": js})
        with open(os.path.join(self.dir, "portfolio.json"), "w") as f:
            json.dump(index, f, indent=2)

    def render(self, force: bool = False) -> List[str]:
        with self._lock:
            if not force and self._rendered_at == self.updates:
                return self.image_paths()
            for rank, ed in enumerate(self.entries):
                png = os.path.join(self.dir, f"rank_{rank}.png")
                fig_to_png_path(episode_figure(
                    self.cfg, ed, title=f"Portfolio rank {rank}",
                    episode=ed.get("found_at_episode"),
                    steps=ed.get("found_at_steps")), png)
            self._rendered_at = self.updates
            return self.image_paths()

    def image_paths(self) -> List[str]:
        return [os.path.join(self.dir, f"rank_{i}.png") for i in range(len(self.entries))]

    def diversity_mm(self) -> float:
        """Mean endpoint displacement between portfolio entries.

        The direct measure of whether the search is finding different layouts
        or polishing one. Near min_point_shift_mm means entries only just
        clear the diversity filter; high means real structural variety.
        Rising best_reward_terminal with flat diversity is the signature of a
        local optimum."""
        if len(self.entries) < 2:
            return 0.0
        eps = [np.asarray(e["endpoints"]) for e in self.entries]
        d = [float(np.mean(np.linalg.norm(eps[i] - eps[j], axis=1)))
             for i in range(len(eps)) for j in range(i + 1, len(eps))]
        return float(np.mean(d))

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            best = _score(self.entries[0]) if self.entries else (0, 0.0)
            return {
                "diversity_mm": self.diversity_mm(),
                "size": len(self.entries), "capacity": self.k,
                "considered": self.considered, "gate_passed": self.gate_passed,
                "best_meets_spec": best[0], "best_reward_terminal": best[1],
                "spec_pass_count": sum(1 for e in self.entries if e["meets_spec"]),
                "best_endpoint_spacing_mm": max(
                    (e["min_endpoint_spacing_mm"] for e in self.entries), default=0.0),
                "worst_reward_terminal": min(
                    (e["reward_terminal"] for e in self.entries), default=0.0)}
