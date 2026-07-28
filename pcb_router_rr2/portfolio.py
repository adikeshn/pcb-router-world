"""Solution portfolio: budget bands x per-band diverse slots.

Gate    : all traces complete AND zero audited clearance violations.
Ranking : lexicographic (meets_spec DESC, reward_terminal DESC).

STRUCTURE
---------
[budget_min, budget_max] is split into `portfolio_bands` equal bands, and each
band holds up to `portfolio_per_band` layouts (total = bands x per_band).

Two different kinds of variety, deliberately handled by two mechanisms:
  * ACROSS length -- stratification. A 100 mm layout competes only for a slot
    in its own band, never against a 75 mm layout. Previously a `q_short`
    reward term was supposed to express "shorter is better" as a tie-breaker,
    but at 0.5 reward points against a top-5 spread of 0.08 it decided the
    entire ranking and every entry collapsed to the minimum budget.
  * ACROSS shape -- the endpoint-diversity filter, applied WITHIN each band.
    Without it a band's slots simply fill with the N highest-scoring
    near-identical layouts, which defeats the point of having N slots.

This matters more now that endpoint spacing is uncapped: long budgets can
spread further and so systematically outscore short ones. Under a single
global top-K that would collapse the portfolio to all-long -- the mirror
image of the old q_short problem. Stratification makes cross-band score
comparison never happen, so the bias is harmless.

RENDERING
---------
PNG rendering is decoupled from `consider()`. With up to 25 entries and
updates firing every few episodes, re-rendering everything inline would eat
significant wall-clock inside the training loop. `consider()` writes only the
JSON index; call `render()` on a logging cadence or at run end.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Tuple

import numpy as np

from .config import Config
from .rendering import episode_figure, fig_to_png_path


def _score(ed: Dict[str, Any]) -> Tuple[int, float]:
    return (1 if ed["meets_spec"] else 0, float(ed["reward_terminal"]))


class Portfolio:
    def __init__(self, cfg: Config, out_dir: str):
        self.cfg = cfg
        self.dir = out_dir
        os.makedirs(self.dir, exist_ok=True)
        self.n_bands = cfg.portfolio_bands if cfg.portfolio_stratify_by_budget else 1
        self.per_band = (cfg.portfolio_per_band if cfg.portfolio_stratify_by_budget
                         else cfg.portfolio_bands * cfg.portfolio_per_band)
        self.capacity = self.n_bands * self.per_band
        self.bands: List[List[Dict[str, Any]]] = [[] for _ in range(self.n_bands)]
        self.entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.updates = 0
        self.considered = 0
        self.gate_passed = 0
        self._rendered_at = -1
        lo, hi = cfg.budget_min_mm, cfg.budget_max_mm
        self.band_edges = [lo + (hi - lo) * i / self.n_bands for i in range(self.n_bands + 1)]

    # ------------------------------------------------------------------ #
    def band_of(self, budget_mm: float) -> int:
        if self.n_bands == 1:
            return 0
        lo, hi = self.cfg.budget_min_mm, self.cfg.budget_max_mm
        if hi <= lo:
            return 0
        f = (float(budget_mm) - lo) / (hi - lo)
        return int(min(self.n_bands - 1, max(0, int(f * self.n_bands))))

    def _is_distinct(self, cand_eps: np.ndarray, entry: Dict[str, Any]) -> bool:
        eps = np.asarray(entry["endpoints"])
        shifts = np.linalg.norm(cand_eps - eps, axis=1)
        need = int(np.ceil(self.cfg.min_moved_frac * len(cand_eps)))
        return int(np.sum(shifts >= self.cfg.min_point_shift_mm)) >= need

    # ------------------------------------------------------------------ #
    def consider(self, ed: Dict[str, Any]) -> bool:
        with self._lock:
            self.considered += 1
            if not ed.get("gate_pass"):
                return False
            self.gate_passed += 1

            b = self.band_of(ed["budget_mm"])
            slots = self.bands[b]
            cand_eps = np.asarray(ed["endpoints"])
            cand_score = _score(ed)

            # 1. collides with an existing entry in this band -> replace only
            #    if strictly better (keeps the band's shapes distinct)
            for i, entry in enumerate(slots):
                if not self._is_distinct(cand_eps, entry):
                    if cand_score > _score(entry):
                        slots[i] = ed
                        self._commit()
                        return True
                    return False

            # 2. distinct from everything in the band and there is room
            if len(slots) < self.per_band:
                slots.append(ed)
                self._commit()
                return True

            # 3. band full -> displace its weakest entry if better
            worst = min(range(len(slots)), key=lambda i: _score(slots[i]))
            if cand_score > _score(slots[worst]):
                slots[worst] = ed
                self._commit()
                return True
            return False

    # ------------------------------------------------------------------ #
    def _commit(self) -> None:
        flat = [e for band in self.bands for e in band]
        flat.sort(key=_score, reverse=True)
        self.entries = flat
        self.updates += 1
        self._save_index()

    def _entry_row(self, rank: int, ed: Dict[str, Any]) -> Dict[str, Any]:
        b = self.band_of(ed["budget_mm"])
        return {"rank": rank, "budget_band": b,
                "band_range_mm": [round(self.band_edges[b], 1),
                                  round(self.band_edges[b + 1], 1)],
                "meets_spec": ed["meets_spec"],
                "reward_terminal": ed["reward_terminal"],
                "budget_mm": ed["budget_mm"],
                "min_endpoint_spacing_mm": ed["min_endpoint_spacing_mm"],
                "min_endpoint_edge_mm": ed["min_endpoint_edge_mm"],
                "min_self_distance_mm": ed["min_self_distance_mm"],
                "mean_path_clearance_mm": ed["mean_path_clearance_mm"],
                "turn_rate": ed["turn_rate"],
                "length_spread_mm": ed["length_spread_mm"],
                "png": os.path.join(self.dir, f"rank_{rank}.png"),
                "json": os.path.join(self.dir, f"rank_{rank}.json")}

    def _save_index(self) -> None:
        index = []
        for rank, ed in enumerate(self.entries):
            row = self._entry_row(rank, ed)
            with open(row["json"], "w") as f:
                json.dump(ed, f)
            index.append(row)
        with open(os.path.join(self.dir, "portfolio.json"), "w") as f:
            json.dump(index, f, indent=2)

    # ------------------------------------------------------------------ #
    def render(self, force: bool = False) -> List[str]:
        """Render PNGs for the current entries. Skipped when nothing has
        changed since the last render."""
        with self._lock:
            if not force and self._rendered_at == self.updates:
                return self.image_paths()
            for rank, ed in enumerate(self.entries):
                png = os.path.join(self.dir, f"rank_{rank}.png")
                b = self.band_of(ed["budget_mm"])
                fig_to_png_path(episode_figure(
                    self.cfg, ed,
                    title=f"Portfolio rank {rank} — band {b} "
                          f"({self.band_edges[b]:.0f}-{self.band_edges[b+1]:.0f} mm)"), png)
            self._rendered_at = self.updates
            return self.image_paths()

    def image_paths(self) -> List[str]:
        return [os.path.join(self.dir, f"rank_{i}.png") for i in range(len(self.entries))]

    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        with self._lock:
            best = _score(self.entries[0]) if self.entries else (0, 0.0)
            out = {"size": len(self.entries), "capacity": self.capacity,
                   "considered": self.considered, "gate_passed": self.gate_passed,
                   "best_meets_spec": best[0], "best_reward_terminal": best[1],
                   "spec_pass_count": sum(1 for e in self.entries if e["meets_spec"]),
                   "best_endpoint_spacing_mm": max(
                       (e["min_endpoint_spacing_mm"] for e in self.entries), default=0.0)}
            if self.cfg.portfolio_stratify_by_budget:
                out["bands_occupied"] = sum(1 for b in self.bands if b)
                for i, band in enumerate(self.bands):
                    lo, hi = self.band_edges[i], self.band_edges[i + 1]
                    out[f"band{i}_{lo:.0f}_{hi:.0f}mm_count"] = len(band)
                    out[f"band{i}_{lo:.0f}_{hi:.0f}mm_best_spacing"] = (
                        max(e["min_endpoint_spacing_mm"] for e in band) if band else 0.0)
            return out
