"""Board specification: outline, connector, pins, obstacles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import Config
from .geometry import ClearanceChecker, Rect


@dataclass
class Board:
    width: float
    height: float
    edge_clearance: float
    connector_rect: Rect
    pins: np.ndarray
    obstacles: List[Rect]

    @classmethod
    def from_config(cls, cfg: Config) -> "Board":
        return cls(cfg.board_width_mm, cfg.board_height_mm, cfg.edge_clearance_mm,
                   tuple(cfg.connector_rect),
                   np.asarray(cfg.pins, dtype=np.float64),
                   [tuple(o) for o in cfg.obstacles])

    @property
    def n_traces(self) -> int:
        return len(self.pins)

    @property
    def keepout_rects(self) -> List[Rect]:
        return [self.connector_rect] + list(self.obstacles)

    def make_checker(self, cfg: Config) -> ClearanceChecker:
        return ClearanceChecker(
            board_w=self.width, board_h=self.height,
            edge_clearance=cfg.edge_clearance_mm,
            keepout_rects=self.keepout_rects,
            obstacle_clearance=cfg.obstacle_clearance_mm,
            trace_clearance=cfg.trace_clearance_mm,
            self_clearance=cfg.self_clearance_mm,
            self_lookback_mm=cfg.self_lookback_mm,
            soft_radius_mm=max(cfg.path_soft_mm, cfg.self_soft_mm))

    def summary(self) -> str:
        L = [f"Board          : {self.width:.1f} x {self.height:.1f} mm "
             f"(edge clearance {self.edge_clearance:.2f} mm)",
             f"Connector      : x[{self.connector_rect[0]:.1f}, {self.connector_rect[2]:.1f}] "
             f"y[{self.connector_rect[1]:.1f}, {self.connector_rect[3]:.1f}]",
             f"Traces / pins  : {self.n_traces}"]
        for i, (x, y) in enumerate(self.pins):
            L.append(f"    pin {i}: ({x:.2f}, {y:.2f})")
        L.append(f"Obstacles      : {len(self.obstacles) or 'none'}")
        for r in self.obstacles:
            L.append(f"    rect x[{r[0]:.1f},{r[2]:.1f}] y[{r[1]:.1f},{r[3]:.1f}]")
        return "\n".join(L)
