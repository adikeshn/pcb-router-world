"""Board specification: outline, connector, pins, obstacles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import Config
from .geometry import ClearanceChecker, Rect


@dataclass
class Board:
    width: float
    height: float
    edge_clearance: float
    connector_rect: Rect
    pins: np.ndarray                 # (n, 2)
    obstacles: List[Rect]

    @classmethod
    def from_config(cls, cfg: Config) -> "Board":
        return cls(
            width=cfg.board_width_mm,
            height=cfg.board_height_mm,
            edge_clearance=cfg.edge_clearance_mm,
            connector_rect=tuple(cfg.connector_rect),
            pins=np.asarray(cfg.pins, dtype=np.float64),
            obstacles=[tuple(o) for o in cfg.obstacles],
        )

    @property
    def n_traces(self) -> int:
        return len(self.pins)

    @property
    def keepout_rects(self) -> List[Rect]:
        return [self.connector_rect] + list(self.obstacles)

    def make_checker(self, cfg: Config) -> ClearanceChecker:
        return ClearanceChecker(
            board_w=self.width,
            board_h=self.height,
            edge_clearance=cfg.edge_clearance_mm,
            keepout_rects=self.keepout_rects,
            obstacle_clearance=cfg.obstacle_clearance_mm,
            trace_clearance=cfg.trace_clearance_mm,
            self_clearance=cfg.self_clearance_mm,
            self_skip_mm=cfg.self_skip_mm,
        )

    def summary(self) -> str:
        lines = [
            f"Board          : {self.width:.1f} x {self.height:.1f} mm "
            f"(edge clearance {self.edge_clearance:.2f} mm)",
            f"Connector      : x[{self.connector_rect[0]:.1f}, {self.connector_rect[2]:.1f}] "
            f"y[{self.connector_rect[1]:.1f}, {self.connector_rect[3]:.1f}]",
            f"Traces / pins  : {self.n_traces}",
        ]
        for i, (x, y) in enumerate(self.pins):
            lines.append(f"    pin {i}: ({x:.2f}, {y:.2f})")
        if self.obstacles:
            lines.append(f"Obstacles      : {len(self.obstacles)}")
            for r in self.obstacles:
                lines.append(f"    rect x[{r[0]:.1f},{r[2]:.1f}] y[{r[1]:.1f},{r[3]:.1f}]")
        else:
            lines.append("Obstacles      : none")
        return "\n".join(lines)
