"""Single source of truth for every tunable parameter.

Everything the notebook exposes as a "hyperparameter" lives here, so the
Colab config cell is just a dict of overrides applied on top of the
defaults below.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import yaml


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Board geometry (mm).  NOTE: the design doc says "180 x 120" but its
    # pin coordinates require height >= 180, so the board is 120 wide (x)
    # by 180 tall (y).  Adjust here if your board differs.
    # ------------------------------------------------------------------ #
    board_width_mm: float = 120.0
    board_height_mm: float = 180.0
    edge_clearance_mm: float = 2.0          # copper-to-board-edge keep-out

    # Connector footprint rectangle (x0, y0, x1, y1) — a keep-out region.
    connector_rect: Tuple[float, float, float, float] = (82.0, 167.0, 98.0, 180.0)

    # Pin start points (x, y).  3-stacked-on-3, 3 mm pitch (representative
    # pitch kept above trace clearance so masks are valid from step one).
    pins: List[Tuple[float, float]] = field(default_factory=lambda: [
        (87.0, 178.5), (90.0, 178.5), (93.0, 178.5),   # top row    (0,1,2)
        (87.0, 171.7), (90.0, 171.7), (93.0, 171.7),   # bottom row (3,4,5)
    ])

    # Extra rectangular keep-out obstacles [(x0, y0, x1, y1), ...]
    obstacles: List[Tuple[float, float, float, float]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Clearances (mm)
    # ------------------------------------------------------------------ #
    trace_clearance_mm: float = 1.33        # between DIFFERENT traces
    self_clearance_mm: float = 0.60         # a trace vs its own older path
    obstacle_clearance_mm: float = 1.00     # trace vs connector/obstacles

    # ------------------------------------------------------------------ #
    # Growth mechanics
    # ------------------------------------------------------------------ #
    step_mm: float = 1.0                    # growth increment per action
    budget_min_mm: float = 25.0             # per-episode sampled growth budget
    budget_max_mm: float = 45.0             #   (post-breakout, per trace)
    self_skip_mm: float = 2.0               # own path arc-length exempt from self check
    ban_reverse: bool = True                # forbid exact 180-degree reversal

    # ------------------------------------------------------------------ #
    # Breakout
    # ------------------------------------------------------------------ #
    breakout_lane_jog_mm: float = 1.5       # lateral jog for top-row lanes
    breakout_clear_margin_mm: float = 3.0   # descend to this far below connector
    breakout_fan_pitch_mm: float = 8.0      # endpoint pitch after fanning
    breakout_fan_step_mm: float = 1.0       # vertical sub-step of the gradual fan
    breakout_fan_safety_mm: float = 0.05    # extra clearance margin during fan
    breakout_runout_mm: float = 4.0         # straight run-out after fan (clean hand-off)

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    n_rays: int = 16
    ray_max_mm: float = 20.0

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #
    # Dense terms (deliberately small; terminal must dominate — see
    # validate.reward_scale_check).
    w_spacing_dense: float = 0.02           # per-round hinge on MIN pairwise tip dist
    spacing_target_mm: float = 16.0         # hinge saturates here (spec 13 + margin)
    w_edge_penalty: float = 0.003           # per-step, tip too close to board edge
    edge_soft_mm: float = 8.0
    w_path_penalty: float = 0.003           # per-step, new segment close to other traces
    path_soft_mm: float = 4.0

    # Terminal terms (gated on full completion + zero violations).
    w_terminal_base: float = 5.0            # flat completion bonus
    w_terminal_quality: float = 5.0         # scales the quality blend below
    q_endpoint_spacing: float = 0.6         # blend: final min endpoint spacing
    q_path_clearance: float = 0.3           # blend: mean path clearance margin
    q_short_budget: float = 0.1             # blend: mild preference for short budgets

    # Fixture spec (LABEL + ranking tier only — never a gate or reward term)
    endpoint_spec_mm: float = 13.0

    # ------------------------------------------------------------------ #
    # Portfolio
    # ------------------------------------------------------------------ #
    portfolio_k: int = 5
    min_moved_frac: float = 0.5             # >= this frac of endpoints must move...
    min_point_shift_mm: float = 13.0        # ...by at least this much vs every entry
    portfolio_dir: str = "portfolio"

    # ------------------------------------------------------------------ #
    # Training / PPO
    # ------------------------------------------------------------------ #
    total_timesteps: int = 2_000_000
    n_envs: int = 8
    seed: int = 0
    learning_rate: float = 3e-4
    n_steps: int = 512                      # per env per rollout
    batch_size: int = 512
    n_epochs: int = 6
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    clip_range: float = 0.2
    max_grad_norm: float = 0.5
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    device: str = "auto"

    # ------------------------------------------------------------------ #
    # Exploration / eval / logging
    # ------------------------------------------------------------------ #
    explorer_every_episodes: int = 200      # run ForcedExplorer every N train episodes
    explorer_episodes: int = 5              # random episodes per burst
    explorer_momentum: float = 0.9          # persistent-walk repeat probability
    eval_every_steps: int = 100_000         # deterministic eval suite cadence
    eval_episodes: int = 8
    render_every_episodes: int = 250        # log a rendered training board every N eps
    log_every_episodes: int = 10            # scalar episode metrics cadence

    wandb_project: str = "pcb-routing"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"              # "online" | "offline" | "disabled"
    out_dir: str = "runs"

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #
    @property
    def n_traces(self) -> int:
        return len(self.pins)

    @property
    def board_diag_mm(self) -> float:
        return float((self.board_width_mm ** 2 + self.board_height_mm ** 2) ** 0.5)

    # ------------------------------------------------------------------ #
    # (De)serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"Unknown config keys: {sorted(unknown)}")
        d = dict(d)
        if "pins" in d:
            d["pins"] = [tuple(p) for p in d["pins"]]
        if "obstacles" in d:
            d["obstacles"] = [tuple(o) for o in d["obstacles"]]
        if "connector_rect" in d:
            d["connector_rect"] = tuple(d["connector_rect"])
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def save_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def override(self, **kwargs) -> "Config":
        """Return a copy with the given fields replaced."""
        d = self.to_dict()
        for k, v in kwargs.items():
            if k not in d:
                raise KeyError(f"Unknown config key: {k}")
            d[k] = v
        return Config.from_dict(d)

    def validate(self) -> None:
        assert self.budget_min_mm <= self.budget_max_mm
        assert self.step_mm > 0
        assert self.n_traces >= 2, "need at least two traces"
        w, h = self.board_width_mm, self.board_height_mm
        for i, (x, y) in enumerate(self.pins):
            assert 0 < x < w and 0 < y <= h, f"pin {i} ({x},{y}) outside board"
        x0, y0, x1, y1 = self.connector_rect
        assert x0 < x1 and y0 < y1, "connector_rect must be (x0,y0,x1,y1)"
