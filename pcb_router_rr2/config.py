"""Single source of truth for every tunable parameter.

See reward_spec_v5_fixed_budget.md and environment_spec.md for the reasoning
behind each value.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import yaml


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Board geometry (mm)
    # ------------------------------------------------------------------ #
    board_width_mm: float = 200.0
    board_height_mm: float = 150.0
    edge_clearance_mm: float = 2.0
    connector_rect: Tuple[float, float, float, float] = (78.0, 38.0, 122.0, 50.0)

    # One pin per physical pad pair, ON the connector edge, separated beyond
    # trace_clearance_mm (both rules enforced by validate()).
    pins: List[Tuple[float, float]] = field(default_factory=lambda: [
        (83.5, 50.0), (89.0, 50.0), (94.5, 50.0), (100.0, 50.0),
        (105.5, 50.0), (111.0, 50.0), (116.5, 50.0),          # 7 on top edge
        (85.0, 38.0), (100.0, 38.0), (115.0, 38.0),           # 3 on bottom edge
    ])
    obstacles: List[Tuple[float, float, float, float]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Clearances (mm)
    # ------------------------------------------------------------------ #
    trace_clearance_mm: float = 1.33
    # Trace vs its OWN older path, HARD floor.  MUST be < step_mm: with
    # adjacency-based exemption two segments separated by one intervening
    # segment sit exactly step_mm apart even on a dead-straight run.
    self_clearance_mm: float = 0.60
    obstacle_clearance_mm: float = 1.00

    # ------------------------------------------------------------------ #
    # Growth
    # ------------------------------------------------------------------ #
    # 2 mm steps halve the episode vs 1 mm: better credit assignment, ~2x
    # throughput, and a wider minimum hairpin.  Measured on this board, the
    # coarser reachable set costs ~4% of legal directions per step but
    # improves random-policy completion several-fold, because boxing in is
    # dominated by self-trapping and there are half as many chances to do it.
    step_mm: float = 2.0
    budget_mm: float = 100.0          # SINGLE fixed budget (no range)
    n_dirs: int = 16                  # 16 headings, 22.5 deg apart
    ban_reverse: bool = True
    use_breakout: bool = False        # agent grows directly from the pins

    # ------------------------------------------------------------------ #
    # Breakout (only when use_breakout=True)
    # ------------------------------------------------------------------ #
    breakout_lane_jog_mm: float = 1.5
    breakout_clear_margin_mm: float = 3.0
    breakout_fan_pitch_mm: float = 8.0
    breakout_fan_step_mm: float = 1.0
    breakout_fan_safety_mm: float = 0.05
    breakout_runout_mm: float = 4.0
    breakout_mode: str = "auto"
    breakout_split_min_room_mm: float = 25.0

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    n_rays: int = 16
    ray_max_mm: float = 20.0
    sort_tip_distances: bool = True   # nearest other trace always in slot 0

    # ------------------------------------------------------------------ #
    # Dense reward — per-episode TOTALS, divided by round/step count
    # ------------------------------------------------------------------ #
    spacing_dense_total: float = 0.80
    dense_spacing_target_mm: float = 16.0

    # Constriction: charged at round end when the most boxed-in trace has
    # fewer than `constriction_comfort` legal directions.  Exists because the
    # dominant failure was one trace walling off another, and a terminal-only
    # signal arrives hundreds of steps after the harmful move.
    constriction_penalty_total: float = 0.40
    constriction_comfort: int = 4

    path_penalty_total: float = 0.25
    path_soft_mm: float = 4.0

    # Self penalty: deliberately weak.  Meandering is a DESIRED strategy
    # (absorbs surplus length locally instead of travelling into other
    # traces' space).  This term only prevents pathologically tight coils.
    self_penalty_total: float = 0.15
    self_soft_mm: float = 3.0
    self_lookback_mm: float = 12.0    # must be >> self_soft_mm; see validate()

    # Turn penalty with a FREE BAND: turns of +/- turn_free_units cost
    # nothing, so approximating an intermediate heading by alternating
    # between adjacent directions is not charged as "turning".
    turn_penalty_total: float = 0.30
    turn_free_units: int = 1

    # Board-edge penalty REMOVED: hard edge clearance already guarantees
    # manufacturability and perimeter routing is legitimate.
    edge_penalty_total: float = 0.0
    edge_soft_mm: float = 8.0

    # ------------------------------------------------------------------ #
    # Terminal reward (gated on completion + zero violations)
    #   terminal = base
    #            + spacing_reward_coeff * sqrt(min_endpoint_spacing_mm)
    #            + w_path_clearance_bonus * capped clearance quality
    # ------------------------------------------------------------------ #
    w_terminal_base: float = 12.0     # risk dial: raise if gate_pass < 40%
    spacing_reward_coeff: float = 1.50    # UNCAPPED, concave (sqrt)
    w_path_clearance_bonus: float = 1.50
    terminal_clearance_target_mm: float = 14.0
    w_endpoint_edge_bonus: float = 0.0    # REMOVED (lost to uncapped spacing)
    terminal_edge_target_mm: float = 15.0
    endpoint_spec_mm: float = 13.0        # LABEL + ranking tier only

    # ------------------------------------------------------------------ #
    # Portfolio — 5 slots, no budget bands (one fixed budget)
    # ------------------------------------------------------------------ #
    portfolio_k: int = 5
    min_moved_frac: float = 0.6
    min_point_shift_mm: float = 20.0
    portfolio_dir: str = "portfolio"

    # ------------------------------------------------------------------ #
    # Training / PPO
    # ------------------------------------------------------------------ #
    total_timesteps: int = 8_000_000
    n_envs: int = 8
    seed: int = 0
    learning_rate: float = 3e-4
    lr_anneal: bool = True            # linear decay to zero
    n_steps: int = 512
    batch_size: int = 512
    n_epochs: int = 6
    gamma: float = 0.999              # 1/(1-g)=1000 vs 500-step episodes
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    clip_range: float = 0.2
    max_grad_norm: float = 0.5
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    device: str = "auto"

    # ------------------------------------------------------------------ #
    # Exploration / eval / logging / checkpointing
    # ------------------------------------------------------------------ #
    explorer_every_episodes: int = 200
    explorer_episodes: int = 5
    explorer_momentum: float = 0.95
    eval_every_steps: int = 100_000
    eval_episodes: int = 5
    render_every_episodes: int = 250
    log_every_episodes: int = 10
    checkpoint_every_steps: int = 250_000
    early_stop_patience_evals: int = 0    # 0 = disabled

    wandb_project: str = "pcb-routing"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    out_dir: str = "runs"

    # ------------------------------------------------------------------ #
    @property
    def n_traces(self) -> int:
        return len(self.pins)

    @property
    def budget_rounds(self) -> int:
        return max(1, int(round(self.budget_mm / self.step_mm)))

    @property
    def episode_steps(self) -> int:
        return self.budget_rounds * self.n_traces

    @property
    def board_diag_mm(self) -> float:
        return float((self.board_width_mm ** 2 + self.board_height_mm ** 2) ** 0.5)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise KeyError(f"Unknown config keys: {sorted(unknown)}")
        d = dict(d)
        for k in ("pins", "obstacles"):
            if k in d:
                d[k] = [tuple(x) for x in d[k]]
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
        d = self.to_dict()
        for k, v in kwargs.items():
            if k not in d:
                raise KeyError(f"Unknown config key: {k}")
            d[k] = v
        return Config.from_dict(d)

    def validate(self) -> None:
        assert self.step_mm > 0 and self.budget_mm > 0
        assert self.n_traces >= 2, "need at least two traces"
        assert self.n_dirs >= 8 and self.n_dirs % 4 == 0, \
            "n_dirs must be a multiple of 4 and at least 8"
        assert self.self_clearance_mm < self.step_mm, (
            f"self_clearance_mm ({self.self_clearance_mm}) must be < step_mm "
            f"({self.step_mm}): with adjacency-based self exemption, two "
            f"segments separated by one intervening segment sit exactly "
            f"step_mm apart even on a straight run, so a larger value would "
            f"make straight-line growth illegal")
        assert self.self_soft_mm < self.self_lookback_mm - self.step_mm, (
            f"self_soft_mm ({self.self_soft_mm}) must be well below "
            f"self_lookback_mm ({self.self_lookback_mm}): the distance to a "
            f"point L mm back along a path is at most L, so a threshold near "
            f"the lookback fires constantly on straight traces. This coupling "
            f"has caused the same bug twice")
        assert 0 <= self.turn_free_units < self.n_dirs // 2
        w, h = self.board_width_mm, self.board_height_mm
        for i, (x, y) in enumerate(self.pins):
            assert 0 < x < w and 0 < y <= h, f"pin {i} ({x},{y}) outside board"
        x0, y0, x1, y1 = self.connector_rect
        assert x0 < x1 and y0 < y1, "connector_rect must be (x0,y0,x1,y1)"
        for i in range(len(self.pins)):
            for j in range(i + 1, len(self.pins)):
                d = math.dist(self.pins[i], self.pins[j])
                assert d > self.trace_clearance_mm, (
                    f"pins {i} and {j} are {d:.2f} mm apart, not beyond "
                    f"trace_clearance_mm={self.trace_clearance_mm}")
        eps = 1e-6
        for i, (x, y) in enumerate(self.pins):
            assert not ((x0 + eps < x < x1 - eps) and (y0 + eps < y < y1 - eps)), (
                f"pin {i} ({x},{y}) is strictly inside the connector footprint; "
                f"place pins on the edge so traces can escape without crossing it")
        assert self.portfolio_k >= 1
        assert self.spacing_reward_coeff > 0
        # gamma must cover the episode
        horizon = 1.0 / max(1e-9, 1.0 - self.gamma)
        if horizon < self.episode_steps:
            raise AssertionError(
                f"gamma={self.gamma} gives a planning horizon of {horizon:.0f} "
                f"steps but episodes are {self.episode_steps} steps. The "
                f"terminal reward would be beyond the agent's sight. Use "
                f"gamma >= {1 - 1/self.episode_steps:.5f}")
