"""Single source of truth for every tunable parameter."""
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

    # Connector footprint rectangle (x0, y0, x1, y1) — a keep-out region.
    connector_rect: Tuple[float, float, float, float] = (78.0, 38.0, 122.0, 50.0)

    # Pin start points (x, y): one representative per physical pad pair.
    # Pins sit ON the connector footprint edge (real connector pads) and are
    # separated beyond trace_clearance_mm, because with no breakout the very
    # first agent segments must already respect trace-to-trace clearance.
    # Both rules are enforced by validate().
    pins: List[Tuple[float, float]] = field(default_factory=lambda: [
        (83.5, 50.0), (89.0, 50.0), (94.5, 50.0), (100.0, 50.0),
        (105.5, 50.0), (111.0, 50.0), (116.5, 50.0),          # 7 on top edge
        (85.0, 38.0), (100.0, 38.0), (115.0, 38.0),           # 3 on bottom edge
    ])

    obstacles: List[Tuple[float, float, float, float]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Clearances (mm)
    # ------------------------------------------------------------------ #
    trace_clearance_mm: float = 1.33        # between DIFFERENT traces
    # Trace vs its OWN older path.  MUST be < step_mm: with adjacency-based
    # exemption (see geometry) two segments separated by one intervening
    # 1 mm segment are exactly 1 mm apart even on a perfectly straight run,
    # so any value >= step_mm would make straight lines illegal.
    self_clearance_mm: float = 0.60
    obstacle_clearance_mm: float = 1.00

    # ------------------------------------------------------------------ #
    # Growth mechanics
    # ------------------------------------------------------------------ #
    step_mm: float = 1.0
    budget_min_mm: float = 75.0
    budget_max_mm: float = 110.0
    ban_reverse: bool = True
    # No-breakout mode (default): the agent starts directly at the pins.
    # While a tip is inside a keep-out clearance halo, only moves that
    # monotonically INCREASE distance from that keep-out are valid; once
    # outside, full clearance applies permanently.
    use_breakout: bool = False

    # ------------------------------------------------------------------ #
    # Breakout (only used when use_breakout=True)
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

    # ------------------------------------------------------------------ #
    # Reward — DENSE
    # Every dense term is specified as a TOTAL budget for a whole episode
    # and divided by the episode's round/step count at reset.  This makes
    # the dense/terminal balance invariant to the sampled growth budget;
    # previously dense scaled with episode length while terminal did not,
    # which inverted the objective at long budgets.
    # ------------------------------------------------------------------ #
    spacing_dense_total: float = 0.80       # max total spacing reward / episode
    dense_spacing_target_mm: float = 16.0   # hinge saturation for the DENSE term
    path_penalty_total: float = 0.25        # max total other-trace crowding penalty
    path_soft_mm: float = 4.0
    self_penalty_total: float = 0.25        # max total SELF-crowding penalty
    self_soft_mm: float = 4.0
    turn_penalty_total: float = 0.30        # max total turning penalty
    edge_penalty_total: float = 0.20        # max total edge-proximity penalty
    edge_soft_mm: float = 8.0

    # ------------------------------------------------------------------ #
    # Reward — TERMINAL (gated on completion + zero violations)
    #
    #   terminal = w_terminal_base                       (flat, for finishing)
    #            + spacing_reward_per_mm * min_endpoint_spacing_mm   (UNCAPPED)
    #            + w_path_clearance_bonus * capped_clearance_quality
    #            + w_endpoint_edge_bonus  * capped_edge_quality
    #
    # Endpoint spacing is UNCAPPED and LINEAR: it is a MINIMUM over endpoint
    # pairs, so every extra millimetre corresponds to a real improvement in
    # the worst pair, and it is the quantity the project actually optimises.
    # A cap here previously pinned q_spacing at 1.0 for every board, which
    # both froze the portfolio ranking and removed the policy's gradient.
    #
    # The other two stay CAPPED, deliberately:
    #  * path clearance is a clipped MEAN over per-step samples. The clip is
    #    what makes it measure crowding rather than spread -- uncapped, the
    #    mean is dominated by traces in open board space and a wandering
    #    trace can outscore a tidily spaced layout. Cap raised 8 -> 14 mm
    #    because clearance does keep genuinely improving past 8 mm.
    #  * endpoint edge distance is a constraint: past probe-access range,
    #    further is worth nothing, and rewarding it would pull endpoints
    #    inward, fighting directly against endpoint spacing.
    # ------------------------------------------------------------------ #
    w_terminal_base: float = 5.0
    spacing_reward_per_mm: float = 0.30     # UNCAPPED linear, reward pts per mm
    w_path_clearance_bonus: float = 1.50    # max pts from path clearance
    w_endpoint_edge_bonus: float = 1.00     # max pts from endpoint edge standoff
    terminal_clearance_target_mm: float = 14.0
    terminal_edge_target_mm: float = 15.0

    endpoint_spec_mm: float = 13.0          # LABEL + ranking tier only

    # ------------------------------------------------------------------ #
    # Portfolio
    # ------------------------------------------------------------------ #
    # [budget_min, budget_max] is split into portfolio_bands equal bands and
    # each band holds up to portfolio_per_band layouts (total =
    # bands x per_band).  Stratification gives variety ACROSS length; the
    # diversity filter below gives variety ACROSS shape WITHIN each band --
    # without it a band's slots fill with near-identical layouts.
    portfolio_bands: int = 5
    portfolio_per_band: int = 5
    portfolio_stratify_by_budget: bool = True
    min_moved_frac: float = 0.5             # >= this frac of endpoints must move...
    min_point_shift_mm: float = 13.0        # ...by >= this vs every entry in the band
    portfolio_dir: str = "portfolio"

    # ------------------------------------------------------------------ #
    # Training / PPO
    # ------------------------------------------------------------------ #
    total_timesteps: int = 4_000_000
    n_envs: int = 8
    seed: int = 0
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 512
    n_epochs: int = 6
    # Effective planning horizon 1/(1-gamma) must cover the episode length
    # (n_traces x budget).  At 10 traces x 110 mm = 1100 steps, gamma=0.999
    # gives a 1000-step horizon; 0.995 would give only 200.
    gamma: float = 0.999
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    clip_range: float = 0.2
    max_grad_norm: float = 0.5
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    device: str = "auto"

    # ------------------------------------------------------------------ #
    # Exploration / eval / logging
    # ------------------------------------------------------------------ #
    explorer_every_episodes: int = 200
    explorer_episodes: int = 5
    explorer_momentum: float = 0.95
    eval_every_steps: int = 100_000
    eval_episodes: int = 8
    render_every_episodes: int = 250
    log_every_episodes: int = 10

    wandb_project: str = "pcb-routing"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    out_dir: str = "runs"

    # ------------------------------------------------------------------ #
    @property
    def n_traces(self) -> int:
        return len(self.pins)

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
        assert self.self_clearance_mm < self.step_mm, (
            f"self_clearance_mm ({self.self_clearance_mm}) must be < step_mm "
            f"({self.step_mm}): with adjacency-based self exemption, two "
            f"segments separated by one intervening segment sit exactly "
            f"step_mm apart even on a straight run, so a larger value would "
            f"make straight-line growth illegal")
        w, h = self.board_width_mm, self.board_height_mm
        for i, (x, y) in enumerate(self.pins):
            assert 0 < x < w and 0 < y <= h, f"pin {i} ({x},{y}) outside board"
        x0, y0, x1, y1 = self.connector_rect
        assert x0 < x1 and y0 < y1, "connector_rect must be (x0,y0,x1,y1)"
        for i in range(len(self.pins)):
            for j in range(i + 1, len(self.pins)):
                d = math.dist(self.pins[i], self.pins[j])
                assert d > self.trace_clearance_mm, (
                    f"pins {i} and {j} are {d:.2f} mm apart, which is not "
                    f"beyond trace_clearance_mm={self.trace_clearance_mm}; "
                    f"widen the pin pitch (each pin represents its pad pair)")
        eps = 1e-6
        for i, (x, y) in enumerate(self.pins):
            inside = (x0 + eps < x < x1 - eps) and (y0 + eps < y < y1 - eps)
            assert not inside, (
                f"pin {i} ({x},{y}) is strictly inside the connector "
                f"footprint {self.connector_rect}; place pins on the "
                f"connector edge so traces can escape without crossing it")
        assert self.portfolio_bands >= 1 and self.portfolio_per_band >= 1
        assert self.spacing_reward_per_mm > 0, (
            "spacing_reward_per_mm is the uncapped linear reward rate for "
            "endpoint spacing and must be positive")
