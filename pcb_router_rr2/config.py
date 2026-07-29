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
    # Hard mask limit on how SHARP a single turn may be, in direction units
    # (1 unit = 360/n_dirs deg).  At 2 units a 45 deg turn is legal in one
    # step while a 90 deg turn needs two -- a mitred corner, the PCB idiom.
    # Sharp single-step corners become geometrically impossible.  If no
    # turn-limited direction is legal the limit lifts for that step, so the
    # constraint shapes routing without CAUSING boxing in.
    #
    # Measured with zero_violation_check (random policy, 40 episodes each):
    #   max_turn   legal dirs   completion   frac_survived   escape valve/ep
    #       1           3          10.0%         0.319            13.9
    #       2           5           5.0%         0.327            11.1
    #       3           7           7.5%         0.379             7.0
    #       8 (none)   16           0.0%         0.204             0.0
    # ANY limit hugely beats none -- constraining turns PREVENTS self-trapping,
    # which is the dominant random-policy failure. 2 is chosen over 3 because a
    # 45 deg single-step turn is the mitred-corner idiom while 67.5 deg is a
    # visible kink; both forbid a single-step 90 deg turn. Raise to 3 if
    # boxing-in persists with a trained policy.
    max_turn_units: int = 2
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
    # POSITIVE dense reward is FARMABLE (a policy can collect crumbs while
    # never completing) so it must stay small relative to the terminal.
    # PENALTIES are NOT farmable -- the only thing to do with one is avoid it
    # by routing well -- so the constraint on them is simply that completing
    # while incurring all of them still beats failing.
    spacing_dense_total: float = 0.80
    dense_spacing_target_mm: float = 16.0

    # Constriction: charged at round end when the most boxed-in trace has
    # fewer than `constriction_comfort` legal directions.  Exists because the
    # dominant failure was one trace walling off another, and a terminal-only
    # signal arrives hundreds of steps after the harmful move.
    constriction_penalty_total: float = 1.50
    constriction_comfort: int = 6

    path_penalty_total: float = 0.60
    path_soft_mm: float = 4.0

    # Self penalty: deliberately weak.  Meandering is a DESIRED strategy
    # (absorbs surplus length locally instead of travelling into other
    # traces' space).  This term only prevents pathologically tight coils.
    self_penalty_total: float = 0.60
    self_soft_mm: float = 3.0
    # Lookback must be SMALLER than the arc length of the tightest fold worth
    # catching (a hairpin spans ~6-8mm of arc at 2mm steps), and larger than
    # self_soft_mm + step_mm so straight runs stay silent.  Both bounds are
    # asserted in validate().
    self_lookback_mm: float = 6.0

    # REVERSAL penalty, not a magnitude penalty.  Sustained turning in one
    # direction is a curve (desirable: absorbs surplus length locally).  Only
    # FLIPPING turn direction on consecutive steps is jitter.  Mean turn
    # magnitude scores a smooth arc and pure jitter identically and cannot
    # separate them; reversal rate is 0 for straight runs, smooth arcs and
    # mitred corners alike, and 1.0 for jitter.
    reversal_penalty_total: float = 0.80
    # How many straight steps may separate two turns and still count as a
    # reversal.  0 would let a policy dodge the penalty entirely by inserting
    # one straight step between alternating turns; a large value would flag two
    # legitimate opposite corners separated by a long straight run.
    reversal_memory_steps: int = 2

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
    # Path clearance is measured two ways and averaged: the MEAN dilutes
    # localised crowding, a PERCENTILE has a cliff below its threshold.
    # Splitting the weight covers both without a second tunable.
    w_path_clearance_bonus: float = 3.00
    terminal_clearance_target_mm: float = 14.0
    clearance_tail_pct: float = 5.0        # percentile used for the tail half
    # Jitter cannot be masked away (it is a legal sequence of legal moves) so
    # it varies between boards and must be SCORED, or it has no influence on
    # which boards the portfolio keeps.  Sharpness is capped structurally by
    # max_turn_units and therefore is not scored.
    w_smoothness_bonus: float = 1.50
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
    total_timesteps: int = 2_500_000   # x3 seeds rather than 1 long run
    n_envs: int = 12                   # match CPU core count
    seed: int = 0
    learning_rate: float = 3e-4
    lr_anneal: bool = True
    # Anneal to a FLOOR, not to zero.  Zero freezes the policy into whatever
    # basin it occupies by mid-run, which is counterproductive when escaping
    # a basin is the goal.
    lr_floor: float = 1e-4
    subproc_vecenv: bool = True       # DummyVecEnv steps all envs SERIALLY
    n_steps: int = 512
    batch_size: int = 512
    n_epochs: int = 6
    gamma: float = 0.999              # 1/(1-g)=1000 vs 500-step episodes
    # GAE credit horizon is 1/(1-gamma*lambda). At 0.95 that is 20 steps, so
    # in a 500-step terminal-dominant episode the observed outcome reaches
    # mid-episode with weight ~2e-6 -- early decisions are credited entirely
    # through the critic. 0.98 extends it to ~48 steps.
    gae_lambda: float = 0.98
    ent_coef: float = 0.01
    clip_range: float = 0.2
    max_grad_norm: float = 0.5
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    device: str = "auto"

    # ------------------------------------------------------------------ #
    # Exploration / eval / logging / checkpointing
    # ------------------------------------------------------------------ #
    # Parameter-noise explorer: perturb ACTOR WEIGHTS, run a whole episode.
    # Action noise makes a trace wobble and average back; weight noise makes
    # the policy behave consistently differently for an entire episode, which
    # is the correlated deviation per-step noise cannot produce.
    explorer_every_episodes: int = 200
    explorer_episodes: int = 5
    explorer_sigma: float = 0.05          # initial weight-noise scale
    explorer_target_action_change: float = 0.15   # adapt sigma to hold this
    explorer_min_gate_pass: float = 0.20  # skip until the policy is competent
    # Forced-prefix rollouts: override one trace's first L rounds, then hand
    # control back.  Targets are chosen automatically by stuckness score.
    prefix_every_episodes: int = 500
    prefix_episodes_per_burst: int = 6
    prefix_fractions: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    eval_every_steps: int = 100_000
    # With 5 episodes eval gate-pass is quantised to {0,.2,.4,.6,.8,1} and is
    # too coarse to select model_best on.
    eval_episodes: int = 20
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
        assert 1 <= self.max_turn_units <= self.n_dirs // 2, (
            f"max_turn_units must be in [1, {self.n_dirs // 2}]")
        assert self.self_soft_mm < self.self_lookback_mm - self.step_mm, (
            f"self_soft_mm ({self.self_soft_mm}) must be below "
            f"self_lookback_mm - step_mm ({self.self_lookback_mm - self.step_mm}): "
            f"the distance to a point L mm back along a path is at most L, so a "
            f"straight run measured at {self.self_lookback_mm} mm lookback reports "
            f"{self.self_lookback_mm - self.step_mm} mm. A threshold at or above "
            f"that fires constantly on perfectly good traces")
        assert 0 < self.clearance_tail_pct < 50
        assert 0 < self.lr_floor <= self.learning_rate
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
