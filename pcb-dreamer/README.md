# World-Model-Based Test Point Placement for SI Fixture Routing

## Overview

This project automates test point placement for Signal Integrity (SI) test fixtures on PCBs using DreamerV3, a model-based reinforcement learning algorithm. The agent learns to place test points on a board such that all spacing, clearance, and length-matching constraints are satisfied while minimizing total trace length and maximizing inter-trace spacing.

The core contribution is framing test point placement as a **generative co-design problem** — unlike prior work that optimizes within a fixed netlist (endpoints known), this agent decides where endpoints go. The world model learns to predict how each placement cascades through coupled constraints, enabling lookahead planning that model-free RL cannot achieve.

---

## 1. Problem Definition

### Input (from CSV/Excel, varies per board instance)

| Parameter | Description | Example |
|-----------|-------------|---------|
| Board dimensions | Width × Height in mm | 135 × 90 |
| Board origin | Bottom-left corner (x, y) | (0, 98.2) |
| Trace count | Number of signal traces to route | 20 |
| Starting points | (x, y) per trace from connector escape routing | (58.94, 107.94) |
| Breakout lengths | Pre-routed trace length per trace | 0.8626 mm |
| Non-routing zones | Rectangular keep-out areas (position, size) | (58.29, 108.04, 17.8, 6.64) |
| UPTHs | Un-plated through holes (center, diameter) | (58.19, 105.89, ø1.9) |
| Tab pads | Mechanical pads (position, size) | (56.15, 113.35, 1.53, 1.22) |

### Fixed Constraints (same across all instances)

| Constraint | Definition | Value |
|------------|-----------|-------|
| Trace width | Fixed | 0.2286 mm |
| Trace to board edge | Edge-to-edge, minimum | 0.26 mm |
| Trace to trace | Edge-to-edge, minimum | 1.1 mm |
| Trace to UPTH | Edge-to-edge, minimum | 0.7 mm |
| Trace to tab pad | Edge-to-edge, minimum | 0.7 mm |
| TP to TP | Center-to-center, minimum | 13 mm |
| TP to board edge | Center-to-edge, minimum | 14 mm |
| TP to connector outline | Center-to-edge, minimum | 3 mm |
| Length matching | All traces must have equal total length | Post-hoc |

### Objectives

| Objective | Weight |
|-----------|--------|
| Minimize total trace length | 0.5 |
| Maximize trace-to-trace distance | 0.5 |
| Length matching (equal total length) | Hard constraint via post-processing |

---

## 2. Three-Stage Pipeline

The problem decomposes into three stages. Only Stage 1 uses learning.

### Stage 1: Test Point Placement (DreamerV3)

The RL agent decides where to place each test point on the board. This is the hard combinatorial problem — test point positions are coupled through spatial exclusion zones (13mm TP-to-TP), routing corridors (1.1mm trace-to-trace), and length relationships (length matching).

### Stage 2: Trace Routing (Deterministic — A*)

Given starting points and test point locations, compute shortest feasible paths using A* with obstacle avoidance. No learning needed. This is invoked during environment evaluation to compute trace lengths for the reward.

### Stage 3: Length Equalization (Deterministic — Meandering)

After routing, shorter traces are meandered (serpentine/accordion patterns) to match the longest trace. This uses established algorithms (Fang et al. DAC '24). The soft length-spread constraint in Stage 1 ensures traces are close enough in length for meandering to be feasible.

---

## 3. What Is the World Model Learning?

The DreamerV3 world model learns a **latent dynamics model of the placement process**. It predicts:

1. **Feasible space dynamics.** Placing test point A at (x, y) creates a 13mm exclusion zone. The world model learns which candidate positions become invalid — including indirect effects (blocking a corridor needed by future test points).

2. **Routing interactions.** The trace from starting point to test point occupies space. The world model predicts whether a placement will cause trace-to-trace violations for not-yet-placed traces.

3. **Length relationships.** Each test point position determines a trace length. The world model learns the mapping from position to length and predicts whether future placements can produce lengths close enough for post-hoc equalization.

4. **Reward.** Given a latent state, the model predicts expected reward — combining constraint satisfaction, total trace length, and spacing quality.

The agent uses this model to **imagine** placement sequences in latent space before committing. It mentally simulates "if I place test point 5 here, will test points 6–15 still have feasible positions with good spacing and similar lengths?" — selecting the placement with the best imagined long-term outcome.

This lookahead is the core advantage over model-free RL (PPO, DQN), which can only learn from trial-and-error without forward simulation through the constraint network.

---

## 4. Action Space

### Constraint-Driven Candidate Grid

The 13mm TP-to-TP and 14mm TP-to-board-edge constraints severely restrict where test points can go. On a typical 135×90mm board, the valid region is roughly 107×62mm. At 13mm spacing, only ~40 non-overlapping positions exist.

For each board instance, a candidate grid is pre-computed:

1. Inset board boundaries by 14mm → valid region
2. Remove positions within 3mm of connector outline
3. Remove positions within clearance of obstacles
4. Discretize on a 5–7mm grid → ~50–200 candidate positions

### Fixed-Size Action with Masking

Use a fixed discrete action space (e.g., 200) with invalid action masking. Per-step, candidates that violate spacing with already-placed test points have logits set to -∞. This keeps the action dimension constant across instances while enforcing validity.

This matches the ~100-action space of DreamerV3+FR — proven tractable.

### Design Decision: Grid Resolution

Informed by the Component-Centric Placement paper (2026), which found that discretization resolution is a critical hyperparameter:
- Too coarse (few candidates): optimal positions fall between grid points, solution quality suffers
- Too fine (many candidates): action space grows, training slows, numerical stability risks

**Recommendation:** Start with ~50 candidates (coarse, ~13mm grid), get learning working, then refine to ~150 candidates (5–7mm grid) and measure quality improvement.

---

## 5. State Representation (64×64 RGB Image)

The environment renders the board state as a 64×64×3 uint8 image, processed by DreamerV3's CNN encoder. Each color channel encodes a different category of information, giving the world model a spatially grounded view of the placement problem.

### Observation Image Channels

**Red channel — Obstacles and boundaries:**
- Rectangular obstacles (NRZ, tab pads) with clearance zones (intensity 150) and cores (intensity 255)
- Circular obstacles (UPTHs) with clearance halos
- Board edge inset by TP_TO_EDGE_MIN (intensity 100)
- Connector outline (intensity 180)
- Dim starting points for all traces (intensity 80)

**Green channel — Placed test points:**
- TP-to-TP exclusion zones (13mm radius, intensity 60)
- Placed TP markers (intensity 255)

**Blue channel — Current decision context:**
- Current trace starting point (bright circle, intensity 255)
- Valid remaining candidate positions (intensity 150)

### Action Space

Fixed discrete action space of 200 candidates (MAX_CANDIDATES). The candidate grid is generated at 6.5mm resolution and padded/truncated to 200 entries. Padding slots are permanently masked invalid.

### DreamerV3 Config

```yaml
encoder: {mlp_keys: '$^', cnn_keys: 'image', cnn_depth: 32, kernel_size: 4, minres: 4}
decoder: {mlp_keys: '$^', cnn_keys: 'image', cnn_depth: 32, kernel_size: 4, minres: 4}
```

CNN-only mode. The image observation captures spatial relationships (obstacle proximity, TP exclusion zones, candidate density) that inform the world model's dynamics predictions.

### Board Randomization

When a seed is provided, the connector cluster (NRZ, UPTHs, tab pads, starting traces) is shifted to a random position on the board while maintaining a 10mm margin from all edges. Board dimensions stay fixed (135×90mm), preserving the candidate grid resolution and action space size. Different seeds produce different spatial layouts for generalization.

---

## 6. Reward Design

### Per-Step Rewards (dense signal for learning)

| Condition | Reward | Rationale |
|-----------|--------|-----------|
| Valid placement (all constraints satisfied) | +1.0 | Positive reinforcement for feasible choices |
| Constraint violation (TP-to-TP, TP-to-edge) | -2.0 | Penalize but don't terminate — agent learns boundary |
| Preserving future options | +0.3 × (remaining_valid / total_candidates) | Avoid painting into a corner early |

### Episode-End Rewards (quality assessment after routing)

| Metric | Reward | Rationale |
|--------|--------|-----------|
| All constraints satisfied, all traces routable | +10.0 | Feasibility bonus |
| Any trace unroutable or hard violation | -10.0 | Feasibility penalty |
| Total trace length | -0.5 × (total_length / baseline) | Minimize length (weight 0.5) |
| Minimum trace-to-trace spacing | +0.5 × (min_spacing / target) | Maximize spacing (weight 0.5) |
| Length spread (soft) | -λ × (max_length - min_length) / mean_length | Keep lengths similar for post-hoc equalization |

### Design Decisions Informed by Prior Work

**Dense per-step rewards (from DreamerV3+FR):** They found PPO fails entirely on sparse-reward PCB tasks. Even DreamerV3 benefits from step-level signal. Our per-step constraint and future-options rewards provide this density.

**Adaptive reward shaping (from Vassallo, DATE '24):** Static rewards led to poor convergence. Consider a curriculum: start with generous constraint tolerances (e.g., 10mm TP-to-TP instead of 13mm) and tighten toward true constraints as training progresses. This gives the agent easy wins early, then forces precision.

**Multi-objective to prevent reward hacking:** Across all placement RL papers, reward hacking is the most common failure mode. The agent finds degenerate solutions (e.g., cluster all test points in one corner). Using both length AND spacing objectives, plus the future-options term, prevents any single term from being gamed.

---

## 7. Episode Structure

1. **Reset:** Load a board instance from CSV or generate procedurally. Compute candidate grid. Initialize empty placements.

2. **Steps 1–N** (N = trace count): Agent picks a candidate position per trace. After each:
   - Mark 13mm exclusion zone around placed point
   - Update candidate mask
   - Compute routed trace length via A*
   - Update observation vector
   - Return per-step reward

3. **After all N placements:**
   - Run full routing (A*) for all traces
   - Run length equalization (meandering)
   - Check all constraints
   - Compute episode-end reward

4. **Episode length:** Exactly N steps (4–20 depending on instance).

---

## 8. Length Matching: Soft Constraint + Post-Hoc Equalization

Exact length matching is handled entirely in post-processing. The RL agent only receives a soft penalty for trace length spread.

### Why Post-Hoc

Post-hoc equalization via serpentine meandering is well-established (Fang et al. DAC '24, Ozdal & Wong 2006, Tseng et al. 2017). Requiring exact matching during placement would make the RL problem dramatically harder without benefit — the agent would need to coordinate all trace lengths simultaneously, turning a tractable placement problem into an intractable global optimization.

### Feasibility Threshold

From Fang et al.: meandering requires a minimum corridor width. From Tseng et al.: dense meanders cause crosstalk-induced speedup, defeating the purpose. These findings give us a concrete feasibility check:

- Compute available meandering space per trace (based on trace-to-trace spacing and nearby obstacles)
- If (max_length - trace_length) exceeds what can be meandered in available space → infeasible
- The soft reward penalty steers the agent away from layouts where this occurs

### Implication for Spacing Objective

Maximizing trace-to-trace spacing serves double duty (from Tseng et al.):
1. Signal integrity during normal operation
2. Providing room for well-distributed (non-dense) meander segments during equalization

This means the spacing and length-matching objectives are aligned, not competing.

---

## 9. Generalization Across Board Configurations

### Spatial Normalization

All coordinates normalized to [0, 1] relative to board dimensions. Different board sizes produce identically-scaled observations. The MLP learns constraint relationships (relative distances, ratios), not absolute positions.

### Variable Trace Count

Fixed maximum (20) with zero-padding. Episode terminates after actual count. Candidate mask handles variable grid sizes.

### Training Distribution

Train on procedurally generated boards:
- Board size: uniform in [80–250mm] × [60–180mm]
- Trace count: uniform in [4–20]
- Obstacle count: uniform in [0–6], random positions
- Starting points: along connector edge with realistic spacing

Or use a dataset of real TE board specs if available.

### Fallback: Board Decomposition

From the Dummy Pad DRL paper (2025): if full-board placement is intractable at 20 traces, decompose the board into quadrants and place test points region-by-region. This reduces per-region trace count to 5–10 while maintaining most constraint interactions.

---

## 10. Design Decisions Informed by Prior Work

| Decision | Source | Lesson |
|----------|--------|--------|
| ~50–200 discrete actions | DreamerV3+FR (2026) | 100 actions proven tractable for PCB tasks |
| Dense per-step rewards | DreamerV3+FR (2026) | PPO fails on sparse rewards; DreamerV3 benefits from density |
| Adaptive reward curriculum | Vassallo, DATE '24 | Static rewards cause convergence failures; tighten constraints over training |
| Start coarse grid, refine later | Component-Centric Placement (2026) | Grid resolution is critical hyperparameter; too fine = intractable |
| Multi-objective reward | All placement RL papers | Single-objective rewards cause degenerate solutions (reward hacking) |
| Maximize spacing for meander quality | Tseng et al. (2017) | Dense meanders cause crosstalk speedup; spacing supports equalization |
| Feasibility threshold from corridor width | Fang et al. (DAC '24) | Meandering needs minimum space; soft constraint ensures this exists |
| Board decomposition as fallback | Dummy Pad DRL (2025) | Sub-board partitioning reduces complexity for large trace counts |
| Soft length constraint, not hard | Ozdal & Wong (2006) | Post-hoc equalization is proven; exact matching during placement is unnecessary |

---

## 11. Comparison to DreamerV3+FR (2026)

| Aspect | DreamerV3+FR | This Work |
|--------|-------------|-----------|
| Decision | Net ordering (which to route next) | Test point placement (where to place) |
| Action space | ~100 discrete (routing decisions) | ~50–200 discrete (candidate positions) |
| Endpoints | Fixed (netlist provided) | Unknown (agent decides) |
| Observation | 64×64 RGB image (CNN) | Structured vector ~500 floats (MLP) |
| Routing | FreeRouting (learned ordering) | A* (deterministic, after placement) |
| Length matching | Not addressed | Soft constraint + post-hoc equalization |
| Generalization | Across layer count (2→6 layers) | Across board geometry and trace count |
| Environment | Java (FreeRouting via JPype) | Pure Python/NumPy (coordinate math) |
| Core claim | World models for routing optimization | World models for generative co-design |

The fundamental difference: DreamerV3+FR optimizes **within** a fixed design (endpoints exist, choose routing order). This work **generates** the design itself (test point layout) — a harder problem class where the topology is being created.

---

## 12. Related Work

### PCB Routing and Length Matching

1. **Fang et al. (DAC '24)** — "Obstacle-Aware Length-Matching Routing for Any-Direction Traces in PCB." DP-based meandering for obstacle-aware length equalization. Assumes fixed endpoints. Our Stage 3 builds on this.

2. **Ozdal & Wong (IEEE TCAD 2006)** — "A Length-Matching Routing Algorithm for High-Performance PCBs." Foundational work on length-matching as routing objective.

3. **Tseng, Li, Ho & Schlichtmann (2017)** — "ILP-based Alleviation of Dense Meander Segments." Post-processing to distribute meanders and reduce crosstalk speedup. Informs our spacing objective.

4. **TRouter — Chen et al. (IEEE TCAD 2023)** — "Thermal-Driven PCB Routing via Nonlocal Crisscross Attention Networks." ML-driven routing for thermal optimization.

### RL for PCB/Chip Placement

5. **DreamerV3+FR (Expert Systems with Applications, 2026)** — World-model RL for PCB routing via FreeRouting. 100 discrete actions, image observations. Direct predecessor.

6. **Vassallo & Bajada (DATE '24)** — "Learning Circuit Placement Techniques Through RL with Adaptive Rewards." Adaptive reward shaping for PCB component placement.

7. **Goldie & Mirhoseini (ISPD '20)** — "Placement Optimization with Deep RL." Google's foundational chip placement RL work.

8. **Component-Centric Placement (arXiv 2026)** — DQN/A2C for PCB placement with discretized free space. Action space discretization strategy.

9. **DRL for Dummy Pad Placement (Expert Systems with Applications, 2025)** — RL for pad placement in discontinuous search spaces. Board decomposition strategy.

### World Models

10. **DreamerV3 — Hafner et al. (arXiv 2023)** — "Mastering Diverse Domains through World Models." Base architecture with fixed hyperparameters across domains.

### The Gap

All prior work assumes fixed endpoints (component placement with known netlists, routing between known pins). **No prior work addresses automated test point placement** where the endpoints themselves are design decisions under coupled spacing, length, and clearance constraints. This is the novel contribution.

---

## 13. Experimental Plan

### Minimum Viable Experiment (Poster)

| Parameter | Value |
|-----------|-------|
| Traces | 4–8 |
| Board geometry | Single instance from TE data |
| Obstacles | 3 (from example spec) |
| Candidate positions | ~50 |
| Training steps | 500K–1M |
| Compute | Google Colab Pro, T4 GPU |
| Training time | ~2–4 hours (MLP-only) |
| Baselines | Random placement, greedy heuristic |

### Metrics

| Metric | What It Shows |
|--------|---------------|
| Completion rate | % of episodes with all constraints satisfied |
| Total trace length | Quality of placement (lower = better) |
| Min trace-to-trace spacing | Quality of placement (higher = better) |
| Length spread (max - min) / mean | Feasibility of post-hoc equalization |
| DreamerV3 vs random vs greedy | Does RL help? |

### Stretch Goals

| Goal | What It Shows |
|------|---------------|
| Train on varied boards, test on held-out | Generalization |
| Scale to 20–20 traces | Scalability |
| Ablation: imagination horizon 5 vs 10 vs 15 | Does world-model lookahead help on coupled constraints? |
| DreamerV3 vs PPO on same environment | Model-based vs model-free advantage |

### Key Hypothesis to Test

**DreamerV3 outperforms PPO specifically on boards where constraints are tightly coupled** (many traces, small board, obstacles creating bottlenecks). On easy boards (few traces, large board), the advantage diminishes because lookahead isn't needed. This is the testable scientific claim.

---

## 14. Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Reward hacking (degenerate layouts) | High | Multi-objective reward; visual inspection of early results; curriculum |
| Reward function needs iteration | High | Budget 2–3 reward redesigns; this is normal in RL |
| Agent doesn't learn within 500K steps | Medium | Check world model loss first — if model learns but policy doesn't, increase imagination horizon or adjust reward |
| Scaling to 20 traces fails | Medium | Board decomposition (Dummy Pad DRL approach); or present 8-trace results as proof of concept |
| Post-hoc meandering infeasible for some layouts | Low | Soft length-spread constraint steers away from these; increase penalty weight λ |
| Colab disconnects mid-training | High | Checkpoint to Google Drive every eval cycle; auto-resume |

---

## 15. Training, Evaluation & Run Tracking

This section documents the tooling for running, evaluating, and comparing
training runs (as opposed to the conceptual design above).

### Named configs

`configs.yaml` includes, in addition to `defaults`/`debug`:

| Config | Purpose | Steps | Notes |
|---|---|---|---|
| `smoke` | Confirm the pipeline runs after a code change | 1,500 | Same architecture as `defaults`, tiny budget |
| `fast_iter` | "Did this change move the needle?" | 10,000 | A few eval cycles, small board recommended |
| `standard` | Real before/after comparison runs | 100,000 | |
| `large` | Final runs | 500,000 | Video prediction logging enabled |

Any budget knob (`--steps`, `--prefill`, `--eval_every`, `--eval_episode_num`,
`--log_every`) can also be overridden directly on the command line without
editing `configs.yaml`, e.g.:

```bash
python train.py --configs defaults fast_iter --num_traces 4 --steps 20000
```

### Run naming & metadata

If `--logdir` isn't given, each run gets an auto-generated directory under
`./logdir/`, named from the configs used, trace count, seed, short git
commit hash, and a timestamp, e.g.:

```
logdir/defaults-fast_iter_t4_s0_553b161_20260612-205039/
```

Each run directory contains `meta.json` (configs used, git commit, full
resolved config) and `metrics.jsonl` (one line per logged step, written by
`tools.Logger`). Because the commit hash is embedded, any run can be traced
back to the exact code that produced it — important once you start comparing
"before fix" vs "after fix" runs across multiple sessions.

### Per-episode diagnostics

In addition to `train_return`/`eval_return`, every episode now logs (as
`train_log_*` / `eval_log_*` scalars, averaged across eval episodes):

- `log_failures`, `log_routable` — A* routing outcome
- `log_total_length`, `log_length_spread`, `log_min_tp_spacing` — geometry
- `log_invalid_actions` — count of masked/invalid placements chosen per episode
- `log_reward_routability`, `log_reward_length`, `log_reward_spread`,
  `log_reward_spacing` — terminal reward broken down by term

These let you see *which* part of the reward or environment behavior is
changing, rather than just the aggregate return.

### Evaluating a checkpoint

```bash
python evaluate_checkpoint.py --logdir logdir/<run_name> --episodes 10
```

Runs deterministic eval episodes, reports the diagnostics above, renders
board layouts (`eval_report/episode_*.png`) for a few episodes, and compares
against the non-learned `eval.py` baselines (random/greedy/spread) on the
same board configuration. Writes `eval_report/summary.json`.

### Comparing runs

```bash
python compare_runs.py --plot eval_return eval_log_routable eval_log_length_spread
```

Scans `./logdir/*/` for `meta.json` + `metrics.jsonl`, prints a summary table
of the latest eval metrics per run, and (with `--plot`) saves overlay plots
to `logdir/comparison_plots/`.