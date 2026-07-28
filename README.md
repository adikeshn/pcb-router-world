# pcb-router-world — `round-robin-v2` (v3.0 update)

Round-robin trace-growth PCB router on **MaskablePPO** (sb3-contrib), with
clearance-inflated geometry, no-breakout growth straight from connector pads,
a length-invariant reward, and a budget-stratified solution portfolio.

## What changed in v3.0

### Geometry / correctness
| Issue | Fix |
|---|---|
| Arc-length self-exemption had a **blind spot**: the legal 3-step fold N, SE, W crosses itself, yet mask and audit both reported it clean | Exemption is now **topological adjacency** (`\|i-j\| <= 1`) — only segments that actually share an endpoint. No blind spot, robust to variable segment length |
| Self-crowding had no gradient — a trace could coil against itself freely down to the hard 0.6 mm wall | `segment_valid` now returns the self-distance too, feeding a new dense **self-proximity penalty** |
| Nothing penalised **turning**, so traces jittered (N, NE, N, NE…) instead of running straight | New dense **turn penalty**, scaled by turn magnitude. A wide meander costs ~4 turns; jitter costs one per step — so clean meanders become strictly cheaper than jitter for the same length-burning purpose |

`self_clearance_mm` **must be < `step_mm`** and this is now validated. Raising it
to force wider meanders (as originally suggested) is impossible: with adjacency
exemption, two segments separated by one intervening 1 mm segment sit exactly
1 mm apart *even on a perfectly straight run*, so any value ≥ 1 mm would make
straight-line growth illegal. Meander width is shaped by the soft penalty instead.

### Reward
| Issue | Fix |
|---|---|
| Dense reward scaled with episode length while terminal stayed fixed. At 110 mm budgets with γ=0.998 the dense term was worth **81 %** of the terminal — nearly inverted | Every dense term is now a per-episode **TOTAL**, divided at reset by that episode's round/step count. Balance is invariant to budget |
| γ=0.998 gives a 500-step planning horizon for 1100-step episodes | γ=**0.999** (1000-step horizon, matched to episode length) |
| Terminal saturated at exactly 10.0: `spacing_target_mm=16` pinned `q_spacing=1.0` for every board, so the portfolio couldn't rank and the policy got no terminal gradient | Terminal targets separated from dense ones and raised: spacing 35 mm, clearance 8 mm |
| `q_short` was worth 0.5 reward points against a top-5 spread of 0.08 — it silently decided the entire ranking, collapsing every entry to the minimum budget | **Removed from the reward.** Length is handled structurally by portfolio stratification |
| Nothing stopped endpoints landing on the board edge | New `q_endpoint_edge` terminal term (target 15 mm) |

### Portfolio
- **Budget-stratified by default**: `[budget_min, budget_max]` split into
  `portfolio_k` bands, best gated layout kept per band. Surfaces the
  length-vs-quality trade-off directly instead of a reward term deciding it.
- Set `portfolio_stratify_by_budget=False` for the original endpoint-diversity filter.
- ForcedExplorer now sweeps budgets across all bands so every band gets pressure.

### Validation & logging
- New `self_crossing_check`: the tight fold must be caught; straight runs and
  90° corners must stay legal (guards against over-correction).
- `reward_scale_check` is **discount-aware** and runs at both ends of the budget
  range — the old raw-sum version passed the broken configuration.
- New metrics: `train/budget_mm_all` vs `train/budget_mm_gated` (are long budgets
  failing the gate or just losing the ranking?), `eval_by_budget/*` per-budget
  breakdown, `train/q_spacing|q_clear|q_edge` saturation watch,
  `train/min_self_gap_mm`, `train/turn_rate`, `portfolio/band*_terminal`.

## Layout
```
pcb_router_rr2/{config,geometry,board,breakout,env,rendering,portfolio,explorer,callbacks,validate,train}.py
configs/default_10trace.yaml
notebooks/train_colab.ipynb
```

## Quick start
```bash
pip install -r requirements.txt
python -m pcb_router_rr2.validate                      # must pass before training
python -m pcb_router_rr2.train --config configs/default_10trace.yaml
```

Outputs in `runs/<name>/`: `config.yaml`, `model_final.zip`, `renders/`,
`portfolio/` (`rank_*.png`, `rank_*.json`, `portfolio.json`).
