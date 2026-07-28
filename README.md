# pcb-router-world — `round-robin-v2` (v5.0)

Round-robin trace-growth PCB router on **MaskablePPO**. Ten traces grow
simultaneously from connector pads in 2 mm steps across 16 compass directions;
illegal moves are masked out, so a completed board is valid by construction.

**Design documents:** `docs/reward_spec.md` (why each reward term exists) and
`docs/environment_spec.md` (how masking, observation and reward are computed).

## Quick start

```bash
pip install -r requirements.txt
python -m pcb_router_rr2.validate                       # must pass first
python -m pcb_router_rr2.train --config configs/default_10trace.yaml
```

Outputs land in `runs/<name>/`: `config.yaml`, `model_best.zip`,
`model_final.zip`, `checkpoints/`, `renders/`, `portfolio/`.

## What changed in v5.0

### Fixed budget

One `budget_mm` instead of a sampled range. The policy learns _a_ routing
strategy rather than a family indexed by budget, γ is matched exactly to one
episode length, and portfolio banding is no longer needed.

### 2 mm steps (was 1 mm)

Halves the episode to 500 steps. Measured on this board: costs ~4% of legal
directions per step, but **improves random-policy completion several-fold**
(4% → 29% at a 30 mm budget, consistent across seeds), because boxing in is
dominated by self-trapping and there are half as many chances to do it. Also
~2× training throughput and a wider minimum hairpin.

### 16 directions (was 8)

Halves the worst-case heading error to 11.25°. Mask cost doubles but
ray-casting dominates per-step time, so the impact is small.

### Turn penalty with a free band

Turns of ±1 direction unit cost nothing. On a discrete grid, travelling at an
intermediate heading _requires_ alternating between adjacent directions — the
old penalty charged 25% of its entire budget for that quantisation artifact.

### Constriction penalty (new)

The dominant failure was one trace walling off another: trace 6 dived across
trace 9's corridor, trace 9 boxed in 40 rounds later. Two fixes, both needed:
per-trace legal-direction counts are now **in the observation** (the policy
previously could not see it), and a penalty fires at the end of the round in
which constriction appears — credit within 10 steps of the culprit instead of 400.

### Self penalty relaxed and reframed

Threshold 5.0 → 3.0 mm, total 0.25 → 0.15. **Meandering is a desired
strategy** — a trace that absorbs surplus length locally stays out of other
traces' space, and penalising it forces the travel that causes blocking. The
term now only prevents pathologically tight coils.

### Terminal reward

Base 5 → **12**, and spacing changed from linear to **concave (√)**, still
uncapped. Pushing spacing 25 → 30 mm previously gained 11.1% of terminal; it
now gains 3.5%. A previous run peaked at 91% gate-pass then collapsed to ~2.5%
chasing marginal spacing.

### Edge terms removed

Both set to 0. Hard edge clearance already guarantees manufacturability and
perimeter routing is legitimate. Zeroed rather than deleted, so the metrics
still log.

### Training-loop hardening

Checkpoints every 250k steps, `model_best.zip` saved on every eval gate-pass
improvement, optional early stop on sustained decline, and linear LR
annealing. The previous run's best policy (step 1.28M) was lost because only
`model_final.zip` was saved.

## Diagnostics

Board images are stamped with the **episode and step count** that produced
them, and failed episodes name the boxed-in trace and ring it in red.

Key metrics: `train/frac_survived` (progress even at 0% completion),
`train/most_boxed_trace` (who is being walled in), `train/min_freedom`,
`train/r_spacing` (uncapped — should climb), `train/q_clear` (capped — watch
for 1.0), `eval/best_gate_pass_rate`, `eval/evals_since_best`.

**The fastest failure tell:** `length_spread` equal to one `step_mm` with
`terminal 0.00` means the episode truncated, even if the board reads
`spec PASS, violations 0`.

## Inference

```python
from pcb_router_rr2.inference import solve
pf = solve(cfg, "runs/<name>/model_best.zip", n_episodes=200)
```

The portfolio is the product — it survived a full policy collapse in a
previous run with its best boards intact.
