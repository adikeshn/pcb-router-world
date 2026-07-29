# pcb-router-world — round-robin trace router

Ten copper traces grow simultaneously from connector pads in 2 mm steps across 16
compass headings. Illegal moves are masked out, so a completed board is valid by
construction; the policy's job is to make it a *good* one. Trained with MaskablePPO.

**Design documents in `docs/`:**
- `environment_spec.md` — the mechanism: masking, observation, where each reward term is computed
- `reward_spec.md` — what "good" means and how to tune it
- `exploration_spec.md` — how the solution space is searched

## Quick start

```bash
pip install -r requirements.txt
python -m pcb_router_rr2.validate            # six acceptance checks, must pass first
python -m pcb_router_rr2.train --config configs/default_10trace.yaml
```

**Use the L4 Colab runtime.** Environment stepping is >90% of wall clock, so CPU
cores matter and the GPU barely does. T4 gives 2 cores; L4 and A100 give 12, and
A100 costs far more for the same cores. Set `n_envs` to the core count and leave
`subproc_vecenv=True` — `DummyVecEnv` steps every environment serially in one
process, which was costing roughly 5× throughput.

Run **3 seeds of 2.5 M sequentially**, not one long run: each seed settles into its
own design family, and the deliverable is a portfolio of *distinct* boards. Sequential
also gets you the first result in ~1.5 h so a broken config can be aborted.

## Structure

```
pcb_router_rr2/
  config.py         every tunable, with the reasoning inline
  geometry.py       clearance-inflated collision, spatial hash, independent audit
  board.py          board / connector / pins / obstacles
  env.py            gymnasium env: masking, turn limit, reward computation
  rendering.py      board renders, episode- and step-stamped
  portfolio.py      diverse top-K, plus the diversity_mm metric
  explorer.py       parameter-noise exploration (+ random episodes for validation)
  forced_prefix.py  auto-targeted forced-prefix rollouts and verdict report
  callbacks.py      W&B logging, checkpointing, eval, collapse guard
  validate.py       six synthetic acceptance checks
  inference.py      generate a portfolio from a trained model
  train.py          training entry point
```

## Key design decisions

**Masking, not penalties, for hard constraints.** Clearance, board bounds, keep-outs
and turn sharpness are all enforced in the action mask, so violations are impossible
rather than discouraged. A soft penalty competing against a much larger reward simply
gets traded away.

**`max_turn_units = 2`** caps a single step at 45°, so a 90° turn needs two steps — a
mitred corner. Measured effect on random-policy survival (`frac_survived`):

| max_turn | legal dirs | completion | frac survived |
|---|---|---|---|
| 1 | 3 | 10.0% | 0.319 |
| **2** | **5** | **5.0%** | **0.327** |
| 3 | 7 | 7.5% | 0.379 |
| none | 16 | **0.0%** | 0.204 |

Any limit hugely beats none: constraining turns *prevents* self-trapping, the dominant
random-policy failure. An escape valve lifts the limit when nothing else is legal, so
it shapes routing without causing boxing in.

**Reversal, not magnitude, for jitter.** Sustained turning in one direction is a curve
and desirable; only *flipping* direction is jitter. Mean turn magnitude scores a smooth
arc and pure jitter identically and cannot separate them.

**Positive dense reward is farmable; penalties are not.** A policy can collect spacing
crumbs while never completing, so positive dense reward stays small. Penalties can only
be avoided by routing well, so the constraint on them is just that completing while
incurring all of them still beats failing. `reward_scale_check` tests both separately.

**Two self-distance measurements, never shared.** The hard clearance check uses
adjacency exemption; the soft coiling penalty uses an arc-length lookback that must sit
*below* the arc length of the tightest fold worth catching.

## Diagnostics

Board images are stamped with the episode and step that produced them; failed episodes
name the boxed-in trace and ring it in red.

**Fastest failure tell:** `length_spread` equal to one `step_mm` with `terminal 0.00`
means the episode truncated, even on a board reading `spec PASS, violations 0`.

Watch `most_boxed_trace` (pinned to one value = a structural block),
`portfolio/diversity_mm` (flat while terminal rises = local optimum),
`evals_since_best` (climbing = the run has stopped improving), and `q_clear_tail`
versus `q_clear_mean` (bulk improving while tails do not).

## Forced-prefix sweep

```python
from pcb_router_rr2.forced_prefix import sweep
report = sweep(cfg, model, recent_episodes=[], portfolio=pf, out_dir=run_dir)
```

Overrides one trace's first L rounds, then returns control to the policy — supplying
the multi-round commitment per-step action noise cannot produce. Targets are chosen
automatically by stuckness score. Returns a verdict: `LOCAL_OPTIMUM`, `PLATEAU`,
`MARGINAL` or `ALL_FAIL`. Runs against any checkpoint in minutes, and is worth doing
before committing to another training run.
