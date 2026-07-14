# pcb-router-world — `round-robin-v2`

Round-robin trace-growth router, rebuilt on **MaskablePPO** (sb3-contrib)
with clearance-inflated geometry, per-step mask recomputation, sampled
length budgets, a dominance-checked reward, and a diverse top-K portfolio.

## What changed vs `round_robin` (v1)

| v1 | v2 |
|---|---|
| DreamerV3 on rendered board images | MaskablePPO on a 68-dim vector obs (tips, headings, ray-casts, masks) |
| Crossing checks on raw segments | All checks clearance-inflated (1.33 mm trace-to-trace enforced everywhere) |
| Residual 0–1 crossings/episode (stale masks) | Masks recomputed per step; brute-force final audit; acceptance test requires **zero** violations over 200 random episodes |
| Fixed `max_length_mm` | Budget sampled per episode (25–45 mm), in the observation and portfolio → total length is searchable again |
| Mean pairwise tip spacing reward | **Min** pairwise hinge (mean is gameable by outliers) + dense path-clearance + edge penalties |
| Turn order randomised per round | Randomised per episode and exposed in obs (robustness without dynamics noise) |
| unimix floor leaks masked actions | True zero-probability masking; redirect kept only as an asserted safety net |
| `meets_spec` label only | Still no gate/reward role, but portfolio ranks spec-passing layouts first (lexicographic) |

Kept from v1: 1 mm steps, 8 directions, equal length by construction,
two-phase length-normalised breakout (now with lane jogs so top-row
descents can't hit bottom-row pins), boxed-in early termination,
ForcedExplorer (now portfolio-seeding only), file-path W&B images.

## Layout

```
pcb_router_rr2/
  config.py     every tunable in one dataclass (YAML round-trip)
  geometry.py   seg/seg distances, spatial hash, ClearanceChecker, audit
  board.py      board / connector / pins / obstacles
  breakout.py   deterministic breakout + validation at construction
  env.py        gymnasium env (RoundRobinTraceEnv) + action_masks()
  rendering.py  preview_figure / episode_figure (Agg, file-path PNGs)
  portfolio.py  diverse top-K, lexicographic (meets_spec, terminal)
  explorer.py   ForcedExplorer random episodes -> portfolio
  callbacks.py  W&B scalars, board images, eval suite, explorer cadence
  validate.py   zero_violation_check + reward_scale_check
  train.py      MaskablePPO training entry
configs/default_6trace.yaml   suggested 6-trace board settings
notebooks/train_colab.ipynb   Colab driver (preview -> validate -> train)
```

## Quick start

```bash
pip install -r requirements.txt

# 1. look at the board before anything else
python -c "from pcb_router_rr2.config import Config; \
           from pcb_router_rr2.rendering import preview_figure; \
           preview_figure(Config(), save_path='preview.png')"

# 2. acceptance tests (must pass before training)
python -m pcb_router_rr2.validate

# 3. train
python -m pcb_router_rr2.train --config configs/default_6trace.yaml
```

Outputs land in `runs/<run_name>/`: `config.yaml`, `model_final.zip`,
`renders/`, and `portfolio/` (`rank_*.png`, `rank_*.json`, `portfolio.json`).

## W&B metrics

- `train/*` — completion, gate-pass, spec-pass, boxed-in, violation,
  redirect rates; endpoint spacing; length spread; steps/sec
- `reward/*` — per-term episode decomposition (spacing_dense,
  edge_penalty, path_penalty, terminal) — watch for any term silently
  dominating
- `eval/*` — deterministic policy across the budget range
- `portfolio/*` — size, gate throughput, best score, spec-pass count
- `board/*` — images: `training_episode`, `eval_best`, `portfolio`,
  `final_portfolio`

## Notes

- The v1 doc says "180 × 120 mm" but its pin coordinates need height ≥ 180;
  this repo uses 120 (x) × 180 (y). Flip in config if your board differs.
- `length_spread_mm` is logged and should be ~0 by construction; a nonzero
  value indicates a breakout-normalisation regression (the safety net that
  replaces the unported `equalize_lengths()`).
