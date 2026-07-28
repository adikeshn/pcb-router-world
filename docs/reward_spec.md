# Reward specification — PCB round-robin router (v5, fixed budget)

Standalone reference: how the reward works, why each term exists, how to tune it.
Companion document: `environment_spec.md` (how the environment computes all of this).

**Configuration this version assumes:** a single user-specified growth budget of
100 mm, **2 mm growth steps**, 16 compass directions, 10 traces on a 200 × 150 mm
board.

---

## 1. The problem

The agent grows 10 copper traces simultaneously from connector pads, one 2 mm step
at a time, cycling through traces in round-robin order. One *round* is one 2 mm
extension for every trace.

With `budget_mm = 100`, `step_mm = 2.0` and 10 traces, **every episode is exactly
50 rounds = 500 steps.**

**Why 2 mm rather than 1 mm.** Raising the step size halves the episode, which is a
better lever than raising γ: γ extends how far the agent can *see*, while a larger
step reduces how far it *needs* to see. It also roughly doubles training throughput
(the end-of-episode audit is O(segments²), so it gets 4× cheaper), and it raises the
tightest geometrically-possible hairpin from 1 mm to 2 mm for free.

The cost is coarser manoeuvring — from any tip the reachable set is 16 points at 2 mm
radius, so threading a narrow corridor is harder. **Verify before committing:** run
`zero_violation_check` at 1 mm and 2 mm and compare boxed-in rates under the random
policy. If 2 mm is materially worse, `step_mm = 1.25` gives a clean 80 rounds. Because every trace advances once per round, all traces finish at exactly the
same length — length matching is guaranteed by construction, not by a reward term, so
there is nothing to game.

The agent's only decision is direction. Illegal directions are masked out before the
policy sees them, so invalid boards are structurally impossible rather than merely
discouraged.

---

## 2. Two layers

| Layer | When paid | Size | Purpose |
|---|---|---|---|
| **Dense** | throughout the episode | ~0.3 net | breadcrumbs — gradient before the agent can finish |
| **Terminal** | once, at round 50 | 12–25 | the actual objective |

The terminal reward is roughly **60× larger** than net dense reward. If the dense
layer could rival it, the policy would optimise breadcrumbs and ignore the goal.

**Rule: dense terms guide, the terminal term decides.**

---

## 3. The gate

Terminal reward is **zero** unless both hold:

1. **Complete** — all 10 traces grew the full 90 rounds
2. **Zero violations** — an independent brute-force audit finds no clearance breach

Diagnostic note: a truncated episode reports `length_spread_mm = 2.00` (one `step_mm`), because the
traces that had already moved in the final round are 2 mm longer than the rest. **If
you see `length spread` equal to one `step_mm` and `terminal 0.00` on a board that otherwise looks fine
— even one reading `spec PASS, violations 0` — the episode boxed in.** This is the
single most useful tell in the render subtitles.

---

## 4. Dense terms

Each is a **total for the whole episode**, divided by the round or step count. At
100 mm / 2 mm steps / 10 traces those divisors are 50 and 500.

| Term | Total | Cadence | Per-event weight |
|---|---|---|---|
| `spacing_dense_total` | **+0.80** | per round | +0.016000 |
| `constriction_penalty_total` | **−0.40** | per round | −0.008000 |
| `path_penalty_total` | **−0.25** | per step | −0.000500 |
| `self_penalty_total` | **−0.15** | per step | −0.000300 |
| `turn_penalty_total` | **−0.30** | per step | −0.000600 |

**Why keep "total ÷ count" when the count is fixed?** It makes the config robust to
*you* changing `budget_mm` later. With per-step weights, switching 100 → 130 mm
silently inflates every dense term by 30% while the terminal prize stays put — the
exact failure that broke an earlier run.

### Spacing (+0.80)

Each round, measure the **minimum** pairwise distance between the 10 trace tips and
pay proportionally, saturating at 16 mm.

Minimum, not mean: a mean is maximised by pushing one outlier far away while the rest
stay clustered. A minimum can only improve by fixing the genuinely worst pair.

### Constriction penalty (−0.40) — NEW, and the most important addition

**The problem it solves.** The dominant failure mode in the previous run was *one
trace walling off another*. Trace 6 would dive down-right across trace 9's only
escape corridor; trace 9 would hook against the connector and box in 40 rounds later.
The policy repeated this every episode and never corrected it.

Two reasons it couldn't:

1. **No visibility.** The observation contained other traces' tip positions but
   nothing about how much room they had. When choosing trace 6's move, the policy
   literally could not see that trace 9 was down to two legal directions.
2. **No credit assignment.** The only signal was terminal reward = 0, arriving ~400
   steps after the harmful move. With γ = 0.999 that is discounted to 67% — not
   nothing, but hopelessly diffuse across 400 intervening decisions.

**The mechanism.** At the end of each round, compute the number of legal directions
`f_i` for every trace. Let `f_min = min(f_i)`. If `f_min` falls below a comfort
threshold, charge a penalty proportional to how constricted it is:

```
penalty = weight × max(0, F_comfort − f_min) / F_comfort        F_comfort = 4
```

All traces having ≥4 options costs nothing. A trace down to 1 option costs 75% of the
per-round maximum.

**Why this fixes credit assignment.** The penalty lands at the end of the round in
which the constriction appeared — within 10 steps of the culprit move instead of 400.
That is the difference between a signal PPO can attribute and one it cannot.

**Paired observation change.** Per-trace legal-direction counts are added to the
observation (see `environment_spec.md` §6). The penalty supplies the gradient; the
observation supplies the information needed to act on it. Neither works alone.

### Path clearance penalty (−0.25)

Zero above 4 mm of separation from another trace, ramping to full at the 1.33 mm hard
wall. The mask alone permits two traces to run at 1.4 mm indefinitely; this
discourages it.

### Self penalty (−0.15) — deliberately weak

**Meandering is a desired strategy, not a defect.** A trace that absorbs surplus
length by meandering locally stays out of other traces' territory. A trace that cannot
meander must instead travel — and travelling across the board is precisely what
produced the blocking failure (trace 6 diving across trace 9's corridor). The
reference production board is full of controlled switchbacks for exactly this reason:
they are how length matching is done in practice.

So this term's job is **not** "discourage coiling". It is narrowly to prevent
*pathologically tight* coiling — the hairpins that ride the 0.60 mm hard floor, which
are an etching and impedance problem rather than a routing one.

- `self_lookback_mm = 12.0` — only path more than 12 mm behind counts
- `self_soft_mm = 3.0` — penalty ramps in below 3 mm (**relaxed from 5.0**)

**The lookback is essential.** Without it the measurement is meaningless: the nearest
piece of a trace's own path is always the segment two steps back, sitting exactly
1 mm away whether the trace is dead straight or tightly coiled. An earlier version
omitted it and the penalty fired on every step of every trace — a flat tax, not a
coiling detector.

**`self_soft_mm` must be well below `self_lookback_mm`.** For any path, the distance
to a point *L* mm back is at most *L*, with equality when straight. A straight trace
measured at 12 mm lookback reports ~11 mm, so a threshold near 12 would fire
constantly on good traces. This coupling has caused the same bug twice and is now
asserted at build time.

| Path shape | measured | at soft 5.0 (old) | at soft 3.0 (new) |
|---|---|---|---|
| straight run | 11.0 mm | silent | silent |
| 8 mm wide meander | 7.0 mm | silent | silent |
| 5 mm meander | 4.0 mm | **fires** | silent |
| 4 mm meander | 3.0 mm | **fires** | silent |
| 2 mm tight coil | 1.0 mm | fires | fires |

The old 5.0 mm threshold penalised 4–5 mm meanders — the exact structures the target
board uses. At 3.0 mm those are free and only genuinely tight coils are charged.

`self_soft_mm` is effectively **the minimum acceptable meander width**. Set it to the
tightest switchback you would accept from a human designer, not to a value that
discourages switchbacks.

### Turn penalty (−0.30) — revised for 16 directions

Charged per direction change, but with a **free band**: turns of ±1 direction unit
(22.5° on a 16-direction grid) cost nothing.

```
turn_cost = max(0, turn_units − 1) / (n_dirs/2 − 1)
```

**Why the free band exists.** On a discrete grid, a trace travelling at a heading
between two representable directions *must* alternate between them. Those alternations
are not turns in any meaningful sense — the trace is going straight at an intermediate
angle — but the naive penalty charged for every one. Measured cost of travelling at
22.5° on an 8-direction grid: **25% of the entire turn budget**, purely as a
quantization artifact.

Moving to 16 directions halves the worst-case heading error (22.5° → 11.25°), and the
free band removes the residual tax. Together they mean the penalty charges for real
turns and nothing else.

**Why not a continuous action space?** It was considered and rejected. Hard action
masking is this environment's most valuable property — 4.7 M steps of training with
zero violations and zero redirects. With continuous headings, validity becomes a set
of angular intervals that MaskablePPO cannot mask, forcing projection or rejection
sampling — reintroducing exactly the "redirect to nearest valid" hack this design
removed. 16 directions buys most of the expressiveness at 2× mask cost (minor, since
ray-casting dominates per-step time) and keeps every guarantee.

### Board-edge penalty — removed

Set to `0.0`. The hard 2 mm edge clearance already guarantees manufacturability and
perimeter routing is legitimate practice. Penalising it fought a real strategy for no
benefit.

---

## 5. Terminal reward

Paid once at round 50, only if the gate passes:

```
terminal = 12.00                                     flat, for finishing validly
         + 1.50 × sqrt(min_endpoint_spacing_mm)      UNCAPPED, concave
         + 1.50 × min(mean_path_clearance, 14)/14    capped
```

### Base (12.00) — raised from 5.00

Separates "did you succeed" from "how good was it", and is **the risk dial**.

The previous run raises the stakes on getting this right. Gate-pass rate reached 91%
at 1.28 M steps and then collapsed to ~2.5% over the following 3.5 M — expected return
per episode fell from 8.7 to 0.3, a 97% destruction. Boards were *better* when they
succeeded (terminal ~12 vs ~9.6) but succeeded almost never.

With base 5 and linear spacing, pushing spacing from 25 → 30 mm gained **11.1%** of
terminal — a large incentive to take risk. With base 12 and concave spacing the same
push gains **3.5%**. Combined, that is roughly a 3× reduction in the appetite for
trading completion against marginal quality.

**Raise this further if `gate_pass_rate` falls below ~40%.** Do not wait for 20%.

### Endpoint spacing — uncapped but concave

`1.50 × sqrt(spacing_mm)`, scaled to match the old linear value exactly at 25 mm.

| Spacing | linear (old) | sqrt (new) |
|---|---|---|
| 15 mm | 4.50 | 5.81 |
| 20 mm | 6.00 | 6.71 |
| 25 mm | 7.50 | 7.50 |
| 30 mm | 9.00 | 8.22 |
| 40 mm | 12.00 | 9.49 |

Still **uncapped** — it never saturates, so it always rewards improvement and always
ranks the portfolio. But returns diminish: the marginal gain from 25 → 30 mm drops
from +1.50 to +0.72.

This is a deliberate reversal. Linear was chosen originally on the argument that
constant pressure at the frontier suits a pure "as good as we can get" objective, and
that reasoning was sound in isolation. The collapse is evidence the pressure was too
strong in practice. Concavity keeps the uncapped property that fixed the *previous*
failure (saturation, which froze both ranking and gradient) while removing the
risk-seeking that caused this one.

### Path clearance (up to 1.50, capped at 14 mm)

The average, over every 1 mm step, of distance to the nearest other trace. Dividing by
14 converts millimetres into a 0–1 grade, multiplied by the 1.50 maximum.

**Capped deliberately.** Physically, crosstalk stops improving meaningfully past a
point. Mechanically, each per-step sample is clipped at 14 mm *before* averaging, and
that clipping is what makes the metric measure *crowding* rather than *spread* —
uncapped, one trace alone in an empty corner would drag the average up and mask real
crowding elsewhere.

Known limitation: an average over 900 samples dilutes localised problems. Two traces
running 2 mm apart for 30 mm costs only ~5% of this bonus. If that becomes practically
important, replace the clipped mean with a 5th-percentile measure or add a "fraction
of steps below 4 mm" term.

### Endpoint-edge bonus — removed

Set to `0.0`. It was losing a rigged fight against uncapped spacing and became
friction rather than control. Test points on the perimeter are acceptable for this
fixture. Zeroed rather than deleted, so `train/min_endpoint_edge_mm` still logs — you
can see where endpoints land without scoring it.

### `endpoint_spec_mm = 13.0` — a label, not a rule

Deliberately not enforced and not rewarded: a board at 12.9 mm is not meaningfully
worse than one at 13.1 mm, and a reward cliff there would make the agent chase an
arbitrary line instead of maximising spacing generally. It only stamps boards
PASS/miss and sorts passing boards first in the portfolio.

---

## 6. Worked example

10 traces, 100 mm budget at 2 mm steps (50 rounds, 500 steps), final min endpoint spacing 25 mm,
mean path clearance 10 mm, meandering freely, no near-boxing.

**During the episode:**

| Term | Budget used | Contribution |
|---|---|---|
| spacing | 60% of +0.80 | **+0.480** |
| constriction | 15% of −0.40 | −0.060 |
| path penalty | 10% of −0.25 | −0.025 |
| self penalty | 2% of −0.15 | −0.003 |
| turn penalty | 25% of −0.30 | −0.075 |
| | **net dense** | **+0.317** |

**At round 50:**

```
terminal = 12.00 + 1.50 × sqrt(25.0) + 1.50 × (10/14)
         = 12.00 + 7.50 + 1.07
         = 20.57
```

**Total ≈ 20.89**, of which the terminal is **98.5%**. A failed episode with identical
routing quality earns only +0.32.

---

## 7. Why these magnitudes (discounting)

PPO optimises the **discounted** sum: a reward *k* steps ahead is multiplied by γ^k.
The quantity `1/(1−γ)` is the effective planning horizon and **must cover the episode
length**.

At 500 steps (100 mm at 2 mm per step):

| γ | horizon | slack vs. episode | terminal base at step 0 | ratio |
|---|---|---|---|---|
| 0.998 | 500 | 1.0× | 4.41 | 0.272 |
| **0.999** | 1,000 | **2.0×** | **7.28** | **0.206** |

**γ = 0.999 now has 2× slack** instead of sitting exactly at the boundary, which is
what 1,000-step episodes forced. The terminal reward is worth 7.28 at step 0 rather
than 4.41 — a 65% stronger signal reaching the decisions that matter most, without
touching γ at all.

For reference, at 1 mm steps (1,000-step episodes) γ = 0.999 gave a ratio of 0.272
with zero slack, and reaching this signal strength would have required γ = 0.9995 and
the higher-variance value estimates that come with it.

Below the 0.35 threshold enforced by `reward_scale_check`. That bound is conservative
— it assumes every dense term maxes simultaneously, which cannot happen since spacing
is positive and the rest are negative. Measured ratios run ~0.05.

**A fixed budget makes γ exact rather than approximate.** With a sampled range one γ
had to serve 750–1,100 step episodes. Now there is one episode length. If you change
`budget_mm`, `step_mm`, or the trace count, **recheck γ**: the rule is
`1/(1−γ) ≥ (budget_mm / step_mm) × n_traces`. At 100 mm / 2 mm × 10 traces that is 500,
comfortably inside γ = 0.999's 1,000-step horizon.

---

## 8. Does this reward produce boards like the reference?

Checked term by term against the target production board.

| Reference feature | Reward treatment | Verdict |
|---|---|---|
| Long straight runs | turn penalty | ✅ rewarded |
| Controlled switchbacks for length matching | self penalty at 3 mm leaves 4 mm+ meanders free | ✅ **fixed in this revision** — the old 5 mm threshold penalised them |
| Wide smooth arcs rather than tight hairpins | turn free band (±1 unit) makes a gradual 180° turn free, a sharp one costly | ✅ rewarded |
| Traces reaching board extremes and corners | uncapped spacing; edge penalty removed | ✅ rewarded |
| Traces staying out of each other's way | constriction penalty | ✅ rewarded |
| Tight parallel bundles at close pitch | path penalty charges below 4 mm | ⚠️ **discouraged, deliberately** |
| Mitred 45° corners | free band prefers gradual curves over sharp corners | ⚠️ deviates |

**Two deliberate deviations, both defensible:**

*Bundling.* The reference packs traces into tight parallel groups because it is
optimising for area on a dense production board. This project's stated priority is
separation — both at endpoints and along paths — because it is a signal-integrity test
fixture. The path penalty is doing what it was asked to. Keep it, but understand that
outputs will look more "spread out" than the reference and that is correct.

*Corner style.* The turn free band makes a 90° turn spread over four 22.5° steps free,
while the same turn in one step costs. So the policy will prefer gradual curves where
the reference uses sharp mitres. Curved routing is acceptable and arguably better for
high-speed signals (no impedance discontinuity at the corner), so this is left as-is.
If a mitred look is wanted, the fix is a penalty on *cumulative* turning over a sliding
window rather than per-step turning.

**The insight that drove the self-penalty change.** Meandering and blocking are linked.
A trace carrying surplus length has two ways to spend it: absorb it locally by
meandering, or travel somewhere far. Penalising meanders forces the second, and
travelling across the board is exactly what produced the blocking failure. So the old
5 mm self threshold was not merely failing to reward the reference style — it was
plausibly *contributing to the dominant failure mode*. Relaxing it and adding the
constriction penalty push from both sides: meandering in place is now free, and
travelling into another trace's space now costs.

---

## 9. Portfolio

**5 slots, no budget bands.** Every board is the same length, so terminal rewards are
directly comparable and a single ranking is honest.

Ranking is lexicographic: `(meets_spec DESC, reward_terminal DESC)`.

**The diversity filter now carries all the variety.** Previously budget bands provided
most of it structurally. A candidate must move at least `min_moved_frac` of its
endpoints by `min_point_shift_mm` versus every existing entry.

**Tighten this.** In the previous run, portfolio ranks 6, 7, 8 and 11 were
topologically near-identical — same fan, same trace-9 hook, endpoints in similar
places — and all passed a threshold of 5-of-10 endpoints moved 13 mm. Suggested
starting point: `min_moved_frac = 0.6`, `min_point_shift_mm = 20.0`. Expect the
portfolio to plateau below 5 entries sometimes; that is a more honest outcome than 5
lookalikes.

**The portfolio is the product.** When the policy collapsed in the previous run, the
portfolio still held the good boards found near the peak. Every episode from every
source — training, eval, forced exploration — is offered to it.

---

## 10. Training-side changes (not reward, but required)

The collapse was preventable and partly a training-loop gap:

| Change | Why |
|---|---|
| **Checkpoint every ~250k steps**, and track best-by-eval-gate-pass | Only `model_final.zip` was saved. The 1.28 M policy — the best one — no longer exists. |
| **Anneal learning rate** linearly to zero | Constant 3e-4 across 4.7 M steps is destabilising late in training. |
| **Early-stop on sustained gate-pass decline** | The run spent 3.5 M steps degrading. |

---

## 11. Tuning guide

| Parameter | Effect | When to change |
|---|---|---|
| `budget_mm` | trace length | design input — **recheck γ** |
| `step_mm` (2.0) | routing resolution **and** episode length | lower for finer manoeuvring at 2× the compute; **recheck γ and `self_clearance_mm < step_mm`** |
| `w_terminal_base` (12.0) | completion vs. quality | **raise if `gate_pass_rate` < 40%** |
| `spacing_reward_coeff` (1.50) | how hard the agent chases spacing | raise to push spacing; costs gate-pass rate |
| `constriction_penalty_total` (0.40) | how strongly traces avoid boxing each other in | raise if boxing-in persists |
| `F_comfort` (4) | when constriction starts to hurt | raise to make the agent keep more room |
| `self_soft_mm` (3.0) | minimum acceptable meander width | **lower** if meanders look over-constrained; raise only to ban tight switchbacks |
| `turn_penalty_total` (0.30) | trace straightness | raise for cleaner corners |
| `terminal_clearance_target_mm` (14) | clearance grading scale | **raise if `q_clear` pins at 1.0** |
| `min_point_shift_mm` | portfolio variety | raise — bands no longer provide it |

**After any change, rerun the acceptance checks:**

- `self_crossing_check` — a tight fold must be caught; straight runs and corners legal
- `zero_violation_check` — 200 random episodes, zero audited violations
- `reward_scale_check` — discount-aware, at the configured budget

---

## 12. Failure modes

| Symptom | Meaning | Fix |
|---|---|---|
| `gate_pass_rate` rises then falls over millions of steps | **policy collapse** — risk-seeking drift | raise `w_terminal_base`; anneal LR; roll back to best checkpoint |
| `length spread 1.00` with `terminal 0.00` | episode truncated — a trace boxed in | check which trace; look for a blocker |
| One trace boxes in repeatedly in the same place | another trace is walling it off | raise `constriction_penalty_total` / `F_comfort` |
| A `q_*` metric pins at 1.0 | that term saturated | raise its target |
| A dense term uses ~0% of budget | inert — verify it measures what you think | test against a synthetic case |
| Dense rises while terminal falls | classic inversion | recheck γ and the scale check |
| `turn_rate` high but traces look straight | quantization artifact, not real turning | more directions / widen the free band |
| Portfolio fills with lookalikes | diversity threshold too loose | raise `min_point_shift_mm` |
| `q_clear` drops as spacing climbs | traces bundling on the perimeter | raise `w_path_clearance_bonus` |

**General lesson from this project's bugs:** every reward term should be tested against
a synthetic case where you know the right answer *before* trusting it in training.
Three separate terms here — the self penalty, the clearance mean, and the turn penalty
— were silently measuring something other than what they were named for, and each was
caught only by constructing a hand-built example and checking the number.
