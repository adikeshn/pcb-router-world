# Reward specification — PCB round-robin trace router

What "good" means, why each term exists, and how to tune it.

Companion documents: `environment_spec.md` (how the environment computes all of this)
and `exploration_spec.md` (how the solution space is searched).

**Reference configuration:** 10 traces, 200 × 150 mm board, `budget_mm = 100`,
`step_mm = 2.0`, 16 directions → **50 rounds = 500 steps** per episode.

---

## 1. Two layers

| Layer | When paid | Size | Purpose |
|---|---|---|---|
| **Dense** | throughout the episode | ~0.4 net | breadcrumbs — gradient before the agent can finish anything |
| **Terminal** | once, at the final round | 12–25 | the actual objective |

The terminal reward is roughly **50× larger** than the net dense reward. This is
load-bearing: if the dense layer could rival it, the policy optimises breadcrumbs and
ignores the goal.

**Rule: dense terms guide, the terminal term decides.**

---

## 2. The gate

Terminal reward is **zero** unless both hold:

1. **Complete** — every trace grew the full 50 rounds
2. **Zero violations** — an independent brute-force audit finds no clearance breach

Nothing earned during an episode compensates for failing the gate: a failure collects
~0.4 in crumbs against 20+ for a pass.

---

## 3. Dense terms

Each is a **total for the whole episode**, divided at reset by the round or step count.
At 50 rounds / 500 steps:

| Term | Episode total | Cadence | Per-event weight |
|---|---|---|---|
| `spacing_dense_total` | **+0.80** | per round | +0.016 |
| `constriction_penalty_total` | **−1.50** | per round | −0.030 |
| `reversal_penalty_total` | **−0.80** | per step | −0.0016 |
| `self_penalty_total` | **−0.60** | per step | −0.0012 |
| `path_penalty_total` | **−0.60** | per step | −0.0012 |
| `edge_penalty_total` | 0.00 | — | disabled |

**Positive dense reward and penalties are constrained differently.** This distinction
governs every weight above:

* **Positive dense reward is farmable** — a policy can accumulate spacing crumbs while
  never completing an episode. It must stay small relative to the terminal. Here it is
  0.80, or ~4% of a typical terminal.
* **Penalties are not farmable.** The only thing a policy can do with a penalty is
  avoid it by routing well. The constraint is not a ratio but simply: *completing while
  incurring every penalty must still beat failing.* Penalties total 3.50 against a
  terminal base of 12, so a maximally-penalised completed board scores ~17 against 0
  for a failure.

Treating both halves with one "dense must be tiny" rule under-weights routing quality
for no safety benefit. Penalties have headroom to ~5.0 total before completion stops
being clearly worth it.

**Why totals rather than per-event weights.** It makes the configuration robust to
changing `budget_mm` or `step_mm`. Specify per-step weights directly and switching from
100 mm to 130 mm silently inflates every dense term by 30% while the terminal prize
stays fixed — quietly inverting the objective at long budgets.

### Spacing (+0.80) — the main guiding signal

Each round, measure the **minimum** pairwise distance between trace tips and pay
proportionally, saturating at `dense_spacing_target_mm = 16`.

Minimum, not mean: a mean is maximised by pushing one outlier far away while the rest
stay clustered. A minimum can only improve by fixing the genuinely worst pair.

Saturation at 16 mm is fine here — during growth "adequately spread" is all that is
needed, and final quality is graded by the terminal term, which does not saturate.

### Constriction (−1.50) — prevents traces walling each other in

At the end of each round, compute the number of legal directions `f_i` for every trace.
With `f_min = min(f_i)`:

```
penalty = weight × max(0, F_comfort − f_min) / F_comfort        F_comfort = 6
```

**The problem it solves.** The dominant failure on a dense connector fan is one trace
diving across another's only escape corridor, with the victim boxing in dozens of
rounds later. Two things are needed and neither works alone:

1. **Visibility** — per-trace legal-direction counts are in the observation
   (`environment_spec.md` §6). Without them the policy cannot see that another trace is
   running out of room; tip positions do not convey it.
2. **Credit assignment** — the only other signal is terminal reward = 0, arriving
   hundreds of steps after the harmful move. This penalty lands at the end of the round
   in which the constriction appeared, within one round of the culprit.

**Why `F_comfort = 6` specifically.** Typical `mean_freedom` on this board is ~8 while
`min_freedom` reaches 0–1: freedom is comfortable on average and collapses abruptly.
A threshold near the mean fires on roughly half of all rounds and degrades into a
constant background tax; a threshold much below it only catches the last few rounds,
after the blocking move is long past. 6 sits below typical but above collapse.

### Self penalty (−0.30) — prevents pathologically tight coils only

**Meandering is a desired strategy, not a defect.** A trace that absorbs surplus length
by meandering locally stays out of other traces' territory; a trace that cannot meander
must travel, and travelling across the board is what causes blocking. Production boards
are full of controlled switchbacks for exactly this reason.

This term's job is narrowly to prevent *pathologically tight* coils — hairpins riding
the hard clearance floor, which are an etching and impedance problem rather than a
routing one.

- `self_lookback_mm = 6.0` — only path more than 6 mm behind counts
- `self_soft_mm = 3.0` — penalty ramps in below 3 mm

**Both parameters are tightly coupled and easy to get wrong.** Two rules:

*Upper bound on the threshold.* For any path, the distance to a point *L* mm back is at
most *L*, with equality when straight. A straight run measured at 6 mm lookback reports
4 mm, so the threshold must sit below that or the term fires constantly on perfectly
good traces. Rule: `self_soft_mm < self_lookback_mm − step_mm`.

*Upper bound on the lookback.* The lookback must be **smaller than the arc length of
the tightest fold worth catching**. At 2 mm steps a hairpin spans only 6–8 mm of arc,
so a 12 mm lookback ignores exactly the structures the term exists to detect.

Measured behaviour at lookback 6 mm / threshold 3 mm:

| path shape | measured | penalty |
|---|---|---|
| straight run | 4.0 mm | silent |
| gentle arc (22.5°/step) | 4.7 mm | silent |
| tightest legal U-turn (45°/step) | ~2.5 mm | fires |

`self_soft_mm` is effectively **the minimum acceptable meander width**. Set it to the
tightest switchback you would accept from a human designer, not to a value that
discourages switchbacks.

### Path clearance penalty (−0.25)

Zero above `path_soft_mm = 4` of separation from another trace, ramping to full at the
hard clearance wall. The mask alone permits two traces to run at 1.4 mm indefinitely.

### Reversal penalty (−0.80) — the jitter mechanism

Charged when the **sign** of the heading change flips between consecutive steps, i.e.
`sign(turn_t) × sign(turn_{t−1}) < 0` with both non-zero. Straight steps between two
corners do not create a reversal — two separate corners are not jitter.

**Why sign and not magnitude.** Sustained turning in one direction is a *curve*, which
is a legitimate and desirable routing structure — controlled meanders are how length is
absorbed locally instead of by travelling into another trace's space. Only *flipping*
direction is jitter. Measured across shapes:

| shape | mean turn magnitude | reversal rate |
|---|---|---|
| straight run | 0.00 | 0.00 |
| smooth arc | 1.00 | **0.00** |
| tight smooth arc | 2.00 | **0.00** |
| mitred 90° corner | 0.33 | **0.00** |
| **jitter** | 1.00 | **1.00** |
| S-curve | 1.00 | 0.09 |

A magnitude-based penalty scores the smooth arc and jitter **identically** — it cannot
distinguish them, and taxes exactly the structure the router should be using. Reversal
is zero for every desirable shape.

**Division of labour with the turn limit.** `max_turn_units` (a hard mask, see
`environment_spec.md` §5.4) caps how *sharp* a turn can be. This penalty catches how
*erratically* the trace turns. Neither substitutes for the other, and neither
double-counts the other.

**Side effect, and a desirable one.** Travelling at a heading between two of the 16
representable directions requires alternating between them, which registers as
reversals. The penalty therefore pushes traces onto the discrete headings and holds
them there — which is how PCB traces are actually drawn. No "free band" exemption is
needed; one existed only to protect quantisation alternation under a magnitude penalty,
and is unnecessary here.

### Board-edge penalty — disabled

Set to 0. Hard edge clearance already guarantees manufacturability and perimeter
routing is legitimate practice. Zeroed rather than deleted so the metric still logs.

---

## 4. Terminal reward

Paid once at round 50, only if the gate passes:

```
terminal = 12.00                                   flat, for finishing validly
         + 1.50 × sqrt(min_endpoint_spacing_mm)    UNCAPPED, concave
         + 3.00 × clearance_quality                 capped, mean AND tail
         + 1.50 × (1 − turn_reversal_rate)          smoothness

clearance_quality = 0.5 × min(mean_clearance, 14)/14
                  + 0.5 × min(p5_clearance,   14)/14
```

### Base (12.00)

Separates "did you succeed" from "how good was it". Without it a barely-valid board
scores little more than a failure and the policy takes reckless risks chasing quality.

**This is the risk dial.** Raise it if `gate_pass_rate` falls below ~40%. At 12, pushing
spacing from 25 → 30 mm gains 3.5% of terminal; at 5 it would gain 11%, which is enough
incentive to trade away completion.

### Endpoint spacing — uncapped, concave

`1.50 × sqrt(min_endpoint_spacing_mm)`. The primary objective.

**Uncapped** because this is a **minimum** over endpoint pairs — every extra millimetre
is a real improvement to the genuinely worst pair, and it cannot be gamed by pushing
already-distant points further apart. A cap here saturates: once every board scores
full marks the portfolio cannot rank its entries and the policy receives no gradient
for further improvement.

**Concave** because linear pressure at the frontier is destabilising. With linear
reward the marginal gain from pushing spacing is constant no matter how good the board
already is, which makes the policy willing to risk failure indefinitely for marginal
gain. Square root keeps the uncapped property while halving the marginal pull at high
spacing:

| spacing | linear (0.30/mm) | sqrt (1.50·√) |
|---|---|---|
| 15 mm | 4.50 | 5.81 |
| 25 mm | 7.50 | 7.50 |
| 40 mm | 12.00 | 9.49 |

### Path clearance (up to 3.00) — half bulk, half worst-stretch

Whole-trace quality, measured two ways and averaged:

* **mean** per-step distance to the nearest other trace — bulk quality
* **5th percentile** of the same samples — the worst stretches

**Why both.** A mean over ~500 samples dilutes localised problems: two traces running
2 mm apart for 30 mm move the mean barely at all. But a percentile alone has a cliff —
crowding affecting under 5% of steps is invisible to it. Splitting the weight evenly
covers both failure shapes without adding a second tunable.

Each sample is clipped at 14 mm **before** aggregation. The clip is what makes this
measure *crowding* rather than *spread*: uncapped, one trace alone in an empty corner
drags the mean up and masks real crowding elsewhere.

**Why the weight is 3.00 and not smaller.** Path separation is priority 3 in the design
brief, and at a weight of 1.50 it accounted for ~6% of the terminal, which is not
enough to change any decision. The test is whether the policy will trade path quality
for endpoint spacing: gaining 2 mm of spacing at 25 mm is worth **+0.29**, while losing
0.1 of `q_clear` costs **0.15** at weight 1.50 — so the trade is accepted — but
**0.30** at weight 3.00, so it is declined. The weight is set where that incentive
flips, not by preference.

**Capped for two reasons.** Physically, crosstalk stops improving meaningfully past a
point. Mechanically, each per-step sample is clipped at 14 mm *before* averaging, and
that clipping is what makes the metric measure *crowding* rather than *spread* —
uncapped, one trace alone in an empty corner drags the average up and masks real
crowding elsewhere.

**Why the weight is 3.00.** Path separation is priority 3 in the design brief; at 1.50
it accounted for ~6% of the terminal, not enough to change any decision. The test is
whether the policy will trade path quality for endpoint spacing: gaining 2 mm of
spacing at 25 mm is worth **+0.29**, while losing 0.1 of clearance quality costs
**0.15** at weight 1.50 — so the trade is taken — but **0.30** at weight 3.00, so it is
declined. The weight sits where that incentive flips.

### Smoothness (up to 1.50) — `1 − turn_reversal_rate`

The fraction of steps at which the trace flips turn direction, over the finished board.
Zero for straight runs, smooth arcs and mitred corners alike; 1.0 for jitter.

**Why this is in the terminal and sharpness is not.** Turn *sharpness* is capped
structurally by `max_turn_units`, so it cannot vary between boards and needs no score.
Jitter cannot be masked away — it is a legal sequence of legal moves — so it varies,
and anything that varies and matters must be scored or it will not affect which boards
the portfolio keeps.

It appears in both layers deliberately, the same pattern endpoint spacing uses: the
dense reversal penalty supplies gradient *during* the episode, this term puts the
finished result into the ranking.

### Endpoint-edge bonus — disabled

Set to 0. It loses a rigged fight against uncapped spacing — moving an endpoint 5 mm
toward the board edge to gain 3 mm of spacing wins far more than the edge term can
charge, so the agent correctly takes that trade every time and the term becomes
friction rather than control. Zeroed rather than deleted so the metric still logs.

### `endpoint_spec_mm = 13.0` — a label, not a rule

The fixture specification. Deliberately not enforced and not rewarded: a board at
12.9 mm is not meaningfully worse than one at 13.1 mm, and a reward cliff there makes
the agent chase an arbitrary line instead of maximising spacing generally. It stamps
boards PASS/miss and sorts passing boards first in the portfolio.

---

## 5. Worked example

10 traces, 50 rounds, 500 steps. Final min endpoint spacing 25 mm, mean path clearance
10 mm, meandering freely, no near-boxing.

**During the episode:**

| Term | Budget used | Contribution |
|---|---|---|
| spacing | 60% of +0.80 | **+0.480** |
| constriction | 8% of −1.50 | −0.120 |
| reversal penalty | 10% of −0.80 | −0.080 |
| self penalty | 5% of −0.60 | −0.030 |
| path penalty | 10% of −0.60 | −0.060 |
| | **net dense** | **+0.190** |

**At round 50:**

```
terminal = 12.00 + 1.50 × sqrt(25.0)
         + 3.00 × (0.5 × 10.9/14 + 0.5 × 4.2/14)
         + 1.50 × (1 − 0.30)
         = 12.00 + 7.50 + 1.62 + 1.05
         = 22.17
```

**Total ≈ 22.36.** Shares of the terminal: base 54%, endpoint spacing 34%, path
clearance 7%, smoothness 5% — **path and shape quality together are 12% of the
objective**. A failed episode with identical routing quality earns only +0.19.

---

## 6. Why these magnitudes — discounting

PPO optimises the **discounted** sum: a reward *k* steps ahead is multiplied by γ^k.
`1/(1−γ)` is the effective planning horizon and **must cover the episode length**.

At 500 steps:

| γ | horizon | slack | terminal base at step 0 | dense/terminal ratio |
|---|---|---|---|---|
| 0.998 | 500 | 1.0× | 4.41 | 0.27 |
| **0.999** | **1,000** | **2.0×** | **7.28** | **0.21** |

γ = 0.999 gives 2× slack. The discount-aware `reward_scale_check` runs **two separate
tests**, because the two halves of the dense layer fail in different ways:

1. **Farmability** — discounted *positive* dense reward vs. discounted terminal base.
   Must be below ~0.35. This is the test that catches a policy farming crumbs instead
   of completing.
2. **Completion dominance** — total penalties vs. terminal base, undiscounted. A board
   that completes while incurring every penalty must still score well above zero.

A single combined test is wrong: it treats penalties as if they were farmable, and
would reject configurations that weight routing quality properly for no safety
benefit.

**`gae_lambda = 0.98`, not the usual 0.95.** GAE's credit horizon is `1/(1−γλ)`. At
λ=0.95 that is 20 steps, so in a 500-step terminal-dominant episode the observed
outcome reaches mid-episode with weight 2×10⁻⁶ — early decisions are credited
*entirely* through the critic's prediction. λ=0.98 extends it to 48 steps. The cost is
variance; fall back to 0.95 if `value_loss` destabilises.

**If you change `budget_mm`, `step_mm` or the trace count, recheck γ:**
`1/(1−γ) ≥ (budget_mm / step_mm) × n_traces`.

---

## 7. Portfolio

**5 slots, ranked lexicographically** by `(meets_spec DESC, reward_terminal DESC)`.
Every board is the same length, so terminal rewards are directly comparable.

**Why ranking uses the terminal alone and not terminal + dense.** Two reasons:

1. **Numerically it would change nothing.** Net dense on a good gated board is ~0.19
   against a terminal of ~22 — under 1% of the total, varying by perhaps ±0.3 between
   boards against a ranking spread of ~2 points. It would essentially never reorder the
   portfolio.
2. **Two of the dense terms do not describe the board.** `spacing_dense` measures tip
   spacing *during growth*, and `constriction` measures how boxed-in traces *got* — a
   near-miss that was survived leaves no mark in the copper. Those are learning
   signals shaped to guide a policy, not properties of the artifact. Ranking on them
   would score the trajectory rather than the board.

The three dense terms that *do* describe the artifact — crowding, coiling and jitter —
belong in the score that does the ranking, and are represented there: crowding via the
clearance tail, jitter via the smoothness term. That is the correct fix for "boards with
good endpoints but poor routing rank highly", rather than adding process terms to the
ranking.

**The diversity filter carries all the variety.** A candidate must move at least
`min_moved_frac = 0.6` of its endpoints by `min_point_shift_mm = 20` versus every
existing entry, or it is treated as a near-duplicate of the entry it collides with and
only replaces it if strictly better. Loose thresholds fill the portfolio with
topologically identical layouts that happen to be the highest scorers.

Expect fewer than 5 entries sometimes; that is a more honest outcome than 5 lookalikes.

**The portfolio is the product.** Every episode from every source — training, eval, and
every exploration mechanism — is offered to it, and each entry records the episode and
step count at which it was found. If the policy degrades, the portfolio still holds the
best boards discovered at any point.

**`portfolio/diversity_mm`** — mean endpoint displacement between entries — measures
directly whether the search is finding different layouts or polishing one. Rising
`best_reward_terminal` with flat diversity is the signature of a local optimum.

---

## 8. Training settings

| Setting | Value | Why |
|---|---|---|
| Total steps | 2.5 M × **3 seeds** | Each seed settles into its own design family; since the deliverable is a portfolio of *distinct* boards, more families beats more polish on one |
| `gamma` | 0.999 | horizon 1,000 vs 500-step episodes |
| `gae_lambda` | 0.98 | see §6 |
| Learning rate | 3e-4 annealed to a **floor of 1e-4** | Constant high LR is destabilising late; annealing to *zero* freezes the policy into whatever basin it occupies by mid-run, which is counterproductive when escaping a basin is the goal |
| `ent_coef` | 0.01 | raise to ~0.02 if learning stalls |
| `n_envs` / `n_steps` / `batch` / `n_epochs` | 8 / 512 / 512 / 6 | 4,096 experiences per update |
| `net_arch` | [256, 256] | vector observation, no CNN needed |
| Checkpoints | every 250 k steps, plus `model_best.zip` on every eval improvement | the best policy is often not the final one |
| `eval_episodes` | **20** | with 5, `eval/gate_pass_rate` is quantised to {0, .2, .4, .6, .8, 1} and is too coarse to select `model_best` on |
| `early_stop_patience_evals` | **10** | stops a run that has ceased improving; arms only once best gate-pass exceeds 20% |

---

## 9. Tuning guide

| Parameter | Effect | When to change |
|---|---|---|
| `w_terminal_base` (12.0) | completion vs. quality | **raise if `gate_pass_rate` < 40%** |
| `spacing_reward_coeff` (1.50) | how hard the agent chases spacing | raise to push spacing; costs gate-pass rate |
| `constriction_penalty_total` (1.50) / `F_comfort` (6) | how strongly traces avoid boxing each other in | raise if one trace keeps dying |
| `self_soft_mm` (3.0) | minimum acceptable meander width | **recheck `self_lookback_mm` whenever you change this** |
| `max_turn_units` (2) | how **sharp** a turn can be | lower for smoother traces; watch `frac_survived` for a manoeuvrability cost |
| `reversal_penalty_total` (0.80) | how **erratically** traces turn | raise if jitter persists |
| `w_path_clearance_bonus` (3.00) | path quality vs. endpoint spacing | raise if traces still crowd to buy spacing |
| `terminal_clearance_target_mm` (14) | clearance grading scale | raise if `q_clear` pins at 1.0 |
| `min_point_shift_mm` (20) | portfolio variety | raise if entries look alike |
| `budget_mm` / `step_mm` | trace length and episode length | **recheck γ** |

**After any change, rerun the acceptance checks:**

- `self_crossing_check` — a tight fold must be caught; straight runs and corners legal
- `self_penalty_check` — silent on straight runs and wide meanders, fires on tight coils
- `turn_penalty_check` — adjacent-direction alternation must be free
- `zero_violation_check` — zero audited violations; compare `frac_survived`, not
  completion, which saturates
- `reward_scale_check` — discount-aware

---

## 10. Failure modes and metric traps

| Symptom | Meaning | Fix |
|---|---|---|
| `length_spread` = one `step_mm`, `terminal 0.00` | episode truncated — a trace boxed in | check `boxed_trace` |
| `most_boxed_trace` pinned to one value | one trace is structurally being walled in — the single most informative failure signal | raise constriction; run a forced-prefix sweep on that trace |
| `evals_since_best` climbing linearly | the run has stopped improving | enable/lower early stopping; check `eval_episodes` is large enough to mean anything |
| `gate_pass_rate` oscillating with no trend | policy sitting at a **feasibility cliff** — small updates flip episodes between success and failure. Usually downstream of one fragile trace | fix the fragile trace first; do not restructure the reward |
| `gate_pass_rate` rises then falls monotonically | risk-seeking drift | raise `w_terminal_base`; roll back to best checkpoint |
| A `q_*` metric pins at 1.0 | that term saturated — no longer ranks or teaches | raise its target |
| A dense term uses ~0% of budget | inert — verify it measures what you think | test against a synthetic case |
| Dense rises while terminal falls | classic inversion | recheck γ and the scale check |
| Portfolio fills with lookalikes | diversity threshold too loose | raise `min_point_shift_mm` |

### Metric traps

**`turn_rate` is not a smoothness metric.** It is *mean* turning, so long straight runs
with a few sharp corners score low while gentle curvature scores high — and it gives a
smooth arc and pure jitter identical scores. Sharpness is capped by the hard turn
limit; erratic turning is measured by **reversal rate**, which is zero for straight
runs, smooth arcs and mitred corners alike, and 1.0 for jitter.

**`mean_self_clearance` is not a coiling metric.** It is pinned near `step_mm` by
geometry regardless of shape. Use `d_self_far` with a correctly-sized lookback.

**Completion rate saturates.** At full budget random-policy completion is ~0%, so two
configurations both reading 0% cannot be compared. Use `frac_survived`.

**Averages hide tails.** The terminal measures mean clearance; the dense penalties
measure tail crowding. Improving the mean pays far more, so the policy improves the
mean. If tails matter, measure tails.

**General rule: every reward term and every metric should be tested against a synthetic
case where the answer is known, before being trusted in training.** Terms that silently
measure something other than their name do not show up as errors — they show up as
flat graphs, and flat graphs are easy to misread as "the policy is ignoring this."
