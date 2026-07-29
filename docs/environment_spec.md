# Environment specification — PCB round-robin trace router

What the agent sees, how legal moves are determined, how each reward term is computed,
and how an episode ends.

Companion documents: `reward_spec.md` (what "good" means) and `exploration_spec.md`
(how the solution space is searched). This document covers **mechanism**.

---

## 1. The problem

Given a fixed PCB — a board outline, a connector footprint with pads on its perimeter,
and any keep-out obstacles — route one copper trace from each pad to a test point
elsewhere on the board, subject to manufacturing clearances.

Requirements, in priority order:

1. **Equal trace lengths.** All traces must finish the same length (signal-integrity
   fixtures require matched propagation delay).
2. **Endpoint spacing.** Test points must be as far apart as possible, with a 13 mm
   fixture minimum.
3. **Path separation.** Traces should stay clear of each other along their whole
   length, not only at endpoints.
4. **Total length.** Shorter is preferred, as a tie-breaker.

The deliverable is not one layout but a **portfolio of distinct valid layouts** for a
fixture engineer to choose among.

### Framing as an MDP

Traces grow simultaneously, one small step at a time, cycling through them in
round-robin order. One *round* is one step for every trace. Because each trace
advances exactly once per round, **all traces are always the same length** —
requirement 1 is satisfied by construction, with no reward term and therefore nothing
to game.

The agent's only decision is a compass direction for whichever trace is currently
active. Illegal directions are masked out before the policy sees them, so an invalid
board is not merely discouraged but structurally impossible.

Standard gymnasium interface; one `step()` moves one trace.

```
reset()  ->  observation
loop:
    action_masks()          -> which of the 16 directions are legal right now
    policy picks an action  -> restricted to legal directions
    step(action)            -> observation, reward, terminated, truncated, info
```

**Reference configuration** used throughout: 10 traces, 200 × 150 mm board,
`budget_mm = 100`, `step_mm = 2.0`, 16 directions. That gives **50 rounds = 500 steps**
per episode.

---

## 2. Board representation

All geometry in millimetres.

| Element | Representation | Role |
|---|---|---|
| Board outline | width × height rectangle | traces stay inside, inset by `edge_clearance_mm` |
| Connector | rectangle `(x0, y0, x1, y1)` | keep-out; traces may not enter |
| Obstacles | list of rectangles | additional keep-outs |
| Pins | list of `(x, y)` | trace start points, **on the connector edge** |

Two rules are enforced at construction and fail loudly:

- **No pin strictly inside the connector footprint.** An interior pin cannot escape
  without its trace crossing the connector body.
- **No two pins closer than `trace_clearance_mm`.** With no breakout phase, the first
  agent segments must already satisfy trace-to-trace clearance.

Traces are stored as polylines, one point appended per step.

---

## 3. Round-robin scheduling

At `reset()` the environment draws a random permutation of trace indices. That order is
**fixed for the whole episode** and exposed in the observation.

Steps 1–10 are round 1, steps 11–20 round 2, and so on.

Ordering is randomised per *episode* rather than per *round*. Per-round randomisation
makes the one-step dynamics unpredictable to the policy, adding noise without any
robustness benefit that per-episode randomisation does not already provide.

**Ordering by spatial proximity is not used.** It sounds appealing — spatially adjacent
traces moving consecutively — but the policy is a feedforward network with no memory
between steps, so consecutive steps being nearby changes nothing about what it can
perceive. GAE already propagates credit over roughly 20–50 steps, more than a full
round. A related change that *does* help is sorting the observation's tip-distance
block by proximity (§6), so "the nearest other trace" always occupies the same slot.

---

## 4. Action space

`Discrete(16)` — sixteen compass headings 22.5° apart, index 0 = North, increasing
clockwise. Each action extends the active trace by exactly `step_mm` along that
heading. Diagonals are unit-normalised, so **every step advances exactly `step_mm`**
regardless of direction.

### Why 2 mm steps

Step size trades routing resolution against episode length, and episode length drives
both credit assignment and throughput. Measured on the reference board, comparing 1 mm
against 2 mm at identical total trace length:

| budget | random-policy completion at 1 mm | at 2 mm |
|---|---|---|
| 20 mm | 33% | **50%** |
| 30 mm | 4% | **29%** |
| 40 mm | 0% | **6.7%** |

Coarser steps cost about 4% of legal directions per step, but **improve completion
several-fold**, because boxing in is dominated by self-trapping and a halved step count
means half as many opportunities to self-trap. Larger steps also raise the tightest
geometrically possible hairpin from 1 mm to 2 mm for free, and roughly double training
throughput.

Use `frac_survived` (§9), not completion rate, when comparing step sizes: at full
budget random completion is near 0% and saturates as a metric.

---

## 5. Action masking

A direction is masked out unless the resulting segment is legal, so the policy assigns
exactly **zero** probability to illegal moves. This is the environment's most important
property.

### 5.1 Recomputed every step, never cached

`action_masks()` recomputes the mask fresh for the active trace at every step, against
the full current segment set. Caching within a round allows one trace to move into
space another has just occupied.

### 5.2 What each candidate is tested against

For each direction the environment forms the candidate segment `a → b` and applies four
tests in order:

**1. Board bounds.** `b` must lie inside the board inset by `edge_clearance_mm`.

**2. Keep-outs, with escape mode.** For each keep-out rectangle:

- If tip `a` is *inside* that rectangle's clearance halo — true for a pin sitting on
  the connector edge — the move is legal only if it never touches the rectangle
  interior **and** strictly increases distance from it by at least
  `escape_progress_mm`. A monotone escape.
- Otherwise the segment may not come within `obstacle_clearance_mm` of the rectangle.

Escape mode exists because traces start on the connector edge, inside its halo, where
the standard rule would leave zero legal moves. Once a tip leaves the halo the standard
rule applies, which also makes re-entering impossible.

**3. Trace-to-trace clearance.** Minimum distance from the candidate to any segment of
any *other* trace must be ≥ `trace_clearance_mm`.

**4. Self clearance.** Minimum distance to the trace's *own* non-adjacent segments must
be ≥ `self_clearance_mm`.

### 5.3 The self-exemption rule

Two segments of the same trace are exempt from the self check only when
**topologically adjacent** (`|i − j| ≤ 1`) — they share an endpoint, so their distance
is trivially zero.

Any wider exemption creates a blind spot. A 3-step sequence N → SE → W is a legal
action sequence whose third segment genuinely *crosses* the first; under an
arc-length window both segments fall inside the window and the crossing is never
checked, by either the mask or the audit.

**Consequence: `self_clearance_mm` must be < `step_mm`.** With adjacency exemption, two
segments separated by one intervening segment sit exactly `step_mm` apart *even on a
dead-straight run*, so any larger value makes straight-line growth illegal. Asserted at
construction.

### 5.4 Turn limit — `max_turn_units = 2`

A step may change heading by at most 2 direction units (±45°), so 5 of 16 directions
are legal at any moment. A trace still on its pin has no heading and all 16 are open.

**Mitred corners by construction.** One unit is 22.5°, so at 2 units a 45° turn is
legal in a single step while **a 90° turn requires two steps** — precisely a mitred
corner, the PCB idiom. Sharp single-step right angles become geometrically impossible.

**Why a mask, not a penalty.** A soft turn penalty competing against a much larger
spacing reward simply gets traded away; a mask cannot be. This is also the only
mechanism preventing "visually messy but well-spaced" boards, and it works by removing
them from the reachable space rather than trying to score them correctly — which
matters because the obvious metric is inverted (§5.6).

**Minimum turn radius** ≈ 2.5 mm at 2 units (≈5 mm at 1 unit, ≈1.9 mm at 3).

**Emergency escape valve.** If no direction within the turn limit is geometrically
legal, the limit lifts for that step and all 16 are considered. Without this the turn
limit would *cause* boxing in rather than shaping routing. Relaxations are counted in
`episode_data.turn_limit_relaxations`; a high count means the limit is too tight for
the board.

`ban_reverse` additionally masks the exact 180° opposite of the current heading,
preventing a trace from retracing the step it just made.

### 5.5 Spatial hash

Segments are indexed in a uniform grid (~4 mm cells). A clearance query collects
candidates from overlapping cells plus a radius halo rather than testing every segment
on the board — roughly 10 distance computations instead of 500 at full episode length.

Average-case, not worst-case; the degenerate case of many segments in one cell is
prevented by the clearance constraints the environment itself enforces.

### 5.6 Two independent aspects of trace shape

"Smoothness" is two separable properties, measured differently and controlled by
different mechanisms. Conflating them produces a term that penalises good routing.

| aspect | question | mechanism |
|---|---|---|
| **Sharpness** | how far can a single step turn? | `max_turn_units` — a hard mask |
| **Erraticism** | does the trace keep flipping turn direction? | reversal penalty — a soft term |

**`turn_rate` measures neither usefully.** It is total turn units ÷ segments, i.e. mean
turning. Long straight runs punctuated by a few sharp corners score **low**, gentle
continuous curvature scores **high**, and a smooth arc and pure jitter score
**identically**:

| shape | mean turn magnitude | reversal rate |
|---|---|---|
| straight run | 0.00 | 0.00 |
| smooth arc | 1.00 | **0.00** |
| tight smooth arc | 2.00 | **0.00** |
| mitred 90° corner | 0.33 | **0.00** |
| **jitter** | 1.00 | **1.00** |
| S-curve | 1.00 | 0.09 |

A reward term built on mean turning therefore penalises the smooth arc as heavily as
jitter, while a `turn_rate`-based *quality* term would rank angular boards above
well-curved ones. **Reversal rate is zero for every desirable shape and 1.0 for
jitter**, which is why it is the term that ships.

`turn_rate` remains logged as a descriptive statistic. It is not used in the reward.

### 5.7 Reversal, defined precisely

A reversal is counted at step *t* when `sign(turn_t) × sign(turn_{t−1}) < 0` and both
are non-zero — the trace was turning one way and is now turning the other, on
consecutive steps.

Straight steps between two corners do **not** create a reversal: two separate corners
in opposite directions are ordinary routing, not jitter. Only immediate alternation
counts.

A side effect worth knowing: travelling at a heading between two of the 16 representable
directions requires alternating between them, which registers as reversals. The penalty
therefore holds traces on the discrete headings, which is how PCB traces are drawn.

### 5.8 Boxed in

If no direction is legal, the episode **truncates immediately** (`truncated=True`) and
the gate fails. `episode_data.boxed_trace` records which trace died.

Truncated episodes show `length_spread_mm` equal to one `step_mm`, because traces
earlier in the round order already took their step. **`length_spread` equal to one step
with `terminal 0.00` is the fastest tell that an episode boxed in**, even on a board
that otherwise reads `spec PASS, violations 0`.

---

## 6. Observation

A flat float32 vector, every component normalised to [−1, 1]. For 10 traces, 16 rays
and 16 directions: **113 values**.

| Block | Size | Contents |
|---|---|---|
| Tip positions | 20 | all tips, `(x/W, y/H)` rescaled |
| Headings | 20 | all current headings as unit `(dx, dy)`; `(0,0)` if not yet moved |
| Active one-hot | 10 | which trace is moving now |
| Remaining | 1 | fraction of budget left |
| Turn order | 10 | each trace's position in this episode's permutation |
| Ray-casts | 16 | distance from the **active** tip to the nearest obstruction in 16 directions, clipped at `ray_max_mm` |
| Action mask | 16 | the legal-direction mask, also passed separately to MaskablePPO |
| Tip distances | 9 | distance from active tip to each other tip, **sorted by proximity** |
| Min pairwise | 1 | current minimum distance between any two tips |
| Per-trace freedom | 10 | number of legal directions available to **each** trace |

**Ray-casts** are the workhorse for local navigation, marching outward in 1 mm
increments — independent of `step_mm`, so perception stays fine-grained even with
coarse growth steps.

**Tip distances are sorted** so the nearest other trace always occupies the same slot.
Unsorted, the policy learns "avoid trace 6" rather than "avoid whichever trace is
near", which does not transfer.

**Per-trace freedom** exists because the dominant failure mode is one trace walling in
another. Positions alone do not convey it — a trace can be far away and still about to
be blocked. Without this block the policy literally cannot see that another trace is
down to two legal directions. Each trace's mask is cached when that trace is active, so
the counts cost essentially nothing.

**No absolute budget value is included** — with a fixed budget it is a constant and
carries no information. `remaining` is kept because it varies within an episode.

---

## 7. Where each reward term is computed

### Per step, inside `step()`

The environment reuses distances already computed during validation; no extra geometry.

| Term | Computed from | Fires when |
|---|---|---|
| **path penalty** | `d_other` from `segment_valid` | `d_other < path_soft_mm` |
| **self penalty** | `d_self_far` — distance to own path more than `self_lookback_mm` behind, via cumulative arc length | `d_self_far < self_soft_mm` |
| **reversal penalty** | sign of heading change vs. the previous step's | `sign(turn_t) x sign(turn_{t-1}) < 0`, both non-zero |

**Two distinct self measurements exist and must not be confused:**

- `d_self_adjacent` — minimum over all non-adjacent own segments. Used for the **hard
  clearance check**. Pinned near `step_mm` by geometry, so it carries no shape
  information.
- `d_self_far` — minimum over own segments more than `self_lookback_mm` back along the
  path. Used for the **soft coiling penalty**. This is the one that detects coiling.

Sharing one measurement between these jobs makes the soft penalty inert: it fires on
every step of every trace regardless of shape.

Also accumulated per step: `min(d_other, terminal_clearance_target_mm)`. **The clip
happens before averaging** — that is what makes the terminal clearance metric measure
crowding rather than spread.

### Per round

| Term | Computed from |
|---|---|
| **spacing reward** | minimum pairwise distance among all tips, hinged at `dense_spacing_target_mm` |
| **constriction penalty** | `f_min` = smallest legal-direction count across all traces, charged when below `constriction_comfort` |

Constriction recomputes all traces' masks fresh at round end so the signal is accurate,
and charges within one round of the move that caused the constriction rather than at
episode end.

### At the final step

1. **Audit.** `audit_paths()` runs a brute-force O(n²) check over every segment pair —
   deliberately *not* sharing code with the spatial hash, so it catches bugs in the
   hash as well as in the policy. Same adjacency self-exemption rule.
2. **Gate.** `complete AND violations == 0`.
3. **Metrics** and **terminal reward** (see `reward_spec.md`).
4. **`episode_data`** — full paths, all metrics, per-term reward breakdown — attached to
   `info` for the callback, the portfolio and W&B.

---

## 8. Episode lifecycle

```
reset()
  ├─ clear the spatial hash
  ├─ seed each trace's path with its pin
  ├─ register each pin as a ZERO-LENGTH segment
  ├─ draw the turn-order permutation
  └─ compute dense weights = totals ÷ (rounds, steps)

step() x 500
  ├─ active trace = order[pointer]
  ├─ mask = action_masks()            [fresh, uncached]
  ├─ if no legal direction  -> TRUNCATE, gate fails
  ├─ apply move, append point, add segment to hash
  ├─ per-step reward terms
  └─ pointer++; on wrap: round++, per-round terms

_finish()
  └─ audit -> gate -> terminal reward -> episode_data
```

**The zero-length pin segment matters.** Registering each pin as a degenerate segment
means one trace's first move already cannot come within `trace_clearance_mm` of
another's pin, before that trace has moved.

---

## 9. Termination and progress measurement

| Outcome | Flags | Gate | Terminal |
|---|---|---|---|
| All traces reached the budget | `terminated=True` | passes if audit is clean | 12–25 |
| A trace boxed in | `truncated=True` | fails | 0 |

Boxing in carries no explicit negative reward — losing the terminal prize *is* the
penalty. The constriction penalty provides the nearer-term signal, since a terminal-only
signal arriving hundreds of steps after the harmful move is too diffuse to attribute.

**`frac_survived`** — the fraction of rounds completed before boxing in — is the primary
progress metric when completion is low. Completion rate saturates near 0% at full budget
and cannot distinguish configurations; `frac_survived` is continuous and has full
dynamic range even when nothing completes.

---

## 10. Guarantees

| Guarantee | Mechanism |
|---|---|
| No clearance violations | masking + independent brute-force audit |
| No trace enters a keep-out | clearance halos in the mask |
| No trace leaves the board | bounds check in the mask |
| All traces exactly equal length | round-robin scheduling |
| No illegal action ever selected | MaskablePPO logit masking |

These are structural, not learned. A completed episode is a valid board by
construction; the policy's job is to make it a *good* one.

---

## 11. Extending

| Change | What it touches |
|---|---|
| Different board / connector / pins | `Config` only; validation catches bad geometry |
| More directions | direction table, mask size, observation size; cost scales linearly and is small next to ray-casting |
| Different budget or step size | `Config`; **recheck γ** — `1/(1−γ) ≥ (budget_mm / step_mm) × n_traces` |
| New reward term | compute in `step()` or at round end; add to `_terms` so it logs separately |
| Different policy network (e.g. graph encoder) | features extractor only; the observation already exposes the relational information a graph encoder would consume |
| Continuous actions | major rewrite — validity becomes angular intervals and hard masking is lost |

**Before trusting any new reward term or metric, test it against a synthetic case where
you know the answer.** Terms that silently measure something other than their name are
the most common and most expensive failure in this system, and they are invisible in
training curves.
