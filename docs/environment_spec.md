# Environment specification — PCB round-robin router

How the environment works: what the agent sees, how legal moves are determined, how
each reward term is computed, and how an episode ends.

Companion document: `reward_spec_v5_fixed_budget.md` (why the reward is shaped the way
it is). This document covers **mechanism**; that one covers **intent**.

Assumed configuration: 10 traces, 200 × 150 mm board, `budget_mm = 100`, 16 directions,
**2 mm steps**.

---

## 1. The agent–environment loop

Standard gymnasium interface. One `step()` call moves **one** trace by `step_mm` (2 mm).

```
reset()  ->  observation
loop:
    action_masks()          -> which of the 16 directions are legal right now
    policy picks an action  -> masked to legal directions only
    step(action)            -> observation, reward, terminated, truncated, info
```

An episode is 50 rounds × 10 traces = **500 steps**. The agent never chooses which
trace moves — that is a fixed schedule (§3).

---

## 2. Board representation

Four kinds of geometry, all in millimetres:

| Element | Representation | Role |
|---|---|---|
| Board outline | width × height rectangle | traces must stay inside, inset by `edge_clearance_mm` |
| Connector | rectangle `(x0, y0, x1, y1)` | keep-out; traces may not enter |
| Obstacles | list of rectangles | additional keep-outs |
| Pins | list of `(x, y)` | trace start points, **on the connector edge** |

Two rules are enforced at construction by `Config.validate()` and fail loudly:

- **No pin may be strictly inside the connector footprint.** An interior pin cannot
  escape without its trace crossing the connector body.
- **No two pins may be closer than `trace_clearance_mm`.** With no breakout phase, the
  first agent segments must already satisfy trace-to-trace clearance.

Traces are stored as polylines — a growing list of points, one appended per step.

---

## 3. Round-robin scheduling

At `reset()` the environment draws a random permutation of the 10 trace indices. That
order is **fixed for the whole episode** and exposed in the observation.

Steps 1–10 are round 1 (each trace moves once, in permutation order), steps 11–20 are
round 2, and so on. Because every trace moves exactly once per round, **all traces are
always the same length** — this is where length matching comes from, not from reward.

Ordering is randomised per episode rather than per round. Per-round randomisation was
tried and rejected: it makes the one-step dynamics unpredictable to the policy, adding
noise for no robustness benefit that per-episode randomisation doesn't already give.

### Why not order by spatial proximity?

A natural proposal is to order traces so that spatially adjacent traces move
consecutively — pick one at random, then its nearest neighbour, and so on — on the
theory that the policy would "see the consequences" of its moves sooner.

**Not adopted, for three reasons:**

1. **The policy has no memory between steps.** It is a feedforward MLP: every decision
   is made purely from the current observation, with no hidden state carried from the
   previous step. Consecutive steps being spatially near each other therefore changes
   nothing about what the network can perceive.
2. **The credit horizon already covers a full round.** With GAE at λ = 0.95, temporal
   difference credit propagates effectively over roughly 20 steps. A round is 10 steps,
   so a move's effect on any other trace in the same round is already well inside the
   window. Reordering within that window buys nothing.
3. **It does not address the failure it is aimed at.** The blocking failure spans
   ~400 steps — trace 6 moves at round 20, trace 9 dies at round 60. No ordering within
   a 10-step round shortens that gap. The constriction penalty does, by charging at the
   end of the round in which the constriction appears.

There is also a cost: proximity order computed from fixed pin positions is
deterministic, removing the ordering randomisation that guards against overfitting to
one sequence. Computed from live tip positions it changes mid-episode, reintroducing
exactly the per-round unpredictability that was already rejected.

**A related change that would help.** Rather than reordering *when* traces move,
reorder *how other traces appear in the observation*. The "tip distances" block
currently lists the other nine traces in fixed index order. Sorting that block by
proximity — so the nearest other trace always occupies the same slot — gives the
network a consistent representation of "what is near me" instead of one that depends on
arbitrary trace numbering. That is a genuine representational improvement and cheap to
implement.

---

## 4. Action space

`Discrete(16)` — sixteen compass headings 22.5° apart, index 0 = North, increasing
clockwise. Each action extends the active trace by exactly `step_mm = 2.0` along that
heading. Diagonals are unit-normalised, so **every step advances exactly 2 mm**
regardless of direction.

Step size is a resolution/horizon trade. A larger step halves the episode (better
credit assignment, ~2× throughput, tighter hairpins geometrically impossible) at the
cost of coarser manoeuvring — the reachable set from any tip is 16 points at 2 mm
radius, so threading a narrow corridor is harder. Measure the boxed-in rate under the
random policy at each candidate value before committing.

Two constraints beyond geometry:

- **`ban_reverse`** — the exact 180° opposite of the trace's current heading is always
  masked out, preventing a trace from retracing the step it just made.
- **First step exemption** — a trace with no prior heading (still sitting on its pin)
  has no reverse to ban.

---

## 5. Action masking — the core mechanism

This is the environment's most important property. A direction is masked out unless
the resulting 1 mm segment is legal, so **the policy assigns exactly zero probability
to illegal moves**. Across 4.7 M steps of the previous run, `violations_mean` and
`redirects_mean` were both flat at zero.

### 5.1 Per-step, never cached

`action_masks()` recomputes the mask fresh for the currently active trace at every
step, against the full current segment set. An earlier version cached masks within a
round, which allowed trace B to move into space trace A had just occupied — the source
of "residual crossings" that were once written off as inherent to sequential growth.
They were not; they were a caching bug.

### 5.2 What each candidate direction is tested against

For each of the 16 directions, the environment forms the candidate segment
`a → b` (from the current tip, 1 mm along that heading) and calls
`segment_valid(trace_id, a, b)`, which applies four tests in order:

**1. Board bounds.** `b` must lie inside the board inset by `edge_clearance_mm`.

**2. Keep-outs, with escape mode.** For each keep-out rectangle:

- If the tip `a` is *inside* that rectangle's clearance halo — true for a pin sitting
  on the connector edge — the move is legal only if it never touches the rectangle
  interior **and** strictly increases distance from the rectangle by at least
  `escape_progress_mm` (0.25 mm). This is a monotone escape.
- Otherwise, the segment simply may not come within `obstacle_clearance_mm` of the
  rectangle.

Escape mode exists because there is no breakout phase: traces start on the connector
edge, inside its clearance halo, where the standard rule would leave zero legal moves.
Once a tip leaves the halo the standard rule applies, which also makes re-entering
impossible. With a breakout enabled, tips never start inside a halo and this reduces
to the standard rule — one code path, no special cases.

**3. Trace-to-trace clearance.** The minimum distance from the candidate segment to
any segment of any *other* trace must be ≥ `trace_clearance_mm` (1.33 mm).

**4. Self clearance.** The minimum distance to the trace's *own* non-adjacent segments
must be ≥ `self_clearance_mm` (0.60 mm).

### 5.3 The self-exemption rule

Two segments of the same trace are exempt from the self check only when they are
**topologically adjacent** (`|i − j| ≤ 1`) — they share an endpoint, so their distance
is trivially zero.

This replaced an arc-length exemption window that had a proven blind spot: the 3-step
sequence N → SE → W is a legal action sequence whose third segment genuinely *crosses*
the first, yet both sat inside the window and were never checked. Neither the mask nor
the audit reported a violation.

**Consequence: `self_clearance_mm` must be < `step_mm`.** With adjacency exemption, two
segments separated by one intervening segment sit exactly `step_mm` apart *even on a
dead-straight run*, so any value ≥ `step_mm` would make straight-line growth illegal.
This is asserted at construction.

At 2 mm steps this constraint has more headroom than at 1 mm: `self_clearance_mm` may
go as high as ~1.9. It is left at 0.60 for now, but raising it toward
`trace_clearance_mm` (1.33) is newly available and would ban near-reversal folds
outright. Weigh that against a likely rise in the boxed-in rate.

### 5.4 Spatial hash

Segments are indexed in a uniform grid (~4 mm cells). A clearance query collects
candidate segments from cells overlapping the query's bounding box plus a radius halo,
rather than testing against every segment on the board.

At 500 segments per episode this is the difference between roughly 900 distance
computations per test and roughly 10. The hash is average-case, not worst-case — but
the degenerate case (many segments in one cell) is geometrically prevented by the
clearance constraints the environment itself enforces.

### 5.5 Boxed in

If **no** direction is legal for the active trace, the trace is boxed in. The episode
**truncates immediately** with `terminated=False, truncated=True`, and the gate fails.

This is why truncated episodes show `length_spread_mm` equal to one `step_mm` (2.00):
traces earlier in the round order already took their step, so they are one step longer
than the rest.

---

## 6. Observation

A flat float32 vector, every component normalised to [−1, 1]. For 10 traces, 16 rays
and 16 directions the layout is **113 values**:

| Block | Size | Contents |
|---|---|---|
| Tip positions | 20 | all 10 tips, `(x/W, y/H)` rescaled to [−1, 1] |
| Headings | 20 | all 10 current headings as unit `(dx, dy)`; `(0,0)` if not yet moved |
| Active one-hot | 10 | which trace is moving now |
| Remaining | 1 | fraction of the budget left |
| Turn order | 10 | each trace's position in this episode's permutation |
| Ray-casts | 16 | distance from the **active** tip to the nearest obstruction in 16 directions, clipped at `ray_max_mm` (20 mm) |
| Action mask | 16 | the legal-direction mask, also supplied separately to MaskablePPO |
| Tip distances | 9 | distance from the active tip to each other tip (see §3 on sorting these by proximity) |
| Min pairwise | 1 | current minimum distance between any two tips |
| **Per-trace freedom** | **10** | **number of legal directions available to each trace** |

**Ray-casts** are the workhorse for local navigation. Each marches outward in 1 mm
increments (independent of `step_mm`, so ray resolution stays fine) until it hits an obstruction — board edge, keep-out halo, another trace, or
the trace's own non-adjacent path — and reports the distance. This gives the policy
local geometry without any image processing.

**Per-trace freedom is new** and exists to fix the blocking failure. Previously the
observation contained other traces' *positions* but nothing about their *options*, so
when choosing trace 6's move the policy could not see that trace 9 was down to two
legal directions. Positions alone do not convey it: a trace can be far away and still
about to be walled off.

Implementation note: each trace's mask is stored whenever that trace is active, so the
counts are at most 9 steps stale and cost essentially nothing. If exact values matter,
recompute all masks once per round.

**Nothing about the budget's absolute value is included** — with a fixed budget it is
a constant and carries no information. `remaining` is kept because it varies within an
episode.

---

## 7. How each reward term is computed

### Per step, inside `step()`

The environment applies the chosen move and reuses the distances already computed
during validation — no extra geometry work.

| Term | Computed from | Condition |
|---|---|---|
| **path penalty** | `d_other`, returned by `segment_valid` | `d_other < path_soft_mm` (4 mm) |
| **self penalty** | `d_self_far` — distance to own path more than `self_lookback_mm` (12 mm) behind, tracked via cumulative arc length | `d_self_far < self_soft_mm` (3 mm) |
| **turn penalty** | direction index vs. previous heading | `turn_units > 1` (free band) |

**Two distinct self measurements exist and must not be confused:**

- `d_self_adjacent` — minimum over all non-adjacent own segments. Used for the **hard
  clearance check**. Pinned near 1.0 mm by geometry; carries no shape information.
- `d_self_far` — minimum over own segments more than 12 mm back along the path. Used
  for the **soft penalty**. This is the one that actually detects coiling.

Sharing a single measurement between these two jobs is what made the self penalty
inert in an earlier version: it fired on every step of every trace regardless of shape.

Also accumulated per step: `min(d_other, terminal_clearance_target_mm)` into a running
list. **The clip happens before averaging** — that is what makes the terminal clearance
metric measure crowding rather than spread.

### Per round (every 10 steps)

| Term | Computed from |
|---|---|
| **spacing reward** | minimum pairwise distance among all 10 tips, hinged at `dense_spacing_target_mm` |
| **constriction penalty** | `f_min` = smallest legal-direction count across all traces; charged when `f_min < F_comfort` (4) |

### At the final step, inside `_finish()`

1. **Audit.** `audit_paths()` runs a brute-force O(n²) check over every segment pair —
   deliberately *not* sharing code with the spatial hash, so it catches bugs in the
   hash as well as in the policy. It uses the same adjacency self-exemption rule.
2. **Gate.** `complete AND violations == 0`.
3. **Metrics.** Minimum endpoint spacing, mean path clearance, minimum self-gap,
   turn rate, length spread, endpoint-to-edge distance.
4. **Terminal reward**, if gated (see reward spec §5).
5. **`episode_data`** — full paths, all metrics, and the per-term reward breakdown —
   attached to `info` for the callback, the portfolio, and W&B.

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
means trace A's first move already cannot come within 1.33 mm of trace B's pin, even
before B has moved.

---

## 9. Termination

| Outcome | Flags | Gate | Terminal |
|---|---|---|---|
| All traces reached the budget | `terminated=True` | pass, if audit is clean | 12–25 |
| A trace boxed in | `truncated=True` | fail | 0 |

Boxing in carries no explicit negative reward — losing the terminal prize *is* the
penalty. The constriction penalty exists to provide a nearer-term signal, since a
terminal-only signal arriving 400 steps after the harmful move is too diffuse for
credit assignment.

---

## 10. Guarantees, and what they rest on

| Guarantee | Mechanism | Evidence |
|---|---|---|
| No clearance violations, ever | masking + independent audit | 4.7 M steps, `violations_mean` = 0 |
| No trace enters the connector or obstacles | keep-out halos in the mask | same |
| No trace leaves the board | bounds check in the mask | same |
| All traces exactly equal length | round-robin scheduling | `length_spread_mm` = 0.00 on every completed episode (a reading of one `step_mm` means it truncated) |
| Policy never selects an illegal action | MaskablePPO logit masking | `redirects_mean` = 0 |

These are structural, not learned. A completed episode is a valid board by
construction — the policy's job is to make it a *good* one.

---

## 11. Extending the environment

| Change | What it touches |
|---|---|
| Different board / connector / pins | `Config` only; validation catches bad geometry |
| More directions | `Discrete(n)`, direction table, mask size, observation size; mask cost scales linearly |
| Different budget or step size | `Config`; **recheck γ** — `1/(1−γ) ≥ (budget_mm / step_mm) × n_traces`. At 100 mm / 2 mm × 10 traces that is 500, comfortably inside γ = 0.999 |
| New reward term | compute in `step()` or at round end; add to `_terms` so it logs separately |
| Continuous actions | major rewrite — validity becomes angular intervals, and hard masking is lost (see reward spec §4) |

**Before trusting any new reward term, test it against a synthetic case where you know
the answer.** Three terms in this project silently measured something other than their
name, and every one was caught by hand-building an example and checking the number
rather than by reading the training curves.
