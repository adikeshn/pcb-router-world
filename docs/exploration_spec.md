# Exploration specification — PCB round-robin trace router

How the solution space is searched, why the algorithm's built-in exploration is not
sufficient, and what compensates.

Companion documents: `reward_spec.md` (what "good" means) and `environment_spec.md`
(how the environment works).

**Reference configuration:** 10 traces, `budget_mm = 100`, `step_mm = 2.0` → 50 rounds
= 500 steps per episode. Each trace makes 50 decisions per episode. Those numbers drive
everything below.

---

## 1. Why this needs its own mechanism

The deliverable is a **portfolio of distinct valid layouts**, not one optimum. That
makes search quality as important as policy quality — and PPO, on its own, is a local
optimiser that will find one family of layouts and refine it indefinitely.

The characteristic failure looks like this: the policy settles on a topology, one trace
ends up structurally disadvantaged by its neighbours' routing, and that trace boxes in
episode after episode for millions of steps without the policy ever trying an
alternative for the *neighbours*. Meanwhile spacing and clearance keep improving, so
every reward signal says things are going well.

---

## 2. Why per-step action noise cannot fix it

**Exploration in PPO is per-step action noise, and per-step noise does not compound.**

For a trace to route in a genuinely different direction it needs roughly 10–15
*consecutive* unusual decisions — it moves once per round, so that is 10–15 rounds of
commitment before any payoff appears. If the policy assigns probability *p* to the
unusual direction:

| consecutive decisions | p = 0.10 | p = 0.30 | p = 0.50 |
|---|---|---|---|
| 5 | 1.0e-5 | 2.4e-3 | 3.1e-2 |
| 10 | 1.0e-10 | 5.9e-6 | 9.8e-4 |
| 15 | 1.0e-15 | 1.4e-8 | 3.1e-5 |

A 2.5 M-step run contains ~5,000 episodes. A converged policy sits nearer p = 0.05–0.15
for a direction it has learned to avoid. At p = 0.10 and 10 decisions, the expected
number of occurrences in a run is effectively zero.

Three mechanisms actively reinforce the trap:

1. **The critic defends the incumbent.** Once the policy reliably scores well, the
   value function learns those states are worth that much. Any deviation produces a
   *negative* advantage and the gradient pushes back.
2. **`clip_range` is designed to prevent large jumps.** The cap per update is what
   makes PPO stable, and it is the same thing that confines it to nearby strategies.
3. **Entropy is the wrong shape of noise.** `ent_coef` keeps individual decisions from
   becoming deterministic but produces *independent* randomness per step. What is
   needed is *correlated* deviation: one commitment that persists.

The analogy: editing a novel one word at a time, keeping changes that improve it. You
will polish every sentence and never restructure the plot, because getting from plot A
to plot B requires a hundred consecutive edits that each individually make the current
draft worse.

**Switching algorithms does not help.** SAC, DQN and other gradient methods share the
weakness. The classes that genuinely address it are non-gradient (population /
evolutionary) and search (MCTS), both discussed in §7.

---

## 3. Mechanism 1 — Forced-prefix rollouts

**What it does.** Take a trained policy. Override the first *L* rounds of one chosen
trace to move in a chosen direction, then hand control back to the policy for
everything else and every subsequent round.

**Why it works.** The multi-round commitment PPO cannot sample is supplied externally,
while the competent part — routing the other traces, and routing this one after the
prefix — still comes from the trained policy. The result is a board that is
*structurally* different but still well routed, rather than the incompetent boards a
random policy produces.

Nothing about this is board-specific. It needs no knowledge of which trace is in
trouble, which direction is promising, or what the board looks like: all three are
derived from logged statistics at run time.

### 3.1 Choosing which traces to perturb — automatic

Targets are ranked by a **stuckness score** over a window of recent episodes, using
only quantities the environment already logs:

| signal | meaning | source |
|---|---|---|
| `box_rate[t]` | fraction of episodes in which trace *t* was the one that boxed in | `episode_data.boxed_trace` |
| `endpoint_spread[t]` | spread of trace *t*'s endpoint across episodes, as a fraction of board diagonal | `episode_data.endpoints` |

```
stuckness[t] = box_rate[t] + (1 − clip(endpoint_spread[t] / spread_ref, 0, 1))
```

These capture the two distinct reasons a trace is worth perturbing:

* **High `box_rate`** — this trace keeps failing, so its current routing does not work.
* **Low `endpoint_spread`** — this trace lands in the same place every episode, so the
  policy is certain about it. Certainty is exactly where an untested alternative hides.

Traces are swept in descending stuckness order until the rollout budget is spent. On a
board with a different weak point, the same rule finds that one.

### 3.2 Choosing which directions to try — automatic

Record the trace's **typical early heading** under the unperturbed policy, then sweep
headings ordered by angular distance from it. Directions the policy already favours are
tested last or skipped.

This makes the sweep an experiment about *alternatives* rather than a re-measurement of
the status quo, and it adapts automatically: a trace the policy sends north gets tested
southward first.

### 3.3 Prefix lengths — scaled, not absolute

Expressed as **fractions of `budget_rounds`**, so the configuration transfers to any
budget or step size:

```
prefix_fractions = [0.1, 0.2, 0.3]        # 5, 10, 15 rounds at 50 rounds
```

Too short and the policy simply undoes the deviation; too long and the forced segment
dominates the board rather than testing an alternative.

### 3.4 The question the sweep answers

Every sweep runs **unperturbed** episodes first to establish a baseline. The output is
not a table of scores but an answer to one question:

> **Does any coherent deviation beat what the policy does on its own?**

| result | meaning | action |
|---|---|---|
| Several prefixes beat baseline | **Local optimum confirmed**, and the escape route is known | seed training from those layouts; the mechanisms below are justified |
| All prefixes ≈ baseline | Not a local optimum — a plateau of near-equivalent layouts | improve the reward's discrimination, not the exploration |
| All prefixes much worse / box in | The policy avoids those routes for a real geometric reason | it is correct; do not build machinery to rediscover a bad idea |

That trichotomy is the point of the mechanism, and it is board-independent.

### 3.5 Cost

| scope | episodes | wall clock |
|---|---|---|
| top-1 stuck trace, 16 dirs, 3 lengths | 48 | ~2 min |
| top-3 stuck traces, 8 dirs, 3 lengths | 72 | ~3 min |
| all 10 traces, 8 dirs, 3 lengths | 240 | ~9 min |

Every gated result is offered to the portfolio, so better layouts are captured
permanently even if training never rediscovers them.

### 3.6 Two modes

* **Offline sweep** — run against any saved checkpoint, produces a report. This is the
  diagnostic, and it should be **the first thing run before any retraining**, because
  it tests the premise everything else rests on.
* **Online** — during training, periodically take the current top-stuck trace, try a
  few forced prefixes, and offer the results to the portfolio. This makes it a standing
  exploration mechanism rather than a manual debugging step, at a cost of a few
  episodes per burst.

---

## 4. Mechanism 2 — Parameter-noise explorer

**What it does.** Perturb the *actor network weights* (`θ' = θ + σ·ε`), run a full
episode with the perturbed policy, then discard the weights.

**Why weight noise rather than action noise.** Action noise makes a trace wobble and
average back to the same place. Weight noise produces a policy that behaves
*consistently differently for an entire episode* — the correlated deviation §2 shows is
required. It is the undirected counterpart of forced-prefix: broader coverage, but it
cannot be aimed at a specific hypothesis. Run both.

**Calibration.** σ is adapted to hold the fraction of actions that differ from the
unperturbed policy in a target band (~10–20%). Too small and nothing changes; too large
and the policy becomes incoherent.

**Gating.** Runs only once eval gate-pass exceeds a threshold (default 20%). Perturbing
an incompetent policy yields incompetent boards.

**This replaces masked-random exploration**, which at full growth budget has a measured
completion rate of ~0% — random walks self-trap long before finishing, contributing
nothing to the portfolio while consuming compute.

---

## 5. Mechanism 3 — Multiple seeds

Run 3 × 2.5 M steps rather than 1 × 7.5 M, and merge the portfolios.

Each seed drifts into its own basin and converges on its own design family. One long
run yields one family polished thoroughly; three yield three. Since the deliverable is
a portfolio of *distinct* good boards, more families beats more polish.

This is the cheapest reliable escape from local optima and is standard practice — RL
results are essentially never reported from a single seed.

---

## 6. Mechanism 4 — Budget curriculum (optional, default off)

Random-policy completion against growth budget, measured on the reference board:

| budget | random completion |
|---|---|
| 20 mm | 50% |
| 30 mm | 29% |
| 40 mm | 6.7% |
| 100 mm | **0%** |

Training at full budget starts at the 0% end, which is why the first valid board can
take hundreds of thousands of steps to appear. A curriculum starting near 30 mm and
ramping to 100 mm lets the policy learn basic skills — fan out, do not crowd, do not
block — while success is common, then extend them.

**Two reservations, which is why it is off by default:**

1. It reintroduces budget variation, scheduled rather than sampled, into a design that
   deliberately fixes the budget for simplicity.
2. Transfer is not guaranteed. The policy observes `remaining` as a *fraction*, so that
   feature's distribution is stable, but absolute geometry differs: short traces
   neither need to spread as far nor block each other as much.

**Test it as a separate run** so its effect is attributable.

---

## 7. Deferred options, with trigger conditions

| Option | Why not now | What would change the decision |
|---|---|---|
| **Graph (GNN) encoder** | Improves *representation*, not exploration — it will not escape a basin already occupied. Genuinely the strongest architectural option available: permutation invariance means learning "avoid the nearby trace" rather than "avoid trace 6", directly relevant to blocking. | The policy keeps failing specifically at blocking after the exploration mechanisms are in place. |
| **Population-based training / CMA-ES** | Structurally the right answer to multi-modal landscapes, but costs N policies in parallel. Multiple seeds is the cheap approximation. | Forced-prefix shows large gaps between basins **and** seeds land in widely different places — i.e. the landscape is confirmed multi-modal. |
| **MCTS with the real simulator** | Priced out for training: at 50 simulations × depth 10 that is ~250,000 env steps per episode versus 500 now, roughly 500×. | Wanting a handful of maximally good boards at inference only, where cost per board does not matter. |
| **Learned world models (Dreamer / MuZero)** | **Inverted for this problem.** A world model exists for environments that cannot be simulated cheaply. This environment *is* a cheap, exact, deterministic simulator — learning a neural approximation of it is strictly worse than calling it. If lookahead is wanted, that is MCTS with the real simulator, not a learned model. | Nothing foreseeable. |
| **RND / intrinsic novelty** | State space is continuous tip positions, so nearly every state is "novel" and the bonus rewards undirected flailing as much as a genuinely different strategy. It also competes with the terminal reward, whose balance is carefully tuned and verified. | Forced-prefix and parameter noise both stall while the landscape is confirmed multi-modal. |

**No lock-in.** The durable asset is the environment — masking, geometry, reward
design, validation suite — and all of it is algorithm-agnostic behind the gymnasium
interface. Any future algorithm plugs into the same environment.

---

## 8. Measuring whether exploration is working

| Metric | Meaning |
|---|---|
| `portfolio/diversity_mm` | Mean endpoint displacement between portfolio entries. **The direct measure.** Near `min_point_shift_mm` means entries only just clear the diversity filter; high means real structural variety. |
| `portfolio/size` | Slots filled. Persistently below capacity means the diversity filter is rejecting everything — the search is producing one design. |
| `train/most_boxed_trace` | Which trace keeps dying. Pinned to one value means a structural block, and identifies the forced-prefix target. |
| `explorer/burst_portfolio_adds` | How often exploration finds something the portfolio keeps. Zero for a long stretch means σ is too small or the policy is too converged. |
| `explorer/action_change_rate` | Fraction of actions the weight perturbation alters. Should sit in the 10–20% band. |
| `eval/evals_since_best` | Climbing linearly means the run has stopped improving and is burning compute. |

**The signature of a local optimum:** `best_reward_terminal` rising while
`portfolio/diversity_mm` stays flat. Quality is improving, variety is not — the search
is polishing one design.

---

## 9. Order of operations

1. **Forced-prefix sweep against the current best checkpoint.** Minutes, no training,
   and it tests the premise everything else depends on.
2. **Train 3 seeds × 2.5 M** with the parameter-noise explorer enabled.
3. **Forced-prefix sweep on each result**, merge all portfolios.
4. Only then consider the curriculum, and only then a graph encoder.

The discipline that matters: change few things per run and make each one attributable.
Steps 1 and 3 are nearly free and answer specific questions; step 2 bundles only
changes that are individually justified and mutually compatible.
