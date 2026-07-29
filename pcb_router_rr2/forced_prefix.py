"""Forced-prefix rollouts.

Override the first L rounds of one trace to a chosen direction, then hand
control back to the trained policy for everything else and every subsequent
round.

The multi-round commitment PPO cannot sample is supplied externally, while the
competent part -- routing the other traces, and routing this one after the
prefix -- still comes from the trained policy. The result is a board that is
structurally different but still well routed, unlike the boards a random
policy produces.

Nothing here is board-specific. Which trace to perturb, which directions to
try and how long to hold them are all derived from logged statistics at run
time:

* WHICH TRACE -- a `stuckness` score from `box_rate` (how often this trace is
  the one that dies) and `endpoint_spread` (how little its endpoint varies).
  High box_rate means its current routing does not work; low spread means the
  policy is certain about it, and certainty is where an untested alternative
  hides.
* WHICH DIRECTION -- ordered by angular distance from the trace's own typical
  early heading, so the sweep tests ALTERNATIVES rather than re-measuring the
  status quo.
* HOW LONG -- fractions of budget_rounds, so the configuration transfers to
  any budget or step size.

The output is not a table of scores but an answer to one question: does any
coherent deviation beat what the policy does on its own?
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .env import RoundRobinTraceEnv, turn_units
from .portfolio import Portfolio


# --------------------------------------------------------------------------- #
def stuckness_scores(episodes: List[Dict], n_traces: int,
                     board_diag_mm: float, spread_ref_frac: float = 0.15
                     ) -> np.ndarray:
    """Rank traces by how much a forced deviation is worth trying.

    Uses only quantities the environment already logs.  Returns one score per
    trace; higher means a better perturbation target."""
    if not episodes:
        return np.zeros(n_traces)
    box = np.zeros(n_traces)
    for ed in episodes:
        t = ed.get("boxed_trace", -1)
        if t is not None and t >= 0:
            box[t] += 1
    box /= len(episodes)

    eps = np.asarray([ed["endpoints"] for ed in episodes])      # (E, n, 2)
    if eps.ndim != 3 or eps.shape[0] < 2:
        spread = np.full(n_traces, spread_ref_frac * board_diag_mm)
    else:
        centroid = eps.mean(axis=0, keepdims=True)
        spread = np.linalg.norm(eps - centroid, axis=2).mean(axis=0)

    ref = max(spread_ref_frac * board_diag_mm, 1e-9)
    static = 1.0 - np.clip(spread / ref, 0.0, 1.0)
    return box + static


def direction_order(episodes: List[Dict], trace: int, n_dirs: int,
                    prefix_rounds: int = 5) -> List[int]:
    """Directions ordered by angular distance from this trace's typical early
    heading, so alternatives are tested first and the status quo last."""
    headings = []
    for ed in episodes:
        pts = np.asarray(ed["paths"][trace])
        if len(pts) < 2:
            continue
        k = min(prefix_rounds, len(pts) - 1)
        v = pts[k] - pts[0]
        if np.linalg.norm(v) < 1e-9:
            continue
        headings.append(np.arctan2(v[0], v[1]))          # 0 = North, clockwise
    if not headings:
        return list(range(n_dirs))
    mean_ang = float(np.arctan2(np.mean(np.sin(headings)), np.mean(np.cos(headings))))
    typical = int(round(mean_ang / (2 * np.pi / n_dirs))) % n_dirs
    return sorted(range(n_dirs), key=lambda k: -turn_units(typical, k, n_dirs))


# --------------------------------------------------------------------------- #
def run_forced_prefix(model, env: RoundRobinTraceEnv, trace: int,
                      direction: int, prefix_rounds: int,
                      seed: Optional[int] = None) -> Dict:
    """One rollout with `trace` forced toward `direction` for `prefix_rounds`.

    The forced action respects masking: if the chosen direction is illegal at
    that moment the nearest legal one by angular distance is used, so a prefix
    can never create a violation."""
    obs, _ = env.reset(seed=seed)
    while True:
        mask = env.action_masks()
        active = env._active()
        if active == trace and env.rounds_done < prefix_rounds and mask.any():
            order = sorted(range(env.nd),
                           key=lambda k: turn_units(direction, k, env.nd))
            action = next(k for k in order if mask[k])
        else:
            action, _ = model.predict(obs, deterministic=False, action_masks=mask)
            action = int(action)
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            return info["episode_data"]


def sweep(cfg: Config, model, recent_episodes: List[Dict],
          portfolio: Optional[Portfolio] = None,
          n_traces_to_try: int = 1, n_dirs_to_try: int = 16,
          n_baseline: int = 8, seed: int = 0,
          out_dir: Optional[str] = None, verbose: bool = True) -> Dict:
    """Run a sweep and return a verdict, not just numbers."""
    env = RoundRobinTraceEnv(cfg, seed=seed)
    rounds = cfg.budget_rounds
    prefix_lens = sorted({max(1, int(round(f * rounds))) for f in cfg.prefix_fractions})

    # --- baseline: what the policy does unperturbed ---------------------- #
    base = []
    for i in range(n_baseline):
        ed = run_policy_episode_local(model, env, seed=seed + 1000 + i)
        base.append(ed)
        if portfolio is not None:
            portfolio.consider(ed)
    base_gated = [e["reward_terminal"] for e in base if e["gate_pass"]]
    base_best = max(base_gated) if base_gated else 0.0
    base_mean = float(np.mean(base_gated)) if base_gated else 0.0

    scores = stuckness_scores(recent_episodes or base, cfg.n_traces, cfg.board_diag_mm)
    targets = list(np.argsort(-scores)[:n_traces_to_try])

    rows = []
    for t in targets:
        dirs = direction_order(recent_episodes or base, int(t), cfg.n_dirs)[:n_dirs_to_try]
        for d in dirs:
            for L in prefix_lens:
                ed = run_forced_prefix(model, env, int(t), int(d), L,
                                       seed=seed + 7919 * int(t) + 31 * d + L)
                if portfolio is not None:
                    portfolio.consider(ed)
                rows.append({"trace": int(t), "direction": int(d),
                             "dir_name": env.DIR_NAMES[d], "prefix_rounds": L,
                             "gate_pass": ed["gate_pass"],
                             "terminal": ed["reward_terminal"],
                             "spacing_mm": ed["min_endpoint_spacing_mm"],
                             "boxed_trace": ed["boxed_trace"]})

    gated = [r for r in rows if r["gate_pass"]]
    best = max(gated, key=lambda r: r["terminal"]) if gated else None
    n_beat = sum(1 for r in gated if r["terminal"] > base_best)

    # --- verdict ---------------------------------------------------------- #
    if best is None:
        verdict = "ALL_FAIL"
        reading = ("No forced prefix completed. The policy avoids those routes for a "
                   "geometric reason and is likely correct -- do not build machinery "
                   "to rediscover a bad idea.")
    elif n_beat >= max(2, len(gated) // 10):
        verdict = "LOCAL_OPTIMUM"
        reading = (f"{n_beat} prefixes beat the unperturbed best. Local optimum "
                   f"confirmed and the escape route is known -- exploration is "
                   f"the bottleneck and is worth investing in.")
    elif best["terminal"] > base_best:
        verdict = "MARGINAL"
        reading = ("One or two prefixes edge out the baseline. Weak evidence of a "
                   "local optimum; treat as a plateau unless it reproduces.")
    else:
        verdict = "PLATEAU"
        reading = ("No prefix beats the baseline. This is a plateau of near-equivalent "
                   "layouts, not a local optimum -- improve the reward's "
                   "discrimination rather than the exploration.")

    result = {"verdict": verdict, "reading": reading,
              "baseline_best": base_best, "baseline_mean": base_mean,
              "baseline_gate_pass_rate": len(base_gated) / max(len(base), 1),
              "stuckness": {int(i): float(scores[i]) for i in range(cfg.n_traces)},
              "targets": [int(t) for t in targets],
              "prefix_rounds_tried": prefix_lens,
              "n_rollouts": len(rows), "n_gated": len(gated), "n_beat_baseline": n_beat,
              "best": best, "rows": rows}

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "forced_prefix_report.json"), "w") as f:
            json.dump(result, f, indent=2)

    if verbose:
        print(f"[forced_prefix] {len(rows)} rollouts | baseline best "
              f"{base_best:.2f} (gate {result['baseline_gate_pass_rate']:.0%})")
        print(f"  stuckness ranking : "
              + ", ".join(f"t{int(i)}={scores[i]:.2f}" for i in np.argsort(-scores)[:5]))
        print(f"  targets swept     : {result['targets']}")
        if best:
            print(f"  best prefix       : trace {best['trace']} -> {best['dir_name']} "
                  f"for {best['prefix_rounds']} rounds | terminal {best['terminal']:.2f} "
                  f"| spacing {best['spacing_mm']:.1f}mm")
        print(f"  beat baseline     : {n_beat} of {len(gated)} gated rollouts")
        print(f"\n  VERDICT: {verdict}\n  {reading}")
    return result


def run_policy_episode_local(model, env, seed=None):
    from .explorer import run_policy_episode
    return run_policy_episode(model, env, deterministic=False, seed=seed)
