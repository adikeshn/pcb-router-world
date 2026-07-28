"""Pre-training acceptance tests.

Every reward term in this project that turned out to be broken was caught by a
synthetic case where the right answer was known in advance, not by reading
training curves. These checks encode those cases.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .config import Config
from .env import RoundRobinTraceEnv, turn_units
from .explorer import run_random_episode
from .geometry import audit_paths


def self_crossing_check(cfg: Config, verbose: bool = True) -> Dict:
    """A 3-step N, SE, W fold is a legal action sequence whose third segment
    genuinely crosses the first. Any exemption window wider than topological
    adjacency reports it as clean. Straight runs and corners must stay legal."""
    s = cfg.step_mm
    sq = s / math.sqrt(2.0)
    fold = np.array([[0.0, 0.0], [0.0, s], [sq, s - sq], [sq - s, s - sq]])
    straight = np.array([[0.0, i * s] for i in range(6)])
    corner = np.array([[0.0, 0], [0, s], [0, 2 * s], [s, 2 * s], [2 * s, 2 * s]])
    v_fold = audit_paths([fold], cfg.trace_clearance_mm, cfg.self_clearance_mm)[0]
    v_str = audit_paths([straight], cfg.trace_clearance_mm, cfg.self_clearance_mm)[0]
    v_cor = audit_paths([corner], cfg.trace_clearance_mm, cfg.self_clearance_mm)[0]
    res = {"fold_violations": v_fold, "straight_violations": v_str,
           "corner_violations": v_cor,
           "passed": v_fold > 0 and v_str == 0 and v_cor == 0}
    if verbose:
        print("[self_crossing_check] synthetic geometry")
        print(f"  tight fold (must be caught)   : {v_fold} "
              f"({'PASS' if v_fold > 0 else 'FAIL - blind spot!'})")
        print(f"  straight run (must be legal)  : {v_str} "
              f"({'PASS' if v_str == 0 else 'FAIL - over-strict!'})")
        print(f"  90-deg corner (must be legal) : {v_cor} "
              f"({'PASS' if v_cor == 0 else 'FAIL - over-strict!'})")
    return res


def self_penalty_check(cfg: Config, verbose: bool = True) -> Dict:
    """The soft self penalty must be SILENT on a straight run and on a wide
    meander, and FIRE on a tight coil. An earlier version measured the segment
    two steps back -- permanently step_mm away -- so it fired on every step of
    every trace, making it a flat tax rather than a coiling detector."""
    L, T, s = cfg.self_lookback_mm, cfg.self_soft_mm, cfg.step_mm

    def d_far(pts):
        segs, ends, acc = [], [], 0.0
        for k in range(len(pts) - 1):
            a, b = pts[k], pts[k + 1]
            acc += math.hypot(b[0] - a[0], b[1] - a[1])
            segs.append((a, b)); ends.append(acc)
        i = len(segs) - 1
        best = math.inf
        from .geometry import seg_seg_dist
        for j in range(len(segs) - 1):
            if ends[i] - ends[j] < L:
                continue
            best = min(best, seg_seg_dist(*segs[i], *segs[j]))
        return best

    n = int(40 / s) + 1
    straight = [(0.0, i * s) for i in range(n)]

    def meander(width):
        pts = [(0.0, 0.0)]; y = 0.0
        legs = max(2, int(10 / s))
        for lap in range(3):
            x = lap * width
            for i in range(legs): pts.append((x, y + (i + 1) * s))
            y += legs * s
            k = max(1, int(width / s))
            for i in range(k): pts.append((x + (i + 1) * s, y))
            for i in range(legs): pts.append((x + k * s, y - (i + 1) * s))
            y -= legs * s
            for i in range(k): pts.append((x + k * s + (i + 1) * s, y))
        return pts

    cases = [("straight run", straight, False),
             ("wide meander", meander(8.0), False),
             ("tight coil", meander(s), True)]
    rows, ok = [], True
    for name, pts, want_fire in cases:
        d = d_far(pts)
        fires = d < T
        rows.append((name, d, fires, want_fire))
        if fires != want_fire:
            ok = False
    if verbose:
        print(f"[self_penalty_check] lookback {L} mm, threshold {T} mm")
        for name, d, fires, want in rows:
            ds = f"{d:.1f}" if math.isfinite(d) else "inf"
            print(f"  {name:14s} d_self_far={ds:>5}  {'FIRES' if fires else 'silent'}"
                  f"   ({'PASS' if fires == want else 'FAIL'})")
    return {"rows": rows, "passed": ok}


def turn_penalty_check(cfg: Config, verbose: bool = True) -> Dict:
    """Alternating between ADJACENT directions is how a discrete grid
    approximates an intermediate heading. It must be free, or the penalty
    taxes a quantisation artifact rather than real turning."""
    nd = cfg.n_dirs
    quant = turn_units(0, 1, nd)                 # adjacent alternation
    real = turn_units(0, nd // 4, nd)            # a real 90-degree turn
    free = cfg.turn_free_units
    res = {"quantisation_units": quant, "real_turn_units": real,
           "quantisation_charged": max(0, quant - free),
           "real_turn_charged": max(0, real - free),
           "passed": quant <= free < real}
    if verbose:
        print(f"[turn_penalty_check] {nd} directions, free band {free} unit(s)")
        print(f"  adjacent alternation ({quant} unit): charged "
              f"{res['quantisation_charged']} ({'PASS - free' if res['quantisation_charged'] == 0 else 'FAIL'})")
        print(f"  real 90-deg turn ({real} units): charged "
              f"{res['real_turn_charged']} ({'PASS - costly' if res['real_turn_charged'] > 0 else 'FAIL'})")
    return res


def zero_violation_check(cfg: Config, n_episodes: int = 100, seed: int = 0,
                         verbose: bool = True) -> Dict:
    """Zero audited violations under a random policy.

    NOTE on `frac_survived`: at full growth budgets the random-policy
    completion rate is near 0%, which SATURATES completion as a metric -- two
    configurations both reading 0% cannot be compared. The mean fraction of
    the episode survived before boxing in is continuous, has full dynamic
    range even at 0% completion, and is the metric to compare when tuning
    step size or clearances."""
    env = RoundRobinTraceEnv(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    total_viol = completes = boxed = 0
    spacings, survived, self_gaps, freedom = [], [], [], []
    for i in range(n_episodes):
        ed = run_random_episode(env, rng,
                                momentum=cfg.explorer_momentum if i % 2 else 0.0)
        total_viol += ed["violations"]
        completes += int(ed["complete"])
        boxed += int(ed["boxed_in"])
        spacings.append(ed["min_endpoint_spacing_mm"])
        survived.append(ed["frac_survived"])
        freedom.append(ed["mean_freedom"])
        if ed["min_self_distance_mm"] >= 0:
            self_gaps.append(ed["min_self_distance_mm"])
    res = {"episodes": n_episodes, "total_violations": total_viol,
           "completion_rate": completes / n_episodes,
           "boxed_in_rate": boxed / n_episodes,
           "frac_survived": float(np.mean(survived)),
           "mean_freedom": float(np.mean(freedom)),
           "min_endpoint_spacing_mean": float(np.mean(spacings)),
           "min_self_gap_min": float(np.min(self_gaps)) if self_gaps else -1.0,
           "passed": total_viol == 0}
    if verbose:
        print(f"[zero_violation_check] {n_episodes} masked-random episodes")
        print(f"  total audited violations : {total_viol} "
              f"({'PASS' if total_viol == 0 else 'FAIL - geometry bug!'})")
        print(f"  completion rate          : {res['completion_rate']:.2%}"
              + ("   <- SATURATED; compare frac_survived instead"
                 if res['completion_rate'] < 0.02 else ""))
        print(f"  frac of episode survived : {res['frac_survived']:.3f}  "
              f"(continuous; use this to compare configs)")
        print(f"  mean legal directions    : {res['mean_freedom']:.2f} of {cfg.n_dirs}")
        print(f"  smallest self-gap seen   : {res['min_self_gap_min']:.2f} mm "
              f"(hard floor {cfg.self_clearance_mm})")
    return res


def reward_scale_check(cfg: Config, n_episodes: int = 8, seed: int = 1,
                       verbose: bool = True) -> Dict:
    """DISCOUNT-AWARE. An earlier raw-sum version passed a configuration in
    which the dense reward was worth ~1.8x the terminal reward from the
    agent's actual point of view."""
    env = RoundRobinTraceEnv(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    steps = cfg.episode_steps
    disc_terminal = cfg.w_terminal_base * (cfg.gamma ** steps)
    mean_disc = ((1 - cfg.gamma ** steps) / (steps * (1 - cfg.gamma))
                 if cfg.gamma < 1 else 1.0)
    dense = []
    for _ in range(n_episodes):
        ed = run_random_episode(env, rng, momentum=cfg.explorer_momentum)
        t = ed["reward_terms"]
        dense.append(sum(abs(t[k]) for k in t if k != "terminal") * mean_disc)
    d = float(np.max(dense))
    ratio = d / max(disc_terminal, 1e-9)
    res = {"episode_steps": steps, "discounted_dense": d,
           "discounted_terminal_base": disc_terminal, "ratio": ratio,
           "horizon": 1.0 / (1.0 - cfg.gamma), "passed": ratio < 0.35}
    if verbose:
        print(f"[reward_scale_check] discount-aware (gamma={cfg.gamma})")
        print(f"  episode {steps} steps | planning horizon "
              f"{res['horizon']:.0f} ({res['horizon']/steps:.1f}x episode)")
        print(f"  discounted dense {d:.3f} vs terminal base {disc_terminal:.3f}")
        print(f"  ratio (want < 0.35): {ratio:.3f} "
              f"({'PASS' if res['passed'] else 'FAIL - rebalance or raise gamma'})")
    return res


def run_all(cfg: Config, n_violation_eps: int = 100, seed: int = 0) -> bool:
    checks = [self_crossing_check(cfg), self_penalty_check(cfg),
              turn_penalty_check(cfg), zero_violation_check(cfg, n_violation_eps, seed),
              reward_scale_check(cfg, seed=seed + 1)]
    ok = all(c["passed"] for c in checks)
    print(f"\n[validate] overall: {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}")
    return ok


if __name__ == "__main__":
    run_all(Config())
