"""Pre-training acceptance tests.

1. zero_violation_check  -- N masked-random episodes must audit to ZERO
   clearance violations (regression test for masking / geometry bugs).
2. self_crossing_check   -- a synthetic tight fold that a segment-count or
   arc-length exemption window would miss MUST be reported as a violation.
3. reward_scale_check    -- DISCOUNT-AWARE. The earlier version compared raw
   undiscounted sums and passed a configuration in which the dense reward was
   worth ~1.8x the terminal reward from the agent's actual (discounted) point
   of view. It now evaluates the quantity PPO really optimises, at BOTH ends
   of the budget range, since the balance degrades as episodes lengthen.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .config import Config
from .env import RoundRobinTraceEnv
from .explorer import run_random_episode
from .geometry import audit_paths


def zero_violation_check(cfg: Config, n_episodes: int = 200, seed: int = 0,
                         verbose: bool = True) -> Dict:
    env = RoundRobinTraceEnv(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    total_viol = completes = boxed = 0
    spacings, self_gaps = [], []
    for i in range(n_episodes):
        ed = run_random_episode(env, rng, momentum=cfg.explorer_momentum if i % 2 else 0.0)
        total_viol += ed["violations"]
        completes += int(ed["complete"])
        boxed += int(ed["boxed_in"])
        spacings.append(ed["min_endpoint_spacing_mm"])
        if ed["min_self_distance_mm"] >= 0:
            self_gaps.append(ed["min_self_distance_mm"])
    res = {"episodes": n_episodes, "total_violations": total_viol,
           "completion_rate": completes / n_episodes,
           "boxed_in_rate": boxed / n_episodes,
           "min_endpoint_spacing_mean": float(np.mean(spacings)),
           "min_self_gap_min": float(np.min(self_gaps)) if self_gaps else -1.0,
           "passed": total_viol == 0}
    if verbose:
        print(f"[zero_violation_check] {n_episodes} masked-random episodes")
        print(f"  total audited violations : {total_viol} "
              f"({'PASS' if total_viol == 0 else 'FAIL - geometry bug!'})")
        print(f"  completion rate          : {res['completion_rate']:.2%}")
        print(f"  boxed-in rate            : {res['boxed_in_rate']:.2%}")
        print(f"  smallest self-gap seen   : {res['min_self_gap_min']:.2f} mm "
              f"(hard floor {cfg.self_clearance_mm} mm)")
        print(f"  mean min endpoint spacing: {res['min_endpoint_spacing_mean']:.1f} mm")
    return res


def self_crossing_check(cfg: Config, verbose: bool = True) -> Dict:
    """A 3-step N, SE, W fold is a legal action sequence whose third segment
    genuinely crosses the first. Any exemption window wider than topological
    adjacency reports it as clean."""
    sq = 1.0 / math.sqrt(2.0)
    pts = np.array([[0.0, 0.0], [0.0, 1.0], [sq, 1 - sq], [sq - 1, 1 - sq]])
    viol, _, min_self = audit_paths([pts], cfg.trace_clearance_mm, cfg.self_clearance_mm)
    straight = np.array([[0.0, 0], [0, 1], [0, 2], [0, 3], [0, 4]])
    corner = np.array([[0.0, 0], [0, 1], [0, 2], [1, 2], [2, 2]])
    v_straight = audit_paths([straight], cfg.trace_clearance_mm, cfg.self_clearance_mm)[0]
    v_corner = audit_paths([corner], cfg.trace_clearance_mm, cfg.self_clearance_mm)[0]
    res = {"fold_violations": viol, "fold_min_self_gap": min_self,
           "straight_run_violations": v_straight, "corner_violations": v_corner,
           "passed": viol > 0 and v_straight == 0 and v_corner == 0}
    if verbose:
        print("[self_crossing_check] synthetic geometry")
        print(f"  tight fold (must be caught)   : {viol} violations "
              f"({'PASS' if viol > 0 else 'FAIL - blind spot!'})")
        print(f"  straight run (must be legal)  : {v_straight} "
              f"({'PASS' if v_straight == 0 else 'FAIL - over-strict!'})")
        print(f"  90-deg corner (must be legal) : {v_corner} "
              f"({'PASS' if v_corner == 0 else 'FAIL - over-strict!'})")
    return res


def reward_scale_check(cfg: Config, n_episodes: int = 24, seed: int = 1,
                       verbose: bool = True) -> Dict:
    """Compare DISCOUNTED cumulative dense reward against the DISCOUNTED
    terminal reward as seen from the start of an episode."""
    env = RoundRobinTraceEnv(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    out = {}
    worst = 0.0
    rows = []
    for budget in (cfg.budget_min_mm, cfg.budget_max_mm):
        steps = int(round(budget / cfg.step_mm)) * cfg.n_traces
        disc_terminal = cfg.w_terminal_base * (cfg.gamma ** steps)
        dense_disc = []
        for _ in range(max(2, n_episodes // 2)):
            ed = run_random_episode(env, rng, budget_mm=budget,
                                    momentum=cfg.explorer_momentum)
            t = ed["reward_terms"]
            raw = abs(t["spacing_dense"]) + abs(t["path_penalty"]) \
                + abs(t["self_penalty"]) + abs(t["turn_penalty"]) + abs(t["edge_penalty"])
            # crumbs arrive roughly uniformly through the episode
            mean_disc = (1 - cfg.gamma ** steps) / (steps * (1 - cfg.gamma)) \
                if cfg.gamma < 1 else 1.0
            dense_disc.append(raw * mean_disc)
        d = float(np.max(dense_disc))
        ratio = d / max(disc_terminal, 1e-9)
        worst = max(worst, ratio)
        rows.append((budget, steps, d, disc_terminal, ratio))
        out[f"budget_{budget:.0f}mm_ratio"] = ratio
    out["worst_ratio"] = worst
    out["passed"] = worst < 0.35
    if verbose:
        print(f"[reward_scale_check] discount-aware (gamma={cfg.gamma})")
        for budget, steps, d, dt, ratio in rows:
            print(f"  budget {budget:6.1f} mm ({steps:5d} steps): discounted dense "
                  f"{d:.3f} vs terminal {dt:.3f} -> ratio {ratio:.3f}")
        print(f"  worst ratio (want < 0.35): {worst:.3f} "
              f"({'PASS' if out['passed'] else 'FAIL - rebalance or raise gamma'})")
    return out


def run_all(cfg: Config, n_violation_eps: int = 200, n_scale_eps: int = 24,
            seed: int = 0) -> bool:
    r0 = self_crossing_check(cfg)
    r1 = zero_violation_check(cfg, n_violation_eps, seed)
    r2 = reward_scale_check(cfg, n_scale_eps, seed + 1)
    ok = r0["passed"] and r1["passed"] and r2["passed"]
    print(f"\n[validate] overall: {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}")
    return ok


if __name__ == "__main__":
    run_all(Config())
