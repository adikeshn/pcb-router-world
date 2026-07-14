"""Pre-training acceptance tests.

1. zero_violation_check — the milestone that proves v1's "residual 0-1
   crossings per episode" are designed out: N masked-random episodes must
   audit to ZERO clearance violations.

2. reward_scale_check — verifies the terminal reward strictly dominates
   the maximum plausible cumulative dense reward (the guard against the
   magnitude-imbalance exploit family from the v1 postmortem).

Run both from the notebook before training; both are also importable.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .config import Config
from .env import RoundRobinTraceEnv
from .explorer import run_random_episode


def zero_violation_check(cfg: Config, n_episodes: int = 200,
                         seed: int = 0, verbose: bool = True) -> Dict:
    env = RoundRobinTraceEnv(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    total_viol = 0
    completes = 0
    boxed = 0
    spacings = []
    for i in range(n_episodes):
        ed = run_random_episode(env, rng)
        total_viol += ed["violations"]
        completes += int(ed["complete"])
        boxed += int(ed["boxed_in"])
        spacings.append(ed["min_endpoint_spacing_mm"])
    result = {
        "episodes": n_episodes,
        "total_violations": total_viol,
        "completion_rate": completes / n_episodes,
        "boxed_in_rate": boxed / n_episodes,
        "min_endpoint_spacing_mean": float(np.mean(spacings)),
        "min_endpoint_spacing_p90": float(np.percentile(spacings, 90)),
        "passed": total_viol == 0,
    }
    if verbose:
        print(f"[zero_violation_check] {n_episodes} masked-random episodes")
        print(f"  total audited violations : {total_viol} "
              f"({'PASS' if total_viol == 0 else 'FAIL — geometry bug!'})")
        print(f"  completion rate          : {result['completion_rate']:.2%}")
        print(f"  boxed-in rate            : {result['boxed_in_rate']:.2%}")
        print(f"  min endpoint spacing     : mean {result['min_endpoint_spacing_mean']:.1f} mm, "
              f"p90 {result['min_endpoint_spacing_p90']:.1f} mm")
    return result


def reward_scale_check(cfg: Config, n_episodes: int = 50,
                       seed: int = 1, verbose: bool = True) -> Dict:
    env = RoundRobinTraceEnv(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    dense_sums = []
    terminals = []
    for _ in range(n_episodes):
        ed = run_random_episode(env, rng)
        t = ed["reward_terms"]
        dense = t["spacing_dense"] + t["edge_penalty"] + t["path_penalty"]
        dense_sums.append(dense)
        if ed["gate_pass"]:
            terminals.append(ed["reward_terminal"])
    max_dense = float(np.max(np.abs(dense_sums)))
    min_terminal_possible = cfg.w_terminal_base   # any gated episode gets >= base
    ratio = max_dense / max(min_terminal_possible, 1e-9)
    result = {
        "max_abs_cumulative_dense": max_dense,
        "terminal_base": min_terminal_possible,
        "dense_to_terminal_ratio": ratio,
        "gated_terminal_mean": float(np.mean(terminals)) if terminals else None,
        "passed": ratio < 0.5,
    }
    if verbose:
        print(f"[reward_scale_check] {n_episodes} masked-random episodes")
        print(f"  max |cumulative dense|   : {max_dense:.3f}")
        print(f"  terminal base (gated)    : {min_terminal_possible:.3f}")
        print(f"  ratio (want < 0.5)       : {ratio:.3f} "
              f"({'PASS' if result['passed'] else 'FAIL — rebalance weights'})")
        if terminals:
            print(f"  mean gated terminal      : {result['gated_terminal_mean']:.3f}")
    return result


def run_all(cfg: Config, n_violation_eps: int = 200,
            n_scale_eps: int = 50, seed: int = 0) -> bool:
    r1 = zero_violation_check(cfg, n_violation_eps, seed)
    r2 = reward_scale_check(cfg, n_scale_eps, seed + 1)
    ok = r1["passed"] and r2["passed"]
    print(f"\n[validate] overall: {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}")
    return ok


if __name__ == "__main__":
    run_all(Config())
