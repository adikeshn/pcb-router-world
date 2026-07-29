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


def turn_limit_check(cfg: Config, verbose: bool = True) -> Dict:
    """The hard turn limit must make a single-step 90-degree corner impossible
    while leaving a single-step 45-degree turn legal -- i.e. corners must be
    mitred. Also verifies the escape valve exists."""
    nd, m = cfg.n_dirs, cfg.max_turn_units
    u45 = nd // 8          # 45 deg in direction units
    u90 = nd // 4          # 90 deg
    legal_dirs = 2 * m + 1
    res = {"legal_dirs": legal_dirs, "u45": u45, "u90": u90,
           "45_in_one_step": u45 <= m, "90_in_one_step": u90 <= m,
           "passed": (u45 <= m) and (u90 > m)}
    if verbose:
        print(f"[turn_limit_check] {nd} directions, max_turn_units={m}")
        print(f"  legal directions per step     : {legal_dirs} of {nd}")
        print(f"  45 deg in one step (want YES) : {res['45_in_one_step']} "
              f"({'PASS' if res['45_in_one_step'] else 'FAIL - over-constrained'})")
        print(f"  90 deg in one step (want NO)  : {res['90_in_one_step']} "
              f"({'PASS - mitred' if not res['90_in_one_step'] else 'FAIL - sharp corners possible'})")
    return res


def reversal_check(cfg: Config, verbose: bool = True) -> Dict:
    """The reversal penalty must be SILENT on every desirable shape and FIRE on
    jitter. Mean turn magnitude cannot do this: it scores a smooth arc and pure
    jitter identically, and ranks long-straights-with-sharp-corners as best."""
    nd = cfg.n_dirs

    def rates(turns, memory=cfg.reversal_memory_steps):
        mag = float(np.mean([abs(t) for t in turns]))
        opp = rev = 0
        prev = 0
        run = 0
        for t in turns:
            sign = 0 if t == 0 else (1 if t > 0 else -1)
            if sign != 0 and prev != 0:
                opp += 1
                if sign * prev < 0:
                    rev += 1
            if sign != 0:
                prev = sign; run = 0
            else:
                run += 1
                if run > memory:
                    prev = 0
        return mag, (rev / opp if opp else 0.0)

    cases = [("straight run",      [0] * 12,                     False),
             ("smooth arc",        [1] * 12,                     False),
             ("tight smooth arc",  [2] * 12,                     False),
             ("mitred 90 corner",  [0, 0, 0, 2, 2, 0, 0, 0, 0],  False),
             ("two opposite corners (long straight between)",
                                   [0, 0, 2, 0, 0, 0, -2, 0, 0], False),
             ("jitter with one straight inserted (must not dodge)",
                                   [1, 0, -1, 0, 1, 0, -1],      True),
             ("JITTER",            [1, -1] * 6,                  True)]
    rows, ok = [], True
    for name, turns, want_fire in cases:
        mag, rev = rates(turns)
        fires = rev > 0.0
        rows.append((name, mag, rev, fires, want_fire))
        if fires != want_fire:
            ok = False
    if verbose:
        print("[reversal_check] synthetic shapes")
        print(f"  {'shape':40} {'mean magnitude':>15} {'reversal':>9}  verdict")
        for name, mag, rev, fires, want in rows:
            print(f"  {name:40} {mag:>15.2f} {rev:>9.2f}  "
                  f"{'FIRES' if fires else 'silent':7} "
                  f"({'PASS' if fires == want else 'FAIL'})")
        print("  note: mean magnitude gives 'smooth arc' and 'JITTER' the same")
        print("        score -- which is why sign, not magnitude, is charged.")
    return {"rows": rows, "passed": ok}


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
                                momentum=0.9 if i % 2 else 0.0)
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
    """TWO tests, because the two halves of the dense layer fail differently.

    1. FARMABILITY -- discounted POSITIVE dense reward vs the discounted
       terminal base. Positive dense reward can be collected while never
       completing an episode, so it must stay small. This is the test that
       catches a policy farming crumbs instead of finishing.
    2. COMPLETION DOMINANCE -- total PENALTIES vs the terminal base. Penalties
       cannot be farmed; the only thing to do with one is avoid it by routing
       well. The requirement is simply that completing while incurring every
       penalty still beats failing.

    A single combined test is wrong: it treats penalties as farmable and would
    reject configurations that weight routing quality properly, for no safety
    benefit."""
    env = RoundRobinTraceEnv(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    steps = cfg.episode_steps
    mean_disc = ((1 - cfg.gamma ** steps) / (steps * (1 - cfg.gamma))
                 if cfg.gamma < 1 else 1.0)
    disc_base = cfg.w_terminal_base * (cfg.gamma ** steps)

    pos, neg = [], []
    for _ in range(n_episodes):
        ed = run_random_episode(env, rng, momentum=0.9)
        t = ed["reward_terms"]
        pos.append(max(0.0, t["spacing_dense"]) * mean_disc)
        neg.append(sum(abs(t[k]) for k in t
                       if k not in ("terminal", "spacing_dense")))

    # theoretical worst case matters more than the sampled one for penalties
    pen_budget = (cfg.constriction_penalty_total + cfg.path_penalty_total
                  + cfg.self_penalty_total + cfg.reversal_penalty_total
                  + cfg.edge_penalty_total)
    farm_ratio = float(np.max(pos)) / max(disc_base, 1e-9)
    worst_complete = (cfg.w_terminal_base
                      + cfg.spacing_reward_coeff * math.sqrt(cfg.endpoint_spec_mm)
                      - pen_budget)

    res = {"episode_steps": steps, "horizon": 1.0 / (1.0 - cfg.gamma),
           "discounted_positive_dense": float(np.max(pos)),
           "discounted_terminal_base": disc_base,
           "farmability_ratio": farm_ratio,
           "penalty_budget": pen_budget,
           "worst_case_completed_score": worst_complete,
           "sampled_penalty_max": float(np.max(neg)) if neg else 0.0,
           "passed": farm_ratio < 0.35 and worst_complete > 0.5 * cfg.w_terminal_base}
    if verbose:
        print(f"[reward_scale_check] discount-aware (gamma={cfg.gamma}, "
              f"{steps} steps, horizon {res['horizon']:.0f})")
        print(f"  1. FARMABILITY  positive dense {res['discounted_positive_dense']:.3f} "
              f"vs terminal base {disc_base:.3f} -> ratio {farm_ratio:.3f} "
              f"(want < 0.35) {'PASS' if farm_ratio < 0.35 else 'FAIL'}")
        print(f"  2. COMPLETION   penalty budget {pen_budget:.2f}; a barely-passing "
              f"board still scores {worst_complete:.2f} "
              f"{'PASS' if worst_complete > 0.5*cfg.w_terminal_base else 'FAIL - penalties too large'}")
        print(f"     (sampled penalties actually incurred: max {res['sampled_penalty_max']:.3f})")
    return res


def run_all(cfg: Config, n_violation_eps: int = 100, seed: int = 0) -> bool:
    checks = [self_crossing_check(cfg), self_penalty_check(cfg),
              turn_limit_check(cfg), reversal_check(cfg),
              zero_violation_check(cfg, n_violation_eps, seed),
              reward_scale_check(cfg, seed=seed + 1)]
    ok = all(c["passed"] for c in checks)
    print(f"\n[validate] overall: {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}")
    return ok


if __name__ == "__main__":
    run_all(Config())
