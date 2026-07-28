"""Generate boards from a trained model.

The policy is the machine; the PORTFOLIO is the product. A single
deterministic rollout returns *a* layout, not the best one, and it may not
even pass the gate. Run many episodes and take the gated, diversity-filtered
top-K.

    from pcb_router_rr2.inference import solve
    pf = solve(cfg, "runs/<name>/model_best.zip", n_episodes=200)

Note the model is trained for ONE board and ONE budget. Truncating early
yields geometrically valid, length-matched boards, but not good ones: the
terminal reward only ever paid at the full budget, so mid-episode tips are
positioned for the final round, not for where they happen to stop.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from sb3_contrib import MaskablePPO

from .config import Config
from .env import RoundRobinTraceEnv
from .portfolio import Portfolio


def run_episode(model, env: RoundRobinTraceEnv, deterministic: bool = True,
                budget_mm: Optional[float] = None, seed: Optional[int] = None):
    opts = {"budget_mm": budget_mm} if budget_mm is not None else None
    obs, _ = env.reset(seed=seed, options=opts)
    while True:
        mask = env.action_masks()
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
        obs, _, term, trunc, info = env.step(int(action))
        if term or trunc:
            return info["episode_data"]


def solve(cfg: Config, model_path: str, n_episodes: int = 200,
          out_dir: Optional[str] = None, deterministic_frac: float = 0.1,
          seed: int = 0, verbose: bool = True) -> Portfolio:
    """Run `n_episodes` and return the filled portfolio.

    A small slice is deterministic (the policy's single best guess); the rest
    are sampled, which is what actually produces layout variety."""
    cfg.validate()
    out_dir = out_dir or os.path.join(cfg.out_dir, "inference")
    model = MaskablePPO.load(model_path, device=cfg.device)
    env = RoundRobinTraceEnv(cfg, seed=seed)
    pf = Portfolio(cfg, out_dir)
    n_det = max(1, int(n_episodes * deterministic_frac))
    gated = 0
    spacings = []
    for i in range(n_episodes):
        ed = run_episode(model, env, deterministic=(i < n_det), seed=seed + i)
        pf.consider(ed, episode=i, steps=None)
        if ed["gate_pass"]:
            gated += 1
            spacings.append(ed["min_endpoint_spacing_mm"])
        if verbose and (i + 1) % max(1, n_episodes // 10) == 0:
            print(f"  {i+1}/{n_episodes} episodes | {gated} passed the gate | "
                  f"portfolio {len(pf.entries)}/{cfg.portfolio_k}")
    pf.render(force=True)
    if verbose:
        print(f"\n{gated}/{n_episodes} episodes passed the gate "
              f"({gated/n_episodes:.0%})")
        if spacings:
            print(f"endpoint spacing: mean {np.mean(spacings):.1f} mm, "
                  f"best {np.max(spacings):.1f} mm")
        print(f"portfolio ({len(pf.entries)} entries) in {out_dir}")
    return pf
