"""ForcedExplorer: full episodes under a masked-random policy.

Purpose in v2 is narrower than in v1: PPO is on-policy so these episodes
never enter training — they exist purely to keep seeding the diversity
portfolio with layouts outside the current policy's basin (the one job
ForcedExplorer was demonstrably good at).

The walk mixes three ingredients, rotated per episode, because they trade
off differently (measured on the 10-trace no-breakout board):
* PERSISTENT (repeat previous direction w.p. `momentum`): completes most
  episodes (~17% at 0.95 vs 0% uniform) but spreads tips only moderately.
* SPREAD-SEEKING (w.p. `spread_bias`, pick the valid direction that most
  increases distance from the nearest other tip): completes fewer but
  reaches near-spec endpoint spacing when it does.  Too much of it traps
  traces against edges (0% completion at bias 0.4), hence the mix.
* UNIFORM: worst completion, kept as a small slice for layout diversity.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .config import Config
from .env import RoundRobinTraceEnv
from .portfolio import Portfolio


def run_random_episode(env: RoundRobinTraceEnv,
                       rng: np.random.Generator,
                       budget_mm: Optional[float] = None,
                       momentum: float = 0.0,
                       spread_bias: float = 0.0) -> Dict:
    from .env import DIRS
    options = {"budget_mm": budget_mm} if budget_mm is not None else None
    env.reset(options=options)
    last: Dict[int, int] = {}
    while True:
        mask = env.action_masks()
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            action = 0
        else:
            tid = env._active()
            prev = last.get(tid, -1)
            r = rng.random()
            if prev >= 0 and mask[prev] and r < momentum:
                action = prev
            elif r < momentum + spread_bias:
                # spread-seeking: valid direction that most increases the
                # distance from the nearest other tip
                tips = np.asarray([p[-1] for p in env.paths])
                tip = tips[tid]
                others = np.delete(tips, tid, axis=0)
                best_a, best_d = int(valid[0]), -1.0
                for k in valid:
                    q = tip + DIRS[k]
                    d = float(np.min(np.linalg.norm(others - q, axis=1)))
                    if d > best_d:
                        best_d, best_a = d, int(k)
                action = best_a
            else:
                action = int(rng.choice(valid))
            last[tid] = action
        _, _, term, trunc, info = env.step(action)
        if term or trunc:
            return info["episode_data"]


class ForcedExplorer:
    def __init__(self, cfg: Config, portfolio: Portfolio, seed: int = 12345):
        self.cfg = cfg
        self.portfolio = portfolio
        self.env = RoundRobinTraceEnv(cfg, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.total_episodes = 0

    def run_burst(self, n_episodes: Optional[int] = None) -> Dict[str, float]:
        n = n_episodes or self.cfg.explorer_episodes
        added = 0
        completes = 0
        # rotation of walk profiles: (momentum, spread_bias)
        profiles = [(self.cfg.explorer_momentum, 0.0),
                    (0.7, 0.25),
                    (self.cfg.explorer_momentum, 0.0),
                    (0.0, 0.0)]
        for k in range(n):
            momentum, spread = profiles[k % len(profiles)]
            ed = run_random_episode(self.env, self.rng,
                                    momentum=momentum, spread_bias=spread)
            self.total_episodes += 1
            completes += int(ed["complete"])
            if self.portfolio.consider(ed):
                added += 1
        return {
            "explorer/episodes_total": float(self.total_episodes),
            "explorer/burst_complete_rate": completes / max(n, 1),
            "explorer/burst_portfolio_adds": float(added),
        }
