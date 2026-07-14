"""ForcedExplorer: full episodes under a masked-random policy.

Purpose in v2 is narrower than in v1: PPO is on-policy so these episodes
never enter training — they exist purely to keep seeding the diversity
portfolio with layouts outside the current policy's basin (the one job
ForcedExplorer was demonstrably good at).

The walk is PERSISTENT (repeat the previous direction with probability
`momentum`, else uniform over the mask): pure uniform random walks
self-coil and box in ~90%+ of episodes, while momentum 0.9 completes
~25% and regularly clears the 13 mm spec — measured on the 6-trace board.
A slice of pure-uniform episodes is kept for layout diversity.
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
                       momentum: float = 0.0) -> Dict:
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
            if prev >= 0 and mask[prev] and rng.random() < momentum:
                action = prev
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
        for k in range(n):
            # mostly persistent walks (complete often), some uniform (diverse)
            momentum = self.cfg.explorer_momentum if k % 4 != 3 else 0.0
            ed = run_random_episode(self.env, self.rng, momentum=momentum)
            self.total_episodes += 1
            completes += int(ed["complete"])
            if self.portfolio.consider(ed):
                added += 1
        return {
            "explorer/episodes_total": float(self.total_episodes),
            "explorer/burst_complete_rate": completes / max(n, 1),
            "explorer/burst_portfolio_adds": float(added),
        }
