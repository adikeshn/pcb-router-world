"""ForcedExplorer: masked-random episodes that seed the portfolio.

PPO is on-policy so these never enter training. They exist only to offer the
portfolio layouts outside the current policy's basin.

Honest caveat: at the full growth budget on this board, random-policy
completion is ~0%, so the explorer contributes little. Do not rely on it for
portfolio diversity -- that is the diversity filter's job.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .config import Config
from .env import RoundRobinTraceEnv
from .portfolio import Portfolio


def run_random_episode(env: RoundRobinTraceEnv, rng: np.random.Generator,
                       budget_mm: Optional[float] = None,
                       momentum: float = 0.0, spread_bias: float = 0.0) -> Dict:
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
                tips = np.asarray([p[-1] for p in env.paths])
                tip = tips[tid]
                others = np.delete(tips, tid, axis=0)
                best_a, best_d = int(valid[0]), -1.0
                for k in valid:
                    d = float(np.min(np.linalg.norm(
                        others - (tip + env.DIRS[k] * env.cfg.step_mm), axis=1)))
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

    def run_burst(self, n_episodes: Optional[int] = None,
                  episode: Optional[int] = None,
                  steps: Optional[int] = None) -> Dict[str, float]:
        n = n_episodes or self.cfg.explorer_episodes
        profiles = [(self.cfg.explorer_momentum, 0.0), (0.7, 0.25),
                    (self.cfg.explorer_momentum, 0.0), (0.0, 0.0)]
        added = completes = 0
        survived = []
        for k in range(n):
            momentum, spread = profiles[k % len(profiles)]
            ed = run_random_episode(self.env, self.rng,
                                    momentum=momentum, spread_bias=spread)
            self.total_episodes += 1
            completes += int(ed["complete"])
            survived.append(ed["frac_survived"])
            if self.portfolio.consider(ed, episode=episode, steps=steps):
                added += 1
        return {"explorer/episodes_total": float(self.total_episodes),
                "explorer/burst_complete_rate": completes / max(n, 1),
                "explorer/burst_frac_survived": float(np.mean(survived)),
                "explorer/burst_portfolio_adds": float(added)}
