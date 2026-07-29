"""Parameter-noise explorer.

Perturbs the ACTOR WEIGHTS, runs a whole episode with the perturbed policy,
then discards the weights.

Why weight noise rather than action noise: action noise makes a trace wobble
and average back to the same place. Weight noise produces a policy that
behaves *consistently differently for an entire episode* -- the correlated,
multi-round deviation that per-step noise cannot produce. Reaching a
structurally different layout needs ~10-15 consecutive unusual decisions,
which at a converged policy's ~0.1 probability per decision is unreachable by
independent sampling.

These episodes never enter PPO training (it is on-policy); they exist only to
offer the portfolio layouts outside the current policy's basin.

Gated on competence: perturbing an incompetent policy yields incompetent
boards, so nothing runs until eval gate-pass clears a threshold.
"""
from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np
import torch

from .config import Config
from .env import RoundRobinTraceEnv
from .portfolio import Portfolio


def run_random_episode(env: RoundRobinTraceEnv, rng: np.random.Generator,
                       momentum: float = 0.0,
                       seed: Optional[int] = None) -> Dict:
    """Masked-random episode with an optional persistent walk.

    Retained as a VALIDATION utility, not an exploration mechanism: at full
    growth budget a random policy completes ~0% of episodes, so it contributes
    nothing to the portfolio. It is useful for the acceptance checks, where the
    point is to exercise the geometry rather than to route well.

    `momentum` repeats the previous direction with that probability; a pure
    coin-flip walk self-traps almost immediately.
    """
    env.reset(seed=seed)
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


def run_policy_episode(model, env: RoundRobinTraceEnv, deterministic: bool = False,
                       seed: Optional[int] = None) -> Dict:
    obs, _ = env.reset(seed=seed)
    while True:
        mask = env.action_masks()
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
        obs, _, term, trunc, info = env.step(int(action))
        if term or trunc:
            return info["episode_data"]


class ParameterNoiseExplorer:
    """Adaptive-sigma weight perturbation.

    Sigma is scaled per-parameter by that tensor's own standard deviation, so
    one global knob works across layers, and is adapted after each burst to
    hold the fraction of changed actions near a target band. Too small and
    nothing changes; too large and the policy becomes incoherent.
    """

    def __init__(self, cfg: Config, portfolio: Portfolio, seed: int = 12345):
        self.cfg = cfg
        self.portfolio = portfolio
        self.env = RoundRobinTraceEnv(cfg, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.sigma = cfg.explorer_sigma
        self.total_episodes = 0
        self.last_action_change = 0.0

    # ------------------------------------------------------------------ #
    def _perturbed(self, model):
        """Return a deep copy of the model with perturbed actor weights."""
        clone = copy.deepcopy(model.policy)
        with torch.no_grad():
            for name, p in clone.named_parameters():
                if "value" in name:          # leave the critic alone
                    continue
                std = float(p.detach().float().std())
                if std == 0.0 or not np.isfinite(std):
                    continue
                noise = torch.randn_like(p) * (self.sigma * std)
                p.add_(noise)
        return clone

    def _measure_action_change(self, model, clone, n_states: int = 64) -> float:
        """Fraction of actions the perturbation alters on real states."""
        obs, _ = self.env.reset()
        base_a, pert_a = [], []
        for _ in range(n_states):
            mask = self.env.action_masks()
            if not mask.any():
                break
            a0, _ = model.predict(obs, deterministic=True, action_masks=mask)
            with torch.no_grad():
                t = torch.as_tensor(obs).float().unsqueeze(0)
                mt = torch.as_tensor(mask).unsqueeze(0)
                dist = clone.get_distribution(t, action_masks=mt)
                a1 = int(dist.distribution.probs.argmax())
            base_a.append(int(a0)); pert_a.append(a1)
            obs, _, term, trunc, _ = self.env.step(int(a0))
            if term or trunc:
                obs, _ = self.env.reset()
        if not base_a:
            return 0.0
        return float(np.mean([a != b for a, b in zip(base_a, pert_a)]))

    # ------------------------------------------------------------------ #
    def run_burst(self, model, n_episodes: Optional[int] = None,
                  episode: Optional[int] = None,
                  steps: Optional[int] = None) -> Dict[str, float]:
        n = n_episodes or self.cfg.explorer_episodes
        added = completes = 0
        survived = []
        change = 0.0
        for _ in range(n):
            clone = self._perturbed(model)
            change = self._measure_action_change(model, clone)
            wrapper = _PolicyWrapper(clone)
            ed = run_policy_episode(wrapper, self.env, deterministic=False)
            self.total_episodes += 1
            completes += int(ed["complete"])
            survived.append(ed["frac_survived"])
            if self.portfolio.consider(ed, episode=episode, steps=steps):
                added += 1

        # adapt sigma toward the target action-change rate
        target = self.cfg.explorer_target_action_change
        if change < target * 0.5:
            self.sigma *= 1.3
        elif change > target * 1.5:
            self.sigma /= 1.3
        self.sigma = float(np.clip(self.sigma, 1e-4, 1.0))
        self.last_action_change = change

        return {"explorer/episodes_total": float(self.total_episodes),
                "explorer/sigma": self.sigma,
                "explorer/action_change_rate": change,
                "explorer/burst_complete_rate": completes / max(n, 1),
                "explorer/burst_frac_survived": float(np.mean(survived)) if survived else 0.0,
                "explorer/burst_portfolio_adds": float(added)}


class _PolicyWrapper:
    """Minimal .predict() shim so a bare policy can drive an episode."""

    def __init__(self, policy):
        self.policy = policy

    def predict(self, obs, deterministic=False, action_masks=None):
        with torch.no_grad():
            t = torch.as_tensor(obs).float().unsqueeze(0)
            mt = None if action_masks is None else torch.as_tensor(action_masks).unsqueeze(0)
            dist = self.policy.get_distribution(t, action_masks=mt)
            a = dist.distribution.probs.argmax() if deterministic else dist.sample()
        return int(a), None
