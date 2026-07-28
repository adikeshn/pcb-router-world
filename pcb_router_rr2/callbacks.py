"""Training callbacks: W&B metrics/images, portfolio harvesting, explorer
scheduling, and a per-budget deterministic eval suite.

The portfolio is fed from ALL THREE episode sources -- training rollouts,
eval episodes, and forced-exploration episodes -- so nothing is filtered by
provenance. `portfolio/considered` counts every episode offered.
"""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .config import Config
from .env import RoundRobinTraceEnv
from .explorer import ForcedExplorer
from .portfolio import Portfolio
from .rendering import episode_figure, fig_to_png_path

try:
    import wandb
except ImportError:
    wandb = None


def _wb_log(data: Dict[str, Any], step: Optional[int] = None) -> None:
    if wandb is not None and wandb.run is not None:
        wandb.log(data, step=step)


class RouterCallback(BaseCallback):
    def __init__(self, cfg: Config, portfolio: Portfolio, run_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.cfg = cfg
        self.portfolio = portfolio
        self.run_dir = run_dir
        os.makedirs(os.path.join(run_dir, "renders"), exist_ok=True)
        self.episodes = 0
        self._window = deque(maxlen=100)
        self._explorer: Optional[ForcedExplorer] = None
        self._eval_env: Optional[RoundRobinTraceEnv] = None
        self._next_eval = cfg.eval_every_steps
        self._t0 = time.time()

    def _init_callback(self) -> None:
        self._explorer = ForcedExplorer(self.cfg, self.portfolio, seed=self.cfg.seed + 777)
        self._eval_env = RoundRobinTraceEnv(self.cfg, seed=self.cfg.seed + 999)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            ed = info.get("episode_data")
            if ed is not None:
                self.episodes += 1
                self._handle_episode(ed)
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.cfg.eval_every_steps
            self._run_eval()
        return True

    # ------------------------------------------------------------------ #
    def _handle_episode(self, ed: Dict[str, Any]) -> None:
        cfg = self.cfg
        changed = self.portfolio.consider(ed)
        self._window.append({
            "complete": ed["complete"], "gate_pass": ed["gate_pass"],
            "meets_spec": ed["meets_spec"], "violations": ed["violations"],
            "min_ep": ed["min_endpoint_spacing_mm"], "terminal": ed["reward_terminal"],
            "redirects": ed["redirects"], "boxed_in": ed["boxed_in"],
            "length_spread": ed["length_spread_mm"], "budget": ed["budget_mm"],
            "turn_rate": ed["turn_rate"], "self_gap": ed["min_self_distance_mm"],
            "mean_self": ed["mean_self_clearance_mm"],
            "ep_edge": ed["min_endpoint_edge_mm"],
            "r_spacing": ed["r_spacing"], "q_clear": ed["q_clear"], "q_edge": ed["q_edge"],
            "terms": ed["reward_terms"]})

        if self.episodes % cfg.log_every_episodes == 0:
            w = list(self._window)
            gated = [e for e in w if e["gate_pass"]]
            terms = {k: float(np.mean([e["terms"][k] for e in w])) for k in w[0]["terms"]}
            data = {
                "train/episodes": self.episodes,
                "train/completion_rate": float(np.mean([e["complete"] for e in w])),
                "train/gate_pass_rate": float(np.mean([e["gate_pass"] for e in w])),
                "train/meets_spec_rate": float(np.mean([e["meets_spec"] for e in w])),
                "train/boxed_in_rate": float(np.mean([e["boxed_in"] for e in w])),
                "train/violations_mean": float(np.mean([e["violations"] for e in w])),
                "train/redirects_mean": float(np.mean([e["redirects"] for e in w])),
                "train/min_endpoint_spacing_mm": float(np.mean([e["min_ep"] for e in w])),
                "train/min_endpoint_edge_mm": float(np.mean([e["ep_edge"] for e in w])),
                "train/length_spread_mm": float(np.mean([e["length_spread"] for e in w])),
                "train/turn_rate": float(np.mean([e["turn_rate"] for e in w])),
                "train/mean_self_clearance_mm": float(np.mean([e["mean_self"] for e in w])),
                "train/reward_terminal_mean": float(np.mean([e["terminal"] for e in w])),
                "train/steps_per_sec": self.num_timesteps / max(time.time() - self._t0, 1e-9),
                # --- budget diagnostics -------------------------------- #
                # budget_mm_all should sit near the midpoint of the sampled
                # range; budget_mm_gated well below it means long budgets are
                # failing the gate rather than merely losing on reward.
                "train/budget_mm_all": float(np.mean([e["budget"] for e in w])),
                "train/budget_mm_gated": (float(np.mean([e["budget"] for e in gated]))
                                          if gated else 0.0),
                "train/budget_mm_expected": 0.5 * (cfg.budget_min_mm + cfg.budget_max_mm),
                # --- terminal components -------------------------------- #
                # r_spacing is UNCAPPED, so it should keep climbing; q_clear
                # and q_edge are capped, and pinning at 1.0 means that term
                # has saturated and stopped providing gradient.
                "train/r_spacing": (float(np.mean([e["r_spacing"] for e in gated]))
                                    if gated else 0.0),
                "train/q_clear": (float(np.mean([e["q_clear"] for e in gated]))
                                  if gated else 0.0),
                "train/q_edge": (float(np.mean([e["q_edge"] for e in gated]))
                                 if gated else 0.0),
                **{f"reward/{k}": v for k, v in terms.items()}}
            sg = [e["self_gap"] for e in w if e["self_gap"] >= 0]
            if sg:
                data["train/min_self_gap_mm"] = float(np.min(sg))
            data.update({f"portfolio/{k}": v for k, v in self.portfolio.summary().items()})
            _wb_log(data, step=self.num_timesteps)

        if self.episodes % cfg.render_every_episodes == 0:
            png = os.path.join(self.run_dir, "renders", f"ep_{self.episodes:07d}.png")
            fig_to_png_path(episode_figure(self.cfg, ed,
                                           title=f"Training episode {self.episodes}"), png)
            if wandb is not None and wandb.run is not None:
                _wb_log({"board/training_episode": wandb.Image(png)}, step=self.num_timesteps)

        # Portfolio images are rendered on the logging cadence rather than on
        # every change: with up to bands x per_band entries, re-rendering
        # inline on each update would cost real wall-clock in the train loop.
        if (changed and wandb is not None and wandb.run is not None
                and self.episodes % cfg.log_every_episodes == 0):
            imgs = [wandb.Image(p, caption=f"rank {i}")
                    for i, p in enumerate(self.portfolio.render())]
            _wb_log({"board/portfolio": imgs, "portfolio/updates": self.portfolio.updates},
                    step=self.num_timesteps)

        if (self._explorer is not None
                and self.episodes % cfg.explorer_every_episodes == 0):
            _wb_log(self._explorer.run_burst(), step=self.num_timesteps)

    # ------------------------------------------------------------------ #
    def _run_eval(self) -> None:
        assert self._eval_env is not None
        cfg = self.cfg
        budgets = np.linspace(cfg.budget_min_mm, cfg.budget_max_mm, cfg.eval_episodes)
        results = []
        data: Dict[str, Any] = {}
        for b in budgets:
            obs, _ = self._eval_env.reset(options={"budget_mm": float(b)})
            while True:
                mask = self._eval_env.action_masks()
                action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
                obs, _, term, trunc, info = self._eval_env.step(int(action))
                if term or trunc:
                    ed = info["episode_data"]
                    results.append(ed)
                    self.portfolio.consider(ed)
                    # per-budget breakdown: answers "are LONG budgets failing
                    # the gate, or just losing the ranking?" directly
                    tag = f"{b:.0f}mm"
                    data[f"eval_by_budget/gate_pass_{tag}"] = float(ed["gate_pass"])
                    data[f"eval_by_budget/min_spacing_{tag}"] = ed["min_endpoint_spacing_mm"]
                    data[f"eval_by_budget/terminal_{tag}"] = ed["reward_terminal"]
                    break
        data.update({
            "eval/completion_rate": float(np.mean([e["complete"] for e in results])),
            "eval/gate_pass_rate": float(np.mean([e["gate_pass"] for e in results])),
            "eval/meets_spec_rate": float(np.mean([e["meets_spec"] for e in results])),
            "eval/min_endpoint_spacing_mm": float(np.mean(
                [e["min_endpoint_spacing_mm"] for e in results])),
            "eval/turn_rate": float(np.mean([e["turn_rate"] for e in results])),
            "eval/reward_terminal_mean": float(np.mean([e["reward_terminal"] for e in results]))})
        _wb_log(data, step=self.num_timesteps)
        best = max(results, key=lambda e: (e["gate_pass"], e["meets_spec"], e["reward_terminal"]))
        png = os.path.join(self.run_dir, "renders", f"eval_{self.num_timesteps:09d}.png")
        fig_to_png_path(episode_figure(self.cfg, best,
                                       title=f"Eval best @ {self.num_timesteps} steps"), png)
        if wandb is not None and wandb.run is not None:
            _wb_log({"board/eval_best": wandb.Image(png)}, step=self.num_timesteps)
