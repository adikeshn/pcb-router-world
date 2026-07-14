"""Training callbacks: W&B metric/image logging, portfolio harvesting,
ForcedExplorer scheduling, and a deterministic eval suite.

Everything logs through wandb; images are written to file paths first
(never BytesIO — see v1 postmortem) and logged from disk.
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
except ImportError:  # pragma: no cover
    wandb = None


def _wb_log(data: Dict[str, Any], step: Optional[int] = None) -> None:
    if wandb is not None and wandb.run is not None:
        wandb.log(data, step=step)


class RouterCallback(BaseCallback):
    """Single callback owning all periodic work, driven by episode ends
    surfaced through the vec-env `infos` (the env attaches `episode_data`)."""

    def __init__(self, cfg: Config, portfolio: Portfolio, run_dir: str,
                 verbose: int = 0):
        super().__init__(verbose)
        self.cfg = cfg
        self.portfolio = portfolio
        self.run_dir = run_dir
        os.makedirs(os.path.join(run_dir, "renders"), exist_ok=True)
        self.episodes = 0
        self._window = deque(maxlen=100)   # recent episode_data (light fields)
        self._explorer: Optional[ForcedExplorer] = None
        self._eval_env: Optional[RoundRobinTraceEnv] = None
        self._next_eval = cfg.eval_every_steps
        self._t0 = time.time()

    # ------------------------------------------------------------------ #
    def _init_callback(self) -> None:
        self._explorer = ForcedExplorer(self.cfg, self.portfolio,
                                        seed=self.cfg.seed + 777)
        self._eval_env = RoundRobinTraceEnv(self.cfg, seed=self.cfg.seed + 999)

    # ------------------------------------------------------------------ #
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        for info in infos:
            ed = info.get("episode_data")
            if ed is None:
                continue
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
            "complete": ed["complete"],
            "gate_pass": ed["gate_pass"],
            "meets_spec": ed["meets_spec"],
            "violations": ed["violations"],
            "min_ep": ed["min_endpoint_spacing_mm"],
            "terminal": ed["reward_terminal"],
            "redirects": ed["redirects"],
            "boxed_in": ed["boxed_in"],
            "length_spread": ed["length_spread_mm"],
            "terms": ed["reward_terms"],
        })

        if self.episodes % cfg.log_every_episodes == 0:
            w = list(self._window)
            terms = {k: float(np.mean([e["terms"][k] for e in w]))
                     for k in w[0]["terms"]}
            data = {
                "train/episodes": self.episodes,
                "train/completion_rate": float(np.mean([e["complete"] for e in w])),
                "train/gate_pass_rate": float(np.mean([e["gate_pass"] for e in w])),
                "train/meets_spec_rate": float(np.mean([e["meets_spec"] for e in w])),
                "train/boxed_in_rate": float(np.mean([e["boxed_in"] for e in w])),
                "train/violations_mean": float(np.mean([e["violations"] for e in w])),
                "train/redirects_mean": float(np.mean([e["redirects"] for e in w])),
                "train/min_endpoint_spacing_mm": float(np.mean([e["min_ep"] for e in w])),
                "train/length_spread_mm": float(np.mean([e["length_spread"] for e in w])),
                "train/reward_terminal_mean": float(np.mean([e["terminal"] for e in w])),
                "train/steps_per_sec": self.num_timesteps / max(time.time() - self._t0, 1e-9),
                **{f"reward/{k}": v for k, v in terms.items()},
            }
            data.update({f"portfolio/{k}": v
                         for k, v in self.portfolio.summary().items()})
            _wb_log(data, step=self.num_timesteps)

        # periodic render of a real training episode
        if self.episodes % cfg.render_every_episodes == 0:
            png = os.path.join(self.run_dir, "renders",
                               f"ep_{self.episodes:07d}.png")
            fig = episode_figure(self.cfg, ed,
                                 title=f"Training episode {self.episodes}")
            fig_to_png_path(fig, png)
            if wandb is not None and wandb.run is not None:
                _wb_log({"board/training_episode": wandb.Image(png)},
                        step=self.num_timesteps)

        # portfolio changed -> log the new board images
        if changed and wandb is not None and wandb.run is not None:
            imgs = [wandb.Image(p, caption=f"rank {i}")
                    for i, p in enumerate(self.portfolio.image_paths())]
            _wb_log({"board/portfolio": imgs,
                     "portfolio/updates": self.portfolio.updates},
                    step=self.num_timesteps)

        # ForcedExplorer burst
        if (self._explorer is not None
                and self.episodes % cfg.explorer_every_episodes == 0):
            stats = self._explorer.run_burst()
            _wb_log(stats, step=self.num_timesteps)

    # ------------------------------------------------------------------ #
    def _run_eval(self) -> None:
        """Deterministic-policy episodes across the budget range."""
        assert self._eval_env is not None
        cfg = self.cfg
        budgets = np.linspace(cfg.budget_min_mm, cfg.budget_max_mm,
                              cfg.eval_episodes)
        results = []
        for b in budgets:
            obs, _ = self._eval_env.reset(options={"budget_mm": float(b)})
            while True:
                mask = self._eval_env.action_masks()
                action, _ = self.model.predict(
                    obs, deterministic=True, action_masks=mask)
                obs, _, term, trunc, info = self._eval_env.step(int(action))
                if term or trunc:
                    ed = info["episode_data"]
                    results.append(ed)
                    self.portfolio.consider(ed)
                    break
        data = {
            "eval/completion_rate": float(np.mean([e["complete"] for e in results])),
            "eval/gate_pass_rate": float(np.mean([e["gate_pass"] for e in results])),
            "eval/meets_spec_rate": float(np.mean([e["meets_spec"] for e in results])),
            "eval/min_endpoint_spacing_mm": float(np.mean(
                [e["min_endpoint_spacing_mm"] for e in results])),
            "eval/reward_terminal_mean": float(np.mean(
                [e["reward_terminal"] for e in results])),
        }
        _wb_log(data, step=self.num_timesteps)
        best = max(results, key=lambda e: (e["gate_pass"], e["meets_spec"],
                                           e["reward_terminal"]))
        png = os.path.join(self.run_dir, "renders",
                           f"eval_{self.num_timesteps:09d}.png")
        fig = episode_figure(self.cfg, best,
                             title=f"Eval best @ {self.num_timesteps} steps")
        fig_to_png_path(fig, png)
        if wandb is not None and wandb.run is not None:
            _wb_log({"board/eval_best": wandb.Image(png)},
                    step=self.num_timesteps)
