"""Training callbacks: W&B metrics/images, portfolio harvesting, explorer
scheduling, per-budget eval, checkpointing, and collapse detection.

Every board image pushed to W&B is stamped with the training episode and step
count that produced it.
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
from .explorer import ParameterNoiseExplorer
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
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        self.episodes = 0
        self._window = deque(maxlen=100)
        # boxed_trace + endpoints + paths, for forced-prefix targeting
        self._recent = deque(maxlen=60)
        self._explorer: Optional[ParameterNoiseExplorer] = None
        self._eval_env: Optional[RoundRobinTraceEnv] = None
        self._next_eval = cfg.eval_every_steps
        self._next_ckpt = cfg.checkpoint_every_steps
        self._t0 = time.time()
        self.best_eval_gate = -1.0
        self.best_eval_step = 0
        self._evals_since_best = 0

    def _init_callback(self) -> None:
        self._explorer = ParameterNoiseExplorer(self.cfg, self.portfolio,
                                                seed=self.cfg.seed + 777)
        self._eval_env = RoundRobinTraceEnv(self.cfg, seed=self.cfg.seed + 999)

    def recent_episodes(self):
        """Light episode records for stuckness scoring (forced-prefix targets)."""
        return list(self._recent)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            ed = info.get("episode_data")
            if ed is not None:
                self.episodes += 1
                self._handle_episode(ed)
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.cfg.eval_every_steps
            if not self._run_eval():
                return False
        if self.num_timesteps >= self._next_ckpt:
            self._next_ckpt += self.cfg.checkpoint_every_steps
            path = os.path.join(self.run_dir, "checkpoints",
                                f"model_{self.num_timesteps:09d}.zip")
            self.model.save(path)
            if self.verbose:
                print(f"[checkpoint] {path}")
        return True

    # ------------------------------------------------------------------ #
    def _handle_episode(self, ed: Dict[str, Any]) -> None:
        cfg = self.cfg
        changed = self.portfolio.consider(ed, episode=self.episodes,
                                          steps=self.num_timesteps)
        self._recent.append({"boxed_trace": ed["boxed_trace"],
                             "endpoints": ed["endpoints"], "paths": ed["paths"]})
        self._window.append({k: ed[k] for k in (
            "complete", "gate_pass", "meets_spec", "violations", "boxed_in",
            "boxed_trace", "frac_survived", "min_endpoint_spacing_mm",
            "min_endpoint_edge_mm", "min_self_distance_mm", "mean_self_far_mm",
            "mean_path_clearance_mm", "p5_path_clearance_mm", "min_freedom",
            "mean_freedom", "turn_rate", "turn_reversal_rate",
            "turn_limit_relaxations", "length_spread_mm", "reward_terminal",
            "redirects", "r_spacing", "q_clear", "q_clear_mean", "q_clear_tail",
            "q_smooth", "q_edge")} | {"terms": ed["reward_terms"]})

        if self.episodes % cfg.log_every_episodes == 0:
            w = list(self._window)
            gated = [e for e in w if e["gate_pass"]]
            mean = lambda k, src=w: float(np.mean([e[k] for e in src])) if src else 0.0
            terms = {k: float(np.mean([e["terms"][k] for e in w])) for k in w[0]["terms"]}
            data = {
                "train/episodes": self.episodes,
                # --- outcome ---
                "train/completion_rate": mean("complete"),
                "train/gate_pass_rate": mean("gate_pass"),
                "train/meets_spec_rate": mean("meets_spec"),
                "train/boxed_in_rate": mean("boxed_in"),
                "train/violations_mean": mean("violations"),
                "train/redirects_mean": mean("redirects"),
                # --- how far episodes get (works even at 0% completion) ---
                "train/frac_survived": mean("frac_survived"),
                # --- quality ---
                "train/min_endpoint_spacing_mm": mean("min_endpoint_spacing_mm"),
                "train/min_endpoint_edge_mm": mean("min_endpoint_edge_mm"),
                "train/mean_path_clearance_mm": mean("mean_path_clearance_mm"),
                "train/length_spread_mm": mean("length_spread_mm"),
                # --- shape / geometry ---
                "train/turn_rate": mean("turn_rate"),
                "train/turn_reversal_rate": mean("turn_reversal_rate"),
                "train/turn_limit_relaxations": mean("turn_limit_relaxations"),
                "train/p5_path_clearance_mm": mean("p5_path_clearance_mm"),
                "train/mean_self_far_mm": mean("mean_self_far_mm"),
                "train/min_freedom": mean("min_freedom"),
                "train/mean_freedom": mean("mean_freedom"),
                # --- terminal components (watch for saturation) ---
                "train/reward_terminal_mean": mean("reward_terminal"),
                "train/r_spacing": mean("r_spacing", gated),
                "train/q_clear": mean("q_clear", gated),
                "train/q_clear_mean": mean("q_clear_mean", gated),
                "train/q_clear_tail": mean("q_clear_tail", gated),
                "train/q_smooth": mean("q_smooth", gated),
                "train/terminal_when_gated": mean("reward_terminal", gated),
                # --- throughput ---
                "train/steps_per_sec": self.num_timesteps / max(time.time() - self._t0, 1e-9),
                **{f"reward/{k}": v for k, v in terms.items()}}
            # which trace boxes in most often -- pinpoints the blocking victim
            boxed = [e["boxed_trace"] for e in w if e["boxed_in"] and e["boxed_trace"] >= 0]
            if boxed:
                vals, counts = np.unique(boxed, return_counts=True)
                data["train/most_boxed_trace"] = int(vals[np.argmax(counts)])
                data["train/most_boxed_frac"] = float(counts.max() / len(w))
            sg = [e["min_self_distance_mm"] for e in w if e["min_self_distance_mm"] >= 0]
            if sg:
                data["train/min_self_gap_mm"] = float(np.min(sg))
            data.update({f"portfolio/{k}": v for k, v in self.portfolio.summary().items()})
            _wb_log(data, step=self.num_timesteps)

        if self.episodes % cfg.render_every_episodes == 0:
            png = os.path.join(self.run_dir, "renders", f"ep_{self.episodes:07d}.png")
            fig_to_png_path(episode_figure(
                self.cfg, ed, title="Training episode",
                episode=self.episodes, steps=self.num_timesteps), png)
            if wandb is not None and wandb.run is not None:
                _wb_log({"board/training_episode": wandb.Image(
                    png, caption=f"episode {self.episodes:,} | {self.num_timesteps:,} steps")},
                    step=self.num_timesteps)

        # portfolio images rendered on the logging cadence, not every change
        if (changed and wandb is not None and wandb.run is not None
                and self.episodes % cfg.log_every_episodes == 0):
            imgs = []
            for i, p in enumerate(self.portfolio.render()):
                e = self.portfolio.entries[i]
                imgs.append(wandb.Image(p, caption=(
                    f"rank {i} | terminal {e['reward_terminal']:.2f} | "
                    f"spacing {e['min_endpoint_spacing_mm']:.1f}mm | "
                    f"found at episode {e.get('found_at_episode')}")))
            _wb_log({"board/portfolio": imgs,
                     "portfolio/updates": self.portfolio.updates},
                    step=self.num_timesteps)

        # Parameter-noise explorer, gated on competence: perturbing an
        # incompetent policy just yields incompetent boards.
        if (self._explorer is not None
                and self.episodes % cfg.explorer_every_episodes == 0):
            if self.best_eval_gate >= cfg.explorer_min_gate_pass:
                _wb_log(self._explorer.run_burst(self.model, episode=self.episodes,
                                                 steps=self.num_timesteps),
                        step=self.num_timesteps)
            else:
                _wb_log({"explorer/skipped_low_gate": 1.0}, step=self.num_timesteps)

        # Online forced-prefix burst: automatically targets the most "stuck"
        # trace rather than a hard-coded one.
        if (cfg.prefix_every_episodes > 0
                and self.episodes % cfg.prefix_every_episodes == 0
                and self.best_eval_gate >= cfg.explorer_min_gate_pass
                and len(self._recent) >= 10):
            try:
                from .forced_prefix import (stuckness_scores, direction_order,
                                            run_forced_prefix)
                recent = list(self._recent)
                sc = stuckness_scores(recent, cfg.n_traces, cfg.board_diag_mm)
                t = int(np.argmax(sc))
                dirs = direction_order(recent, t, cfg.n_dirs)
                rounds = max(1, int(round(cfg.prefix_fractions[0] * cfg.budget_rounds)))
                adds = 0
                for d in dirs[:cfg.prefix_episodes_per_burst]:
                    ed2 = run_forced_prefix(self.model, self._eval_env, t, d, rounds)
                    if self.portfolio.consider(ed2, episode=self.episodes,
                                               steps=self.num_timesteps):
                        adds += 1
                _wb_log({"prefix/target_trace": float(t),
                         "prefix/target_stuckness": float(sc[t]),
                         "prefix/portfolio_adds": float(adds)},
                        step=self.num_timesteps)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    def _run_eval(self) -> bool:
        """Deterministic eval. Returns False to stop training early."""
        assert self._eval_env is not None
        cfg = self.cfg
        results = []
        for i in range(cfg.eval_episodes):
            obs, _ = self._eval_env.reset(seed=cfg.seed + 10_000 + i)
            while True:
                mask = self._eval_env.action_masks()
                action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
                obs, _, term, trunc, info = self._eval_env.step(int(action))
                if term or trunc:
                    ed = info["episode_data"]
                    results.append(ed)
                    self.portfolio.consider(ed, episode=self.episodes,
                                            steps=self.num_timesteps)
                    break
        m = lambda k: float(np.mean([e[k] for e in results]))
        gate = m("gate_pass")
        data = {"eval/completion_rate": m("complete"), "eval/gate_pass_rate": gate,
                "eval/meets_spec_rate": m("meets_spec"),
                "eval/frac_survived": m("frac_survived"),
                "eval/min_endpoint_spacing_mm": m("min_endpoint_spacing_mm"),
                "eval/mean_path_clearance_mm": m("mean_path_clearance_mm"),
                "eval/turn_rate": m("turn_rate"), "eval/min_freedom": m("min_freedom"),
                "eval/reward_terminal_mean": m("reward_terminal")}

        # --- best-model tracking (a previous run lost its best policy) --- #
        if gate > self.best_eval_gate:
            self.best_eval_gate = gate
            self.best_eval_step = self.num_timesteps
            self._evals_since_best = 0
            self.model.save(os.path.join(self.run_dir, "model_best.zip"))
        else:
            self._evals_since_best += 1
        data["eval/best_gate_pass_rate"] = self.best_eval_gate
        data["eval/best_at_step"] = self.best_eval_step
        data["eval/evals_since_best"] = self._evals_since_best
        _wb_log(data, step=self.num_timesteps)

        best = max(results, key=lambda e: (e["gate_pass"], e["meets_spec"],
                                           e["reward_terminal"]))
        png = os.path.join(self.run_dir, "renders", f"eval_{self.num_timesteps:09d}.png")
        fig_to_png_path(episode_figure(
            self.cfg, best, title="Eval best",
            episode=self.episodes, steps=self.num_timesteps), png)
        if wandb is not None and wandb.run is not None:
            _wb_log({"board/eval_best": wandb.Image(
                png, caption=f"episode {self.episodes:,} | {self.num_timesteps:,} steps")},
                step=self.num_timesteps)

        # --- collapse guard --------------------------------------------- #
        if (cfg.early_stop_patience_evals > 0
                and self._evals_since_best >= cfg.early_stop_patience_evals
                and self.best_eval_gate > 0.2):
            print(f"\n[early stop] eval gate-pass has not improved on "
                  f"{self.best_eval_gate:.0%} (step {self.best_eval_step:,}) for "
                  f"{self._evals_since_best} evals. Best model is saved as "
                  f"model_best.zip.")
            return False
        return True
