"""Training entry point."""
from __future__ import annotations

import argparse
import os
import time
from typing import Optional

# Headless script path: choose a non-interactive backend BEFORE matplotlib is
# imported, but never override one the caller already set (e.g. a notebook's
# "%matplotlib inline" -- see rendering.py).
os.environ.setdefault("MPLBACKEND", "Agg")

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .callbacks import RouterCallback
from .config import Config
from .env import RoundRobinTraceEnv, mask_fn
from .portfolio import Portfolio

try:
    import wandb
except ImportError:
    wandb = None


def make_env(cfg: Config, seed: int):
    def _thunk():
        return Monitor(ActionMasker(RoundRobinTraceEnv(cfg, seed=seed), mask_fn))
    return _thunk


def linear_schedule(initial: float, floor: float):
    """Anneal linearly from `initial` down to `floor`, not to zero.

    A constant high rate is destabilising late in training, but annealing all
    the way to zero freezes the policy into whatever basin it occupies by
    mid-run -- counterproductive when escaping a basin is the goal."""
    def f(progress_remaining: float) -> float:
        return floor + progress_remaining * (initial - floor)
    return f


def train(cfg: Config, resume_model: Optional[str] = None) -> str:
    cfg.validate()
    run_name = cfg.wandb_run_name or f"rr5_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(cfg.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg.save_yaml(os.path.join(run_dir, "config.yaml"))

    if wandb is not None and cfg.wandb_mode != "disabled":
        wandb.init(project=cfg.wandb_project, name=run_name, config=cfg.to_dict(),
                   mode=cfg.wandb_mode, dir=run_dir)

    portfolio = Portfolio(cfg, os.path.join(run_dir, cfg.portfolio_dir))
    # DummyVecEnv steps every env SERIALLY in one process on one core.
    # SubprocVecEnv is the one that actually uses multiple cores; on a
    # 12-core machine that is the difference between ~200 and ~1500 env
    # steps/sec, and env stepping is >90% of wall clock.
    vec_cls = SubprocVecEnv if cfg.subproc_vecenv and cfg.n_envs > 1 else DummyVecEnv
    venv = vec_cls([make_env(cfg, cfg.seed + i) for i in range(cfg.n_envs)])

    try:
        import tensorboard  # noqa: F401
        tb_dir = os.path.join(run_dir, "tb")
    except ImportError:
        tb_dir = None

    lr = (linear_schedule(cfg.learning_rate, cfg.lr_floor)
          if cfg.lr_anneal else cfg.learning_rate)

    if resume_model:
        model = MaskablePPO.load(resume_model, env=venv, device=cfg.device)
        print(f"Resumed from {resume_model}")
    else:
        model = MaskablePPO("MlpPolicy", venv, learning_rate=lr,
                            n_steps=cfg.n_steps, batch_size=cfg.batch_size,
                            n_epochs=cfg.n_epochs, gamma=cfg.gamma,
                            gae_lambda=cfg.gae_lambda, ent_coef=cfg.ent_coef,
                            clip_range=cfg.clip_range, max_grad_norm=cfg.max_grad_norm,
                            policy_kwargs=dict(net_arch=list(cfg.net_arch)),
                            tensorboard_log=tb_dir, seed=cfg.seed,
                            device=cfg.device, verbose=1)

    print(f"episode = {cfg.budget_rounds} rounds x {cfg.n_traces} traces = "
          f"{cfg.episode_steps} steps | gamma {cfg.gamma} -> horizon "
          f"{1/(1-cfg.gamma):.0f} ({1/(1-cfg.gamma)/cfg.episode_steps:.1f}x episode)")
    print(f"vec env = {vec_cls.__name__} x {cfg.n_envs} | max_turn_units "
          f"{cfg.max_turn_units} ({2*cfg.max_turn_units+1} of {cfg.n_dirs} dirs legal)")

    callback = RouterCallback(cfg, portfolio, run_dir, verbose=1)
    try:
        import tqdm, rich  # noqa: F401
        progress = True
    except ImportError:
        progress = False
    try:
        model.learn(total_timesteps=cfg.total_timesteps, callback=callback,
                    progress_bar=progress)
    except KeyboardInterrupt:
        print("Interrupted - saving current model and portfolio.")
    except Exception as exc:
        # Save before exiting: the portfolio is the deliverable and a mid-run
        # exception would otherwise skip every save below.
        import traceback
        print(f"\n*** TRAINING CRASHED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("\nSaving model and portfolio before exit.\n")

    model.save(os.path.join(run_dir, "model_final.zip"))

    # Forced-prefix sweep on the finished policy: answers whether the run
    # landed in a local optimum, and feeds any better layouts to the portfolio.
    try:
        from .forced_prefix import sweep
        print()
        sweep(cfg, model, callback.recent_episodes(), portfolio=portfolio,
              n_traces_to_try=1, n_dirs_to_try=cfg.n_dirs, n_baseline=8,
              seed=cfg.seed, out_dir=run_dir)
    except Exception as exc:                      # never let this kill a run
        print(f"[forced_prefix] skipped: {exc}")

    portfolio.render(force=True)
    print(f"\nmodel_final.zip saved")
    if callback.best_eval_gate >= 0:
        print(f"model_best.zip  = step {callback.best_eval_step:,} "
              f"(eval gate-pass {callback.best_eval_gate:.0%})")
    print(f"portfolio ({len(portfolio.entries)} entries) in "
          f"{os.path.join(run_dir, cfg.portfolio_dir)}")

    if wandb is not None and wandb.run is not None:
        for k, v in portfolio.summary().items():
            wandb.run.summary[f"portfolio/{k}"] = v
        wandb.run.summary["eval/best_gate_pass_rate"] = callback.best_eval_gate
        wandb.run.summary["eval/best_at_step"] = callback.best_eval_step
        imgs = []
        for i, p in enumerate(portfolio.render(force=True)):
            e = portfolio.entries[i]
            imgs.append(wandb.Image(p, caption=(
                f"rank {i} | terminal {e['reward_terminal']:.2f} | "
                f"found at episode {e.get('found_at_episode')}")))
        if imgs:
            wandb.log({"board/final_portfolio": imgs})
        wandb.finish()
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.timesteps:
        cfg = cfg.override(total_timesteps=args.timesteps)
    train(cfg, resume_model=args.resume)


if __name__ == "__main__":
    main()
