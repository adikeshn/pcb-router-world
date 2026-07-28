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
from stable_baselines3.common.vec_env import DummyVecEnv

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


def train(cfg: Config, resume_model: Optional[str] = None) -> str:
    cfg.validate()
    run_name = cfg.wandb_run_name or f"rr3_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(cfg.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg.save_yaml(os.path.join(run_dir, "config.yaml"))

    if wandb is not None and cfg.wandb_mode != "disabled":
        wandb.init(project=cfg.wandb_project, name=run_name, config=cfg.to_dict(),
                   mode=cfg.wandb_mode, dir=run_dir)

    portfolio = Portfolio(cfg, os.path.join(run_dir, cfg.portfolio_dir))
    venv = DummyVecEnv([make_env(cfg, cfg.seed + i) for i in range(cfg.n_envs)])

    try:
        import tensorboard  # noqa: F401
        tb_dir = os.path.join(run_dir, "tb")
    except ImportError:
        tb_dir = None

    if resume_model:
        model = MaskablePPO.load(resume_model, env=venv, device=cfg.device)
        print(f"Resumed from {resume_model}")
    else:
        model = MaskablePPO("MlpPolicy", venv, learning_rate=cfg.learning_rate,
                            n_steps=cfg.n_steps, batch_size=cfg.batch_size,
                            n_epochs=cfg.n_epochs, gamma=cfg.gamma,
                            gae_lambda=cfg.gae_lambda, ent_coef=cfg.ent_coef,
                            clip_range=cfg.clip_range, max_grad_norm=cfg.max_grad_norm,
                            policy_kwargs=dict(net_arch=list(cfg.net_arch)),
                            tensorboard_log=tb_dir, seed=cfg.seed,
                            device=cfg.device, verbose=1)

    callback = RouterCallback(cfg, portfolio, run_dir)
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

    portfolio.render(force=True)   # final render of every entry
    model_path = os.path.join(run_dir, "model_final.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    print(f"Portfolio ({len(portfolio.entries)} entries) in "
          f"{os.path.join(run_dir, cfg.portfolio_dir)}")

    if wandb is not None and wandb.run is not None:
        for k, v in portfolio.summary().items():
            wandb.run.summary[f"portfolio/{k}"] = v
        imgs = [wandb.Image(p, caption=f"rank {i}")
                for i, p in enumerate(portfolio.render(force=True))]
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
