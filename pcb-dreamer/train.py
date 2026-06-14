"""
Train DreamerV3 on PCB Test Point Placement.

Usage:
    python train.py --configs defaults          # full training (GPU)
    python train.py --configs defaults debug    # quick CPU test
"""

import argparse
import functools
import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml
import torch
from torch import distributions as torchd

# Prevent PyTorch distribution validation errors on discrete actions
torch.distributions.Distribution.set_default_validate_args(False)

import exploration as expl
import models
import tools
from envs import wrappers
from envs.dreamer_wrapper import PCBDreamerEnv
from parallel import Parallel, Dummy
from dreamer import Dreamer
from best_solution import DiverseSolutionTracker

to_np = lambda x: x.detach().cpu().numpy()


def make_env(mode, env_id, seed=0, num_traces=8, reward_version="v1"):
    env = PCBDreamerEnv(num_traces=num_traces, seed=seed + env_id,
                        reward_version=reward_version)
    env = wrappers.OneHotAction(env)
    env = wrappers.TimeLimit(env, num_traces)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env


def get_inner_env(env):
    """Unwrap the wrapper chain down to the raw TPPlacementEnv."""
    raw = env
    while not hasattr(raw, "_inner"):
        raw = raw.env
    return raw._inner


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pathlib.Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "nogit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["defaults"])
    parser.add_argument("--logdir", type=str, default=None,
                        help="Explicit logdir. If omitted, an auto-named "
                             "run directory under ./logdir/ is used.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for the auto-generated run directory "
                             "(ignored if --logdir is set).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_traces", type=int, default=8)
    parser.add_argument("--reward_version", type=str, default="v1",
                        choices=["v1", "v2", "v3"],
                        help="Reward formulation: v1 (original), v2 (graded "
                             "routability + capped terms), or v3 (length "
                             "matching prioritized above all else).")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of diverse solutions to keep.")
    parser.add_argument("--diversity_shift", type=float, default=13.0,
                        help="Min distance (mm) a test point must move to count "
                             "as relocated when judging layout diversity.")
    parser.add_argument("--diversity_frac", type=float, default=0.5,
                        help="Min fraction of test points that must each be "
                             "relocated for two layouts to be considered distinct.")
    parser.add_argument("--actor_entropy", type=float, default=None,
                        help="Override the actor entropy coefficient (config "
                             "default 3e-3). Higher = more exploration, useful "
                             "for filling a diverse top-K portfolio; too high "
                             "stops the policy committing to good layouts. "
                             "Try ~1e-2 first.")
    # Optional overrides for run-budget knobs, so a named config doesn't
    # need to be edited/duplicated just to change run length.
    parser.add_argument("--steps", type=float, default=None)
    parser.add_argument("--prefill", type=int, default=None)
    parser.add_argument("--eval_every", type=float, default=None)
    parser.add_argument("--eval_episode_num", type=int, default=None)
    parser.add_argument("--log_every", type=float, default=None)
    args = parser.parse_args()

    # Load config
    config_path = pathlib.Path(__file__).parent / "configs.yaml"
    configs = yaml.YAML(typ="safe").load(config_path.read_text())
    config = {}
    for name in args.configs:
        assert name in configs, f"Config '{name}' not found in {list(configs.keys())}"
        config.update(configs[name])

    for key in ["steps", "prefill", "eval_every", "eval_episode_num", "log_every"]:
        override = getattr(args, key)
        if override is not None:
            config[key] = override

    if args.actor_entropy is not None:
        # config["actor"] is a dict from yaml; override just the entropy coeff.
        config["actor"] = dict(config["actor"])
        config["actor"]["entropy"] = args.actor_entropy
        print(f"Actor entropy coefficient override: {args.actor_entropy}")

    commit = get_git_commit()
    if args.logdir is not None:
        logdir_str = args.logdir
    else:
        run_name = args.run_name or (
            f"{'-'.join(args.configs)}_t{args.num_traces}_s{args.seed}"
            f"_{commit}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        logdir_str = str(pathlib.Path("./logdir") / run_name)

    config["logdir"] = logdir_str
    config["seed"] = args.seed
    if args.device:
        config["device"] = args.device
    if config["device"] == "cuda:0" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config["device"] = "cpu"

    config["time_limit"] = args.num_traces

    # Convert to namespace
    config = argparse.Namespace(**config)

    tools.set_seed_everywhere(config.seed)
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print(f"Logdir: {logdir}")
    print(f"Device: {config.device}")
    print(f"Steps: {int(config.steps)}")
    print(f"Traces: {args.num_traces}")

    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)

    meta_path = logdir / "meta.json"
    if not meta_path.exists():
        meta = {
            "configs": args.configs,
            "num_traces": args.num_traces,
            "seed": args.seed,
            "reward_version": args.reward_version,
            "git_commit": commit,
            "created": datetime.now().isoformat(timespec="seconds"),
            "config": {
                k: (str(v) if isinstance(v, pathlib.Path) else v)
                for k, v in vars(config).items()
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Run metadata: {meta_path}")

    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Creating environments...")
    train_envs = [Dummy(make_env("train", i, config.seed, args.num_traces, args.reward_version))
                  for i in range(config.envs)]
    eval_envs = [Dummy(make_env("eval", i, config.seed, args.num_traces, args.reward_version))
                 for i in range(config.envs)]

    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    print(f"Action space: {config.num_actions} candidates")

    train_eps = tools.load_episodes(config.traindir, limit=config.dataset_size)
    eval_eps = tools.load_episodes(config.evaldir, limit=1)

    # Prefill
    state = None
    prefill = max(0, config.prefill - count_steps(config.traindir))
    if prefill > 0:
        print(f"Prefilling ({prefill} steps)...")
        random_actor = tools.OneHotDist(
            torch.zeros(config.num_actions).repeat(config.envs, 1)
        )
        def random_agent(o, d, s):
            action = random_actor.sample()
            return {"action": action, "logprob": random_actor.log_prob(action)}, None

        state = tools.simulate(
            random_agent, train_envs, train_eps, config.traindir,
            logger, limit=config.dataset_size, steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Prefill done. {len(train_eps)} episodes.")

    # Create agent
    print("Creating DreamerV3 agent...")
    dataset = tools.from_generator(
        tools.sample_episodes(train_eps, config.batch_length), config.batch_size
    )
    eval_dataset = tools.from_generator(
        tools.sample_episodes(eval_eps, config.batch_length), config.batch_size
    )

    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config, logger, dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    # Resume from checkpoint
    if (logdir / "latest.pt").exists():
        print("Resuming from checkpoint...")
        ckpt = torch.load(logdir / "latest.pt", map_location=config.device)
        agent.load_state_dict(ckpt["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, ckpt["optims_state_dict"])
        agent._should_pretrain._once = False

    # Train
    print(f"\n{'='*50}")
    print("Training DreamerV3 on PCB Test Point Placement")
    print(f"{'='*50}")

    # Diverse top-K solution tracker: this run trains on a single fixed board,
    # so the whole run is a search. Keep the best K *distinct* valid layouts
    # found across all episodes (train + eval), a portfolio of alternatives.
    tracker = DiverseSolutionTracker(
        logdir, get_inner_env(train_envs[0]).board,
        k=args.top_k,
        min_point_shift=args.diversity_shift,
        min_moved_frac=args.diversity_frac,
    )

    def track_episode(env, is_eval):
        status = tracker.update(
            get_inner_env(env), agent._step,
            source=("eval" if is_eval else "train"),
        )
        if status:
            print(f"  [solutions] {status}")

    while agent._step < config.steps + config.eval_every:
        logger.write()

        if config.eval_episode_num > 0:
            print(f"\n[Step {agent._step}] Eval...")
            tools.simulate(
                functools.partial(agent, training=False),
                eval_envs, eval_eps, config.evaldir,
                logger, is_eval=True, episodes=config.eval_episode_num,
                episode_callback=track_episode,
            )
            if config.video_pred_log:
                try:
                    video_pred = agent._wm.video_pred(next(eval_dataset))
                    logger.video("eval_openl", to_np(video_pred))
                except StopIteration:
                    pass

        print(f"[Step {agent._step}] Training...")
        state = tools.simulate(
            agent, train_envs, train_eps, config.traindir,
            logger, limit=config.dataset_size,
            steps=config.eval_every, state=state,
            episode_callback=track_episode,
        )

        torch.save({
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }, logdir / "latest.pt")

    if tracker.solutions:
        print(f"\nTop {len(tracker.solutions)} distinct valid layouts found "
              f"(ranked, 0=best, length-matching first):")
        for rank, s in enumerate(tracker.solutions):
            print(f"  [{rank}] spread={s['length_spread']:.3f} "
                  f"total_length={s['total_length']:.1f}mm "
                  f"min_spacing={s['min_tp_spacing']:.1f}mm "
                  f"(found @ step {s['step']}, {s['source']})")
        print(f"  -> {logdir / 'solutions'}/ (solution_*.json + solution_*.png)")
    else:
        print("\nNo fully-routable solution found during training.")

    for env in train_envs + eval_envs:
        try: env.close()
        except: pass

    print("\nDone! tensorboard --logdir", logdir)


if __name__ == "__main__":
    main()