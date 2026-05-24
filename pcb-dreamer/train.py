"""
Train DreamerV3 on PCB Test Point Placement.

Usage:
    python train.py --configs defaults          # full training (GPU)
    python train.py --configs defaults debug    # quick CPU test
"""

import argparse
import functools
import os
import pathlib
import sys

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
from parallel import Parallel, Damy
from dreamer import Dreamer

to_np = lambda x: x.detach().cpu().numpy()


def make_env(mode, env_id, seed=0, num_traces=8):
    env = PCBDreamerEnv(num_traces=num_traces, seed=seed + env_id)
    env = wrappers.OneHotAction(env)
    env = wrappers.TimeLimit(env, num_traces)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["defaults"])
    parser.add_argument("--logdir", type=str, default="./logdir/pcb")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_traces", type=int, default=8)
    args = parser.parse_args()

    # Load config
    config_path = pathlib.Path(__file__).parent / "configs.yaml"
    configs = yaml.YAML(typ="safe").load(config_path.read_text())
    config = {}
    for name in args.configs:
        assert name in configs, f"Config '{name}' not found in {list(configs.keys())}"
        config.update(configs[name])

    config["logdir"] = args.logdir
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

    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Creating environments...")
    train_envs = [Damy(make_env("train", i, config.seed, args.num_traces))
                  for i in range(config.envs)]
    eval_envs = [Damy(make_env("eval", i, config.seed + 100, args.num_traces))
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

    while agent._step < config.steps + config.eval_every:
        logger.write()

        if config.eval_episode_num > 0:
            print(f"\n[Step {agent._step}] Eval...")
            tools.simulate(
                functools.partial(agent, training=False),
                eval_envs, eval_eps, config.evaldir,
                logger, is_eval=True, episodes=config.eval_episode_num,
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
        )

        torch.save({
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }, logdir / "latest.pt")

    for env in train_envs + eval_envs:
        try: env.close()
        except: pass

    print("\nDone! tensorboard --logdir", logdir)


if __name__ == "__main__":
    main()
