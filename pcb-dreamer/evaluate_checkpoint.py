"""
Evaluate a trained DreamerV3 checkpoint on PCB test point placement.

Loads `meta.json` from the run's logdir to reconstruct the exact config
used for training, runs N deterministic eval episodes, reports diagnostic
metrics (routability, total length, length spread, TP spacing, invalid
placements), renders board layouts for a few episodes, and compares against
the non-learned baselines from eval.py (random / greedy / spread).

Usage:
    python evaluate_checkpoint.py --logdir ./logdir/<run_name>
    python evaluate_checkpoint.py --logdir ./logdir/<run_name> \
        --checkpoint ckpt_step50000.pt --episodes 20

Output (written to <logdir>/eval_report/):
    summary.json     - model metrics + baseline comparison
    episode_*.png     - rendered board layouts for the first few episodes
"""

import argparse
import json
import os
import pathlib

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import torch

torch.distributions.Distribution.set_default_validate_args(False)

import tools
from dreamer import Dreamer
from train import make_env
from envs.board import load_te_example, generate_candidate_grid
from envs.visualize import plot_board
import eval as heuristic_eval


def load_config(logdir: pathlib.Path):
    meta = json.loads((logdir / "meta.json").read_text())
    cfg_dict = dict(meta["config"])
    cfg_dict["traindir"] = pathlib.Path(cfg_dict["traindir"])
    cfg_dict["evaldir"] = pathlib.Path(cfg_dict["evaldir"])
    return argparse.Namespace(**cfg_dict), meta


def strip_log_keys(obs):
    return {k: v for k, v in obs.items() if not k.startswith("log_")}


def find_inner_env(env):
    """Unwrap the wrapper chain down to the raw PCBDreamerEnv / TPPlacementEnv."""
    raw = env
    while not hasattr(raw, "_inner"):
        raw = raw.env
    return raw._inner


def run_episode(env, agent, agent_state):
    obs = env.reset()
    done = np.array([False])
    obs_batched = {k: np.array([v]) for k, v in strip_log_keys(obs).items()}
    ep_reward = 0.0
    info = {}
    while True:
        policy_output, agent_state = agent(obs_batched, done, agent_state, training=False)
        action = {k: np.array(policy_output[k][0].detach().cpu()) for k in policy_output}
        obs, reward, d, info = env.step(action)
        ep_reward += float(reward)
        done = np.array([d])
        if d:
            break
        obs_batched = {k: np.array([v]) for k, v in strip_log_keys(obs).items()}
    return ep_reward, info, agent_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="latest.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render_episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None,
                         help="Board seed to evaluate on. Default: the seed the "
                              "run trained on (from meta.json) -- i.e. the same "
                              "board. Pass a different value (e.g. a held-out seed) "
                              "to measure out-of-distribution generalization.")
    parser.add_argument("--skip_baselines", action="store_true")
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir)
    config, meta = load_config(logdir)
    num_traces = meta["num_traces"]

    # Default to the training board so eval reflects the single-board task.
    eval_seed = args.seed if args.seed is not None else meta.get("seed", 0)
    if eval_seed != meta.get("seed", 0):
        print(f"[eval] NOTE: evaluating on seed={eval_seed}, which differs from "
              f"the training seed={meta.get('seed', 0)} (out-of-distribution board).")

    outdir = logdir / "eval_report"
    outdir.mkdir(exist_ok=True)

    env = make_env("eval", 0, seed=eval_seed, num_traces=num_traces)
    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    logger = tools.Logger(outdir, 0)
    agent = Dreamer(env.observation_space, env.action_space, config, logger, None).to(config.device)
    agent.requires_grad_(False)

    ckpt_path = logdir / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location=config.device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()

    inner = find_inner_env(env)

    metric_keys = [
        "return", "failures", "routable", "total_length",
        "length_spread", "min_tp_spacing", "invalid_actions",
    ]
    metrics = {k: [] for k in metric_keys}

    agent_state = None
    for ep in range(args.episodes):
        ep_reward, info, agent_state = run_episode(env, agent, agent_state)
        metrics["return"].append(ep_reward)
        metrics["failures"].append(info.get("failures", 0))
        metrics["routable"].append(info.get("routable", 0.0))
        metrics["total_length"].append(info.get("total_length", 0.0))
        metrics["length_spread"].append(info.get("length_spread", 0.0))
        metrics["min_tp_spacing"].append(info.get("min_tp_spacing", 0.0))
        metrics["invalid_actions"].append(info.get("episode_invalid_actions", 0))

        if ep < args.render_episodes:
            plot_board(
                inner.board,
                test_points=inner.placed_tps,
                paths=inner.routed_paths,
                candidates=inner.candidates[:inner._real_count],
                title=(f"Episode {ep}: failures={info.get('failures', 0)}, "
                       f"length={info.get('total_length', 0):.0f}mm, "
                       f"spread={info.get('length_spread', 0):.2f}"),
                filename=str(outdir / f"episode_{ep}.png"),
            )

    summary = {
        "checkpoint": str(ckpt_path),
        "episodes": args.episodes,
        "model": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                  for k, v in metrics.items()},
    }

    print(f"\n--- Model ({ckpt_path.name}, {args.episodes} episodes, "
          f"{num_traces} traces) ---")
    for k, v in metrics.items():
        print(f"  {k:>15s}: {np.mean(v):8.3f} +/- {np.std(v):.3f}")

    if not args.skip_baselines:
        board = load_te_example(num_traces=num_traces)
        candidates, real_count = generate_candidate_grid(board, resolution=6.5)
        candidates = candidates[:real_count]
        for name, fn in [
            ("random", heuristic_eval.run_random_baseline),
            ("greedy", heuristic_eval.run_greedy_baseline),
            ("spread", heuristic_eval.run_spread_baseline),
        ]:
            results = fn(board, candidates, num_traces, args.episodes)
            fails = [r["failures"] for r in results]
            lengths = [r["total_length"] for r in results]
            spreads = [r["spread"] for r in results]
            summary[name] = {
                "failures": {"mean": float(np.mean(fails)), "std": float(np.std(fails))},
                "total_length": {"mean": float(np.mean(lengths)), "std": float(np.std(lengths))},
                "length_spread": {"mean": float(np.mean(spreads)), "std": float(np.std(spreads))},
            }
            print(f"\n--- {name.capitalize()} baseline ---")
            print(f"  failures:      {np.mean(fails):8.3f} +/- {np.std(fails):.3f}")
            print(f"  total_length:  {np.mean(lengths):8.3f} +/- {np.std(lengths):.3f}")
            print(f"  length_spread: {np.mean(spreads):8.3f} +/- {np.std(spreads):.3f}")

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nReport saved to {outdir}/ (summary.json + episode_*.png)")


if __name__ == "__main__":
    main()
