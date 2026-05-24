"""
Evaluate a trained DreamerV3 agent on PCB test point placement.

Usage:
    python eval.py --logdir ./logdir/pcb --episodes 5
    python eval.py --logdir /path/to/drive/run1 --episodes 10
"""

import argparse
import pathlib
import sys
import os

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import torch
import matplotlib.pyplot as plt

torch.distributions.Distribution.set_default_validate_args(False)

import tools
from envs.pcb_env import TPPlacementEnv
from envs.board import load_te_example, generate_toy_board, generate_candidate_grid
from envs.routing import route_all_traces
from envs.visualize import plot_board


def run_random_baseline(board, candidates, num_traces, num_episodes=5):
    """Run random placement as a baseline."""
    from envs.board import check_tp_spacing
    results = []
    rng = np.random.RandomState(0)

    for ep in range(num_episodes):
        placed = []
        for i in range(num_traces):
            indices = rng.permutation(len(candidates))
            found = False
            for idx in indices:
                cx, cy = candidates[idx]
                if check_tp_spacing(placed, cx, cy):
                    placed.append((cx, cy))
                    found = True
                    break
            if not found:
                placed.append(tuple(candidates[rng.randint(len(candidates))]))

        paths, lengths, failures = route_all_traces(board, placed, resolution=1.0)
        finite = [l for l in lengths if l < float('inf')]
        results.append({
            "placed": placed, "paths": paths, "lengths": lengths,
            "failures": failures, "total_length": sum(finite) if finite else 0,
            "spread": (max(finite) - min(finite)) / np.mean(finite) if len(finite) > 1 else 0,
        })
    return results


def run_greedy_baseline(board, candidates, num_traces, num_episodes=5):
    """Greedy: for each trace, pick the closest valid candidate to its starting point."""
    from envs.board import check_tp_spacing
    results = []

    for ep in range(num_episodes):
        placed = []
        for i in range(num_traces):
            trace = board.traces[i]
            # Sort candidates by distance to starting point
            dists = np.sqrt(
                (candidates[:, 0] - trace.start_x) ** 2 +
                (candidates[:, 1] - trace.start_y) ** 2
            )
            order = np.argsort(dists)
            found = False
            for idx in order:
                cx, cy = candidates[idx]
                if check_tp_spacing(placed, cx, cy):
                    placed.append((cx, cy))
                    found = True
                    break
            if not found:
                placed.append(tuple(candidates[order[0]]))

        paths, lengths, failures = route_all_traces(board, placed, resolution=1.0)
        finite = [l for l in lengths if l < float('inf')]
        results.append({
            "placed": placed, "paths": paths, "lengths": lengths,
            "failures": failures, "total_length": sum(finite) if finite else 0,
            "spread": (max(finite) - min(finite)) / np.mean(finite) if len(finite) > 1 else 0,
        })
    return results


def run_trained_agent(logdir, num_traces, num_episodes=5):
    """Run the trained DreamerV3 agent."""
    from envs.dreamer_wrapper import PCBDreamerEnv
    from envs import wrappers
    from parallel import Damy
    from dreamer import Dreamer
    import ruamel.yaml as yaml
    import functools

    # Load config
    config_path = pathlib.Path(__file__).parent / "configs.yaml"
    configs = yaml.YAML(typ="safe").load(config_path.read_text())
    config = configs["defaults"]
    config["logdir"] = str(logdir)
    config["traindir"] = str(logdir / "train_eps")
    config["evaldir"] = str(logdir / "eval_eps")
    config["time_limit"] = num_traces
    config["device"] = "cpu"
    config = argparse.Namespace(**config)
    config.num_actions = 162  # will be overwritten

    # Create env
    env = PCBDreamerEnv(num_traces=num_traces, seed=999)
    env = wrappers.OneHotAction(env)
    env = wrappers.TimeLimit(env, num_traces)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    eval_env = Damy(env)

    config.num_actions = eval_env.action_space.n if hasattr(eval_env.action_space, "n") else \
    eval_env.action_space.shape[0]

    # Create agent and load checkpoint
    logger = tools.Logger(logdir, 0)
    train_eps = tools.load_episodes(pathlib.Path(config.traindir), limit=100)
    dataset = tools.from_generator(
        tools.sample_episodes(train_eps, config.batch_length), config.batch_size
    )

    agent = Dreamer(
        eval_env.observation_space, eval_env.action_space,
        config, logger, dataset,
    )

    ckpt_path = logdir / "latest.pt"
    if not ckpt_path.exists():
        print(f"No checkpoint found at {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu")
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    print(f"Loaded checkpoint from {ckpt_path}")

    # Run episodes
    results = []
    inner_env = TPPlacementEnv(num_traces=num_traces)

    for ep in range(num_episodes):
        obs = eval_env.reset()
        done = False
        state = None
        total_reward = 0

        while not done:
            with torch.no_grad():
                obs_torch = {
                    k: torch.tensor(np.array([v]), dtype=torch.float32 if k != "image" else torch.uint8)
                    for k, v in obs.items()
                }
                action_output, state = agent._policy(
                    {k: v.to(config.device) for k, v in obs_torch.items()},
                    state, training=False
                )
                action = {"action": action_output["action"].cpu()}

            obs, reward, done, info = eval_env.step(action)
            total_reward += reward

        # Extract placement results from the inner env
        # Re-run to get placements (the wrapper doesn't expose inner state easily)
        placed = []
        inner_env.reset(seed=999 + ep)
        obs_inner, _ = inner_env.reset(seed=999 + ep)
        for step in range(num_traces):
            # Use same actions the agent would take
            # For now, just record the info
            pass

        results.append({
            "reward": total_reward,
            "info": info,
        })
        print(f"  Episode {ep + 1}: reward={total_reward:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="./logdir/pcb")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--num_traces", type=int, default=8)
    parser.add_argument("--outdir", type=str, default="./eval_results")
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    board = generate_toy_board(num_traces=args.num_traces)
    candidates = generate_candidate_grid(board, resolution=6.5)
    print(f"Board: {board.width}x{board.height}mm, {args.num_traces} traces, {len(candidates)} candidates")

    # === Random baseline ===
    print("\n--- Random Baseline ---")
    random_results = run_random_baseline(board, candidates, args.num_traces, args.episodes)
    for i, r in enumerate(random_results):
        print(f"  Ep {i + 1}: failures={r['failures']}, length={r['total_length']:.0f}mm, spread={r['spread']:.2f}")
        plot_board(board, test_points=r["placed"], paths=r["paths"], candidates=candidates,
                   title=f"Random #{i + 1}: {r['failures']} failures, {r['total_length']:.0f}mm",
                   filename=str(outdir / f"random_{i + 1}.png"))

    # === Greedy baseline ===
    print("\n--- Greedy Baseline ---")
    greedy_results = run_greedy_baseline(board, candidates, args.num_traces, args.episodes)
    for i, r in enumerate(greedy_results):
        print(f"  Ep {i + 1}: failures={r['failures']}, length={r['total_length']:.0f}mm, spread={r['spread']:.2f}")
        plot_board(board, test_points=r["placed"], paths=r["paths"], candidates=candidates,
                   title=f"Greedy #{i + 1}: {r['failures']} failures, {r['total_length']:.0f}mm",
                   filename=str(outdir / f"greedy_{i + 1}.png"))

    # === Trained agent ===
    if (logdir / "latest.pt").exists():
        print("\n--- Trained Agent ---")
        agent_results = run_trained_agent(logdir, args.num_traces, args.episodes)
    else:
        print(f"\nNo checkpoint at {logdir / 'latest.pt'} — skipping trained agent eval.")
        agent_results = None

    # === Summary ===
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    def summarize(name, results):
        if not results:
            return
        fails = [r["failures"] for r in results]
        lengths = [r["total_length"] for r in results]
        spreads = [r["spread"] for r in results]
        print(f"{name:>15s}: failures={np.mean(fails):.1f}±{np.std(fails):.1f}, "
              f"length={np.mean(lengths):.0f}±{np.std(lengths):.0f}mm, "
              f"spread={np.mean(spreads):.2f}±{np.std(spreads):.2f}")

    summarize("Random", random_results)
    summarize("Greedy", greedy_results)
    if agent_results:
        print(f"{'Trained':>15s}: see episode rewards above (full eval requires env access)")

    print(f"\nPlots saved to {outdir}/")


if __name__ == "__main__":
    main()