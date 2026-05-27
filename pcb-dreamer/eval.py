"""
Evaluate baselines on PCB test point placement.

Baselines place TPs (one per trace, in order), then route via
A* (default) or FreeRouting (--freerouting flag).

Usage:
    python eval.py --episodes 5 --num_traces 10
    python eval.py --episodes 5 --num_traces 10 --freerouting
"""

import argparse
import pathlib
import os

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np

from envs.board import load_te_example, generate_candidate_grid, check_tp_spacing
from envs.routing import route_all_traces, validate_routing_constraints
from envs.visualize import plot_board


def evaluate_placement(board, placed_tps, use_freerouting=False):
    """Route and score a placement."""
    if use_freerouting:
        from envs.freerouting import route_with_freerouting
        paths, lengths, failures = route_with_freerouting(board, placed_tps)
    else:
        paths, lengths, failures = route_all_traces(board, placed_tps)

    validation = validate_routing_constraints(board, paths)
    finite = [l for l in lengths if l < float('inf')]
    spread = ((max(finite) - min(finite)) / np.mean(finite)
              if len(finite) > 1 else 0)
    return {
        "placed": placed_tps, "paths": paths, "lengths": lengths,
        "failures": failures,
        "total_length": sum(finite) if finite else 0,
        "spread": spread,
        "validation": validation,
    }


def run_random_baseline(board, candidates, num_traces, num_episodes=5,
                        use_freerouting=False):
    """Random TP placement, one per trace in order."""
    results = []
    rng = np.random.RandomState(0)
    for ep in range(num_episodes):
        placed = []
        for i in range(num_traces):
            for idx in rng.permutation(len(candidates)):
                cx, cy = candidates[idx]
                if check_tp_spacing(placed, cx, cy):
                    placed.append((cx, cy))
                    break
            else:
                placed.append(tuple(candidates[rng.randint(len(candidates))]))
        results.append(evaluate_placement(board, placed, use_freerouting))
    return results


def run_greedy_baseline(board, candidates, num_traces, num_episodes=5,
                        use_freerouting=False):
    """For each trace, pick the closest valid candidate."""
    results = []
    for ep in range(num_episodes):
        placed = []
        for i in range(num_traces):
            trace = board.traces[i]
            dists = np.hypot(candidates[:, 0] - trace.start_x,
                             candidates[:, 1] - trace.start_y)
            for idx in np.argsort(dists):
                cx, cy = candidates[idx]
                if check_tp_spacing(placed, cx, cy):
                    placed.append((cx, cy))
                    break
            else:
                placed.append(tuple(candidates[np.argsort(dists)[0]]))
        results.append(evaluate_placement(board, placed, use_freerouting))
    return results


def run_spread_baseline(board, candidates, num_traces, num_episodes=5,
                        use_freerouting=False):
    """Place TPs far from connector, well-spread, then assign in order."""
    results = []
    cx_conn = board.connector_x + board.connector_w / 2
    cy_conn = board.connector_y + board.connector_h / 2
    dists = np.hypot(candidates[:, 0] - cx_conn, candidates[:, 1] - cy_conn)
    order = np.argsort(-dists)

    for ep in range(num_episodes):
        placed = []
        for idx in order:
            if len(placed) >= num_traces:
                break
            cx, cy = candidates[idx]
            if check_tp_spacing(placed, cx, cy):
                placed.append((cx, cy))
        while len(placed) < num_traces:
            for idx in range(len(candidates)):
                if len(placed) >= num_traces:
                    break
                cx, cy = candidates[idx]
                if check_tp_spacing(placed, cx, cy):
                    placed.append((cx, cy))
        results.append(evaluate_placement(board, placed, use_freerouting))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--num_traces", type=int, default=10)
    parser.add_argument("--freerouting", action="store_true",
                        help="Use FreeRouting instead of A*")
    args = parser.parse_args()

    router_name = "FreeRouting" if args.freerouting else "A*"
    outdir = pathlib.Path("eval_results")
    outdir.mkdir(exist_ok=True)

    board = load_te_example(num_traces=args.num_traces)
    candidates = generate_candidate_grid(board, resolution=6.5)
    print(f"Board: {board.width}x{board.height}mm, {len(board.traces)} traces, "
          f"{len(candidates)} candidates, router={router_name}")

    def print_results(name, results):
        for i, r in enumerate(results):
            v = r["validation"]
            t2t = (f"{v['trace_to_trace_min']:.2f}"
                   if v['trace_to_trace_min'] < float('inf') else "n/a")
            print(f"  Ep {i + 1}: failures={r['failures']}, "
                  f"length={r['total_length']:.0f}mm, "
                  f"spread={r['spread']:.2f}, t2t={t2t}mm")
            plot_board(board, test_points=r["placed"], paths=r["paths"],
                       candidates=candidates,
                       title=f"{name} #{i + 1}: {r['failures']} fail, "
                             f"{r['total_length']:.0f}mm",
                       filename=str(outdir / f"{name.lower()}_{i + 1}.png"))

    print(f"\n--- Random Baseline ({router_name}) ---")
    random_results = run_random_baseline(
        board, candidates, args.num_traces, args.episodes, args.freerouting)
    print_results("Random", random_results)

    print(f"\n--- Greedy Baseline ({router_name}) ---")
    greedy_results = run_greedy_baseline(
        board, candidates, args.num_traces, args.episodes, args.freerouting)
    print_results("Greedy", greedy_results)

    print(f"\n--- Spread Baseline ({router_name}) ---")
    spread_results = run_spread_baseline(
        board, candidates, args.num_traces, args.episodes, args.freerouting)
    print_results("Spread", spread_results)

    # Summary
    print(f"\n--- Summary ({router_name}) ---")
    for name, results in [("Random", random_results),
                          ("Greedy", greedy_results),
                          ("Spread", spread_results)]:
        fails = [r["failures"] for r in results]
        lengths = [r["total_length"] for r in results]
        spreads = [r["spread"] for r in results]
        valid = sum(1 for r in results
                    if r.get("validation", {}).get("all_valid", False))
        print(f"  {name:>10s}: failures={np.mean(fails):.1f}+/-{np.std(fails):.1f}, "
              f"length={np.mean(lengths):.0f}mm, "
              f"spread={np.mean(spreads):.2f}, "
              f"valid={valid}/{len(results)}")

    print(f"\nPlots saved to {outdir}/")


if __name__ == "__main__":
    main()