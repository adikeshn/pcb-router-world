"""
Compare metrics across training runs.

Scans `<logdir_root>/*/` for run directories containing `meta.json` and
`metrics.jsonl` (written by train.py), and prints a side-by-side summary
table of the latest eval metrics for each run, plus optional overlay plots
of metrics vs training step.

Usage:
    python compare_runs.py
    python compare_runs.py --logdir_root ./logdir
    python compare_runs.py --runs run_a run_b run_c
    python compare_runs.py --plot eval_return eval_log_routable eval_log_length_spread
"""

import argparse
import json
import pathlib


DEFAULT_METRICS = [
    "eval_return",
    "eval_log_routable",
    "eval_log_failures",
    "eval_log_length_spread",
    "eval_log_invalid_actions",
]


def load_run(run_dir: pathlib.Path):
    meta_path = run_dir / "meta.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not meta_path.exists() or not metrics_path.exists():
        return None

    meta = json.loads(meta_path.read_text())
    rows = []
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    eval_rows = [r for r in rows if "eval_return" in r]
    return {"name": run_dir.name, "dir": run_dir, "meta": meta, "rows": rows, "eval_rows": eval_rows}


def find_runs(logdir_root: pathlib.Path, names=None):
    if names:
        candidates = [logdir_root / n for n in names]
    else:
        candidates = sorted(p for p in logdir_root.iterdir() if p.is_dir())

    runs = []
    for c in candidates:
        run = load_run(c)
        if run is not None:
            runs.append(run)
    return runs


def print_summary(runs, metrics):
    print(f"{'run':<45s} {'commit':<9s} {'traces':>6s} {'steps':>8s}", end="")
    for m in metrics:
        print(f" {m:>26s}", end="")
    print()

    for run in runs:
        meta = run["meta"]
        last_train_step = run["rows"][-1]["step"] if run["rows"] else 0
        print(f"{run['name']:<45.45s} "
              f"{meta.get('git_commit', '?'):<9s} "
              f"{meta.get('num_traces', '?'):>6} "
              f"{last_train_step:>8.0f}", end="")

        if run["eval_rows"]:
            last_eval = run["eval_rows"][-1]
        else:
            last_eval = {}
        for m in metrics:
            val = last_eval.get(m)
            cell = f"{val:.3f}" if val is not None else "n/a"
            print(f" {cell:>26s}", end="")
        print()

    print("\n(values are from the most recent eval cycle of each run)")


def make_plots(runs, metrics, outdir: pathlib.Path):
    import matplotlib.pyplot as plt

    outdir.mkdir(exist_ok=True, parents=True)
    for m in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        any_data = False
        for run in runs:
            xs = [r["step"] for r in run["eval_rows"] if m in r]
            ys = [r[m] for r in run["eval_rows"] if m in r]
            if xs:
                ax.plot(xs, ys, marker="o", markersize=3, label=run["name"])
                any_data = True
        if not any_data:
            plt.close(fig)
            continue
        ax.set_xlabel("env step")
        ax.set_ylabel(m)
        ax.set_title(m)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fname = outdir / f"{m}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir_root", type=str, default="./logdir")
    parser.add_argument("--runs", nargs="+", default=None,
                         help="Specific run directory names under logdir_root. "
                              "Default: all runs found.")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                         help="metrics.jsonl keys to report/plot.")
    parser.add_argument("--plot", nargs="*", default=None,
                         help="If given, save overlay plots for these metrics "
                              "(or --metrics if no names given) to "
                              "<logdir_root>/comparison_plots/.")
    args = parser.parse_args()

    logdir_root = pathlib.Path(args.logdir_root)
    runs = find_runs(logdir_root, args.runs)

    if not runs:
        print(f"No runs with meta.json + metrics.jsonl found under {logdir_root}/")
        return

    print_summary(runs, args.metrics)

    if args.plot is not None:
        plot_metrics = args.plot if len(args.plot) > 0 else args.metrics
        make_plots(runs, plot_metrics, logdir_root / "comparison_plots")


if __name__ == "__main__":
    main()
