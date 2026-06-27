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
from forced_explore import ForcedExplorer

to_np = lambda x: x.detach().cpu().numpy()


def make_env(mode, env_id, seed=0, num_traces=8, reward_version="v1",
             board_width=135.0, board_height=90.0,
             grow=False, max_length_mm=60.0, img_size=128, step_mm=2.0,
             trace_indices=None, dense_reward_weight=0.005, spacing_threshold=5.0):
    if grow:
        from envs.pcb_grow_dreamer import PCBGrowDreamerEnv
        env = PCBGrowDreamerEnv(num_traces=num_traces, seed=seed + env_id,
                                max_length_mm=max_length_mm, img_size=img_size,
                                board_width=board_width, board_height=board_height,
                                step_mm=step_mm, trace_indices=trace_indices,
                                dense_reward_weight=dense_reward_weight,
                                spacing_threshold=spacing_threshold)
        env = wrappers.OneHotAction(env)
        # Episode length for the growth env is its internal step cap.
        env = wrappers.TimeLimit(env, env._inner.episode_steps)
        env = wrappers.SelectAction(env, key="action")
        env = wrappers.UUID(env)
        return env
    env = PCBDreamerEnv(num_traces=num_traces, seed=seed + env_id,
                        reward_version=reward_version,
                        board_width=board_width, board_height=board_height)
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
    parser.add_argument("--board_width", type=float, default=135.0,
                        help="Board width in mm (default 135, the TE example).")
    parser.add_argument("--board_height", type=float, default=90.0,
                        help="Board height in mm (default 90, the TE example).")
    # Trace-growth mode: grow every trace 1mm/step (round-robin) instead of
    # placing endpoints. Length equality is structural; reward is endpoint
    # spacing. See envs/pcb_grow_env.py.
    parser.add_argument("--grow", action="store_true", default=False,
                        help="Use the trace-growth env instead of endpoint "
                             "placement. Implies a vector trace_id observation "
                             "and an 8-direction action space.")
    parser.add_argument("--max_length_mm", type=float, default=60.0,
                        help="(grow mode) Target length each trace grows to. "
                             "Episode length scales with this.")
    parser.add_argument("--step_mm", type=float, default=2.0,
                        help="(grow mode) Agent step size in mm (default 2). "
                             "3-4mm gives cleaner geometry than 1mm with fewer "
                             "decisions per episode.")
    parser.add_argument("--spacing_threshold", type=float, default=4.0,
                        help="(grow mode) Minimum pairwise endpoint spacing "
                             "for a valid solution (default 4mm). Start low "
                             "so random policy seeds positive reward; raise "
                             "toward 13mm across curriculum runs.")
    parser.add_argument("--dense_reward_weight", type=float, default=0.005,
                        help="(grow mode) Weight on the per-step tip-spacing "
                             "reward (default 0.005). Lower = less penalty for "
                             "curling; higher = faster convergence but biases "
                             "against non-monotone paths.")
    parser.add_argument("--trace_indices", type=str, default="1,2,3,4,11,12,13,14",
                        help="(grow mode) Comma-separated 1-based trace indices "
                             "from the actual TE board to route "
                             "(default: 1,2,3,4,11,12,13,14 — left cluster, both rows).")
    parser.add_argument("--grow_img_size", type=int, default=128,
                        help="(grow mode) Render resolution (default 128). "
                             "256 is sharper but uses more GPU memory.")
    # Forced-exploration: periodic random episodes for portfolio diversity.
    parser.add_argument("--force_explore_every", type=int, default=1000,
                        help="Run a forced-random-episode batch every N env steps "
                             "(0 to disable). Should be a multiple of eval_every "
                             "so it aligns with training cycles. Default 1000.")
    parser.add_argument("--force_explore_episodes", type=int, default=200,
                        help="Number of masked-random episodes per batch. More = "
                             "denser portfolio coverage per batch. Default 200.")
    parser.add_argument("--force_explore_start", type=int, default=0,
                        help="Step at which forced exploration begins (default 0).")
    # Optional overrides for run-budget knobs, so a named config doesn't
    # need to be edited/duplicated just to change run length.
    parser.add_argument("--steps", type=float, default=None)
    parser.add_argument("--prefill", type=int, default=None)
    parser.add_argument("--eval_every", type=float, default=None)
    parser.add_argument("--eval_episode_num", type=int, default=None)
    parser.add_argument("--log_every", type=float, default=None)
    # Memory-control overrides (important for high-res grow mode on small GPUs).
    parser.add_argument("--batch_size", type=int, default=None,
                        help="World-model training batch size (config default "
                             "16). Lower to fit high-res images on small GPUs.")
    parser.add_argument("--batch_length", type=int, default=None,
                        help="World-model training sequence length (config "
                             "default 32). Lower to reduce memory.")
    parser.add_argument("--cnn_depth", type=int, default=None,
                        help="Base CNN channel depth for encoder/decoder "
                             "(config default 32). Lower (e.g. 16) cuts memory "
                             "substantially at high render resolution.")
    # Weights & Biases integration (optional)
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="pcb-dreamer",
                        help="wandb project name (default: pcb-dreamer).")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="wandb entity/team (default: personal account).")
    parser.add_argument("--wandb_key", type=str, default=None,
                        help="wandb API key (alternative to WANDB_API_KEY env var).")
    args = parser.parse_args()

    # Parse trace_indices string to list of ints
    args.trace_indices_list = [int(x.strip()) for x in args.trace_indices.split(",") if x.strip()]

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

    # Memory-control overrides.
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.batch_length is not None:
        config["batch_length"] = args.batch_length
    if args.cnn_depth is not None:
        config["encoder"] = dict(config["encoder"])
        config["decoder"] = dict(config["decoder"])
        config["encoder"]["cnn_depth"] = args.cnn_depth
        config["decoder"]["cnn_depth"] = args.cnn_depth

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

    # Trace-growth mode reconfigures the observation pipeline: a larger image
    # (so 1mm steps are visible) plus a trace_id vector the encoder must read.
    if args.grow:
        gs = args.grow_img_size
        config["size"] = [gs, gs]
        # Encoder/decoder must consume the image AND the trace_id vector.
        config["encoder"] = dict(config["encoder"])
        config["decoder"] = dict(config["decoder"])
        config["encoder"]["mlp_keys"] = "trace_id"
        config["encoder"]["cnn_keys"] = "image"
        config["decoder"]["mlp_keys"] = "trace_id"
        config["decoder"]["cnn_keys"] = "image"
        # Episode length is the growth env's internal step cap; compute it from
        # the same formula the env uses (1.5x ideal). num_rounds = max_length.
        num_rounds = int(round(args.max_length_mm / args.step_mm))
        n_traces_actual = len(args.trace_indices_list) if args.trace_indices_list else args.num_traces
        config["time_limit"] = int(num_rounds * n_traces_actual * 3)  # matches env's 3x cap
        print(f"[grow] img={gs}x{gs}, step={args.step_mm}mm, max_length={args.max_length_mm}mm, "
              f"rounds={num_rounds}, episode_steps={config['time_limit']}")

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
            "board_width": args.board_width,
            "board_height": args.board_height,
            "force_explore_every": args.force_explore_every,
            "force_explore_episodes": args.force_explore_episodes,
            "git_commit": commit,
            "created": datetime.now().isoformat(timespec="seconds"),
            "config": {
                k: (str(v) if isinstance(v, pathlib.Path) else v)
                for k, v in vars(config).items()
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Run metadata: {meta_path}")

    # Weights & Biases (optional)
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            if args.wandb_key:
                wandb.login(key=args.wandb_key, relogin=True)
            run_id = logdir.name  # use run dir name as stable id for resume
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_id,
                id=run_id,
                resume="allow",
                config={
                    "configs": args.configs,
                    "num_traces": args.num_traces,
                    "seed": args.seed,
                    "reward_version": args.reward_version,
                    "board_width": args.board_width,
                    "board_height": args.board_height,
                    "top_k": args.top_k,
                    "diversity_shift": args.diversity_shift,
                    "diversity_frac": args.diversity_frac,
                    "force_explore_every": args.force_explore_every,
                    "force_explore_episodes": args.force_explore_episodes,
                    "git_commit": commit,
                    **{k: (str(v) if isinstance(v, pathlib.Path) else v)
                       for k, v in vars(config).items()},
                },
                dir=str(logdir),
            )
            print(f"wandb run: {wandb_run.url}")
        except Exception as e:
            print(f"[wandb] init failed ({e}), continuing without wandb.")
            wandb_run = None

    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step, wandb_run=wandb_run)

    print("Creating environments...")
    train_envs = [Dummy(make_env("train", i, config.seed, args.num_traces, args.reward_version,
                                 args.board_width, args.board_height,
                                 grow=args.grow, max_length_mm=args.max_length_mm,
                                 img_size=args.grow_img_size, step_mm=args.step_mm,
                                 trace_indices=args.trace_indices_list,
                                 dense_reward_weight=args.dense_reward_weight,
                                 spacing_threshold=args.spacing_threshold))
                  for i in range(config.envs)]
    eval_envs = [Dummy(make_env("eval", i, config.seed, args.num_traces, args.reward_version,
                                args.board_width, args.board_height,
                                grow=args.grow, max_length_mm=args.max_length_mm,
                                img_size=args.grow_img_size))
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
    if args.grow:
        from grow_solution import DiverseGrowTracker
        tracker = DiverseGrowTracker(
            logdir, get_inner_env(train_envs[0]).board,
            k=args.top_k,
            min_point_shift=args.diversity_shift,
            min_moved_frac=args.diversity_frac,
        )
    else:
        tracker = DiverseSolutionTracker(
            logdir, get_inner_env(train_envs[0]).board,
            k=args.top_k,
            min_point_shift=args.diversity_shift,
            min_moved_frac=args.diversity_frac,
        )

    # Forced exploration: periodic masked-random episodes that bypass the
    # learned policy. These feed the portfolio tracker directly without going
    # into the replay buffer, ensuring the portfolio sees the full board even
    # when the trained policy has converged to a narrow spatial region.
    config.force_explore_every = args.force_explore_every
    config.force_explore_episodes = args.force_explore_episodes
    config.force_explore_start = args.force_explore_start

    def _make_explore_env():
        if args.grow:
            from envs.pcb_grow_dreamer import PCBGrowDreamerEnv
            return PCBGrowDreamerEnv(
                num_traces=args.num_traces, seed=42,
                max_length_mm=args.max_length_mm, img_size=args.grow_img_size,
                board_width=args.board_width, board_height=args.board_height,
                step_mm=args.step_mm, trace_indices=args.trace_indices_list,
                dense_reward_weight=args.dense_reward_weight,
                spacing_threshold=args.spacing_threshold,
            )
        return PCBDreamerEnv(
            num_traces=args.num_traces, seed=42,
            reward_version=args.reward_version,
            board_width=args.board_width,
            board_height=args.board_height,
        )

    explorer = ForcedExplorer(logdir, _make_explore_env, config)

    # Throttle for training-episode board renders (don't log every episode)
    _last_board_log = [0]
    _BOARD_LOG_INTERVAL = 2000  # log a training board image every ~2000 steps

    def _log_board_image(inner, tag, caption_extra=""):
        """Render the current board state of `inner` env and log to wandb."""
        if wandb_run is None:
            return
        try:
            import wandb as _wandb
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import io
            b = inner.board
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.plot([b.x_min,b.x_max,b.x_max,b.x_min,b.x_min],
                    [b.y_min,b.y_min,b.y_max,b.y_max,b.y_min], "k-", lw=2)
            for obs_r in b.rect_obstacles:
                xn,yn,xx,yx = obs_r.bounds
                ax.add_patch(plt.Rectangle((xn,yn),xx-xn,yx-yn,
                             fill=True,color="salmon",alpha=0.5,zorder=2))
            cmap = plt.get_cmap("tab10")
            for ti, p in enumerate(inner.paths):
                if len(p) < 2: continue
                arr = np.array(p)
                ax.plot(arr[:,0],arr[:,1],"-",color=cmap(ti%10),lw=1.5,zorder=4)
                ax.plot(arr[-1,0],arr[-1,1],"o",color=cmap(ti%10),
                        markersize=8,markeredgecolor="k",zorder=6)
                ax.plot(arr[0,0],arr[0,1],"s",color=cmap(ti%10),markersize=4,zorder=5)
            mm = inner._terminal_metrics
            ax.set_xlim(b.x_min-5,b.x_max+5); ax.set_ylim(b.y_min-5,b.y_max+5)
            ax.set_aspect("equal")
            ax.set_title(f"Step {agent._step}  |  "
                         f"min={mm.get('min_tp_spacing',0):.1f}mm "
                         f"mean={mm.get('mean_tp_spacing',0):.1f}mm "
                         f"pen={mm.get('endpoint_penalty',0):.2f} "
                         f"complete={mm.get('all_complete',0):.0f}{caption_extra}",
                         fontsize=10)
            ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0); plt.close(fig)
            wandb_run.log({tag: _wandb.Image(buf)}, step=agent._step)
        except Exception as e:
            print(f"[wandb board render] {e}")

    def track_episode(env, is_eval):
        inner = get_inner_env(env)
        status = tracker.update(
            inner, agent._step,
            source=("eval" if is_eval else "train"),
        )
        # Periodically log the live TRAINING board state (not just at eval and
        # not just when the portfolio updates) so you can watch the policy's
        # actual output evolve during training.
        if (not is_eval) and (agent._step - _last_board_log[0] >= _BOARD_LOG_INTERVAL):
            _last_board_log[0] = agent._step
            _log_board_image(inner, "board/training_episode")
        if status:
            print(f"  [solutions] {status}")
            if wandb_run is not None:
                best = tracker.solutions[0] if tracker.solutions else None
                wandb_run.log({
                    "portfolio/size": len(tracker.solutions),
                    "portfolio/best_spacing": best["min_tp_spacing"] if best else float("nan"),
                    "portfolio/best_reward": best.get("reward_terminal", float("nan")) if best else float("nan"),
                    "portfolio/best_total_length": best["total_length"] if best else float("nan"),
                }, step=agent._step)
                # Upload solution PNGs immediately when portfolio updates
                try:
                    import wandb as _wandb
                    sol_imgs = {}
                    for rank, sol in enumerate(tracker.solutions[:3]):
                        png = tracker.outdir / f"solution_{rank}.png"
                        if png.exists():
                            sol_imgs[f"portfolio/solution_{rank}"] = _wandb.Image(
                                str(png),
                                caption=(f"rank={rank} reward={sol.get('reward_terminal',0):.1f} "
                                         f"min={sol['min_tp_spacing']:.1f}mm "
                                         f"mean={sol.get('mean_tp_spacing',0):.1f}mm "
                                         f"step={sol['step']}")
                            )
                    if sol_imgs:
                        wandb_run.log(sol_imgs, step=agent._step)
                except Exception as e:
                    print(f"[wandb portfolio imgs] {e}")

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

            # Render current board state to wandb at every eval checkpoint.
            # Uses the last eval episode's env state (already run above).
            if wandb_run is not None and args.grow:
                try:
                    import wandb as _wandb
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    import io

                    inner = get_inner_env(eval_envs[0])
                    b = inner.board
                    fig, ax = plt.subplots(figsize=(10, 7))
                    # Board outline
                    ax.plot([b.x_min,b.x_max,b.x_max,b.x_min,b.x_min],
                            [b.y_min,b.y_min,b.y_max,b.y_max,b.y_min], "k-", lw=2)
                    # Connector / obstacles
                    for obs_r in b.rect_obstacles:
                        xn,yn,xx,yx = obs_r.bounds
                        ax.add_patch(plt.Rectangle((xn,yn),xx-xn,yx-yn,
                                                   fill=True,color="salmon",alpha=0.5,zorder=2))
                    # Traces from last episode
                    cmap = plt.get_cmap("tab10")
                    for ti, p in enumerate(inner.paths):
                        if len(p) < 2: continue
                        arr = np.array(p)
                        ax.plot(arr[:,0],arr[:,1],"-",color=cmap(ti%10),lw=1.5,zorder=4)
                        ax.plot(arr[-1,0],arr[-1,1],"o",color=cmap(ti%10),
                                markersize=8,markeredgecolor="k",zorder=6)
                        ax.plot(arr[0,0],arr[0,1],"s",color=cmap(ti%10),
                                markersize=4,zorder=5)
                    m = inner._terminal_metrics
                    min_sp  = m.get("min_tp_spacing", 0)
                    mean_sp = m.get("mean_tp_spacing", 0)
                    pen     = m.get("endpoint_penalty", 0)
                    complete = m.get("all_complete", 0)
                    ax.set_xlim(b.x_min-5, b.x_max+5)
                    ax.set_ylim(b.y_min-5, b.y_max+5)
                    ax.set_aspect("equal")
                    ax.set_title(
                        f"Step {agent._step}  |  "
                        f"min={min_sp:.1f}mm  mean={mean_sp:.1f}mm  "
                        f"pen={pen:.2f}  complete={complete:.0f}",
                        fontsize=10)
                    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
                    fig.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                    buf.seek(0)
                    plt.close(fig)
                    wandb_run.log({
                        "board/last_eval_episode": _wandb.Image(
                            buf,
                            caption=(f"min={min_sp:.1f}mm mean={mean_sp:.1f}mm "
                                     f"pen={pen:.2f}")
                        )
                    }, step=agent._step)
                except Exception as e:
                    print(f"[wandb board render] {e}")

        print(f"[Step {agent._step}] Training...")
        state = tools.simulate(
            agent, train_envs, train_eps, config.traindir,
            logger, limit=config.dataset_size,
            steps=config.eval_every, state=state,
            episode_callback=track_episode,
        )

        # Forced-exploration batch: run masked-random episodes to ensure the
        # portfolio sees the full board, not just the region the policy favours.
        if args.force_explore_every > 0:
            if explorer.should_run(agent._step):
                print(f"  [explore] running {args.force_explore_episodes} "
                      f"random episodes @ step {agent._step}...")
                n_new, n_routable = explorer.run(tracker, agent._step)
                explorer._last_ran = agent._step
                print(f"  [explore] done: {n_routable} routable, "
                      f"{n_new} portfolio updates")
                if wandb_run is not None:
                    wandb_run.log({
                        "explore/routable": n_routable,
                        "explore/portfolio_updates": n_new,
                    }, step=agent._step)

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

    # Log final portfolio to wandb as a Table
    if wandb_run is not None and tracker.solutions:
        try:
            import wandb
            cols = ["rank", "length_spread", "total_length_mm", "min_tp_spacing_mm",
                    "step", "source"]
            rows = [
                [i, s["length_spread"], s["total_length"], s["min_tp_spacing"],
                 s["step"], s["source"]]
                for i, s in enumerate(tracker.solutions)
            ]
            wandb_run.log({"portfolio": wandb.Table(columns=cols, data=rows)})
        except Exception as e:
            print(f"[wandb] portfolio table upload failed: {e}")
        wandb_run.finish()

    for env in train_envs + eval_envs:
        try: env.close()
        except: pass

    print("\nDone! tensorboard --logdir", logdir)


if __name__ == "__main__":
    main()