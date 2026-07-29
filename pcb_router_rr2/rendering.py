"""Board rendering.

Does NOT call matplotlib.use(): forcing "Agg" at import time overrides a
notebook's "%matplotlib inline" and makes plt.show() silently no-op. Headless
script paths set MPLBACKEND=Agg in the environment instead (see train.py).
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .board import Board
from .breakout import build_breakout
from .config import Config

_TRACE_COLORS = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                 "#42d4f4", "#f032e6", "#bfef45", "#469990", "#9A6324",
                 "#800000", "#808000", "#000075", "#a9a9a9"]


def _draw_board(ax, cfg: Config, board: Board) -> None:
    ax.add_patch(Rectangle((0, 0), board.width, board.height, fill=False,
                           ec="black", lw=2, zorder=1))
    e = board.edge_clearance
    ax.add_patch(Rectangle((e, e), board.width - 2 * e, board.height - 2 * e,
                           fill=False, ec="grey", lw=0.8, ls="--", zorder=1))
    x0, y0, x1, y1 = board.connector_rect
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#c9a95c", ec="black",
                           alpha=0.85, zorder=2))
    ax.annotate("connector", ((x0 + x1) / 2, (y0 + y1) / 2), ha="center",
                va="center", fontsize=7, zorder=3)
    for r in board.obstacles:
        ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1],
                               fc="#888888", ec="black", alpha=0.8, zorder=2))
    m = 0.06 * max(board.width, board.height)
    ax.annotate(f"{board.width:.0f} mm", (board.width / 2, -m * 0.5),
                ha="center", va="top", fontsize=9)
    ax.annotate(f"{board.height:.0f} mm", (-m * 0.5, board.height / 2),
                ha="right", va="center", fontsize=9, rotation=90)
    ax.set_xlim(-m, board.width + m * 0.4)
    ax.set_ylim(-m, board.height + m * 0.4)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def render_board(cfg: Config, paths: Optional[Sequence] = None,
                 breakout_points: Optional[Sequence[int]] = None,
                 title: str = "", subtitle: str = "",
                 highlight_trace: int = -1,
                 save_path: Optional[str] = None):
    board = Board.from_config(cfg)
    fig, ax = plt.subplots(figsize=(7, 7 * board.height / board.width / 1.05))
    _draw_board(ax, cfg, board)
    for i, (x, y) in enumerate(board.pins):
        c = _TRACE_COLORS[i % len(_TRACE_COLORS)]
        ax.plot(x, y, "o", ms=5, mfc=c, mec="black", zorder=6)
        ax.annotate(str(i), (x, y + 0.012 * board.height), ha="center",
                    fontsize=6, zorder=6)
    if paths is not None:
        for i, p in enumerate(paths):
            p = np.asarray(p, dtype=float)
            c = _TRACE_COLORS[i % len(_TRACE_COLORS)]
            lw = 3.0 if i == highlight_trace else 1.8
            bp = breakout_points[i] if breakout_points else 0
            if bp > 1:
                ax.plot(p[:bp, 0], p[:bp, 1], color=c, lw=1.4, ls=":", alpha=.9, zorder=4)
                ax.plot(p[bp - 1:, 0], p[bp - 1:, 1], color=c, lw=lw, zorder=5)
            else:
                ax.plot(p[:, 0], p[:, 1], color=c, lw=lw, zorder=5)
            ax.plot(p[-1, 0], p[-1, 1], "s", ms=6, mfc=c, mec="black", zorder=6)
            if i == highlight_trace:
                # ring the trace that boxed in, so failures are diagnosable
                ax.plot(p[-1, 0], p[-1, 1], "o", ms=16, mfc="none",
                        mec="red", mew=2.0, zorder=7)
    if title:
        ax.set_title(title, fontsize=10)
    if subtitle:
        ax.annotate(subtitle, (0.5, -0.05), xycoords="axes fraction", ha="center",
                    fontsize=7.5, color="#444444")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def preview_figure(cfg: Config, save_path: Optional[str] = None):
    if cfg.use_breakout:
        bo = build_breakout(cfg, Board.from_config(cfg))
        lens = [float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) for p in bo]
        return render_board(cfg, paths=[p.tolist() for p in bo],
                            breakout_points=[len(p) for p in bo],
                            title="Pre-training layout preview",
                            subtitle=(f"dotted = breakout ({lens[0]:.1f} mm each) | "
                                      f"squares = hand-off tips | dashed = edge clearance"),
                            save_path=save_path)
    return render_board(
        cfg, title="Pre-training layout preview",
        subtitle=(f"{cfg.n_traces} traces | no breakout: agent grows directly from "
                  f"the numbered pins\n{cfg.budget_mm:.0f} mm budget at "
                  f"{cfg.step_mm:.1f} mm steps = {cfg.budget_rounds} rounds, "
                  f"{cfg.episode_steps} steps | {cfg.n_dirs} directions | "
                  f"dashed = edge clearance"),
        save_path=save_path)


def episode_subtitle(cfg: Config, ed: dict) -> str:
    """Compact metric line. Note length_spread == step_mm means the episode
    TRUNCATED (a trace boxed in) even if the board otherwise looks fine."""
    trunc = ""
    if not ed["complete"]:
        trunc = f" | BOXED IN: trace {ed['boxed_trace']} at round {ed['rounds_done']}/{int(ed['budget_mm']/cfg.step_mm)}"
    return (f"budget {ed['budget_mm']:.0f} mm | min endpoint spacing "
            f"{ed['min_endpoint_spacing_mm']:.1f} mm | spec "
            f"{'PASS' if ed['meets_spec'] else 'miss'} | violations {ed['violations']}"
            f"{trunc}\n"
            f"clearance mean {ed['mean_path_clearance_mm']:.1f} / p5 "
            f"{ed['p5_path_clearance_mm']:.1f} mm | reversal "
            f"{ed['turn_reversal_rate']:.2f} | min self-gap "
            f"{ed['min_self_distance_mm']:.2f} mm | min freedom {ed['min_freedom']}\n"
            f"survived {ed['frac_survived']:.0%} | turn-limit relaxations "
            f"{ed['turn_limit_relaxations']} | length spread "
            f"{ed['length_spread_mm']:.2f} mm | terminal {ed['reward_terminal']:.2f}")


def episode_figure(cfg: Config, ed: dict, title: str = "",
                   episode: Optional[int] = None, steps: Optional[int] = None,
                   save_path: Optional[str] = None):
    """`episode` / `steps` are stamped into the title so every board pushed to
    W&B says exactly when it was produced."""
    stamp = []
    if episode is not None:
        stamp.append(f"episode {episode:,}")
    if steps is not None:
        stamp.append(f"{steps:,} steps")
    full = title + (f"  —  {' | '.join(stamp)}" if stamp else "")
    return render_board(cfg, paths=ed["paths"], breakout_points=ed["breakout_points"],
                        title=full, subtitle=episode_subtitle(cfg, ed),
                        highlight_trace=ed["boxed_trace"] if not ed["complete"] else -1,
                        save_path=save_path)


def fig_to_png_path(fig, path: str) -> str:
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def close_fig(fig) -> None:
    plt.close(fig)
