"""Board rendering.

NOTE: this module deliberately does NOT call matplotlib.use(). Forcing "Agg"
at import time overrides a notebook's "%matplotlib inline" and makes plt.show()
silently no-op. Headless script paths set MPLBACKEND=Agg in the environment
instead (see train.py).
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
            bp = breakout_points[i] if breakout_points else 0
            if bp > 1:
                ax.plot(p[:bp, 0], p[:bp, 1], color=c, lw=1.4, ls=":", alpha=.9, zorder=4)
                ax.plot(p[bp - 1:, 0], p[bp - 1:, 1], color=c, lw=1.8, zorder=5)
            else:
                ax.plot(p[:, 0], p[:, 1], color=c, lw=1.8, zorder=5)
            ax.plot(p[-1, 0], p[-1, 1], "s", ms=6, mfc=c, mec="black", zorder=6)
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
                            subtitle=(f"dotted = deterministic breakout ({lens[0]:.1f} mm each) | "
                                      f"squares = agent hand-off tips | dashed = edge clearance"),
                            save_path=save_path)
    return render_board(cfg, title="Pre-training layout preview",
                        subtitle=(f"{cfg.n_traces} traces | no breakout: agent grows directly "
                                  f"from the numbered pins | budgets {cfg.budget_min_mm:.0f}"
                                  f"-{cfg.budget_max_mm:.0f} mm | dashed = edge clearance"),
                        save_path=save_path)


def episode_figure(cfg: Config, ed: dict, title: str = "",
                   save_path: Optional[str] = None):
    sub = (f"budget {ed['budget_mm']:.0f} mm | min endpoint spacing "
           f"{ed['min_endpoint_spacing_mm']:.1f} mm | spec "
           f"{'PASS' if ed['meets_spec'] else 'miss'} | violations {ed['violations']}\n"
           f"min self-gap {ed.get('min_self_distance_mm', -1):.2f} mm | turn rate "
           f"{ed.get('turn_rate', 0):.2f} | endpoint-to-edge "
           f"{ed.get('min_endpoint_edge_mm', 0):.1f} mm | length spread "
           f"{ed['length_spread_mm']:.2f} mm | terminal {ed['reward_terminal']:.2f}")
    return render_board(cfg, paths=ed["paths"], breakout_points=ed["breakout_points"],
                        title=title, subtitle=sub, save_path=save_path)


def fig_to_png_path(fig, path: str) -> str:
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def close_fig(fig) -> None:
    plt.close(fig)
