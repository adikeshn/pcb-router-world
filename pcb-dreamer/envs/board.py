"""
Board representation for SI test fixture routing.

Encodes board geometry, obstacles, starting points, and constraints.
Provides candidate grid generation and constraint checking.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Obstacle:
    """Rectangular obstacle or keep-out zone."""
    cx: float  # center x
    cy: float  # center y
    width: float
    height: float
    clearance: float  # minimum trace-to-obstacle distance
    name: str = ""

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max)."""
        hw, hh = self.width / 2, self.height / 2
        return (self.cx - hw, self.cy - hh, self.cx + hw, self.cy + hh)


@dataclass
class CircularObstacle:
    """Circular obstacle (UPTH, via, etc.)."""
    cx: float
    cy: float
    radius: float
    clearance: float
    name: str = ""


@dataclass
class TraceSpec:
    """Specification for one trace to be routed."""
    start_x: float
    start_y: float
    breakout_length: float
    index: int


@dataclass
class BoardSpec:
    """Complete board specification loaded from data."""
    # Board dimensions
    origin_x: float
    origin_y: float
    width: float
    height: float

    # Obstacles
    rect_obstacles: List[Obstacle] = field(default_factory=list)
    circ_obstacles: List[CircularObstacle] = field(default_factory=list)

    # Traces
    traces: List[TraceSpec] = field(default_factory=list)

    # Connector outline (simplified as rectangle)
    connector_x: float = 0.0
    connector_y: float = 0.0
    connector_w: float = 0.0
    connector_h: float = 0.0

    @property
    def x_min(self):
        return self.origin_x

    @property
    def y_min(self):
        return self.origin_y

    @property
    def x_max(self):
        return self.origin_x + self.width

    @property
    def y_max(self):
        return self.origin_y + self.height


# ---------------------------------------------------------------------------
# Fixed constraints (same across all instances)
# ---------------------------------------------------------------------------
TRACE_WIDTH = 0.2286  # mm
TRACE_TO_EDGE_MIN = 0.26  # mm, edge-to-edge
TRACE_TO_TRACE_MIN = 1.1  # mm, edge-to-edge
TRACE_TO_UPTH_MIN = 0.7  # mm, edge-to-edge
TRACE_TO_TABPAD_MIN = 0.7  # mm, edge-to-edge
TP_TO_TP_MIN = 13.0  # mm, center-to-center
TP_TO_EDGE_MIN = 14.0  # mm, center-to-edge
TP_TO_CONNECTOR_MIN = 3.0  # mm, center-to-edge


def load_te_example() -> BoardSpec:
    """Load the TE example board from the provided Excel data."""
    board = BoardSpec(
        origin_x=0.0,
        origin_y=98.2,
        width=135.0,
        height=90.0,
    )

    # Non-routing zone (rectangular obstacle)
    nrz_x = 58.294
    nrz_y = 108.044
    nrz_w = 17.8
    nrz_h = 6.64
    board.rect_obstacles.append(Obstacle(
        cx=nrz_x + nrz_w / 2,
        cy=nrz_y + nrz_h / 2,
        width=nrz_w,
        height=nrz_h,
        clearance=TRACE_TO_EDGE_MIN,
        name="non_routing_zone",
    ))

    # Connector outline (approximate from non-routing zone + tab pads region)
    board.connector_x = 56.0
    board.connector_y = 105.0
    board.connector_w = 22.0
    board.connector_h = 11.0

    # UPTHs (circular obstacles)
    board.circ_obstacles.append(CircularObstacle(
        cx=58.194, cy=105.894, radius=1.9 / 2,
        clearance=TRACE_TO_UPTH_MIN, name="UPTH_1",
    ))
    board.circ_obstacles.append(CircularObstacle(
        cx=76.194, cy=105.894, radius=1.9 / 2,
        clearance=TRACE_TO_UPTH_MIN, name="UPTH_2",
    ))

    # Tab pads (rectangular obstacles)
    board.rect_obstacles.append(Obstacle(
        cx=56.151 + 1.526 / 2, cy=113.346 + 1.216 / 2,
        width=1.526, height=1.216,
        clearance=TRACE_TO_TABPAD_MIN, name="tab_pad_1",
    ))
    board.rect_obstacles.append(Obstacle(
        cx=76.711 + 1.526 / 2, cy=113.346 + 1.216 / 2,
        width=1.526, height=1.216,
        clearance=TRACE_TO_TABPAD_MIN, name="tab_pad_2",
    ))

    # 20 starting points from fanout data
    top_row_x = [58.9442, 59.8442, 60.7442, 61.6442, 62.5442,
                 63.4442, 64.3442, 65.2442, 69.1442, 70.0442]
    top_y = 107.9436
    bot_y = 114.7446
    breakout = 0.8626

    for i, x in enumerate(top_row_x):
        board.traces.append(TraceSpec(start_x=x, start_y=top_y,
                                     breakout_length=breakout, index=i))
    for i, x in enumerate(top_row_x):
        board.traces.append(TraceSpec(start_x=x, start_y=bot_y,
                                     breakout_length=breakout, index=10 + i))

    return board


def generate_toy_board(num_traces: int = 8, seed: int = 0) -> BoardSpec:
    """
    Generate a toy board with properly spaced starting points
    that guarantee routing feasibility.

    Starting points are spaced at 3mm apart (well above the 1.33mm
    minimum needed for trace-to-trace clearance), placed along the
    bottom-center of the board.
    """
    board = BoardSpec(
        origin_x=0.0,
        origin_y=0.0,
        width=135.0,
        height=90.0,
    )

    # Starting point spacing: must be > TRACE_TO_TRACE_MIN + TRACE_WIDTH
    # Using 3mm for comfortable margin
    start_spacing = 3.0
    total_span = (num_traces - 1) * start_spacing
    start_x_offset = (board.width - total_span) / 2  # center horizontally
    start_y = 5.0  # near bottom edge, outside edge clearance

    for i in range(num_traces):
        board.traces.append(TraceSpec(
            start_x=start_x_offset + i * start_spacing,
            start_y=start_y,
            breakout_length=0.8626,
            index=i,
        ))

    # Add some obstacles in the middle to make routing interesting
    rng = np.random.RandomState(seed)

    # A rectangular obstacle in the center
    board.rect_obstacles.append(Obstacle(
        cx=board.width / 2,
        cy=board.height / 2,
        width=15.0, height=8.0,
        clearance=TRACE_TO_EDGE_MIN,
        name="obstacle_center",
    ))

    # Two smaller obstacles
    board.rect_obstacles.append(Obstacle(
        cx=board.width * 0.3,
        cy=board.height * 0.6,
        width=8.0, height=5.0,
        clearance=TRACE_TO_EDGE_MIN,
        name="obstacle_left",
    ))
    board.rect_obstacles.append(Obstacle(
        cx=board.width * 0.7,
        cy=board.height * 0.4,
        width=8.0, height=5.0,
        clearance=TRACE_TO_EDGE_MIN,
        name="obstacle_right",
    ))

    # Two circular obstacles (like UPTHs)
    board.circ_obstacles.append(CircularObstacle(
        cx=board.width * 0.4, cy=board.height * 0.35,
        radius=1.5, clearance=TRACE_TO_UPTH_MIN,
        name="via_1",
    ))
    board.circ_obstacles.append(CircularObstacle(
        cx=board.width * 0.6, cy=board.height * 0.65,
        radius=1.5, clearance=TRACE_TO_UPTH_MIN,
        name="via_2",
    ))

    # Connector outline (around starting points)
    board.connector_x = start_x_offset - 2.0
    board.connector_y = 0.0
    board.connector_w = total_span + 4.0
    board.connector_h = 8.0

    return board


def generate_candidate_grid(board: BoardSpec, resolution: float = 6.5) -> np.ndarray:
    """
    Generate valid test point candidate positions.

    Returns array of shape (N, 2) with (x, y) positions.
    """
    candidates = []

    # Valid region: inset by TP_TO_EDGE_MIN from board edges
    x_lo = board.x_min + TP_TO_EDGE_MIN
    x_hi = board.x_max - TP_TO_EDGE_MIN
    y_lo = board.y_min + TP_TO_EDGE_MIN
    y_hi = board.y_max - TP_TO_EDGE_MIN

    # Generate grid
    xs = np.arange(x_lo, x_hi + resolution / 2, resolution)
    ys = np.arange(y_lo, y_hi + resolution / 2, resolution)

    for x in xs:
        for y in ys:
            if _is_valid_tp_position(board, x, y):
                candidates.append((x, y))

    return np.array(candidates, dtype=np.float64)


def _is_valid_tp_position(board: BoardSpec, x: float, y: float) -> bool:
    """Check if (x, y) is a valid test point position."""
    # Check board edge clearance
    if (x - board.x_min < TP_TO_EDGE_MIN or board.x_max - x < TP_TO_EDGE_MIN or
            y - board.y_min < TP_TO_EDGE_MIN or board.y_max - y < TP_TO_EDGE_MIN):
        return False

    # Check connector outline clearance
    conn_xmin = board.connector_x
    conn_xmax = board.connector_x + board.connector_w
    conn_ymin = board.connector_y
    conn_ymax = board.connector_y + board.connector_h
    # Distance from point to rectangle (0 if inside)
    dx = max(conn_xmin - x, 0, x - conn_xmax)
    dy = max(conn_ymin - y, 0, y - conn_ymax)
    dist_to_conn = np.sqrt(dx**2 + dy**2)
    # If point is inside connector, dist is 0
    if dx == 0 and dy == 0 and conn_xmin <= x <= conn_xmax and conn_ymin <= y <= conn_ymax:
        return False  # inside connector
    if dist_to_conn < TP_TO_CONNECTOR_MIN:
        return False  # too close to connector

    # Check rectangular obstacles
    for obs in board.rect_obstacles:
        xmin, ymin, xmax, ymax = obs.bounds
        buf = obs.clearance  # trace-to-obstacle clearance
        if xmin - buf < x < xmax + buf and ymin - buf < y < ymax + buf:
            return False

    # Check circular obstacles
    for obs in board.circ_obstacles:
        dist = np.sqrt((x - obs.cx)**2 + (y - obs.cy)**2)
        if dist < obs.radius + obs.clearance:
            return False

    return True


def check_tp_spacing(placed_tps: List[Tuple[float, float]], x: float, y: float) -> bool:
    """Check if new TP at (x,y) satisfies spacing with all placed TPs."""
    for px, py in placed_tps:
        dist = np.sqrt((x - px)**2 + (y - py)**2)
        if dist < TP_TO_TP_MIN:
            return False
    return True