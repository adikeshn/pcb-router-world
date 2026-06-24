"""
Board representation for SI test fixture routing.

Encodes board geometry, obstacles, starting points, and constraints.
Provides candidate grid generation and constraint checking.

Board data comes from TE Connectivity AutoLayout_Example01 Excel/PowerPoint.
Starting-point spacing is adjusted to exactly satisfy the minimum trace-to-
trace clearance constraint (original 0.9mm spacing violates it; we use
TRACE_TO_TRACE_MIN + TRACE_WIDTH ≈ 1.3286mm).

When a seed is provided, the connector cluster (NRZ, UPTHs, tab pads,
starting traces) is shifted to a random position on the board while
maintaining minimum margin from all edges.
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
    clearance: float  # minimum trace-to-obstacle distance (edge-to-edge)
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
    clearance: float  # edge-to-edge
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
    origin_x: float
    origin_y: float
    width: float
    height: float

    rect_obstacles: List[Obstacle] = field(default_factory=list)
    circ_obstacles: List[CircularObstacle] = field(default_factory=list)
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
# Fixed constraints (from TE AutoLayout Example01 Excel)
# ---------------------------------------------------------------------------
TRACE_WIDTH = 0.2286        # mm
TRACE_TO_EDGE_MIN = 0.26    # mm, edge-to-edge
TRACE_TO_TRACE_MIN = 1.1    # mm, edge-to-edge
TRACE_TO_UPTH_MIN = 0.7     # mm, edge-to-edge
TRACE_TO_TABPAD_MIN = 0.7   # mm, edge-to-edge
TP_TO_TP_MIN = 13.0         # mm, center-to-center
TP_TO_EDGE_MIN = 14.0       # mm, center-to-edge (from PCB Routine Material)
TP_TO_CONNECTOR_MIN = 3.0   # mm, center-to-edge (from PCB Routine Material)

# Minimum center-to-center trace spacing that satisfies edge-to-edge clearance.
TRACE_MIN_CENTER_TO_CENTER = TRACE_TO_TRACE_MIN + TRACE_WIDTH  # 1.3286mm

# Fixed action-space size so it stays constant across board geometries.
MAX_CANDIDATES = 400


def _respaced_x(original_x: List[float], min_spacing: float) -> List[float]:
    """
    Re-space a list of x-coordinates so the gaps between consecutive
    points are at least `min_spacing`, preserving the group center.

    If the original spacing already satisfies the constraint, positions
    are returned unchanged.
    """
    n = len(original_x)
    if n <= 1:
        return list(original_x)
    center = np.mean(original_x)
    new_x = [center + (i - (n - 1) / 2) * min_spacing for i in range(n)]
    return new_x


def load_te_example(num_traces: int = 10, seed: int = None,
                    board_width: float = 135.0,
                    board_height: float = 90.0) -> BoardSpec:
    """
    Load the TE AutoLayout Example01 board.

    Board geometry and obstacles from TE Excel/PowerPoint data.
    Starting points: 5 per row (top + bottom), centered within the
    non-routing zone at TRACE_MIN_CENTER_TO_CENTER spacing.

    Args:
        num_traces: how many traces to include (1–12). Top row first,
                    then bottom row. 6 per row at minimum spacing.
        seed:       if provided, the connector cluster (NRZ, obstacles,
                    starting traces) is shifted to a random position on
                    the board, giving each seed a different layout while
                    preserving internal geometry.
        board_width, board_height: outer board dimensions (mm). Defaults
                    match the TE example (135 x 90). Enlarging gives the
                    router more space and yields more test-point candidates;
                    the connector cluster keeps its real hardware size.
    """
    board = BoardSpec(
        origin_x=0.0,
        origin_y=98.2,
        width=board_width,
        height=board_height,
    )

    # ------------------------------------------------------------------
    # Compute random offset for the connector cluster
    # ------------------------------------------------------------------
    # Original cluster reference positions (absolute coords)
    # Connector outline: (55.0, 104.5) with size (24.0, 12.0)
    # Cluster center:
    _orig_conn_x = 55.0
    _orig_conn_y = 104.5
    _orig_conn_w = 24.0
    _orig_conn_h = 12.0

    # Margin from board edge to keep the cluster fully inside
    _edge_margin = 10.0  # mm

    if seed is not None:
        rng = np.random.RandomState(seed)

        # Valid range for the connector's bottom-left corner
        x_lo = board.x_min + _edge_margin
        x_hi = board.x_max - _orig_conn_w - _edge_margin
        y_lo = board.y_min + _edge_margin
        y_hi = board.y_max - _orig_conn_h - _edge_margin

        new_conn_x = rng.uniform(x_lo, x_hi)
        new_conn_y = rng.uniform(y_lo, y_hi)

        dx = new_conn_x - _orig_conn_x
        dy = new_conn_y - _orig_conn_y
    else:
        dx = 0.0
        dy = 0.0

    # ------------------------------------------------------------------
    # Obstacles (exact TE data, shifted by dx/dy)
    # ------------------------------------------------------------------

    # Non-routing zone: bottom-left (58.294, 108.044), 17.8 × 6.64 mm
    nrz_x, nrz_y, nrz_w, nrz_h = 58.294 + dx, 108.044 + dy, 17.8, 6.64
    board.rect_obstacles.append(Obstacle(
        cx=nrz_x + nrz_w / 2,
        cy=nrz_y + nrz_h / 2,
        width=nrz_w, height=nrz_h,
        clearance=TRACE_TO_EDGE_MIN,
        name="non_routing_zone",
    ))

    # UPTHs
    board.circ_obstacles.append(CircularObstacle(
        cx=58.194 + dx, cy=105.894 + dy, radius=1.9 / 2,
        clearance=TRACE_TO_UPTH_MIN, name="UPTH_1",
    ))
    board.circ_obstacles.append(CircularObstacle(
        cx=76.194 + dx, cy=105.894 + dy, radius=1.9 / 2,
        clearance=TRACE_TO_UPTH_MIN, name="UPTH_2",
    ))

    # Tab pads: bottom-left corners given, convert to center
    board.rect_obstacles.append(Obstacle(
        cx=56.151 + dx + 1.526 / 2, cy=113.346 + dy + 1.216 / 2,
        width=1.526, height=1.216,
        clearance=TRACE_TO_TABPAD_MIN, name="tab_pad_1",
    ))
    board.rect_obstacles.append(Obstacle(
        cx=76.711 + dx + 1.526 / 2, cy=113.346 + dy + 1.216 / 2,
        width=1.526, height=1.216,
        clearance=TRACE_TO_TABPAD_MIN, name="tab_pad_2",
    ))

    # Connector outline (encompasses NRZ + tab pads + UPTH region)
    board.connector_x = _orig_conn_x + dx
    board.connector_y = _orig_conn_y + dy
    board.connector_w = _orig_conn_w
    board.connector_h = _orig_conn_h

    # ------------------------------------------------------------------
    # Starting points — 3 per row × 2 rows = 6 traces, centered on NRZ.
    # (3-stacked-on-3 layout. With num_traces=6 this gives two rows of
    # three rather than a single straight line of six.)
    # Spacing = TRACE_MIN_CENTER_TO_CENTER (1.3286mm), just sufficient
    # for trace-to-trace clearance.
    # ------------------------------------------------------------------
    nrz_cx = nrz_x + nrz_w / 2
    n_per_row = 3
    min_sp = TRACE_MIN_CENTER_TO_CENTER  # 1.3286mm

    # positions centered on the NRZ x-center at minimum spacing
    start_xs = [nrz_cx + (i - (n_per_row - 1) / 2) * min_sp
                for i in range(n_per_row)]

    top_y = 107.9436 + dy   # just below NRZ bottom
    bot_y = 114.7446 + dy   # just above NRZ top
    breakout = 0.8626

    all_traces = []
    # Top row
    for i, x in enumerate(start_xs):
        all_traces.append(TraceSpec(
            start_x=x, start_y=top_y,
            breakout_length=breakout, index=i,
        ))
    # Bottom row
    for i, x in enumerate(start_xs):
        all_traces.append(TraceSpec(
            start_x=x, start_y=bot_y,
            breakout_length=breakout, index=n_per_row + i,
        ))

    board.traces = all_traces[:num_traces]
    return board


def generate_candidate_grid(board: BoardSpec, resolution: float = 6.5,
                            max_candidates: int = MAX_CANDIDATES
                            ) -> Tuple[np.ndarray, int]:
    """
    Generate valid test point candidate positions.

    Returns:
        candidates: array of shape (max_candidates, 2) with (x, y) positions,
                    padded with (x_min, y_min) entries beyond the real count.
        real_count: number of genuine (non-padding) candidates.
    """
    candidates = []

    x_lo = board.x_min + TP_TO_EDGE_MIN
    x_hi = board.x_max - TP_TO_EDGE_MIN
    y_lo = board.y_min + TP_TO_EDGE_MIN
    y_hi = board.y_max - TP_TO_EDGE_MIN

    xs = np.arange(x_lo, x_hi + resolution / 2, resolution)
    ys = np.arange(y_lo, y_hi + resolution / 2, resolution)

    for x in xs:
        for y in ys:
            if _is_valid_tp_position(board, x, y):
                candidates.append((x, y))

    # If there are more valid candidates than fit in the fixed-size action
    # space, SUBSAMPLE EVENLY across the full set rather than truncating to
    # the first N. The grid is built column-major (x outer), so a plain
    # truncation keeps only the leftmost columns and leaves the right side of
    # the board with no candidates. Evenly-spaced index selection preserves
    # spatial coverage across the whole board.
    if len(candidates) > max_candidates:
        idx = np.linspace(0, len(candidates) - 1, max_candidates)
        idx = np.unique(np.round(idx).astype(int))
        candidates = [candidates[i] for i in idx]

    real_count = len(candidates)

    # Pad to fixed size with dummy entries
    while len(candidates) < max_candidates:
        candidates.append((board.x_min, board.y_min))

    return np.array(candidates, dtype=np.float64), real_count


def _is_valid_tp_position(board: BoardSpec, x: float, y: float) -> bool:
    """Check if (x, y) is a valid test point position."""
    # Board edge clearance
    if (x - board.x_min < TP_TO_EDGE_MIN or board.x_max - x < TP_TO_EDGE_MIN or
            y - board.y_min < TP_TO_EDGE_MIN or board.y_max - y < TP_TO_EDGE_MIN):
        return False

    # Connector outline clearance
    conn_xmin = board.connector_x
    conn_xmax = board.connector_x + board.connector_w
    conn_ymin = board.connector_y
    conn_ymax = board.connector_y + board.connector_h
    dx = max(conn_xmin - x, 0, x - conn_xmax)
    dy = max(conn_ymin - y, 0, y - conn_ymax)
    if dx == 0 and dy == 0 and conn_xmin <= x <= conn_xmax and conn_ymin <= y <= conn_ymax:
        return False  # inside connector
    dist_to_conn = np.sqrt(dx**2 + dy**2)
    if dist_to_conn < TP_TO_CONNECTOR_MIN:
        return False  # too close to connector

    # Rectangular obstacles
    for obs in board.rect_obstacles:
        xmin, ymin, xmax, ymax = obs.bounds
        buf = obs.clearance
        if xmin - buf < x < xmax + buf and ymin - buf < y < ymax + buf:
            return False

    # Circular obstacles
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

def load_actual_te_board(
    trace_indices: List[int] = None,
) -> BoardSpec:
    """
    Load the EXACT TE AutoLayout Example01 board from the slide spec.

    All coordinates taken directly from the PowerPoint slides:
      - Board: bottom-left (0, 98.2), width=135, height=90
      - Non-routing zone: BL (58.294, 108.044), w=17.8, h=6.64
      - UPTH 1: center (58.194, 105.894), dia=1.9
      - UPTH 2: center (76.194, 105.894), dia=1.9
      - Tab Pad 1: BL (56.151, 113.346), w=1.526, h=1.216
      - Tab Pad 2: BL (76.711, 113.346), w=1.526, h=1.216
      - Traces 1-10 (bottom row): y=107.9436, x = 58.9442 + (n-1)*0.9
      - Traces 11-20 (top row):   y=114.7844, x = 58.9442 + (n-1)*0.9
        (top y inferred by mirroring through NRZ center; matches diagram)

    Args:
        trace_indices: 1-based list of traces to include, e.g. [1,2,3,4,11,12,13,14].
                       Default: [1,2,3,4,11,12,13,14].

    Returns:
        BoardSpec with exact geometry, no randomisation, no re-spacing.
    """
    if trace_indices is None:
        trace_indices = [1, 2, 3, 4, 11, 12, 13, 14]

    board = BoardSpec(
        origin_x=0.0,
        origin_y=98.2,
        width=135.0,
        height=90.0,
    )

    # Non-routing zone (treated as a rectangular obstacle)
    nrz_bl_x, nrz_bl_y = 58.294, 108.044
    nrz_w, nrz_h = 17.8, 6.64
    nrz_cx = nrz_bl_x + nrz_w / 2
    nrz_cy = nrz_bl_y + nrz_h / 2
    board.rect_obstacles.append(Obstacle(
        cx=nrz_cx, cy=nrz_cy,
        width=nrz_w, height=nrz_h,
        clearance=0.0,   # zero extra clearance; hard boundary
        name="NRZ",
    ))

    # Tab Pad 1
    tp1_cx = 56.151 + 1.526 / 2
    tp1_cy = 113.346 + 1.216 / 2
    board.rect_obstacles.append(Obstacle(
        cx=tp1_cx, cy=tp1_cy,
        width=1.526, height=1.216,
        clearance=TRACE_TO_TABPAD_MIN,
        name="TabPad1",
    ))

    # Tab Pad 2
    tp2_cx = 76.711 + 1.526 / 2
    tp2_cy = 113.346 + 1.216 / 2
    board.rect_obstacles.append(Obstacle(
        cx=tp2_cx, cy=tp2_cy,
        width=1.526, height=1.216,
        clearance=TRACE_TO_TABPAD_MIN,
        name="TabPad2",
    ))

    # UPTH 1 (circular, use radius + drill-to-trace clearance)
    board.circ_obstacles.append(CircularObstacle(
        cx=58.194, cy=105.894,
        radius=1.9 / 2,
        clearance=TRACE_TO_UPTH_MIN,
        name="UPTH1",
    ))

    # UPTH 2
    board.circ_obstacles.append(CircularObstacle(
        cx=76.194, cy=105.894,
        radius=1.9 / 2,
        clearance=TRACE_TO_UPTH_MIN,
        name="UPTH2",
    ))

    # Connector outline: encloses NRZ + tab pads + UPTHs with a small margin.
    # Used for TP_TO_CONNECTOR_MIN clearance on test-point endpoints.
    board.connector_x = 55.0
    board.connector_y = 104.5
    board.connector_w = 24.0
    board.connector_h = 12.0

    # All 20 trace starting points (1-based index), exact from CSV.
    # Bottom row (1-10): y=107.9436. Top row (11-20): y=114.7446.
    # Note the non-uniform pitch: traces 1-8 and 11-18 are at 0.9mm spacing,
    # but there is a physical gap in the connector between pins 8 and 9
    # (and 18 and 19), so traces 9,10,19,20 have a shifted x origin.
    bottom_y = 107.9436
    top_y    = 114.7446   # exact from CSV (not inferred)

    x_positions = {
        1:  58.9442,  2:  59.8442,  3:  60.7442,  4:  61.6442,
        5:  62.5442,  6:  63.4442,  7:  64.3442,  8:  65.2442,
        9:  69.1442,  10: 70.0442,
        11: 58.9442,  12: 59.8442,  13: 60.7442,  14: 61.6442,
        15: 62.5442,  16: 63.4442,  17: 64.3442,  18: 65.2442,
        19: 69.1442,  20: 70.0442,
    }

    all_traces = {}
    for n in range(1, 21):
        y = bottom_y if n <= 10 else top_y
        all_traces[n] = TraceSpec(start_x=x_positions[n], start_y=y,
                                  breakout_length=0.8626,
                                  index=n)

    board.traces = [all_traces[i] for i in trace_indices if i in all_traces]
    return board
