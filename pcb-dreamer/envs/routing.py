"""
A* pathfinding for trace routing on a discretized PCB grid.

The board is discretized at a given resolution (e.g., 0.5mm per cell).
Obstacles and previously routed traces are rasterized with clearance buffers.
A* finds the shortest feasible path from start to test point.
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional
from envs.board import (
    BoardSpec, TRACE_WIDTH, TRACE_TO_TRACE_MIN,
    TRACE_TO_EDGE_MIN, TRACE_TO_UPTH_MIN, TRACE_TO_TABPAD_MIN,
)


class RoutingGrid:
    """
    Discretized grid for A* routing.

    Each cell is either free (0) or blocked (1).
    Obstacles are rasterized with clearance buffers.
    """

    def __init__(self, board: BoardSpec, resolution: float = 0.5):
        self.board = board
        self.res = resolution

        # Grid dimensions
        self.cols = int(np.ceil(board.width / resolution))
        self.rows = int(np.ceil(board.height / resolution))

        # Occupancy grid: 0 = free, 1 = blocked
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        # Rasterize board edge clearance
        self._rasterize_edge_clearance()

        # Rasterize obstacles
        self._rasterize_obstacles()

        # Clear starting points — traces physically originate here,
        # so these cells must be free even if inside the connector region.
        self._clear_starting_points()

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid (col, row)."""
        col = int((x - self.board.x_min) / self.res)
        row = int((y - self.board.y_min) / self.res)
        col = np.clip(col, 0, self.cols - 1)
        row = np.clip(row, 0, self.rows - 1)
        return col, row

    def _grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Convert grid (col, row) to world coordinates."""
        x = self.board.x_min + (col + 0.5) * self.res
        y = self.board.y_min + (row + 0.5) * self.res
        return x, y

    def _rasterize_edge_clearance(self):
        """Block cells within TRACE_TO_EDGE_MIN of board edges."""
        buf_cells = int(np.ceil((TRACE_TO_EDGE_MIN + TRACE_WIDTH / 2) / self.res))
        self.grid[:buf_cells, :] = 1  # bottom edge
        self.grid[-buf_cells:, :] = 1  # top edge
        self.grid[:, :buf_cells] = 1  # left edge
        self.grid[:, -buf_cells:] = 1  # right edge

    def _rasterize_obstacles(self):
        """Rasterize all obstacles with clearance buffers."""
        for obs in self.board.rect_obstacles:
            xmin, ymin, xmax, ymax = obs.bounds
            buf = obs.clearance + TRACE_WIDTH / 2
            c0, r0 = self._world_to_grid(xmin - buf, ymin - buf)
            c1, r1 = self._world_to_grid(xmax + buf, ymax + buf)
            r0, r1 = max(0, r0), min(self.rows, r1 + 1)
            c0, c1 = max(0, c0), min(self.cols, c1 + 1)
            self.grid[r0:r1, c0:c1] = 1

        for obs in self.board.circ_obstacles:
            buf = obs.radius + obs.clearance + TRACE_WIDTH / 2
            c0, r0 = self._world_to_grid(obs.cx - buf, obs.cy - buf)
            c1, r1 = self._world_to_grid(obs.cx + buf, obs.cy + buf)
            for r in range(max(0, r0), min(self.rows, r1 + 1)):
                for c in range(max(0, c0), min(self.cols, c1 + 1)):
                    wx, wy = self._grid_to_world(c, r)
                    if np.sqrt((wx - obs.cx)**2 + (wy - obs.cy)**2) < buf:
                        self.grid[r, c] = 1

    def _clear_starting_points(self):
        """
        Carve out free cells around each trace starting point.
        Starting points are inside the connector region which is blocked,
        but traces physically originate there so routing must be able to start.
        """
        clear_radius = int(np.ceil(2.0 / self.res))  # 2mm radius free zone
        for trace in self.board.traces:
            sc, sr = self._world_to_grid(trace.start_x, trace.start_y)
            for dr in range(-clear_radius, clear_radius + 1):
                for dc in range(-clear_radius, clear_radius + 1):
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if dr * dr + dc * dc <= clear_radius * clear_radius:
                            self.grid[nr, nc] = 0

    def rasterize_trace_path(self, path: List[Tuple[int, int]]):
        """
        Mark a routed trace path as blocked (with clearance buffer)
        so subsequent traces avoid it.
        """
        buf_cells = int(np.ceil((TRACE_TO_TRACE_MIN + TRACE_WIDTH) / self.res))
        for col, row in path:
            r0 = max(0, row - buf_cells)
            r1 = min(self.rows, row + buf_cells + 1)
            c0 = max(0, col - buf_cells)
            c1 = min(self.cols, col + buf_cells + 1)
            self.grid[r0:r1, c0:c1] = 1

    def find_path(
        self, start_x: float, start_y: float, end_x: float, end_y: float
    ) -> Optional[Tuple[List[Tuple[int, int]], float]]:
        """
        A* from (start_x, start_y) to (end_x, end_y) in world coordinates.

        Returns (path_as_grid_cells, path_length_mm) or None if no path exists.
        """
        sc, sr = self._world_to_grid(start_x, start_y)
        ec, er = self._world_to_grid(end_x, end_y)

        # Ensure start and end cells are free
        self.grid[sr, sc] = 0
        self.grid[er, ec] = 0

        # 8-directional movement
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
        diag_cost = np.sqrt(2)

        open_set = []
        heapq.heappush(open_set, (0.0, sc, sr))
        came_from = {}
        g_score = {(sc, sr): 0.0}

        def heuristic(c, r):
            return np.sqrt((c - ec)**2 + (r - er)**2)

        visited = set()

        while open_set:
            f, cc, cr = heapq.heappop(open_set)

            if (cc, cr) in visited:
                continue
            visited.add((cc, cr))

            if cc == ec and cr == er:
                # Reconstruct path
                path = [(cc, cr)]
                while (cc, cr) in came_from:
                    cc, cr = came_from[(cc, cr)]
                    path.append((cc, cr))
                path.reverse()

                # Compute path length in mm
                length = 0.0
                for i in range(len(path) - 1):
                    dc = path[i + 1][0] - path[i][0]
                    dr = path[i + 1][1] - path[i][1]
                    length += np.sqrt(dc**2 + dr**2) * self.res
                return path, length

            for dc, dr in neighbors:
                nc, nr = cc + dc, cr + dr
                if 0 <= nc < self.cols and 0 <= nr < self.rows:
                    if self.grid[nr, nc] == 0 and (nc, nr) not in visited:
                        move_cost = diag_cost if (dc != 0 and dr != 0) else 1.0
                        tentative_g = g_score[(cc, cr)] + move_cost
                        if tentative_g < g_score.get((nc, nr), float('inf')):
                            g_score[(nc, nr)] = tentative_g
                            came_from[(nc, nr)] = (cc, cr)
                            f_score = tentative_g + heuristic(nc, nr)
                            heapq.heappush(open_set, (f_score, nc, nr))

        return None  # No path found

    def path_to_world(self, path: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """Convert grid path to world coordinates."""
        return [self._grid_to_world(c, r) for c, r in path]


def route_all_traces(
    board: BoardSpec,
    test_points: List[Tuple[float, float]],
    resolution: float = 0.5,
) -> Tuple[List[Optional[List[Tuple[float, float]]]], List[float], int]:
    """
    Route all traces from starting points to test points using A*.

    Args:
        board: Board specification
        test_points: List of (x, y) test point positions, one per trace
        resolution: Grid resolution in mm

    Returns:
        paths: List of world-coordinate paths (or None if unroutable)
        lengths: List of trace lengths in mm (including breakout)
        failures: Number of traces that couldn't be routed
    """
    grid = RoutingGrid(board, resolution)
    paths = []
    lengths = []
    failures = 0

    for i, trace in enumerate(board.traces):
        if i >= len(test_points):
            break

        tp_x, tp_y = test_points[i]
        result = grid.find_path(trace.start_x, trace.start_y, tp_x, tp_y)

        if result is None:
            # Still include start→tp straight line for visualization
            paths.append([(trace.start_x, trace.start_y), (tp_x, tp_y)])
            lengths.append(float('inf'))
            failures += 1
        else:
            grid_path, path_length = result
            world_path = grid.path_to_world(grid_path)
            # Prepend actual starting point so trace visually begins correctly
            world_path.insert(0, (trace.start_x, trace.start_y))
            paths.append(world_path)
            lengths.append(path_length + trace.breakout_length)

            # Mark this trace path as blocked for subsequent traces
            grid.rasterize_trace_path(grid_path)

    return paths, lengths, failures