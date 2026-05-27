"""
FreeRouting interface for PCB trace routing.

Generates Specctra DSN input files from board + test points,
runs FreeRouting as a subprocess, and parses SES output files
to extract routed paths, lengths, and failure counts.

Requires: Java 17+, freerouting.jar in project root or on PATH.
Download from: https://github.com/freerouting/freerouting/releases
"""

import os
import re
import subprocess
import tempfile
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

from envs.board import (
    BoardSpec, TRACE_WIDTH, TRACE_TO_TRACE_MIN,
    TRACE_TO_EDGE_MIN, TRACE_TO_UPTH_MIN, TRACE_TO_TABPAD_MIN,
)

# Look for freerouting.jar in common locations
_JAR_SEARCH = [
    "freerouting.jar",
    "../freerouting.jar",
    os.path.expanduser("~/freerouting.jar"),
    "/opt/freerouting/freerouting.jar",
]


def find_freerouting_jar() -> Optional[str]:
    """Find freerouting.jar on disk."""
    env_path = os.environ.get("FREEROUTING_JAR")
    if env_path and os.path.isfile(env_path):
        return env_path
    for p in _JAR_SEARCH:
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def write_dsn(
    board: BoardSpec,
    test_points: List[Tuple[float, float]],
    filepath: str,
):
    """
    Write a Specctra .dsn file for FreeRouting.

    Each trace i connects starting point i to test point i.
    """
    n = min(len(board.traces), len(test_points))
    lines = []
    a = lines.append

    a('(pcb pcb_test_fixture')
    a('  (parser')
    a('    (string_quote ")')
    a('    (host_cad "pcb-dreamer")')
    a('    (host_version "1.0"))')
    a('  (resolution mm 1000)')
    a('  (unit mm)')
    a('')

    # ---- Structure: board outline + keepouts + rules ----
    a('  (structure')

    # Board boundary
    x0, y0 = board.x_min, board.y_min
    x1, y1 = board.x_max, board.y_max
    a(f'    (boundary')
    a(f'      (path pcb {TRACE_WIDTH:.4f}')
    a(f'        {x0:.4f} {y0:.4f}')
    a(f'        {x1:.4f} {y0:.4f}')
    a(f'        {x1:.4f} {y1:.4f}')
    a(f'        {x0:.4f} {y1:.4f}')
    a(f'        {x0:.4f} {y0:.4f}))')

    # Keepout zones — rectangular obstacles
    for obs in board.rect_obstacles:
        xmin, ymin, xmax, ymax = obs.bounds
        a(f'    (keepout "{obs.name}"')
        a(f'      (rect pcb {xmin:.4f} {ymin:.4f} {xmax:.4f} {ymax:.4f}))')

    # Keepout zones — circular obstacles
    for obs in board.circ_obstacles:
        a(f'    (keepout "{obs.name}"')
        a(f'      (circle pcb {obs.radius * 2:.4f} {obs.cx:.4f} {obs.cy:.4f}))')

    # Design rules
    a(f'    (rule')
    a(f'      (width {TRACE_WIDTH:.4f})')
    a(f'      (clearance {TRACE_TO_TRACE_MIN:.4f})')
    a(f'      (clearance {TRACE_TO_UPTH_MIN:.4f} (type "smd_to_turn"))')
    a(f'      (clearance {TRACE_TO_TABPAD_MIN:.4f} (type "pad_to_turn")))')

    # Single layer
    a('    (layer "F.Cu"')
    a('      (type signal))')

    a('  )')  # end structure
    a('')

    # ---- Placement: starting points + test points as components ----
    a('  (placement')
    for i in range(n):
        t = board.traces[i]
        a(f'    (component "START_{i}"')
        a(f'      (place "s{i}" {t.start_x:.4f} {t.start_y:.4f} front 0))')
    for i in range(n):
        tx, ty = test_points[i]
        a(f'    (component "TP_{i}"')
        a(f'      (place "t{i}" {tx:.4f} {ty:.4f} front 0))')
    a('  )')  # end placement
    a('')

    # ---- Library: pad definitions ----
    a('  (library')
    for i in range(n):
        a(f'    (image "START_{i}"')
        a(f'      (pin "Pad" "1" 0 0))')
        a(f'    (image "TP_{i}"')
        a(f'      (pin "Pad" "1" 0 0))')
    a('    (padstack "Pad"')
    a(f'      (shape (circle "F.Cu" {TRACE_WIDTH:.4f} 0 0)))')
    a('  )')  # end library
    a('')

    # ---- Network: each net connects start_i to tp_i ----
    a('  (network')
    for i in range(n):
        a(f'    (net "net_{i}"')
        a(f'      (pins "s{i}"-"1" "t{i}"-"1"))')
    a('  )')  # end network
    a('')

    a(')')  # end pcb

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))


def run_freerouting(
    dsn_path: str,
    ses_path: str,
    jar_path: Optional[str] = None,
    timeout: int = 30,
    max_passes: int = 20,
) -> bool:
    """
    Run FreeRouting on a DSN file, produce SES output.

    Returns True if FreeRouting completed successfully.
    """
    if jar_path is None:
        jar_path = find_freerouting_jar()
    if jar_path is None:
        raise FileNotFoundError(
            "freerouting.jar not found. Set FREEROUTING_JAR env var or "
            "place it in the project root. Download from: "
            "https://github.com/freerouting/freerouting/releases"
        )

    cmd = [
        "java", "-jar", jar_path,
        "-de", dsn_path,
        "-do", ses_path,
        "-mp", str(max_passes),
        "-mt", str(timeout),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 10,
        )
        return os.path.isfile(ses_path)
    except subprocess.TimeoutExpired:
        return os.path.isfile(ses_path)
    except FileNotFoundError:
        raise FileNotFoundError("Java not found. FreeRouting requires Java 17+.")


def parse_ses(
    ses_path: str,
    num_nets: int,
) -> Tuple[List[Optional[List[Tuple[float, float]]]], List[float], int]:
    """
    Parse a Specctra SES file to extract routed paths.

    Returns (paths, lengths, failures) matching the route_all_traces API.
    """
    paths: List[Optional[List[Tuple[float, float]]]] = [None] * num_nets
    lengths: List[float] = [float('inf')] * num_nets

    if not os.path.isfile(ses_path):
        return paths, lengths, num_nets

    with open(ses_path, 'r') as f:
        content = f.read()

    # Extract wires per net
    # Pattern: (net net_N (wire (path F.Cu width x1 y1 x2 y2 ...)))
    net_pattern = re.compile(
        r'\(net\s+"?net_(\d+)"?\s*((?:\(wire\s*\(path[^)]*\)\s*\)\s*)*)\)',
        re.DOTALL,
    )
    wire_pattern = re.compile(
        r'\(wire\s*\(path\s+\S+\s+[\d.]+\s+([\d.\s-]+)\)\s*\)',
    )

    for net_match in net_pattern.finditer(content):
        net_idx = int(net_match.group(1))
        if net_idx >= num_nets:
            continue

        wire_section = net_match.group(2)
        all_points: List[Tuple[float, float]] = []

        for wire_match in wire_pattern.finditer(wire_section):
            coords_str = wire_match.group(1).strip()
            nums = [float(x) for x in coords_str.split()]
            for j in range(0, len(nums) - 1, 2):
                pt = (nums[j], nums[j + 1])
                if not all_points or pt != all_points[-1]:
                    all_points.append(pt)

        if all_points and len(all_points) >= 2:
            paths[net_idx] = all_points
            length = sum(
                np.hypot(all_points[k + 1][0] - all_points[k][0],
                         all_points[k + 1][1] - all_points[k][1])
                for k in range(len(all_points) - 1)
            )
            lengths[net_idx] = length

    failures = sum(1 for p in paths if p is None)
    return paths, lengths, failures


def route_with_freerouting(
    board: BoardSpec,
    test_points: List[Tuple[float, float]],
    jar_path: Optional[str] = None,
    timeout: int = 30,
) -> Tuple[List[Optional[List[Tuple[float, float]]]], List[float], int]:
    """
    Full FreeRouting pipeline: write DSN → run → parse SES.

    Drop-in replacement for routing.route_all_traces().
    """
    n = min(len(board.traces), len(test_points))
    if n == 0:
        return [], [], 0

    with tempfile.TemporaryDirectory() as tmpdir:
        dsn_path = os.path.join(tmpdir, "board.dsn")
        ses_path = os.path.join(tmpdir, "board.ses")

        write_dsn(board, test_points, dsn_path)
        success = run_freerouting(dsn_path, ses_path, jar_path, timeout)

        if not success:
            return [None] * n, [float('inf')] * n, n

        return parse_ses(ses_path, n)