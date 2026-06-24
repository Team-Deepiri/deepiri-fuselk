"""Helical Quadtree Rotational Mapper (HQRM)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from deepiri_fuselk.helix.coordinate_mapper import field_line_pitch, q_profile

KERNEL_SIZE = 49
MAX_DEPTH = 7


@dataclass
class Square:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    depth: int
    rotation: float
    children: list[Square] = field(default_factory=list)


@dataclass
class HQRMResult:
    kernel: list[Square]
    heat_variance: float
    o_point: tuple[float, float]
    converged: bool


def _shear_in_square(square: Square, q0: float, q95: float) -> float:
    """Magnetic shear proxy — global q-gradient scale (legacy sim behavior)."""
    del square
    q_vals = q_profile(np.linspace(0, 1, 10), q0=q0, q95=q95)
    return abs(float(np.gradient(q_vals)[-1]))


def _rotate_square(square: Square, angle: float) -> Square:
    return Square(
        x_min=square.x_min,
        x_max=square.x_max,
        y_min=square.y_min,
        y_max=square.y_max,
        depth=square.depth,
        rotation=angle,
    )


def _split_square(square: Square) -> list[Square]:
    mx = 0.5 * (square.x_min + square.x_max)
    my = 0.5 * (square.y_min + square.y_max)
    d = square.depth + 1
    rot = square.rotation
    return [
        Square(square.x_min, mx, square.y_min, my, d, rot),
        Square(mx, square.x_max, square.y_min, my, d, rot),
        Square(square.x_min, mx, my, square.y_max, d, rot),
        Square(mx, square.x_max, my, square.y_max, d, rot),
    ]


def _sample_heat(square: Square, heat_field: np.ndarray) -> float:
    n = heat_field.shape[0]
    cx = 0.5 * (square.x_min + square.x_max)
    cy = 0.5 * (square.y_min + square.y_max)
    ix = int(np.clip((cx + 1) / 2 * (n - 1), 0, n - 1))
    iy = int(np.clip((cy + 1) / 2 * (n - 1), 0, n - 1))
    return float(heat_field[iy, ix])


def build_hqrm_kernel(
    heat_field: np.ndarray,
    shear_threshold: float = 0.3,
    max_depth: int = MAX_DEPTH,
    target_leaves: int = KERNEL_SIZE,
    q0: float = 1.0,
    q95: float = 3.5,
) -> list[Square]:
    """
    Breadth-first HQRM subdivision (VISION §2).

    Stops once ``target_leaves`` patches are collected or max depth is reached.
    """
    root = Square(-1.0, 1.0, -1.0, 1.0, 0, 0.0)
    queue: list[Square] = [root]
    leaves: list[Square] = []

    while queue and len(leaves) < target_leaves:
        square = queue.pop(0)
        shear = _shear_in_square(square, q0, q95)
        cx = 0.5 * (square.x_min + square.x_max)
        cy = 0.5 * (square.y_min + square.y_max)
        angle = float(field_line_pitch(np.array([cx]), np.array([cy]), q0, q95)[0])
        square = _rotate_square(square, angle)

        should_split = shear > shear_threshold and square.depth < max_depth
        if should_split:
            queue.extend(_split_square(square))
        else:
            leaves.append(square)

    return leaves[:target_leaves]


def recursive_split(
    square: Square,
    heat_field: np.ndarray,
    grid: np.ndarray,
    shear_threshold: float = 0.3,
    max_depth: int = MAX_DEPTH,
    q0: float = 1.0,
    q95: float = 3.5,
) -> Square:
    """Legacy depth-first split — prefer :func:`build_hqrm_kernel`."""
    _ = grid  # retained for API compatibility
    shear = _shear_in_square(square, q0, q95)
    cx = 0.5 * (square.x_min + square.x_max)
    cy = 0.5 * (square.y_min + square.y_max)
    angle = float(field_line_pitch(np.array([cx]), np.array([cy]), q0, q95)[0])
    square = _rotate_square(square, angle)

    if shear > shear_threshold and square.depth < max_depth:
        square.children = [
            recursive_split(child, heat_field, grid, shear_threshold, max_depth, q0, q95)
            for child in _split_square(square)
        ]
    return square


def extract_7x7_kernel(root: Square) -> list[Square]:
    """Collect finest-level squares (7×7 kernel target)."""
    leaves: list[Square] = []

    def walk(node: Square) -> None:
        if node.children:
            for child in node.children:
                walk(child)
        else:
            leaves.append(node)

    walk(root)
    return leaves[:KERNEL_SIZE]


def run_hqrm(
    heat_field: np.ndarray,
    shear_threshold: float = 0.3,
    variance_threshold: float = 0.07,
) -> HQRMResult:
    """Run HQRM on a 2D heat field; lock when 7×7 kernel variance < threshold."""
    kernel = build_hqrm_kernel(
        heat_field,
        shear_threshold=shear_threshold,
    )

    if not kernel:
        return HQRMResult(kernel=[], heat_variance=1.0, o_point=(0.0, 0.0), converged=False)

    values = [_sample_heat(sq, heat_field) for sq in kernel]
    variance = float(np.var(values))
    centroids = np.array([[(s.x_min + s.x_max) / 2, (s.y_min + s.y_max) / 2] for s in kernel])
    weights = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    weight_sum = float(weights.sum())
    if weight_sum > 1e-12:
        center = weights @ centroids / weight_sum
        cx, cy = float(center[0]), float(center[1])
    else:
        cx = float(np.mean(centroids[:, 0]))
        cy = float(np.mean(centroids[:, 1]))
    converged = variance < variance_threshold and len(kernel) >= 7
    return HQRMResult(
        kernel=kernel,
        heat_variance=variance,
        o_point=(cx, cy),
        converged=converged,
    )
