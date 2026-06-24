"""Helical Quadtree Rotational Mapper (HQRM)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from deepiri_fuselk.helix.coordinate_mapper import field_line_pitch, q_profile


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
    cx = 0.5 * (square.x_min + square.x_max)
    cy = 0.5 * (square.y_min + square.y_max)
    r = np.sqrt(cx**2 + cy**2)
    _ = q_profile(np.array([r]), q0=q0, q95=q95)[0]
    return abs(float(np.gradient(q_profile(np.linspace(0, 1, 10), q0=q0, q95=q95))[-1]))


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


def recursive_split(
    square: Square,
    heat_field: np.ndarray,
    grid: np.ndarray,
    shear_threshold: float = 0.3,
    max_depth: int = 7,
    q0: float = 1.0,
    q95: float = 3.5,
) -> Square:
    """Recursively subdivide and rotate squares along field line pitch."""
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
    """Collect finest-level squares at depth 7 (7x7 kernel target)."""
    leaves: list[Square] = []

    def walk(node: Square) -> None:
        if node.children:
            for child in node.children:
                walk(child)
        else:
            leaves.append(node)

    walk(root)
    return leaves[:49]


def run_hqrm(
    heat_field: np.ndarray,
    shear_threshold: float = 0.3,
    variance_threshold: float = 0.07,
) -> HQRMResult:
    """Run HQRM on a 2D heat field; lock when 7x7 kernel variance < 7%."""
    n = heat_field.shape[0]
    grid = np.linspace(-1, 1, n)
    root = Square(-1.0, 1.0, -1.0, 1.0, 0, 0.0)
    tree = recursive_split(root, heat_field, grid, shear_threshold=shear_threshold)
    kernel = extract_7x7_kernel(tree)

    if not kernel:
        return HQRMResult(kernel=[], heat_variance=1.0, o_point=(0.0, 0.0), converged=False)

    values = []
    for sq in kernel:
        ix = int(np.clip((sq.x_min + 1) / 2 * (n - 1), 0, n - 1))
        iy = int(np.clip((sq.y_min + 1) / 2 * (n - 1), 0, n - 1))
        values.append(heat_field[iy, ix])

    variance = float(np.var(values))
    cx = float(np.mean([(s.x_min + s.x_max) / 2 for s in kernel]))
    cy = float(np.mean([(s.y_min + s.y_max) / 2 for s in kernel]))
    converged = variance < variance_threshold and len(kernel) >= 7
    return HQRMResult(
        kernel=kernel,
        heat_variance=variance,
        o_point=(cx, cy),
        converged=converged,
    )
