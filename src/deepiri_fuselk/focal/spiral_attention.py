"""Helical spiral attention for field-line-aligned feature extraction."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


@lru_cache(maxsize=32)
def spiral_attention_weights(size: int, pitch: float = 0.5, n_heads: int = 4) -> np.ndarray:
    """Multi-head helical attention mask following field line pitch."""
    grid = np.linspace(0, 2 * np.pi, size)
    heads = []
    for h in range(n_heads):
        phase = 2 * np.pi * h / n_heads
        row = 0.5 + 0.5 * np.sin(grid * pitch + phase)
        col = 0.5 + 0.5 * np.cos(grid * pitch * 0.7 + phase)
        heads.append(np.outer(row, col))
    stacked = np.stack(heads, axis=0)
    return _softmax(stacked, axis=0).sum(axis=0)


def _diagonal_blend(attended: np.ndarray) -> np.ndarray:
    """Helical field-line mixing along diagonals (vectorized accumulation)."""
    n = attended.shape[0]
    accum = attended.astype(np.float64, copy=True)
    weight = np.ones_like(accum)
    smooth_kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)

    for offset in range(-n // 4, n // 4 + 1):
        diag = np.diagonal(attended, offset=offset)
        if len(diag) <= 1:
            continue
        smoothed = np.convolve(diag, smooth_kernel, mode="same")
        if offset >= 0:
            rows = np.arange(len(smoothed))
            cols = rows + offset
        else:
            cols = np.arange(len(smoothed))
            rows = cols - offset
        valid = (rows >= 0) & (rows < n) & (cols >= 0) & (cols < n)
        accum[rows[valid], cols[valid]] += 0.5 * smoothed[valid]
        weight[rows[valid], cols[valid]] += 0.5

    return (accum / weight).astype(attended.dtype, copy=False)


def apply_spiral_attention(features: np.ndarray, pitch: float = 0.5) -> np.ndarray:
    """Apply multi-head spiral attention to 2D feature map."""
    size = features.shape[0]
    w = spiral_attention_weights(size, pitch)
    attended = features * w
    return _diagonal_blend(attended)


def query_key_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    pitch: float = 0.5,
) -> np.ndarray:
    """Scaled dot-product attention with helical positional bias."""
    n = query.shape[0]
    scale = np.sqrt(n)
    scores = (query @ key.T) / scale
    pos = np.arange(n)
    helical_bias = np.sin((pos[:, None] - pos[None, :]) * pitch * 0.1)
    scores = scores + helical_bias
    weights = _softmax(scores, axis=1)
    return weights @ value
