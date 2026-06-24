"""Helical spiral attention for field-line-aligned feature extraction."""

from __future__ import annotations

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


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


def apply_spiral_attention(features: np.ndarray, pitch: float = 0.5) -> np.ndarray:
    """Apply multi-head spiral attention to 2D feature map."""
    size = features.shape[0]
    w = spiral_attention_weights(size, pitch)
    attended = features * w
    # Field-line mixing: average along diagonals (helical direction)
    n = size
    mixed = attended.copy()
    for offset in range(-n // 4, n // 4 + 1):
        diag = np.diagonal(attended, offset=offset)
        if len(diag) > 1:
            smoothed = np.convolve(diag, [0.25, 0.5, 0.25], mode="same")
            for i, val in enumerate(smoothed):
                r = i if offset >= 0 else i - offset
                c = i + offset if offset >= 0 else i
                if 0 <= r < n and 0 <= c < n:
                    mixed[r, c] = 0.5 * mixed[r, c] + 0.5 * val
    return mixed


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
