"""Spiral attention layer stub (rotationally-aware focal extraction)."""

from __future__ import annotations

import numpy as np


def spiral_attention_weights(size: int, pitch: float = 0.5) -> np.ndarray:
    """Build attention mask that follows helical pitch."""
    grid = np.linspace(0, 2 * np.pi, size)
    weights = 0.5 + 0.5 * np.sin(grid * pitch + np.arange(size)[:, None] * 0.1)
    weights /= weights.sum()
    return weights


def apply_spiral_attention(features: np.ndarray, pitch: float = 0.5) -> np.ndarray:
    """Apply spiral attention to feature map."""
    size = features.shape[0]
    w = spiral_attention_weights(size, pitch)
    if features.ndim == 2:
        return features * w
    return features * w[:, 0]
