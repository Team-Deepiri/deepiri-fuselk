"""Focal diagnostics subpackage."""

from deepiri_fuselk.focal.focal_heatmap import focal_heatmap, from_hqrm, singularity_gradient
from deepiri_fuselk.focal.spiral_attention import apply_spiral_attention, spiral_attention_weights

__all__ = [
    "apply_spiral_attention",
    "focal_heatmap",
    "from_hqrm",
    "singularity_gradient",
    "spiral_attention_weights",
]
