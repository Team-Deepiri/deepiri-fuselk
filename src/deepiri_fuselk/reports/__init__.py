"""Performance report export — Markdown, JSON, and auto-generated PDF."""

from __future__ import annotations

from deepiri_fuselk.reports.dossier import export_dossier
from deepiri_fuselk.reports.pdf_export import render_performance_pdf
from deepiri_fuselk.reports.performance_report import (
    PerformanceReport,
    from_fusion_cell,
    from_odl_benchmark,
    from_reactor_run,
    from_workbench,
)

__all__ = [
    "PerformanceReport",
    "export_dossier",
    "from_fusion_cell",
    "from_odl_benchmark",
    "from_reactor_run",
    "from_workbench",
    "render_performance_pdf",
]
