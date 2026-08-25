"""Write a complete physicist dossier: JSON + Markdown + PDF."""

from __future__ import annotations

import json
from pathlib import Path

from deepiri_fuselk.reports.pdf_export import render_performance_pdf
from deepiri_fuselk.reports.performance_report import PerformanceReport


def export_dossier(
    report: PerformanceReport,
    out_dir: str | Path,
    *,
    stem: str | None = None,
) -> dict[str, Path]:
    """
    Export a citeable performance pack for physicists.

    Writes three artifacts sharing one stem:
    - ``{stem}_performance.json`` — machine-readable metrics
    - ``{stem}_performance.md`` — lab-notebook Markdown
    - ``{stem}_performance.pdf`` — branded A4 PDF for sharing / archive
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = stem or "fuselk_performance"
    json_path = out / f"{name}_performance.json"
    md_path = out / f"{name}_performance.md"
    pdf_path = out / f"{name}_performance.pdf"

    json_path.write_text(json.dumps(report.to_dict(), indent=2))
    md_path.write_text(report.to_markdown())
    render_performance_pdf(report, pdf_path)
    return {"json": json_path.resolve(), "markdown": md_path.resolve(), "pdf": pdf_path.resolve()}
