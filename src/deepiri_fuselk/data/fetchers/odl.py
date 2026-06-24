"""Download Open Density Limit Database and normalize to fuselk shots."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Any

from deepiri_fuselk.data.fetchers.manifest import FetchRecord, now_iso
from deepiri_fuselk.data.normalize import odl_csv_to_shots
from deepiri_fuselk.data.sources import get_source

ODL_CSV_URL = (
    "https://raw.githubusercontent.com/MIT-PSFC/open_density_limit_database/main/data/DL_DataFrame.csv"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_odl(
    root: Path,
    *,
    force: bool = False,
    max_discharges: int | None = 50,
) -> FetchRecord:
    src = get_source("odl")
    raw_dir = root / "raw" / "odl"
    shots_dir = root / "shots"
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / "DL_DataFrame.csv"
    if force or not csv_path.exists():
        urllib.request.urlretrieve(ODL_CSV_URL, csv_path)

    written = odl_csv_to_shots(csv_path, shots_dir, max_discharges=max_discharges)
    return FetchRecord(
        source_id="odl",
        status="ok",
        fetched_at=now_iso(),
        files=[str(csv_path.relative_to(root)), *[str(p.relative_to(root)) for p in written]],
        shots_written=len(written),
        details={
            "checksum_sha256": _sha256(csv_path),
            "url": ODL_CSV_URL,
            "license": src.license if src else "CC-BY",
            "device": src.device if src else "Alcator C-Mod",
        },
    )
