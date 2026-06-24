"""Generate synthetic fuselk training corpus and sample shots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deepiri_fuselk.data.fetchers.manifest import FetchRecord, now_iso
from deepiri_fuselk.data.imas_loader import export_imas_hdf5, synthetic_imas_shot
from deepiri_fuselk.sim.shot_corpus import generate_corpus


def fetch_synthetic(
    root: Path,
    *,
    n_shots: int = 100,
    grid_size: int = 32,
    seed: int = 42,
) -> FetchRecord:
    corpus_dir = root / "corpus"
    shots_dir = root / "shots"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    shots_dir.mkdir(parents=True, exist_ok=True)

    corpus = generate_corpus(n_shots=n_shots, grid_size=grid_size, seed=seed)
    corpus_path = corpus_dir / "elm_corpus.npz"
    np.savez_compressed(
        corpus_path,
        features=corpus.features_matrix(),
        labels=corpus.labels(),
        elm_rate=np.array([corpus.elm_rate]),
        grid_size=np.array([grid_size]),
    )

    shot_paths: list[Path] = []
    for i in range(min(n_shots, 20)):
        sid = f"SYN{seed + i:04d}"
        shot = synthetic_imas_shot(sid, size=grid_size, seed=seed + i)
        shot_paths.append(export_imas_hdf5(shot, shots_dir / f"{sid}.h5"))

    return FetchRecord(
        source_id="synthetic",
        status="ok",
        fetched_at=now_iso(),
        files=[
            str(corpus_path.relative_to(root)),
            *[str(p.relative_to(root)) for p in shot_paths],
        ],
        shots_written=len(shot_paths),
        details={
            "corpus_frames": len(corpus.frames),
            "elm_rate": corpus.elm_rate,
            "grid_size": grid_size,
            "seed": seed,
        },
    )
