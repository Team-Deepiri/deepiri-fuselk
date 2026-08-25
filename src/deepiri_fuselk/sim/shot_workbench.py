"""Shot Workbench — the scrubbable pulse analysis desk no open stack has.

Locksmith reframe
-----------------
Physicists don't need another prettier SOLPS wrapper. They need one timeline
where *experimental labels*, *HELIX lock*, *disruption fusion*, *Venturi
actions*, and *conserved P_rad* share the same clock — plus a counterfactual:
what if control had stayed open-loop?

That closed identity (data → diagnose → act → prove the gauges match the
render) is the system that makes the "you can't trust a browser sim" blocker
irrelevant.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from deepiri_fuselk.data.imas_loader import load_imas_hdf5
from deepiri_fuselk.data.notebook_loaders import (
    ensure_fetched_data,
    list_shots,
    load_odl_meta,
    resolve_data_root,
)
from deepiri_fuselk.data.paths import under_root
from deepiri_fuselk.helix.disruption_filament import disruption_filament
from deepiri_fuselk.helix.helix_engine import HelixEngine
from deepiri_fuselk.models.disruption_detector import DisruptionDetector
from deepiri_fuselk.models.elm_predictor import ELMPredictor
from deepiri_fuselk.sim.shot_replay import ShotReplayer, ShotReplayResult
from deepiri_fuselk.viz.radiance import radiance_field


@dataclass
class TimelinePoint:
    index: int
    time_s: float | None
    odl_label: int | None
    density: float | None
    p_dis: float
    snr: float
    action: str
    peak_heat: float
    divertor_uniformity: float
    controlled: bool


@dataclass
class CounterfactualDelta:
    """Open-loop vs closed-loop (Venturi) on the same shot."""

    mean_p_dis_open: float
    mean_p_dis_closed: float
    delta_p_dis: float  # closed − open (negative = control helped)
    mean_uniformity_open: float
    mean_uniformity_closed: float
    delta_uniformity: float  # closed − open (positive = more uniform)
    first_odl_onset_s: float | None
    first_action_s: float | None
    lead_time_s: float | None  # action before label (positive = early)


@dataclass
class WorkbenchReport:
    """Physicist-facing pulse analysis artifact."""

    shot_id: str
    device: str
    generated_at: str
    n_frames: int
    odl_label_rate: float | None
    closed_loop: ShotReplayResult
    open_loop_timeline: list[TimelinePoint]
    closed_loop_timeline: list[TimelinePoint]
    counterfactual: CounterfactualDelta
    filament: dict
    radiance_p_rad_mw: float
    thesis: str = (
        "fuselk Shot Workbench: one timeline for experimental labels, HELIX, "
        "disruption fusion, Venturi actuation, and conserved continuum P_rad — "
        "with open-loop vs closed-loop counterfactual on the same discharge."
    )

    def to_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "device": self.device,
            "generated_at": self.generated_at,
            "n_frames": self.n_frames,
            "odl_label_rate": self.odl_label_rate,
            "thesis": self.thesis,
            "closed_loop": self.closed_loop.to_report(),
            "counterfactual": asdict(self.counterfactual),
            "filament": self.filament,
            "radiance_p_rad_mw": self.radiance_p_rad_mw,
            "closed_loop_timeline": [asdict(p) for p in self.closed_loop_timeline],
            "open_loop_timeline": [asdict(p) for p in self.open_loop_timeline],
        }

    def to_markdown(self) -> str:
        cf = self.counterfactual
        lines = [
            f"# Shot Workbench — `{self.shot_id}`",
            "",
            f"**Device:** {self.device}  ",
            f"**Generated:** {self.generated_at}  ",
            f"**Frames:** {self.n_frames}",
            "",
            f"> {self.thesis}",
            "",
            "## Counterfactual (closed-loop Venturi − open-loop)",
            "",
            "| Metric | Open-loop | Closed-loop | Δ |",
            "|--------|-----------|-------------|---|",
            f"| Mean P_dis | {cf.mean_p_dis_open:.3f} | {cf.mean_p_dis_closed:.3f} | {cf.delta_p_dis:+.3f} |",
            f"| Divertor uniformity | {cf.mean_uniformity_open:.3f} | {cf.mean_uniformity_closed:.3f} | {cf.delta_uniformity:+.3f} |",
            "",
            f"- ODL onset: `{cf.first_odl_onset_s}` s",
            f"- First mitigating action: `{cf.first_action_s}` s",
            f"- Lead time (action before label): `{cf.lead_time_s}` s",
            "",
            "## Continuum / filament",
            "",
            f"- Radiance P_rad (bremsstrahlung model): **{self.radiance_p_rad_mw:.3f} MW**",
            f"- Disruption filament pitch: **{float(self.filament.get('pitch_rad', 0.0)):.4f} rad**",
            f"- Filament device proxy: `{self.filament.get('device', '—')}`",
            "",
            "## Closed-loop actions",
            "",
        ]
        actions = self.closed_loop.to_report().get("actions", [])
        uniq = sorted(set(actions))
        lines.append(f"Distinct actions: {', '.join(uniq) if uniq else 'none'}")
        lines.append("")
        lines.append("## Timeline (closed-loop, first 12 rows)")
        lines.append("")
        lines.append("| i | t [s] | ODL | P_dis | SNR | action | Q_peak |")
        lines.append("|---|-------|-----|-------|-----|--------|--------|")
        for p in self.closed_loop_timeline[:12]:
            t = f"{p.time_s:.3f}" if p.time_s is not None else "—"
            odl = "1" if p.odl_label == 1 else ("0" if p.odl_label == 0 else "—")
            lines.append(
                f"| {p.index} | {t} | {odl} | {p.p_dis:.2f} | {p.snr:.2f} | "
                f"`{p.action}` | {p.peak_heat:.2f} |"
            )
        lines.append("")
        lines.append("---")
        lines.append("*Generated by deepiri-fuselk Shot Workbench.*")
        return "\n".join(lines)


def _timeline_from_replay(result: ShotReplayResult, *, controlled: bool) -> list[TimelinePoint]:
    points: list[TimelinePoint] = []
    for fr in result.frames:
        heat = fr.step.heat_flux if controlled else fr.step.raw_heat
        points.append(
            TimelinePoint(
                index=fr.index,
                time_s=fr.time_s,
                odl_label=fr.odl_label,
                density=fr.density,
                p_dis=float(fr.step.disruption.probability),
                snr=float(fr.step.helix.phase_locked_snr),
                action=fr.step.action_taken if controlled else "open_loop",
                peak_heat=float(np.max(heat)),
                divertor_uniformity=float(fr.step.kpis.divertor_uniformity),
                controlled=controlled,
            )
        )
    return points


def _open_loop_scrub(path: Path, n_steps: int, seed: int = 42) -> list[TimelinePoint]:
    """HELIX + disruption only — raw heat never passes through Venturi."""
    from deepiri_fuselk.data.notebook_loaders import imas_to_synthetic_shot
    from deepiri_fuselk.sim.fusion_kpis import divertor_uniformity

    shot = load_imas_hdf5(path)
    meta = load_odl_meta(path)
    times = shot.time
    labels = meta["density_limit_phase"] if meta else None
    densities = meta["density"] if meta else None
    n_time = len(times) if times is not None and len(times) else 1
    steps = min(n_steps, max(n_time, 1))

    engine = HelixEngine()
    detector = DisruptionDetector(ELMPredictor())
    ece = imas_to_synthetic_shot(shot)
    points: list[TimelinePoint] = []

    for i in range(steps):
        t_idx = int(i * (n_time - 1) / max(steps - 1, 1)) if n_time > 1 else 0
        helix = engine.process(ece.heat_field, ece.raw_signal, ece.angles)
        q_min = float(np.min(shot.q_profile.values))
        beta_n = float(np.mean(shot.Te_profile.values)) / 2000.0
        assess = detector.assess(helix, q_min=q_min, beta_n=beta_n)
        points.append(
            TimelinePoint(
                index=i,
                time_s=float(times[t_idx]) if times is not None and len(times) else None,
                odl_label=int(labels[t_idx]) if labels is not None and len(labels) else None,
                density=float(densities[t_idx])
                if densities is not None and len(densities)
                else None,
                p_dis=float(assess.probability),
                snr=float(helix.phase_locked_snr),
                action="open_loop",
                peak_heat=float(np.max(ece.heat_field)),
                divertor_uniformity=divertor_uniformity(ece.heat_field),
                controlled=False,
            )
        )
    return points


def _counterfactual(
    closed: list[TimelinePoint], opened: list[TimelinePoint]
) -> CounterfactualDelta:
    def mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    p_open = [p.p_dis for p in opened]
    p_closed = [p.p_dis for p in closed]
    u_open = [p.divertor_uniformity for p in opened]
    u_closed = [p.divertor_uniformity for p in closed]

    first_odl = next((p.time_s for p in closed if p.odl_label == 1), None)
    first_act = next(
        (
            p.time_s
            for p in closed
            if p.action not in ("none", "monitor", "open_loop", "") and p.action
        ),
        None,
    )
    lead = None
    if first_odl is not None and first_act is not None:
        lead = float(first_odl - first_act)

    return CounterfactualDelta(
        mean_p_dis_open=mean(p_open),
        mean_p_dis_closed=mean(p_closed),
        delta_p_dis=mean(p_closed) - mean(p_open),
        mean_uniformity_open=mean(u_open),
        mean_uniformity_closed=mean(u_closed),
        delta_uniformity=mean(u_closed) - mean(u_open),
        first_odl_onset_s=first_odl,
        first_action_s=first_act,
        lead_time_s=lead,
    )


def resolve_shot_path(
    shot: str | Path,
    *,
    data_root: Path | None = None,
    ensure_data: bool = False,
) -> Path:
    """Resolve a filesystem path or ODL/synthetic shot id to an HDF5 archive."""
    candidate = Path(shot)
    if candidate.is_file():
        # Explicit absolute/relative archives from the operator are allowed.
        return candidate.resolve()
    root = under_root(data_root or resolve_data_root())
    if ensure_data:
        root = ensure_fetched_data(root, n_shots=40, max_odl=20)
    needle = str(shot).strip()
    # Accept bare discharge ids, CMOD_*, SYN*, or stem matches.
    for path in list_shots(root):
        name = path.stem
        discharge = name.removeprefix("CMOD_").removeprefix("SYN_")
        if needle in (name, path.name, discharge) or needle in name:
            return path
    raise FileNotFoundError(
        f"Shot '{shot}' not found under {under_root(root, 'shots')}. "
        "Pass an HDF5 path or run `fuselk data fetch` first."
    )


class ShotWorkbench:
    """Pulse analysis desk: scrub + counterfactual + export."""

    def __init__(self, grid_size: int = 32, data_root: Path | None = None) -> None:
        self.grid_size = grid_size
        self.data_root = data_root
        self.replayer = ShotReplayer(grid_size=grid_size)

    def analyze(
        self,
        shot: str | Path,
        *,
        n_steps: int = 24,
        seed: int = 42,
        ensure_data: bool = False,
    ) -> WorkbenchReport:
        path = resolve_shot_path(shot, data_root=self.data_root, ensure_data=ensure_data)
        closed = self.replayer.scrub(path=path, n_steps=n_steps, seed=seed)
        closed_tl = _timeline_from_replay(closed, controlled=True)
        open_tl = _open_loop_scrub(path, n_steps=n_steps, seed=seed)
        cf = _counterfactual(closed_tl, open_tl)

        imas = load_imas_hdf5(path)
        from deepiri_fuselk.devices.registry import DeviceRegistry

        # Alcator C-Mod ODL shots ≈ DIII-D scale for filament / volume proxies
        # (public ODL has no IMAS device geometry IDS).
        device = DeviceRegistry().get("DIII-D")
        te_raw = float(np.mean(imas.Te_profile.values))
        te = te_raw / 1000.0 if te_raw > 50 else te_raw  # eV → keV when needed
        ne = float(np.mean(imas.ne_profile.values)) / 1e19
        heat = float(np.max(imas.heat_field)) * 10.0
        rad = radiance_field(
            te_kev=max(te, 0.5),
            ne_1e19=max(ne, 0.1),
            heat_flux_mw_m2=heat,
            major_radius_m=device.major_radius_m,
            minor_radius_m=device.minor_radius_m,
            elongation=device.elongation,
            triangularity=device.triangularity,
        )
        filament = disruption_filament(
            device,
            q95=float(imas.q_profile.values[-1]) if len(imas.q_profile.values) else 3.0,
            fracture_vector=closed.frames[-1].step.helix.fracture_vector if closed.frames else None,
        )

        return WorkbenchReport(
            shot_id=closed.shot_id,
            device=closed.device,
            generated_at=datetime.now(UTC).isoformat(),
            n_frames=len(closed.frames),
            odl_label_rate=closed.odl_label_rate,
            closed_loop=closed,
            open_loop_timeline=open_tl,
            closed_loop_timeline=closed_tl,
            counterfactual=cf,
            filament=filament.to_dict(),
            radiance_p_rad_mw=rad.p_rad_mw,
        )

    def analyze_batch(
        self,
        data_root: Path | None = None,
        *,
        max_shots: int = 6,
        n_steps: int = 16,
        ensure_data: bool = True,
    ) -> list[WorkbenchReport]:
        root = data_root or resolve_data_root()
        if ensure_data:
            root = ensure_fetched_data(root, n_shots=40, max_odl=max(max_shots, 10))
        paths = list_shots(root, source="cmod")[:max_shots]
        if not paths:
            paths = list_shots(root, source="synthetic")[:max_shots]
        return [self.analyze(p, n_steps=n_steps) for p in paths]

    def export(
        self,
        report: WorkbenchReport,
        out_dir: str | Path,
        *,
        stem: str | None = None,
        pdf: bool = True,
    ) -> dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        name = stem or report.shot_id
        json_path = out / f"{name}_workbench.json"
        md_path = out / f"{name}_workbench.md"
        json_path.write_text(json.dumps(report.to_dict(), indent=2))
        md_path.write_text(report.to_markdown())
        paths: dict[str, Path] = {"json": json_path, "markdown": md_path}
        if pdf:
            from deepiri_fuselk import __version__
            from deepiri_fuselk.reports import export_dossier, from_workbench

            dossier = export_dossier(
                from_workbench(report, version=__version__),
                out,
                stem=name,
            )
            paths["pdf"] = dossier["pdf"]
            # Keep workbench markdown/json as primary scrub artifacts; dossier
            # PDF is the shareable performance pack.
        return paths
