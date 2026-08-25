"""Shot-faithful replay: scrub IMAS/ODL archives through the control stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from deepiri_fuselk.data.imas_loader import IMASShot, load_imas_hdf5
from deepiri_fuselk.data.notebook_loaders import load_odl_meta
from deepiri_fuselk.sim.reactor_cell import ReactorCell, ReactorRun, ReactorStep


@dataclass
class ReplayFrame:
    """One scrub position through an archived discharge."""

    index: int
    time_s: float | None
    step: ReactorStep
    odl_label: int | None = None
    density: float | None = None


@dataclass
class ShotReplayResult:
    """Full scrub of one IMAS shot through ReactorCell."""

    shot_id: str
    device: str
    frames: list[ReplayFrame] = field(default_factory=list)
    mean_disruption: float = 0.0
    mean_snr: float = 0.0
    odl_label_rate: float | None = None
    final_score: float = 0.0

    def to_report(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "device": self.device,
            "n_frames": len(self.frames),
            "mean_disruption": self.mean_disruption,
            "mean_snr": self.mean_snr,
            "odl_label_rate": self.odl_label_rate,
            "final_score": self.final_score,
            "actions": [f.step.action_taken for f in self.frames],
        }


class ShotReplayer:
    """
    Drive ReactorCell from a fetched IMAS shot (C-Mod ODL or SYN).

    Each scrub step reuses the archived heat field + profiles so HELIX/Venturi
    see real diagnostics instead of freshly synthesized ECE.
    """

    def __init__(self, cell: ReactorCell | None = None, grid_size: int = 32) -> None:
        self.cell = cell or ReactorCell(grid_size=grid_size, train_elm=False)
        self.grid_size = self.cell.grid_size

    def attach(self, shot: IMASShot) -> None:
        self.cell.imas = shot
        self.cell.grid_size = shot.heat_field.shape[0]
        self.cell.hybrid.venturi.reset()
        self.cell._step = 0
        self.cell._kpi_acc.reset()

    def load(self, path: str | Path) -> IMASShot:
        shot = load_imas_hdf5(path)
        self.attach(shot)
        return shot

    def scrub(
        self,
        shot: IMASShot | None = None,
        *,
        n_steps: int | None = None,
        path: str | Path | None = None,
        seed: int = 42,
    ) -> ShotReplayResult:
        if path is not None:
            shot = self.load(path)
        elif shot is not None:
            self.attach(shot)
        else:
            shot = self.cell.imas

        meta = load_odl_meta(path) if path is not None else None
        times = shot.time
        labels = meta["density_limit_phase"] if meta else None
        densities = meta["density"] if meta else None

        # Prefer timebase length for scrub; fall back to requested steps
        n_time = len(times) if times is not None and len(times) else 1
        steps = n_steps if n_steps is not None else min(n_time, 40)

        frames: list[ReplayFrame] = []
        for i in range(steps):
            # Map scrub index onto ODL timebase when available
            t_idx = int(i * (n_time - 1) / max(steps - 1, 1)) if n_time > 1 else 0
            t_s = float(times[t_idx]) if times is not None and len(times) else None
            label = int(labels[t_idx]) if labels is not None and len(labels) else None
            dens = float(densities[t_idx]) if densities is not None and len(densities) else None

            # Amplify island proxy when density-limit phase is active
            island = 0.35 + (0.45 if label == 1 else 0.0)
            step = self._step_with_shot(shot, seed=seed + i, island_amplitude=island)
            frames.append(
                ReplayFrame(
                    index=i,
                    time_s=t_s,
                    step=step,
                    odl_label=label,
                    density=dens,
                )
            )

        probs = [f.step.disruption.probability for f in frames]
        snrs = [f.step.helix.phase_locked_snr for f in frames]
        odl_rate = float(np.mean(labels)) if labels is not None and len(labels) else None
        final = frames[-1].step.kpis.score() if frames else 0.0

        return ShotReplayResult(
            shot_id=shot.shot_id,
            device=shot.device,
            frames=frames,
            mean_disruption=float(np.mean(probs)) if probs else 0.0,
            mean_snr=float(np.mean(snrs)) if snrs else 0.0,
            odl_label_rate=odl_rate,
            final_score=final,
        )

    def _step_with_shot(
        self,
        shot: IMASShot,
        *,
        seed: int,
        island_amplitude: float,
    ) -> ReactorStep:
        self.cell._step += 1
        result = self.cell._pipeline.process(
            grid_size=self.cell.grid_size,
            seed=seed,
            q_profile_values=np.array(shot.q_profile.values, dtype=np.float64),
            te_profile_values=np.array(shot.Te_profile.values, dtype=np.float64),
            imas=shot,
            island_amplitude=island_amplitude,
        )
        from deepiri_fuselk.barrier.breeding_blanket import tritium_breeding_ratio
        from deepiri_fuselk.sim.fusion_kpis import FusionKPIs, divertor_uniformity

        self.cell._kpi_acc.update(
            snr=result.helix.phase_locked_snr,
            reward=result.control.venturi.reward,
            elm_probability=result.disruption.probability,
        )
        tbr = tritium_breeding_ratio(
            self.cell._fuel_cycle.pde.state,
            self.cell._fuel_cycle.pde_params,
        )
        kpis = FusionKPIs(
            tritium_breeding_ratio=tbr,
            elm_free_fraction=self.cell._kpi_acc.elm_free_fraction,
            divertor_uniformity=divertor_uniformity(result.control.final_heat),
            disruption_risk=result.disruption.probability,
            muon_gain=self.cell._fuel_cycle.muon_fpm,
            helix_snr_mean=self.cell._kpi_acc.helix_snr_mean,
            venturi_mean_reward=self.cell._kpi_acc.venturi_mean_reward,
            q_min=result.q_min,
            beta_n=result.beta_n,
        )
        return ReactorStep(
            step=self.cell._step,
            kpis=kpis,
            disruption=result.disruption,
            heat_flux=result.control.final_heat,
            action_taken=result.recommended_action,
            helix=result.helix,
            raw_heat=result.shot.heat_field,
            seed=seed,
        )


def replay_to_reactor_run(result: ShotReplayResult) -> ReactorRun:
    """Convert scrub frames into a ReactorRun for existing report tooling."""
    run = ReactorRun(steps=[f.step for f in result.frames], final_score=result.final_score)
    return run
