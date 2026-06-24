"""Simulation subpackage."""

from deepiri_fuselk.sim.domain_randomizer import RandomizedDomain, randomize_domain
from deepiri_fuselk.sim.fusion_kpis import FusionKPIs, divertor_uniformity, elm_free_fraction
from deepiri_fuselk.sim.solps_wrapper import SOLPSConfig, SOLPSResult, SOLPSWrapper
from deepiri_fuselk.sim.synthetic_data_gen import SyntheticShot, generate_ece_shot

__all__ = [
    "DigitalTwin",
    "FusionKPIs",
    "RandomizedDomain",
    "ReactorCell",
    "ReactorRun",
    "ReactorStep",
    "SOLPSConfig",
    "SOLPSResult",
    "SOLPSWrapper",
    "ShotCorpus",
    "SyntheticShot",
    "TwinHistory",
    "TwinState",
    "divertor_uniformity",
    "elm_free_fraction",
    "generate_corpus",
    "generate_ece_shot",
    "randomize_domain",
]


def __getattr__(name: str):
    """Lazy imports to avoid circular dependency with control/models."""
    if name == "DigitalTwin":
        from deepiri_fuselk.sim.digital_twin import DigitalTwin

        return DigitalTwin
    if name == "TwinHistory":
        from deepiri_fuselk.sim.digital_twin import TwinHistory

        return TwinHistory
    if name == "TwinState":
        from deepiri_fuselk.sim.digital_twin import TwinState

        return TwinState
    if name == "ReactorCell":
        from deepiri_fuselk.sim.reactor_cell import ReactorCell

        return ReactorCell
    if name == "ReactorRun":
        from deepiri_fuselk.sim.reactor_cell import ReactorRun

        return ReactorRun
    if name == "ReactorStep":
        from deepiri_fuselk.sim.reactor_cell import ReactorStep

        return ReactorStep
    if name == "ShotCorpus":
        from deepiri_fuselk.sim.shot_corpus import ShotCorpus

        return ShotCorpus
    if name == "generate_corpus":
        from deepiri_fuselk.sim.shot_corpus import generate_corpus

        return generate_corpus
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
