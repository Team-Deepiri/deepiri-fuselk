"""Tests for muon catalytic cycle."""

from deepiri_fuselk.muon.photon_stripper import PhotonStripperConfig, can_strip, stripping_rate
from deepiri_fuselk.muon.rate_network import BREAKEVEN_FUSIONS, RateNetworkParams, run_rate_network


def test_photon_stripper_energy_threshold():
    cfg = PhotonStripperConfig(energy_kev=2.0)
    assert can_strip(cfg)


def test_photon_stripping_rate_positive():
    cfg = PhotonStripperConfig()
    assert stripping_rate(cfg) > 0


def test_rate_network_runs():
    result = run_rate_network(t_span=(0.0, 1e-6))
    assert result.populations.shape[0] == 4
    assert result.fusions_per_muon >= 0


def test_boosted_stripping_improves_network():
    baseline = run_rate_network(
        t_span=(0.0, 1e-5),
        params=RateNetworkParams(R_photon=0.0, R_proton=0.0),
    )
    boosted = run_rate_network(
        t_span=(0.0, 1e-5),
        params=RateNetworkParams(R_photon=0.5, R_proton=0.3),
    )
    assert boosted.fusions_per_muon >= baseline.fusions_per_muon


def test_breakeven_constant():
    assert BREAKEVEN_FUSIONS == 284.0
