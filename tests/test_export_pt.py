"""Tests for directions.export_pt: the Group A .pt handoff bridge.

The payload-assembly path (build_pt_payloads) is pure numpy and runs
everywhere. The actual torch write/read is guarded with importorskip so it
runs where the model extra is installed and is skipped (not failed) on a
machine without a working torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jspace_binding.directions.export_pt import (
    build_pt_payloads,
    export_pt,
    export_pt_from_npz,
    load_fitted_by_site,
)
from jspace_binding.directions.fit import fit_all, save_directions
from jspace_binding.types import InjectionSite

D = 16
N = 40


def _require_torch():
    """Import torch or skip. Unlike pytest.importorskip this also skips on a
    broken build (an x86_64 wheel on an arm64 host raises OSError, not
    ImportError)."""
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip(f"working torch unavailable: {exc}")
    return torch


def _planted(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(D)
    return raw / np.linalg.norm(raw)


def _role_activations(planted: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    agent = planted + 0.3 * rng.standard_normal((N, D))
    patient = -planted + 0.3 * rng.standard_normal((N, D))
    return agent, patient


def _multi_entity(entities=("doctor", "nurse", "chef")) -> dict:
    return {
        e: _role_activations(_planted(i), seed=i + 100) for i, e in enumerate(entities)
    }


def _two_site_fit(entities=("doctor", "nurse", "chef")) -> dict:
    acts = _multi_entity(entities)
    return {
        InjectionSite.FINAL_TOKEN: fit_all(acts, InjectionSite.FINAL_TOKEN, n_bootstrap=20),
        InjectionSite.ENTITY_TOKEN: fit_all(acts, InjectionSite.ENTITY_TOKEN, n_bootstrap=20),
    }


def test_payload_structure_and_unit_norms() -> None:
    payloads = build_pt_payloads(_two_site_fit())
    assert set(payloads) == {"doctor", "nurse", "chef"}
    doctor = payloads["doctor"]
    assert doctor["entity"] == "doctor"
    assert doctor["d_jspace"] == D
    # Sites appear in configured InjectionSite order; final_token is primary.
    assert list(doctor["sites"]) == ["final_token", "entity_token"]
    assert doctor["primary_site"] == "final_token"
    for site in doctor["sites"].values():
        assert np.linalg.norm(site["fitted"]) == pytest.approx(1.0)
        assert np.linalg.norm(site["shuffled"]) == pytest.approx(1.0)
        assert np.linalg.norm(site["generic_loo"]) == pytest.approx(1.0)  # >=2 entities
        assert isinstance(site["raw_norm"], float)
        assert isinstance(site["stability"], float)


def test_payload_preserves_provenance_metadata() -> None:
    metadata = {
        "model_id": "test/model",
        "layer_band": [4, 8],
        "fitting_corpus": "data/fitting.jsonl",
        "git_commit": "abc123",
    }
    payload = build_pt_payloads(_two_site_fit(), metadata=metadata)["doctor"]
    assert payload["metadata"] == metadata


def test_payload_single_entity_has_no_generic_loo() -> None:
    """A single-entity fit leaves generic_loo undefined — carried as None, not
    a zero vector Group A could push into a silent no-op."""
    fit = {InjectionSite.FINAL_TOKEN: fit_all(_multi_entity(("doctor",)), InjectionSite.FINAL_TOKEN, n_bootstrap=20)}
    payload = build_pt_payloads(fit)["doctor"]
    assert payload["sites"]["final_token"]["generic_loo"] is None
    # fitted is still a real unit direction.
    assert np.linalg.norm(payload["sites"]["final_token"]["fitted"]) == pytest.approx(1.0)


def test_payload_fitted_matches_source_direction() -> None:
    fit = _two_site_fit()
    payloads = build_pt_payloads(fit)
    src = fit[InjectionSite.FINAL_TOKEN].direction("nurse", "fitted")
    assert payloads["nurse"]["sites"]["final_token"]["fitted"] == pytest.approx(src)


def test_build_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no directions"):
        build_pt_payloads({})


def test_load_fitted_by_site_skips_absent_and_errors_on_none(tmp_path: Path) -> None:
    fit = fit_all(_multi_entity(), InjectionSite.FINAL_TOKEN, n_bootstrap=20)
    save_directions(fit, tmp_path)  # only final_token on disk
    loaded = load_fitted_by_site(tmp_path)
    assert set(loaded) == {InjectionSite.FINAL_TOKEN}
    with pytest.raises(FileNotFoundError, match="fit_directions"):
        load_fitted_by_site(tmp_path / "empty")


# --- torch-dependent round trip (skipped without a working torch) ---


def test_export_pt_roundtrip(tmp_path: Path) -> None:
    torch = _require_torch()
    fit = _two_site_fit()
    metadata = {"model_id": "test/model", "git_commit": "abc123"}
    written = export_pt(fit, tmp_path, metadata=metadata)
    assert [p.name for p in written] == [
        "chef_role_direction.pt",
        "doctor_role_direction.pt",
        "nurse_role_direction.pt",
    ]
    loaded = torch.load(tmp_path / "doctor_role_direction.pt", weights_only=False)
    assert loaded["entity"] == "doctor"
    assert loaded["metadata"] == metadata
    assert set(loaded["sites"]) == {"final_token", "entity_token"}
    fitted = loaded["sites"]["final_token"]["fitted"]
    assert fitted.dtype == torch.float32
    assert float(fitted.norm()) == pytest.approx(1.0, abs=1e-5)
    # matches the numpy source
    src = fit[InjectionSite.FINAL_TOKEN].direction("doctor", "fitted")
    assert fitted.numpy() == pytest.approx(src.astype(np.float32), abs=1e-6)


def test_export_pt_from_npz(tmp_path: Path) -> None:
    _require_torch()
    fit = _two_site_fit()
    for site_fit in fit.values():
        save_directions(site_fit, tmp_path)
    written = export_pt_from_npz(tmp_path)
    assert {p.name for p in written} == {
        "chef_role_direction.pt",
        "doctor_role_direction.pt",
        "nurse_role_direction.pt",
    }
