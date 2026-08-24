"""The calibrate -> run_primary handoff (primary.resolve_calibrated_strengths).

Pins the behaviour that replaced the manual "copy the two printed values into
the YAML" step: values are loaded from data/calibration.json automatically, an
explicit config value still overrides, a record fitted at another site is never
applied silently, and a push experiment with no usable coefficient is refused
before the sweep instead of degrading to a pure swap.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from jspace_binding.config import Config
from jspace_binding.experiments.primary import resolve_calibrated_strengths
from jspace_binding.types import EditType, InjectionSite


def _config_with_calibration(tmp_path: Path, **model_overrides) -> Config:
    """A default config whose calibration path points inside tmp_path."""
    config = Config()
    config = replace(config, paths=replace(config.paths, calibration=tmp_path / "calibration.json"))
    if model_overrides:
        config = replace(config, model=replace(config.model, **model_overrides))
    return config


def _write_record(path: Path, *, site: str, push: float | None, alpha: float | None) -> None:
    record: dict = {"site": site}
    record["push_coefficient"] = {"value": push} if push is not None else None
    record["alpha"] = {"value": alpha} if alpha is not None else None
    path.write_text(json.dumps(record), encoding="utf-8")


def test_dry_run_bypasses_calibration(tmp_path):
    # DummyModel has no strength dial: no file needed, nothing filled, no exit.
    config = _config_with_calibration(tmp_path)  # push/alpha default to None
    out = resolve_calibrated_strengths(config, InjectionSite.ENTITY_TOKEN, dry_run=True)
    assert out.model.push_coefficient is None
    assert out.model.alpha is None


def test_loads_matching_site_record(tmp_path):
    config = _config_with_calibration(tmp_path)
    _write_record(config.paths.calibration, site="entity_token", push=12.0, alpha=3.0)
    out = resolve_calibrated_strengths(config, InjectionSite.ENTITY_TOKEN)
    assert out.model.push_coefficient == 12.0
    assert out.model.alpha == 3.0


def test_explicit_config_value_overrides_record(tmp_path):
    config = _config_with_calibration(tmp_path, push_coefficient=8.0)
    _write_record(config.paths.calibration, site="entity_token", push=12.0, alpha=None)
    out = resolve_calibrated_strengths(config, InjectionSite.ENTITY_TOKEN)
    assert out.model.push_coefficient == 8.0  # config wins over the record


def test_wrong_site_record_refuses_when_push_unset(tmp_path):
    config = _config_with_calibration(tmp_path)
    _write_record(config.paths.calibration, site="final_token", push=12.0, alpha=None)
    with pytest.raises(SystemExit) as exc:
        resolve_calibrated_strengths(config, InjectionSite.ENTITY_TOKEN)
    assert exc.value.code == 2


def test_missing_calibration_refuses_push_experiment(tmp_path):
    config = _config_with_calibration(tmp_path)  # no file written, push edits present
    with pytest.raises(SystemExit) as exc:
        resolve_calibrated_strengths(config, InjectionSite.ENTITY_TOKEN)
    assert exc.value.code == 2


def test_no_push_edittypes_allows_missing_coefficient(tmp_path):
    config = _config_with_calibration(tmp_path)
    config = replace(config, experiment=replace(config.experiment, edit_types=(EditType.NO_EDIT,)))
    out = resolve_calibrated_strengths(config, InjectionSite.ENTITY_TOKEN)
    assert out.model.push_coefficient is None  # nothing to scale, no refusal
