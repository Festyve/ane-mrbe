"""Tests for reading the released Jacobian-lens artifact.

These pin the schema of the ACTUAL release (neuronpedia/jacobian-lens,
inspected 2026-07): a torch .pt holding

    {"J": {int_layer: (d_model, d_model) tensor},
     "source_layers": [...], "d_model": int, "n_prompts": int}

The repo previously expected flat "layer_{L}" keys in an .npz/.safetensors,
which meant the real artifact could not be loaded at all. The npz path is
kept working, so both spellings are covered here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jspace_binding.model.qwen_jlens import LensFormatError, QwenJLensModel

D = 4


def test_reads_released_pt_schema(tmp_path: Path) -> None:
    """The nested {"J": {int: matrix}} release flattens to layer_{L} keys."""
    torch = pytest.importorskip("torch")
    path = tmp_path / "Qwen3.6-27B_jacobian_lens_n1000.pt"
    torch.save(
        {
            "J": {0: torch.eye(D), 48: torch.eye(D) * 2},
            "source_layers": [0, 48],
            "d_model": D,
            "n_prompts": 1000,
        },
        path,
    )
    arrays = QwenJLensModel._read_arrays(path)

    assert set(arrays) == {"layer_0", "layer_48", "__d_model__"}
    assert tuple(arrays["layer_48"].shape) == (D, D)
    assert int(arrays["__d_model__"]) == D


def test_pt_without_J_mapping_fails_loudly(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "wrong.pt"
    torch.save({"jacobians": {0: torch.eye(D)}}, path)

    with pytest.raises(LensFormatError, match="no 'J' mapping"):
        QwenJLensModel._read_arrays(path)


def test_pt_holding_a_bare_tensor_fails_loudly(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "bare.pt"
    torch.save(torch.eye(D), path)

    with pytest.raises(LensFormatError, match="expected a dict"):
        QwenJLensModel._read_arrays(path)


def test_npz_flat_schema_still_reads(tmp_path: Path) -> None:
    """The pre-existing flat spelling keeps working."""
    path = tmp_path / "lens.npz"
    np.savez(path, layer_0=np.eye(D), layer_1=np.eye(D))
    arrays = QwenJLensModel._read_arrays(path)

    assert set(arrays) == {"layer_0", "layer_1"}
    assert arrays["layer_0"].shape == (D, D)


# --------------------------------------------------------------------- #
# _load_lens against a release-shaped local directory                    #
# --------------------------------------------------------------------- #

SUBPATH = "qwen3.6-27b/jlens/Salesforce-wikitext"


def _release_dir(tmp_path: Path, layers: dict, d_model: int, subpath: str = SUBPATH) -> Path:
    """Mirror the published layout: {root}/{model}/jlens/{corpus}/*.pt."""
    import torch

    root = tmp_path / "lens_repo"
    target = root / subpath
    target.mkdir(parents=True)
    torch.save(
        {"J": layers, "source_layers": sorted(layers), "d_model": d_model, "n_prompts": 1000},
        target / "Qwen3.6-27B_jacobian_lens_n1000.pt",
    )
    return root


def _model_for(root: Path, band: tuple[int, int], d_model: int, subpath: str = SUBPATH):
    """A QwenJLensModel with the heavy bits stubbed, so _load_lens is exercised alone."""
    from dataclasses import replace

    import torch

    from jspace_binding.config import ModelConfig

    config = replace(
        ModelConfig(),
        backend="qwen_jlens",
        lens_repo=str(root),
        lens_subpath=subpath,
        layer_band=band,
    )
    model = QwenJLensModel(config)
    stub = torch.nn.Linear(d_model, 1)  # gives _device() something to read
    model._model = stub
    model._w_u = torch.zeros(7, d_model)  # (n_vocab, d_model)
    return model


def test_load_lens_reads_scoped_release_layout(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    root = _release_dir(tmp_path, {47: torch.eye(D), 48: torch.eye(D) * 3}, D)
    model = _model_for(root, (48, 48), D)

    model._load_lens()

    assert set(model._jacobians) == {48}
    assert tuple(model._jacobians[48].shape) == (D, D)


def test_load_lens_rejects_another_models_lens(tmp_path: Path) -> None:
    """A lens fitted at a different d_model must not be silently loaded."""
    torch = pytest.importorskip("torch")
    root = _release_dir(tmp_path, {48: torch.eye(D + 2)}, D + 2)
    model = _model_for(root, (48, 48), D)

    with pytest.raises(LensFormatError, match="another model's lens"):
        model._load_lens()


def test_load_lens_reports_layer_outside_coverage(tmp_path: Path) -> None:
    """The real release covers layers 0-62 of a 64-layer model."""
    torch = pytest.importorskip("torch")
    root = _release_dir(tmp_path, {0: torch.eye(D), 62: torch.eye(D)}, D)
    model = _model_for(root, (63, 63), D)

    with pytest.raises(LensFormatError, match="outside the lens artifact's coverage"):
        model._load_lens()


def test_load_lens_reports_bad_subpath(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    root = _release_dir(tmp_path, {48: torch.eye(D)}, D)
    model = _model_for(root, (48, 48), D, subpath="qwen3.5-27b/jlens/Salesforce-wikitext")

    with pytest.raises(LensFormatError, match="not found under"):
        model._load_lens()


def test_decoder_layer_resolves_both_model_shapes() -> None:
    """Qwen3.5's AutoModelForCausalLM resolves to the text tower directly
    (.model.layers); Gemma3's resolves to the vision+text wrapper, whose
    decoder layers sit one level deeper (.model.language_model.layers).
    Hit for real on the A100: 'Gemma3Model' object has no attribute 'layers'."""
    from types import SimpleNamespace

    from jspace_binding.model.qwen_jlens import QwenJLensModel

    backend = object.__new__(QwenJLensModel)

    qwen_shape = SimpleNamespace(model=SimpleNamespace(layers=["L0", "L1"]))
    backend._model = qwen_shape
    assert backend._decoder_layer(1) == "L1"

    class _Wrapper:  # no .layers attribute, like Gemma3Model
        language_model = SimpleNamespace(layers=["G0", "G1", "G2"])

    backend._model = SimpleNamespace(model=_Wrapper())
    assert backend._decoder_layer(2) == "G2"


def test_read_arrays_accepts_stacked_jacobians(tmp_path) -> None:
    """camilablank/workspace-lenses (R-lens + its matched J-lens) stacks the
    per-layer Jacobians into one (n_layers, d, d) tensor with the layer indices
    in "source_layers", where neuronpedia nests {int layer -> matrix}. Same
    content, different packing. Rows must map onto the layers source_layers
    names -- these releases do not start at layer 0, so positional indexing
    would silently read the wrong layer's Jacobian."""
    import torch

    from jspace_binding.model.qwen_jlens import LensFormatError, QwenJLensModel

    d, layers = 4, [12, 13, 14]
    stacked = torch.arange(len(layers) * d * d, dtype=torch.float32).reshape(len(layers), d, d)
    path = tmp_path / "lens.pt"
    torch.save({"J": stacked, "source_layers": layers, "d_model": d, "n_prompts": 1000}, path)

    arrays = QwenJLensModel._read_arrays(path)  # noqa: SLF001
    assert set(arrays) == {"layer_12", "layer_13", "layer_14", "__d_model__"}
    assert torch.equal(torch.as_tensor(arrays["layer_13"]), stacked[1])  # row -> named layer
    assert arrays["__d_model__"] == d

    # A stack whose length disagrees with source_layers cannot be mapped.
    bad = tmp_path / "bad.pt"
    torch.save({"J": stacked, "source_layers": [12, 13], "d_model": d}, bad)
    with pytest.raises(LensFormatError, match="source_layers"):
        QwenJLensModel._read_arrays(bad)  # noqa: SLF001
