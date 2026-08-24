"""Export fitted role-directions as torch ``.pt`` files.

``directions.fit`` stores directions as pure-numpy ``directions_{site}.npz``;
this is the bridge to a torch-based editing stack, writing one
``{entity}_role_direction.pt`` per entity across every injection site.
:func:`build_pt_payloads` stays pure numpy, and only :func:`export_pt` imports
torch, lazily.

Payload schema (``{entity}_role_direction.pt``; a plain dict, load with
``torch.load(path, weights_only=False)``)::

    entity:       str
    metadata:     dict                    # model/layer/corpus/commit provenance
    d_jspace:     int                     # direction dimensionality
    primary_site: str                     # the first site present (final_token)
    variants:     list[str]               # variant keys present per site
    sites:        dict[str, dict] keyed by InjectionSite.value, each:
        fitted:      float32 tensor (d,)  # UNIT role-direction (primary push)
        shuffled:    float32 tensor (d,)  # unit shuffled-label control
        generic_loo: float32 tensor (d,) or None  # None on a single-entity fit
        raw_norm:    float                # |mean_agent - mean_patient| pre-norm
        stability:   float                # bootstrap mean-cosine pilot check

Pushes apply ``coefficient * fitted`` (fitted is unit-norm), so the raw norm
and stability travel as diagnostics only — exactly the contract the fit module
documents.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from jspace_binding.directions.fit import FittedDirections, load_directions
from jspace_binding.types import InjectionSite

# generic_loo rows are stored as zeros when the corpus has a single entity
# (leave-one-out is undefined); treat anything at/under this norm as absent.
_ZERO_NORM = 1e-12

_VARIANTS = ("fitted", "shuffled", "generic_loo")


def load_fitted_by_site(
    directory: str | Path, sites: tuple[InjectionSite, ...] | None = None
) -> dict[InjectionSite, FittedDirections]:
    """Load every ``directions_{site}.npz`` present under ``directory``.

    ``sites`` defaults to all members of :class:`InjectionSite`; a site whose
    ``.npz`` is absent is skipped (a fit run may cover only some sites). Raises
    if none are found, so a wrong directory fails loudly instead of writing
    empty ``.pt`` files.
    """
    sites = sites or tuple(InjectionSite)
    directory = Path(directory)
    found: dict[InjectionSite, FittedDirections] = {}
    for site in sites:
        if (directory / f"directions_{site.value}.npz").exists():
            found[site] = load_directions(directory, site)
    if not found:
        raise FileNotFoundError(
            f"no directions_*.npz under {directory}; run scripts/fit_directions.py first"
        )
    return found


def build_pt_payloads(
    directions_by_site: dict[InjectionSite, FittedDirections],
    metadata: Mapping[str, object] | None = None,
) -> dict[str, dict]:
    """Assemble one per-entity payload of numpy arrays + metadata.

    Pure numpy (no torch) so the handoff structure is testable without the
    model extra; :func:`export_pt` converts these arrays to tensors. Every
    entity present in any site is emitted; an entity missing from some site
    simply has no entry for that site. A degenerate ``generic_loo`` row (zero
    norm, i.e. a single-entity fit) is carried as ``None`` rather than a
    zero vector Group A might push into a silent no-op.
    """
    if not directions_by_site:
        raise ValueError("build_pt_payloads: no directions supplied")

    # Stable site order: the configured InjectionSite order, restricted to
    # what is present. The first is tagged primary_site for convenience.
    ordered_sites = [s for s in InjectionSite if s in directions_by_site]
    entities = sorted({e for d in directions_by_site.values() for e in d.entities})

    payloads: dict[str, dict] = {}
    for entity in entities:
        sites_payload: dict[str, dict] = {}
        d_jspace: int | None = None
        for site in ordered_sites:
            fitted = directions_by_site[site]
            if entity not in fitted.entities:
                continue
            i = fitted.index(entity)
            loo = fitted.generic_loo[i]
            generic = None if float(np.linalg.norm(loo)) < _ZERO_NORM else np.asarray(loo)
            sites_payload[site.value] = {
                "fitted": np.asarray(fitted.fitted[i]),
                "shuffled": np.asarray(fitted.shuffled[i]),
                "generic_loo": generic,
                "raw_norm": float(fitted.raw_norms[i]),
                "stability": float(fitted.stability[i]),
            }
            d_jspace = int(fitted.fitted[i].shape[0])
        if not sites_payload:  # entity present in the set union but no rows — defensive
            continue
        payloads[entity] = {
            "entity": entity,
            "metadata": dict(metadata or {}),
            "d_jspace": d_jspace,
            "primary_site": next(iter(sites_payload)),
            "variants": list(_VARIANTS),
            "sites": sites_payload,
        }
    return payloads


def _torch():
    try:
        import torch
    except (ModuleNotFoundError, OSError) as exc:  # pragma: no cover - env-dependent
        # ModuleNotFoundError: extra not installed. OSError: a broken build
        # (e.g. an x86_64 torch wheel on an arm64 machine) — same remedy.
        raise ModuleNotFoundError(
            "writing .pt files needs a working torch; install the model extra "
            "for this platform: pip install -e '.[model]'"
        ) from exc
    return torch


def _to_tensor(array, torch):
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def export_pt(
    directions_by_site: dict[InjectionSite, FittedDirections],
    out_dir: str | Path,
    metadata: Mapping[str, object] | None = None,
) -> list[Path]:
    """Write one ``{entity}_role_direction.pt`` per entity under ``out_dir``.

    Returns the written paths (sorted by entity). Torch is imported lazily; a
    missing/broken torch raises ModuleNotFoundError with an install hint rather
    than at module import, so callers without the model extra can still import
    and use :func:`build_pt_payloads`.
    """
    torch = _torch()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = build_pt_payloads(directions_by_site, metadata=metadata)

    written: list[Path] = []
    for entity, payload in sorted(payloads.items()):
        tensor_sites = {
            site_name: {
                "fitted": _to_tensor(site_data["fitted"], torch),
                "shuffled": _to_tensor(site_data["shuffled"], torch),
                "generic_loo": (
                    None
                    if site_data["generic_loo"] is None
                    else _to_tensor(site_data["generic_loo"], torch)
                ),
                "raw_norm": site_data["raw_norm"],
                "stability": site_data["stability"],
            }
            for site_name, site_data in payload["sites"].items()
        }
        out = {**payload, "sites": tensor_sites}
        path = out_dir / f"{entity}_role_direction.pt"
        torch.save(out, path)
        written.append(path)
    return written


def export_pt_from_npz(
    directory: str | Path,
    out_dir: str | Path | None = None,
    sites: tuple[InjectionSite, ...] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> list[Path]:
    """Load fitted ``.npz`` directions from ``directory`` and export ``.pt``.

    Convenience for regenerating the Group A handoff from an existing fit
    without re-running the model. ``out_dir`` defaults to ``directory``.
    """
    directions_by_site = load_fitted_by_site(directory, sites)
    return export_pt(
        directions_by_site,
        out_dir if out_dir is not None else directory,
        metadata=metadata,
    )
