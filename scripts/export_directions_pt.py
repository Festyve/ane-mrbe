#!/usr/bin/env python3
"""Export fitted role-directions as torch .pt files for Group A.

Reads the directions_{site}.npz files that scripts/fit_directions.py already
wrote (config.paths.directions) and converts them to one
{entity}_role_direction.pt per entity, bundling every injection site. No model
and no re-fitting — pure conversion — so it runs anywhere a working torch is
installed.

Usage:
    export_directions_pt.py --config configs/default.yaml [--in DIR] [--out DIR]
                            [--fit-corpus PATH] [--entity doctor,nurse]

--in defaults to config.paths.directions; --out defaults to --in. --entity
restricts the export to a subset (e.g. just the concept pair handed off first);
by default every fitted entity is exported. --fit-corpus identifies the corpus
used to fit the directions for the exported provenance metadata. Progress goes
to stderr; a JSON summary of what was written goes to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from jspace_binding.config import Config
from jspace_binding.directions.export_pt import build_pt_payloads, export_pt, load_fitted_by_site


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _export_metadata(config: Config, corpus_path: Path, directions_dir: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_id": config.model.model_id,
        "lens_repo": config.model.lens_repo,
        "layer_band": None
        if config.model.layer_band is None
        else list(config.model.layer_band),
        "injection_sites": [site.value for site in config.experiment.injection_sites],
        "direction_variant": config.directions.variant,
        "directions_dir": str(directions_dir),
        "fitting_corpus": str(corpus_path),
        "fitting_corpus_sha256": _sha256(corpus_path),
        "exemplars_per_role": config.directions.exemplars_per_role,
        "bootstrap_resamples": config.directions.n_bootstrap,
        "direction_seed": config.directions.seed,
        "push_coefficient": config.model.push_coefficient,
        "git_commit": _git_commit(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fitted role-directions as .pt files.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--in",
        dest="in_dir",
        type=Path,
        default=None,
        help="dir holding directions_{site}.npz (default: config paths.directions)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output dir for .pt files (default: same as --in)",
    )
    parser.add_argument(
        "--fit-corpus",
        type=Path,
        default=None,
        help="corpus used to fit directions (default: config.paths.fitting_corpus)",
    )
    parser.add_argument(
        "--entity",
        type=lambda s: tuple(e.strip() for e in s.split(",") if e.strip()),
        default=None,
        help="comma-separated entities to export (default: all fitted entities)",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    in_dir = args.in_dir or config.paths.directions
    out_dir = args.out or in_dir
    fit_corpus = args.fit_corpus or config.paths.fitting_corpus

    directions_by_site = load_fitted_by_site(in_dir)
    sites = sorted(s.value for s in directions_by_site)
    print(
        f"loaded directions for {len(directions_by_site)} site(s) [{', '.join(sites)}] "
        f"from {in_dir}",
        file=sys.stderr,
    )

    if args.entity:
        available = set(build_pt_payloads(directions_by_site))
        missing = [e for e in args.entity if e not in available]
        if missing:
            sys.exit(
                f"requested entities not fitted: {', '.join(missing)}; "
                f"available: {', '.join(sorted(available))}"
            )
        # Drop unwanted entities from each site's fit before writing.
        directions_by_site = _restrict(directions_by_site, set(args.entity))

    metadata = _export_metadata(config, fit_corpus, in_dir)
    written = export_pt(directions_by_site, out_dir, metadata=metadata)
    for path in written:
        print(f"wrote {path}", file=sys.stderr)

    summary = {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "sites": sites,
        "written": [str(p) for p in written],
    }
    print(json.dumps(summary, indent=2))


def _restrict(directions_by_site, keep):
    """Return a copy of the per-site fits keeping only entities in ``keep``.

    Rebuilds each FittedDirections with the selected rows so downstream
    export sees a consistent (entities, matrices) shape.
    """
    import numpy as np

    from jspace_binding.directions.fit import FittedDirections

    restricted = {}
    for site, fitted in directions_by_site.items():
        idx = [i for i, e in enumerate(fitted.entities) if e in keep]
        restricted[site] = FittedDirections(
            site=fitted.site,
            entities=tuple(fitted.entities[i] for i in idx),
            fitted=fitted.fitted[idx],
            shuffled=fitted.shuffled[idx],
            generic_loo=fitted.generic_loo[idx],
            raw_norms=np.asarray([fitted.raw_norms[i] for i in idx]),
            stability=np.asarray([fitted.stability[i] for i in idx]),
        )
    return restricted


if __name__ == "__main__":
    main()
