#!/usr/bin/env python3
"""Export residual activations as the Parquet the NLA verbaliser consumes.

Bridges this pipeline to kitft/natural_language_autoencoders, whose
`nla_inference.py` takes a Parquet with an `activation_vector` column of
d_model-wide float lists. Every other column is provenance.

    python scripts/export_nla_activations.py --config configs/gemma3_27b_it_nla.yaml \
        --out data/gemma3_27b_nla/activations.parquet

Three things that will silently ruin the run:

1. LAYER. The released Gemma-3-27B NLA reads block 41 of 62 where our workspace
   band is 46, and those are different representations. The config MUST set
   layer_band to [41, 41]; this script refuses otherwise.
2. SCALE. We export RAW residual vectors. The injection rescale belongs to their
   inference script and is read from the checkpoint's nla_meta.yaml. Applying it
   twice produces the documented failure mode: output in Chinese on every row.
3. PAIRING. Both cells of a family are emitted and joinable on family_id, which
   is what makes the minimal-pair comparison possible.

Controls are structural rather than extra rows — probe_activation anchors at the
target entity's own token, so an entity absent from the sentence has no token to
read. All three are computed from this one file at analysis time:

  test      WITHIN family: agent cell vs patient cell. Same words, same
            entities, only the role differs.
  positive  ACROSS families: different sentences entirely. If these do not
            separate, the within-pair null is uninterpretable.
  null      SHUFFLED pairing: cells from different families treated as a pair,
            giving the band a real within-pair difference must clear.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jspace_binding.config import Config  # noqa: E402
from jspace_binding.model.factory import (  # noqa: E402
    add_backend_args,
    build_model,
    preflight_or_exit,
)
from jspace_binding.stimuli.generate import generate_families  # noqa: E402
from jspace_binding.types import InjectionSite, Position, Role  # noqa: E402

_NLA_LAYER_BY_MODEL = {
    "google/gemma-3-27b-it": 41,
    "google/gemma-3-12b-it": 32,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_args(parser)
    parser.add_argument("--out", type=Path, required=True, help="output .parquet path")
    parser.add_argument(
        "--site",
        default=InjectionSite.ENTITY_TOKEN.value,
        choices=[s.value for s in InjectionSite],
    )
    parser.add_argument("--limit", type=int, default=None, help="families to export")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        sys.exit("needs pandas + pyarrow: pip install pandas pyarrow")

    config = Config.from_yaml(args.config)
    site = InjectionSite(args.site)

    expected = _NLA_LAYER_BY_MODEL.get(config.model.model_id)
    if expected is None:
        print(
            f"WARNING no released NLA is known for {config.model.model_id}; "
            "check the layer by hand against the checkpoint's nla_meta.yaml",
            file=sys.stderr,
        )
    else:
        band = config.model.layer_band
        if band is None or band[0] != expected or band[1] != expected:
            sys.exit(
                f"layer_band is {band}, but the released NLA for "
                f"{config.model.model_id} reads block {expected}. Exporting from a "
                f"different layer gives the verbaliser vectors it was not trained "
                f"on. Set layer_band: [{expected}, {expected}] in the config."
            )

    model = build_model(config, dry_run=args.dry_run, dummy_mode=args.dummy_mode)
    # No sites: passing them makes preflight demand fitted directions, and this
    # script fits nothing and pushes nothing -- probe_activation only reads. The
    # NLA layer (41) is not where directions are fitted (46), so requiring them
    # would ask for a fit nobody wants at a layer nobody intervenes at.
    preflight_or_exit(model)

    families = generate_families(config)
    if args.limit:
        families = families[: args.limit]

    rows: list[dict[str, object]] = []
    for family in families:
        entity = family.concept_pair.entity
        for role in Role:
            for position in Position:
                sentence = family.cell(role, position).sentence
                acts = model.probe_activation(sentence, entity, site)
                rows.append(
                    {
                        # The column their script reads. Everything else is ours.
                        "activation_vector": [float(x) for x in acts["residual"]],
                        "family_id": family.family_id,
                        "pair_id": family.concept_pair.pair_id,
                        "construction": family.construction.value,
                        "entity": entity,
                        "other_entity": family.other_entity,
                        "role": role.value,
                        "position": position.value,
                        "site": site.value,
                        "sentence": sentence,
                        "condition": "real",
                    }
                )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    dim = len(rows[0]["activation_vector"]) if rows else 0
    print(f"wrote {args.out}", file=sys.stderr)
    print(f"  rows        {len(df)}", file=sys.stderr)
    print(f"  d_model     {dim}", file=sys.stderr)
    print(f"  layer       {config.model.layer_band}", file=sys.stderr)
    print(f"  site        {site.value}", file=sys.stderr)
    print(f"  conditions  {df['condition'].value_counts().to_dict()}", file=sys.stderr)
    print(
        "\nNext: launch SGLang on the AV checkpoint, then run their nla_inference.py\n"
        "against this file. Do NOT pre-scale the vectors -- their script applies\n"
        "injection_scale from nla_meta.yaml.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
