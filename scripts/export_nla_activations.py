#!/usr/bin/env python3
"""Export residual activations as the Parquet the NLA verbaliser consumes.

Bridges our pipeline to kitft/natural_language_autoencoders. Their
`nla_inference.py` takes a Parquet with an `activation_vector` column of
d_model-wide float lists; everything else here is provenance so a row can be
traced back to the stimulus that produced it.

    python scripts/export_nla_activations.py --config configs/gemma3_27b_it_nla.yaml \
        --out data/gemma3_27b_nla/activations.parquet

THREE THINGS THAT WILL SILENTLY RUIN THE RUN if not respected:

1. LAYER. The released Gemma-3-27B NLA reads block 41 of 62; our workspace band
   is 46. Those are different representations -- our own layer sweep showed the
   jspace-minus-random difference flipping sign between L24 and L57 -- so the
   config passed here MUST set layer_band to [41, 41]. This script refuses
   otherwise rather than exporting activations the verbaliser was not trained
   on.

2. SCALE. We export RAW residual vectors. The injection rescale
   (v * injection_scale / ||v||) belongs to their inference script and is read
   from the checkpoint's nla_meta.yaml -- injection_scale is 60000.0 for
   Gemma-3-27B, and `mse_scale` 73.321 is sqrt(5376), the same number as
   Gemma's embedding scale but used for AR scoring, not injection. Do not
   pre-scale here: doing it twice produces the documented failure mode, which
   is output in Chinese on every row.

3. PAIRING. Rows are emitted so that the agent and patient cells of the same
   family are both present and joinable on family_id. That is what makes the
   minimal-pair comparison possible: if the NLA is blind to role, the two cells
   verbalise to reconstructions that do not differ. Exporting only one role
   would make the whole comparison unanswerable.

CONTROLS ARE STRUCTURAL, not extra rows. An earlier draft tried to export a
non-participant entity's activation as the control, which is incoherent here:
probe_activation anchors at the target entity's own token, so an entity absent
from the sentence has no token to read. The contrasts live in the exported data
instead:

  test      WITHIN family: agent cell vs patient cell. Same words, same
            entities, only the role differs. If the NLA is blind to role their
            reconstructions do not separate.
  positive  ACROSS families: different sentences entirely. If these do not
            separate either, the verbaliser is not discriminating anything and
            the within-pair null is uninterpretable -- the NLA equivalent of
            our intervention-strength check.
  null      SHUFFLED pairing: compare cells drawn from different families as if
            they were a pair. Gives the band a real within-pair difference must
            clear.

All three are computed from this one file at analysis time, which is why the
export must carry both role cells of every family (see PAIRING above).
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
    preflight_or_exit(model, (site,))

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
