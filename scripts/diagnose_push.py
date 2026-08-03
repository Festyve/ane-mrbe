"""Disambiguate a sign-inverted calibration: causal inversion vs generic disruption.

Motivated by Gemma-3-12B at entity_token: the toward-agent push produced a clean
dose-response curve (|shift| 0.88 -> 1.87, saturating) but with NEGATIVE sign —
the opposite of Qwen3.6-27B. Two readings, indistinguishable from the
calibration sweep alone, because it only ever pushes the fitted direction:

- causal inversion: the decode-side direction is causally coupled but acts with
  opposite sign in this model. Toward-PATIENT should then shift the probe
  POSITIVE (mirror), while a strength-matched random direction does ~nothing.
- generic disruption: at these coefficients ANY direction drags the entity's
  probe log-odds down; the "effect" is out-of-distribution damage, not role
  content. Random direction then shifts negative just like the real push.

Reads the fitting corpus only — the primary stimuli stay untouched.
"""

from __future__ import annotations

import argparse
import json
import sys

from jspace_binding.config import Config
from jspace_binding.experiments.calibrate import logit
from jspace_binding.model.factory import add_backend_args, build_model, preflight_or_exit
from jspace_binding.stimuli.fitting_corpus import generate_fitting_corpus
from jspace_binding.types import EditSpec, EditType, InjectionSite, PushSign, Role


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_args(parser)
    parser.add_argument("--site", default=InjectionSite.ENTITY_TOKEN.value,
                        choices=[s.value for s in InjectionSite])
    parser.add_argument("--coefficient", type=float, required=True,
                        help="strength for all conditions (e.g. the |shift| dose-response peak)")
    parser.add_argument("--n", type=int, default=20, help="patient-role sentences to average over")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    model = build_model(config, dry_run=args.dry_run, dummy_mode=args.dummy_mode)
    site = InjectionSite(args.site)
    preflight_or_exit(model, (site,))

    corpus = generate_fitting_corpus(
        config.direction_entities(),
        config.directions.exemplars_per_role,
        counterparts=config.counterpart_entities(),
    )
    patients = [ex for ex in corpus if ex.role is Role.PATIENT][: args.n]

    no_edit = EditSpec(edit_type=EditType.NO_EDIT)
    conditions = {
        "role_push_toward_agent": lambda ex: EditSpec(
            edit_type=EditType.ROLE_PUSH, entity=ex.entity,
            sign=PushSign.TOWARD_AGENT, coefficient=args.coefficient),
        "role_push_toward_patient": lambda ex: EditSpec(
            edit_type=EditType.ROLE_PUSH, entity=ex.entity,
            sign=PushSign.TOWARD_PATIENT, coefficient=args.coefficient),
        "random_direction": lambda ex: EditSpec(
            edit_type=EditType.RANDOM_DIRECTION,
            sign=PushSign.TOWARD_AGENT, coefficient=args.coefficient, seed=0),
    }

    shifts: dict[str, float] = {}
    for name, make_spec in conditions.items():
        total = 0.0
        for ex in patients:
            base = model.answer_distribution(ex.sentence, ex.role_probe, no_edit,
                                             site, (ex.entity, ex.other))
            edited = model.answer_distribution(ex.sentence, ex.role_probe, make_spec(ex),
                                               site, (ex.entity, ex.other))
            total += logit(edited[ex.entity]) - logit(base[ex.entity])
        shifts[name] = total / len(patients)
        print(f"{name:>26}: mean shift {shifts[name]:+.3f}", file=sys.stderr)

    agent, patient, rand = (shifts["role_push_toward_agent"],
                            shifts["role_push_toward_patient"],
                            shifts["random_direction"])
    if abs(rand) >= 0.5 * max(abs(agent), abs(patient), 1e-9):
        verdict = ("GENERIC DISRUPTION: the strength-matched random direction moves the "
                   "probe comparably to the fitted push. The negative calibration shifts "
                   "are OOD damage, not role content — treat as an intervention-strength/"
                   "specificity failure at this site.")
    elif agent < 0 < patient:
        verdict = ("CAUSAL INVERSION: the two push signs mirror around zero and the random "
                   "control is quiet. The fitted direction is causally coupled with "
                   "opposite sign to Qwen — a reportable cross-model finding. The primary "
                   "sweep is sign-symmetric, so it can proceed; the binding score handles "
                   "polarity.")
    elif agent > 0:
        verdict = "EXPECTED POLARITY at this strength — re-run calibration around this coefficient."
    else:
        verdict = ("MIXED: same-signed pushes with a quiet random control fit neither story "
                   "cleanly. Bring the numbers back before proceeding.")
    print(json.dumps({"site": site.value, "coefficient": args.coefficient,
                      "n": len(patients), "shifts": shifts, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
