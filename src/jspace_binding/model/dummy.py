"""GPU-free synthetic backend with planted ground truth.

Lets the whole pipeline be validated against known answers before the real
backend is loaded. Three modes:

- ``binding``: the workspace binds fillers to roles continuously, so a uniform
  ROLE_PUSH interacts with the role the sentence assigned (the crossover) and
  the pipeline must recover a large positive binding score.
- ``bag``: the workspace is an unordered bag of concepts, so the same push
  shifts the entity's log-odds by a constant regardless of role and the score
  must land at ~0 inside the null band.
- ``recruitment``: binds ON DEMAND. Identical to ``binding`` except at E4's
  question-conditional read, and deliberately so — RQ1/RQ2/primary never vary
  the task, so telling always-on from on-demand is exactly what E4 is for.

All behaviour is defined in LOGIT space and mapped through a sigmoid, so
"uniform log-odds shift" is exact. Stdlib only. answer_distribution ignores
edit.coefficient/alpha, so calibration against the dummy returns the smallest
grid value.

fitting_activation is NOT consumed by answer_distribution — it validates the
directions.fit plumbing, not the vector-to-behaviour link, which only the real
backend can test.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections.abc import Sequence

from jspace_binding.stimuli.vocab import PROFESSION_CUE, PROFESSION_ENTITIES
from jspace_binding.types import (
    DIRECTION_PUSH_EDIT_TYPES,
    EditSpec,
    EditType,
    InjectionSite,
    PushSign,
)


# Local twin of analysis.binding_score.logit: this module is stdlib-only by
# charter (must run on any machine), so it cannot import the numpy-backed
# analysis package.
def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


_L_AGENT = _logit(0.75)  # entity's answer logit when it IS the agent (ROLE probe)
_L_PATIENT = _logit(0.05)  # ... when it is the patient
_L_ABSENT = _logit(0.01)  # a token not mentioned in the sentence
_L_MENTIONED = _logit(0.45)  # NEUTRAL probe: a mentioned profession
_L_SWAPPED_IN = _logit(0.55)  # NEUTRAL probe: the identity-swap counterpart

_BIG = 2.5  # binding mode: push interacting with the sentence's role assignment
_SMALL = 0.3  # binding mode: push against an already-assigned role (near-saturated)
_BAG_SHIFT = 0.8  # bag mode: uniform role-blind log-odds increment
_POSITION_BIAS = 0.25  # added to the entity's logit when it is surface-FIRST
_SIGMA = 0.05  # logit-space jitter std dev

_FIT_DIM = 32  # synthetic J-space dimensionality for fitting_activation
_FIT_NOISE = 0.3
# Synthetic orthogonal-remainder dimensionality (RQ1). Much larger than
# _FIT_DIM so the residual is mostly NOT jspace, as on the real model. Small
# values let a random jspace-sized subspace capture the planted signal wherever
# it sat, making RQ1's capacity control unable to fail.
_ORTH_DIM = 240

_L_DEGRADED = _logit(0.30)  # post-ablation level: role info flat / recall mildly hit
_L_MILD_AGENT = _logit(0.65)  # random-subspace ablation: mild, order-preserving
_L_MILD_PATIENT = _logit(0.08)
_L_MILD_MENTIONED = _logit(0.40)

# CONCEPT probe (the RQ2 recall control). The baseline gap is deliberately
# modest, calibrated to the +1.42 log-odds measured on Qwen2.5-1.5B, so the
# dummy's control carries the same headroom to fall as the real one.
_L_CONCEPT_MATCH = _logit(0.64)  # participant whose profession fits the cue
_L_CONCEPT_MISMATCH = _logit(0.31)  # the other participant  -> gap ~1.38
# ABLATE_JSPACE, binding mode: concepts live outside the workspace and survive
# the edit, so the semantic contrast narrows but keeps its sign.
_L_CONCEPT_MATCH_DEGRADED = _logit(0.57)
_L_CONCEPT_MISMATCH_DEGRADED = _logit(0.42)  # -> gap ~0.60, ~56% of baseline lost
# ABLATE_RANDOM_SUBSPACE: mild in both modes, the capacity-matched comparison.
_L_CONCEPT_MATCH_MILD = _logit(0.62)
_L_CONCEPT_MISMATCH_MILD = _logit(0.35)  # -> gap ~1.11, ~20% of baseline lost

_PUSH_EDITS = DIRECTION_PUSH_EDIT_TYPES

# "recruitment" is H3: binding built on demand. It behaves exactly like
# "binding" outside recruitment_activation, so every `self.mode == "bag"` test
# below correctly routes it down the binding branch.
MODES: tuple[str, ...] = ("binding", "bag", "recruitment")


class DummyModel:
    """Synthetic WorkspaceModel + FittingActivationSource.

    Probabilities are full-softmax reads and need not sum to 1. The tests pin
    every number below.

    ROLE probe (asks for the agent): agent logit(0.75), patient logit(0.05),
    absent counterpart logit(0.01). Adjustments to the ENTITY's logit, with the
    other participant mirrored oppositely:
    - ROLE_PUSH, binding: TOWARD_AGENT adds +2.5 when the entity was patient
      and +0.3 when already agent; TOWARD_PATIENT is the mirror. The gap
      shrinks from opposite sides under the two signs — the crossover.
    - ROLE_PUSH, bag: sign * 0.8 regardless of role. Uniform in log-odds.
    - The three push controls: no adjustment, in both modes.
    - IDENTITY_SWAP: the counterpart inherits the entity's logit and the entity
      drops to absent (defensive; the sweep pairs it with NEUTRAL only).

    NEUTRAL probe: IDENTITY_SWAP gives entity 0.05, counterpart 0.55, other
    0.45 — the edit propagates role-independently; every other edit gives both
    mentioned participants 0.45 and the counterpart 0.01.

    CONCEPT probe (RQ2's recall control): cue-matching participant logit(0.64)
    against logit(0.31), a ~1.38 gap with somewhere to fall. Nothing here reads
    the entity's role, so it is role-blind by construction. ABLATE_JSPACE
    narrows it to ~0.60 in binding mode (concepts live outside the workspace)
    and flattens it in bag mode (concepts WERE the workspace), so bag yields no
    binding-specific deficit. ABLATE_RANDOM_SUBSPACE is mild in both (~1.11).

    Finally: +0.25 when the entity is surface-first (a position bias the
    analysis must cancel), plus N(0, 0.05) jitter seeded from a content hash
    rather than call order, so reordered runs produce identical trial files.
    """

    def __init__(self, mode: str = "binding", seed: int = 0) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown dummy mode {mode!r}; expected one of {MODES}")
        self.mode = mode
        self.seed = seed

    # ------------------------------------------------------------------ #
    # WorkspaceModel                                                     #
    # ------------------------------------------------------------------ #

    def answer_distribution(
        self,
        sentence: str,
        probe: str,
        edit: EditSpec,
        site: InjectionSite,
        answer_tokens: Sequence[str],
    ) -> dict[str, float]:
        """Synthetic probabilities per the class table.

        `answer_tokens` must follow AnswerSet.tokens order: (entity,
        counterpart, other) — with no design-cell metadata in the call,
        ordering is the only channel telling the dummy which token plays
        which part. A 2-token (entity, other) form is accepted for
        calibration reads, which have no identity-swap counterpart.
        """
        if len(answer_tokens) == 3:
            entity, counterpart, other = answer_tokens
        elif len(answer_tokens) == 2:
            entity, other = answer_tokens
            counterpart = None
        else:
            raise ValueError(
                "DummyModel expects (entity, counterpart, other) or (entity, other), "
                f"got {len(answer_tokens)} tokens"
            )
        entity_first, entity_is_agent = self._locate_entity(sentence, entity, other)
        concept_target = self._probe_concept_target(probe)
        if concept_target is not None:
            logits = self._concept_logits(concept_target, entity, other, edit)
        else:
            logits = self._logits(self._probe_is_role(probe), edit, entity_is_agent)
        l_entity, l_counterpart, l_other = logits
        if concept_target is None and self._probe_is_recipient(probe):
            # The RECIPIENT probe's answer is the NON-agent, so the entity's
            # and the other participant's readouts swap. The push interaction
            # lives in the logit values, so this mirrors the whole structure
            # and PROBE_ORIENTATION undoes the inverted polarity.
            l_entity, l_other = l_other, l_entity
        if entity_first:
            l_entity += _POSITION_BIAS
        jitter = self._jitter(sentence, probe, edit, site)
        pairs = zip(
            (entity, counterpart, other), (l_entity, l_counterpart, l_other), jitter, strict=True
        )
        return {token: _sigmoid(logit + j) for token, logit, j in pairs if token is not None}

    def _concept_logits(
        self, concept_target: str, entity: str, other: str, edit: EditSpec
    ) -> tuple[float, float, float]:
        """CONCEPT-probe ground truth: (entity, counterpart, other) logits.

        Role-blind by construction — nothing here reads entity_is_agent — which
        is what makes it a control rather than a second binding measure.

        Ablation behaviour mirrors _ablation_logits' concept story:
        - binding mode: the workspace holds ROLES, concepts are recoverable
          elsewhere, so ABLATE_JSPACE narrows the semantic gap but keeps its
          sign — the control registers damage without collapsing.
        - bag mode: the workspace holds the CONCEPTS, so ablating it flattens
          the semantic contrast to a tie. Recall then degrades at least as hard
          as binding and no binding-SPECIFIC deficit survives, which is the
          ground truth test_rq2_bag_mode asserts.
        """
        if concept_target == entity:
            match_is_entity = True
        elif concept_target == other:
            match_is_entity = False
        else:
            raise ValueError(
                f"CONCEPT probe asks for {concept_target!r}, which is neither "
                f"participant ({entity!r}, {other!r}); the probe and the family "
                "are mismatched"
            )

        if edit.edit_type is EditType.ABLATE_JSPACE:
            if self.mode == "bag":
                # Concepts lived in the workspace: the contrast is gone.
                hi = lo = _L_CONCEPT_MISMATCH_DEGRADED
            else:
                hi, lo = _L_CONCEPT_MATCH_DEGRADED, _L_CONCEPT_MISMATCH_DEGRADED
        elif edit.edit_type is EditType.ABLATE_RANDOM_SUBSPACE:
            hi, lo = _L_CONCEPT_MATCH_MILD, _L_CONCEPT_MISMATCH_MILD
        else:
            hi, lo = _L_CONCEPT_MATCH, _L_CONCEPT_MISMATCH

        l_entity, l_other = (hi, lo) if match_is_entity else (lo, hi)
        return (l_entity, _L_ABSENT, l_other)

    def _logits(
        self, probe_is_role: bool, edit: EditSpec, entity_is_agent: bool
    ) -> tuple[float, float, float]:
        """(entity, counterpart, other) logits before position bias and jitter."""
        if edit.edit_type in (EditType.ABLATE_JSPACE, EditType.ABLATE_RANDOM_SUBSPACE):
            return self._ablation_logits(probe_is_role, edit.edit_type, entity_is_agent)
        if not probe_is_role:
            if edit.edit_type is EditType.IDENTITY_SWAP:
                return (_L_PATIENT, _L_SWAPPED_IN, _L_MENTIONED)
            return (_L_MENTIONED, _L_ABSENT, _L_MENTIONED)

        l_entity = _L_AGENT if entity_is_agent else _L_PATIENT
        l_other = _L_PATIENT if entity_is_agent else _L_AGENT
        if edit.edit_type is EditType.IDENTITY_SWAP:
            return (_L_ABSENT, l_entity, l_other)
        if edit.edit_type is EditType.ROLE_PUSH:
            if edit.sign is None:
                raise ValueError("ROLE_PUSH EditSpec must carry a PushSign")
            sign = 1.0 if edit.sign is PushSign.TOWARD_AGENT else -1.0
            if self.mode == "bag":
                return (l_entity + sign * _BAG_SHIFT, _L_ABSENT, l_other)
            # binding: the push interacts with the role already assigned.
            pushing_against_assignment = (edit.sign is PushSign.TOWARD_AGENT) != entity_is_agent
            delta = _BIG if pushing_against_assignment else _SMALL
            return (l_entity + sign * delta, _L_ABSENT, l_other - sign * delta)
        if edit.edit_type in _PUSH_EDITS or edit.edit_type is EditType.NO_EDIT:
            # Controls and baseline: sentence effectively untouched.
            return (l_entity, _L_ABSENT, l_other)
        raise ValueError(f"DummyModel: unhandled edit type {edit.edit_type!r}")

    def _ablation_logits(
        self, probe_is_role: bool, edit_type: EditType, entity_is_agent: bool
    ) -> tuple[float, float, float]:
        """RQ2 ground truth.

        binding mode — the workspace is where binding lives: ABLATE_JSPACE
        flattens the ROLE readout to a tie while recall only degrades mildly.
        bag mode — the workspace holds only concepts: ABLATE_JSPACE breaks
        recall outright and thereby the role answer too, so binding degrades no
        more than recall and no binding-specific deficit survives.
        ABLATE_RANDOM_SUBSPACE is mild and order-preserving throughout.
        """
        if edit_type is EditType.ABLATE_RANDOM_SUBSPACE:
            if probe_is_role:
                l_entity = _L_MILD_AGENT if entity_is_agent else _L_MILD_PATIENT
                l_other = _L_MILD_PATIENT if entity_is_agent else _L_MILD_AGENT
                return (l_entity, _L_ABSENT, l_other)
            return (_L_MILD_MENTIONED, _L_ABSENT, _L_MILD_MENTIONED)
        # ABLATE_JSPACE:
        if probe_is_role:
            return (_L_DEGRADED, _L_ABSENT, _L_DEGRADED)  # role tie in both modes
        if self.mode == "bag":
            return (_L_ABSENT, _L_ABSENT, _L_ABSENT)  # recall collapses to a tie
        return (_L_DEGRADED, _L_ABSENT, _L_DEGRADED)  # mild but still above absent

    # ------------------------------------------------------------------ #
    # FittingActivationSource                                            #
    # ------------------------------------------------------------------ #

    def fitting_activation(
        self, sentence: str, entity: str, site: InjectionSite
    ) -> list[float]:
        """Synthetic J-space activation for a fitting-corpus sentence.

        binding mode: +/- the entity's planted unit direction (sign = the
        entity's role in the sentence, inferred lexically) plus Gaussian
        noise, so directions.fit must recover the planted direction with high
        bootstrap stability. bag mode: pure noise — no role information —
        so the stability pilot check must come out low.
        """
        role_sign = 1.0 if self._infer_role_is_agent(sentence, entity) else -1.0
        planted = self._planted_direction(entity, site)
        rng = self._content_rng("fit", sentence, entity, site.value)
        noise = [rng.gauss(0.0, _FIT_NOISE) for _ in range(_FIT_DIM)]
        if self.mode == "bag":
            return noise
        return [role_sign * p + n for p, n in zip(planted, noise, strict=True)]

    def fitting_gradient(
        self, sentence: str, role_probe: str, entity: str, other: str, site: InjectionSite
    ) -> list[float]:
        """Synthetic readout gradient for the LRE-style estimator.

        binding mode: the entity's planted direction plus noise, with NO role
        sign — a gradient of the agent readout points toward agent on every
        exemplar, so the pooled mean must recover the planted direction with
        high stability. bag mode: pure noise, so stability must come out low.
        """
        planted = self._planted_direction(entity, site)
        rng = self._content_rng("fitgrad", sentence, entity, other, site.value)
        noise = [rng.gauss(0.0, _FIT_NOISE) for _ in range(_FIT_DIM)]
        if self.mode == "bag":
            return noise
        return [p + n for p, n in zip(planted, noise, strict=True)]

    def _planted_direction(self, entity: str, site: InjectionSite) -> list[float]:
        """Deterministic pseudo-random unit vector per (entity, site)."""
        return self._unit_vector("planted", entity, site.value, dim=_FIT_DIM)

    def _unit_vector(self, *key: str, dim: int) -> list[float]:
        rng = self._content_rng(*key)
        raw = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]

    # ------------------------------------------------------------------ #
    # ProbeActivationSource (RQ1)                                        #
    # ------------------------------------------------------------------ #

    def probe_activation(
        self, sentence: str, entity: str, site: InjectionSite,
        capacity_seed: int | None = None,
    ) -> dict[str, list[float]]:
        """Synthetic RQ1 activation sources with mode-dependent ground truth.

        binding mode plants the role signal in the J-space component, so a
        probe must decode from "jspace" and "residual" but sit at chance on
        "orthogonal"; bag mode plants it in the orthogonal remainder instead
        (the model still binds, just elsewhere) and inverts that. The synthetic
        residual basis is axis-aligned: residual = jspace ++ orthogonal.

        The planted direction is FILLER-GENERAL, unlike fitting_activation's
        per-entity ones, because RQ1 splits leave-one-pair-out and only
        cross-lexical role information should be decodable.
        """
        role_sign = 1.0 if self._infer_role_is_agent(sentence, entity) else -1.0
        # "recruitment" plants role in jspace like "binding"; they diverge
        # only once a question is in context, which is what E4 reads.
        location = "orthogonal" if self.mode == "bag" else "jspace"
        return self._probe_sources(
            role_sign,
            location,
            self._content_rng("probe", sentence, entity, site.value),
            key=site.value,
        )

    def _probe_sources(
        self, role_sign: float, location: str | None, rng: random.Random, key: str
    ) -> dict[str, list[float]]:
        """The four probe sources with the role signal planted at `location`.

        `location` is "jspace", "orthogonal", or None. None means the signal is
        ABSENT, which is distinct from "somewhere else" and load-bearing: under
        recruitment the model is not binding at all while answering a bag
        question, so role must not be decodable from the remainder either.

        Shared by RQ1 and E4 so the two cannot drift.
        """
        if location not in ("jspace", "orthogonal", None):
            raise ValueError(f"unknown signal location {location!r}")
        j_noise = [rng.gauss(0.0, _FIT_NOISE) for _ in range(_FIT_DIM)]
        o_noise = [rng.gauss(0.0, _FIT_NOISE) for _ in range(_ORTH_DIM)]
        jspace, orthogonal = j_noise, o_noise
        if location == "jspace":
            planted = self._unit_vector("probe-role", key, dim=_FIT_DIM)
            jspace = [role_sign * p + n for p, n in zip(planted, j_noise, strict=True)]
        elif location == "orthogonal":
            planted = self._unit_vector("probe-role-orth", key, dim=_ORTH_DIM)
            orthogonal = [role_sign * p + n for p, n in zip(planted, o_noise, strict=True)]
        residual = jspace + orthogonal
        return {
            "jspace": jspace,
            "orthogonal": orthogonal,
            "residual": residual,
            "random_subspace": self._random_subspace_projection(residual, key),
        }

    def recruitment_activation(
        self, sentence: str, probe: str, entity: str
    ) -> dict[str, list[float]]:
        """E4 ground truth. Read with a question in context, so the three modes
        finally separate:

        - binding: role decodable from jspace under BOTH questions.
        - recruitment: decodable under the ROLE question only, at chance under
          the bag question.
        - bag: never in jspace; the signal sits in the orthogonal remainder.
        """
        role_sign = 1.0 if self._infer_role_is_agent(sentence, entity) else -1.0
        asking_role = self._probe_is_role(probe)
        if self.mode == "bag":
            location = "orthogonal"  # binding happens, just never in the workspace
        elif self.mode == "recruitment":
            # Bound only when the task asks: under the bag question the model
            # is not binding at all, so the signal is absent, not relocated.
            location = "jspace" if asking_role else None
        else:  # binding: always on, question-independent
            location = "jspace"
        return self._probe_sources(
            role_sign,
            location,
            self._content_rng("recruit", sentence, probe, entity),
            # Same basis key as RQ1's final-token read, so a difference can
            # only come from the planted signal, not a different draw.
            key=InjectionSite.FINAL_TOKEN.value,
        )

    def _random_subspace_projection(self, residual: list[float], key: str) -> list[float]:
        """RQ1 capacity control: `residual` projected onto a random subspace of
        the same rank as jspace (analysis.probes.PROBE_SOURCES).

        Fixed per (seed, key), not per sentence: it is a readout BASIS, the
        same for every example, exactly as jspace is. Drawing it per sentence
        would make it noise and pin it at chance — a control that looks like it
        passes while testing nothing.

        The encoded ground truth is that it overlaps the signal-carrying
        subspace partially, so it decodes above chance but below that subspace.
        """
        dim = len(residual)
        basis = [
            self._unit_vector("probe-random-subspace", key, str(k), dim=dim)
            for k in range(_FIT_DIM)
        ]
        # Project onto the span, expressed back in the residual basis so the
        # probe sees the same feature count as every other source.
        out = [0.0] * dim
        for axis in basis:
            coefficient = sum(a * r for a, r in zip(axis, residual, strict=True))
            for i, a in enumerate(axis):
                out[i] += coefficient * a
        return out

    # ------------------------------------------------------------------ #
    # Lexical inference helpers                                          #
    # ------------------------------------------------------------------ #

    def _locate_entity(self, sentence: str, entity: str, other: str) -> tuple[bool, bool]:
        """Return (entity_is_first, entity_is_agent), reconstructed lexically.

        The dummy sees only the sentence string, so the design cell is
        inferred. Position = whether the entity precedes the other
        participant. Agent = the participant after a "by" phrase or a "whom"
        when one separates the two (passives, object clefts, object relatives
        — "X whom Y verbed" makes Y the agent), else the earlier participant
        (active order).
        """
        low = sentence.lower()
        entity_idx = self._find(low, entity)
        other_idx = self._find(low, other)
        if entity_idx is None or other_idx is None:
            raise ValueError(
                f"DummyModel cannot locate both participants in {sentence!r} "
                f"(entity {entity!r}, other {other!r})"
            )
        entity_first = entity_idx < other_idx
        for match in re.finditer(r"\b(?:by|whom)\b", low):
            entity_after, other_after = entity_idx > match.end(), other_idx > match.end()
            if entity_after != other_after:
                return entity_first, entity_after
        return entity_first, entity_first

    def _infer_role_is_agent(self, sentence: str, entity: str) -> bool:
        """Role inference when the other participant is unknown (fitting
        corpus and RQ1 probe reads): the entity is the agent iff it follows a
        "by" (passive agent phrase) or a "whom" (object cleft/relative — in
        "X whom Y verbed", Y is the agent); otherwise iff it is the earliest
        profession mentioned (active order)."""
        low = sentence.lower()
        entity_idx = self._find(low, entity)
        if entity_idx is None:
            raise ValueError(f"DummyModel cannot locate {entity!r} in {sentence!r}")
        marker = re.search(r"\b(?:by|whom)\b", low)
        if marker is not None:
            return entity_idx > marker.end()
        indices = [
            idx for p in PROFESSION_ENTITIES if (idx := self._find(low, p)) is not None
        ]
        if not indices:
            raise ValueError(
                f"DummyModel cannot infer roles in {sentence!r}: no vocab profession "
                "found (stimuli.vocab.PROFESSION_ENTITIES) — extend the vocabulary "
                "before using non-profession entities"
            )
        return entity_idx == min(indices)

    @staticmethod
    def _find(lowered_sentence: str, token: str) -> int | None:
        match = re.search(rf"\b{re.escape(token.lower())}\b", lowered_sentence)
        return None if match is None else match.start()

    @staticmethod
    def _probe_is_role(probe: str) -> bool:
        """Classify the probe by its wording (stimuli.templates): the role
        probe asks "Who ...", the neutral probe "Which ... mentioned".

        The dative RECIPIENT probe also asks "Who ...", so it lands here as a
        role probe by design — answer_distribution then swaps its readout via
        _probe_is_recipient. Both are role-diagnostic; they differ in polarity.

        CONCEPT probes are also "Which one ...?" and must be classified before
        this is called (see _probe_concept_target) — they are role-blind but
        have their own readout, so falling through to the neutral branch would
        score them as recall and silently discard the semantic contrast.
        """
        low = probe.lower()
        is_role = re.search(r"\bwho\b", low) is not None
        is_neutral = re.search(r"\bwhich\b", low) is not None or "mentioned" in low
        if is_role == is_neutral:
            raise ValueError(f"cannot classify probe as role/neutral: {probe!r}")
        return is_role

    @staticmethod
    def _probe_concept_target(probe: str) -> str | None:
        """The profession a CONCEPT probe is asking for, or None if this is not
        a concept probe.

        Matched on the cue text rather than on the profession name, because the
        cues deliberately share no stem with the professions they identify
        (stimuli.vocab.PROFESSION_CUE) — that is what stops the real model
        answering by surface match, and it means the dummy cannot shortcut it
        either.
        """
        low = probe.lower()
        for profession, cue in PROFESSION_CUE.items():
            if cue.lower() in low:
                return profession
        return None

    @staticmethod
    def _probe_is_recipient(probe: str) -> bool:
        """True for the dative RECIPIENT probe (stimuli.templates).

        It is the PASSIVE "Who was ... by someone?"; every other role probe is
        the active "Who ...?". Wording-based, like _probe_is_role — the
        WorkspaceModel protocol passes only the probe string, not its kind.
        """
        low = probe.lower()
        return re.search(r"\bwho was\b", low) is not None and "by someone" in low

    def _content_rng(self, *parts: str) -> random.Random:
        key = "\x1f".join((str(self.seed), *parts))
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def _jitter(
        self, sentence: str, probe: str, edit: EditSpec, site: InjectionSite
    ) -> tuple[float, float, float]:
        """Three N(0, 0.05) logit draws in fixed (entity, counterpart, other)
        order, seeded from a content hash so results are call-order independent."""
        sign = "none" if edit.sign is None else edit.sign.value
        rng = self._content_rng(sentence, probe, edit.edit_type.value, sign, site.value)
        return (rng.gauss(0.0, _SIGMA), rng.gauss(0.0, _SIGMA), rng.gauss(0.0, _SIGMA))
