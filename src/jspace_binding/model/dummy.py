"""GPU-free synthetic backend encoding the proposal's predictions.

Exists so the full pipeline — direction fitting, runner, binding score, stats,
plots — can be validated against known ground truth before the real Qwen
backend is loaded. Two modes:

- ``binding``: the workspace binds fillers to roles. A uniform ROLE_PUSH
  interacts with the role the sentence already assigned — the agent-pole push
  moves patient sentences most, the patient-pole push moves agent sentences
  most (the crossover) — so the pipeline must recover a large positive
  binding score that clears the control null band.
- ``bag``: the workspace is an unordered bag of concepts. The same push shifts
  the entity's answer log-odds by a constant increment regardless of role, so
  the log-odds binding score must land at ~0 inside the band. (In raw
  probability the two conditions move by different amounts — floor/ceiling —
  which is exactly why the analysis reads log-odds; see the proposal's
  Motivation.)

All synthetic behavior is defined in LOGIT space and mapped through a sigmoid,
so "uniform log-odds shift" is exact rather than approximate.

Stdlib only: this backend must run on any machine. answer_distribution ignores
edit.coefficient/alpha (the dummy has no strength dial), so calibration
scripts running against it simply return the smallest grid value.

Note on fitting_activation: the fitted vectors are NOT consumed by
answer_distribution — the dummy's answers come from its mode table. The
synthetic activations validate the directions.fit plumbing (a planted
direction is recoverable and bootstrap-stable in binding mode; unstable pure
noise in bag mode), not the vector-to-behavior link, which only the real
backend can test.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections.abc import Sequence

from jspace_binding.stimuli.vocab import PROFESSION_ENTITIES
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
_ORTH_DIM = 16  # synthetic orthogonal-remainder dimensionality (RQ1)

_L_DEGRADED = _logit(0.30)  # post-ablation level: role info flat / recall mildly hit
_L_MILD_AGENT = _logit(0.65)  # random-subspace ablation: mild, order-preserving
_L_MILD_PATIENT = _logit(0.08)
_L_MILD_MENTIONED = _logit(0.40)

_PUSH_EDITS = DIRECTION_PUSH_EDIT_TYPES


class DummyModel:
    """Synthetic WorkspaceModel + FittingActivationSource.

    Probabilities are full-softmax reads and need not sum to 1. Baseline
    logits at the ROLE probe (which asks for the agent): whichever mentioned
    participant is the agent reads logit(0.75), the patient logit(0.05), the
    absent counterpart logit(0.01).

    ROLE probe adjustments to the ENTITY's logit (other mirrored oppositely,
    counterpart untouched):
    - ROLE_PUSH, binding: TOWARD_AGENT adds +2.5 when the entity was patient,
      +0.3 when already agent; TOWARD_PATIENT subtracts 2.5 when agent, 0.3
      when patient. Gap shrinks from opposite sides under the two signs — the
      crossover.
    - ROLE_PUSH, bag: adds sign * 0.8 regardless of role. Uniform in log-odds.
    - NULL_NON_PARTICIPANT / RANDOM_DIRECTION / SHUFFLED_LABEL_DIRECTION,
      both modes: no adjustment — controls must not move the readout.
    - IDENTITY_SWAP (defensive; the sweep pairs it with NEUTRAL only): the
      counterpart inherits the entity's baseline logit and the entity drops
      to absent.

    NEUTRAL probe, both modes: IDENTITY_SWAP -> entity logit(0.05),
    counterpart logit(0.55), other logit(0.45) — the edit demonstrably
    propagates role-independently; every other edit -> both mentioned
    participants logit(0.45), counterpart logit(0.01).

    Then: +0.25 to the entity's logit when it is surface-FIRST (a position
    bias the analysis must cancel by position-averaging); + N(0, 0.05) logit
    jitter per token, seeded from a content hash of (seed, sentence, probe,
    edit_type, sign, site) — never from call order — so repeated and reordered
    runs produce byte-identical trial files.
    """

    def __init__(self, mode: str = "binding", seed: int = 0) -> None:
        if mode not in ("binding", "bag"):
            raise ValueError(f"unknown dummy mode {mode!r}; expected 'binding' or 'bag'")
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
        logits = self._logits(self._probe_is_role(probe), edit, entity_is_agent)
        l_entity, l_counterpart, l_other = logits
        if self._probe_is_recipient(probe):
            # The dative RECIPIENT probe asks who RECEIVED, so the answer is the
            # NON-agent: the entity's and the other participant's readouts swap.
            # The push interaction lives in the logit VALUES, so swapping mirrors
            # the whole structure — under binding the planted crossover survives
            # with inverted polarity (which analysis.PROBE_ORIENTATION undoes),
            # and under bag it still cancels to ~0. Without this the dummy scores
            # the recipient probe as if it were the role probe, and dry runs
            # report a large spurious NEGATIVE dative score.
            l_entity, l_other = l_other, l_entity
        if entity_first:
            l_entity += _POSITION_BIAS
        jitter = self._jitter(sentence, probe, edit, site)
        pairs = zip(
            (entity, counterpart, other), (l_entity, l_counterpart, l_other), jitter, strict=True
        )
        return {token: _sigmoid(logit + j) for token, logit, j in pairs if token is not None}

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
        """RQ2 ground truth (proposal, §4 Ablation).

        binding mode = the workspace is where binding lives (concepts are
        redundantly recoverable elsewhere): ABLATE_JSPACE flattens the ROLE
        readout to a tie (role accuracy -> chance) while NEUTRAL recall only
        degrades mildly and stays correct. bag mode = the workspace holds only
        concepts, binding computed elsewhere: ABLATE_JSPACE breaks concept
        recall outright (mentioned participants collapse to the absent
        token's level -> recall to chance) and thereby the ROLE answer too, so
        binding degrades NO MORE than recall — no binding-specific deficit.
        ABLATE_RANDOM_SUBSPACE is mild and order-preserving in both modes and
        under both probes: the capacity-matched comparison bar.
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
        self, sentence: str, entity: str, site: InjectionSite
    ) -> dict[str, list[float]]:
        """Synthetic RQ1 activation sources with mode-dependent ground truth.

        binding mode: the role signal lives in the J-space component (the
        orthogonal remainder is pure noise), so a linear probe must decode
        role from "jspace" and "residual" but sit at chance on "orthogonal".
        bag mode: the workspace holds no role information — the signal is
        planted in the ORTHOGONAL remainder instead (the model still binds,
        just elsewhere), so "jspace" must probe at chance while "orthogonal"
        and "residual" decode. The synthetic residual basis is axis-aligned:
        residual = jspace coords ++ orthogonal coords.

        The planted RQ1 direction is FILLER-GENERAL (shared across entities,
        unlike fitting_activation's per-entity directions): the RQ1 split is
        leave-one-pair-out, so only role information that generalizes across
        lexical items should be decodable — the dummy's ground truth is that
        such information exists, in the mode-appropriate subspace.
        """
        role_sign = 1.0 if self._infer_role_is_agent(sentence, entity) else -1.0
        rng = self._content_rng("probe", sentence, entity, site.value)
        j_noise = [rng.gauss(0.0, _FIT_NOISE) for _ in range(_FIT_DIM)]
        o_noise = [rng.gauss(0.0, _FIT_NOISE) for _ in range(_ORTH_DIM)]
        if self.mode == "binding":
            planted = self._unit_vector("probe-role", site.value, dim=_FIT_DIM)
            jspace = [role_sign * p + n for p, n in zip(planted, j_noise, strict=True)]
            orthogonal = o_noise
        else:
            planted = self._unit_vector("probe-role-orth", site.value, dim=_ORTH_DIM)
            jspace = j_noise
            orthogonal = [role_sign * p + n for p, n in zip(planted, o_noise, strict=True)]
        return {"jspace": jspace, "orthogonal": orthogonal, "residual": jspace + orthogonal}

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
        """Classify the probe by its draft wording (stimuli.templates): the
        role probe asks "Who ...", the neutral probe "Which ... mentioned".

        The dative RECIPIENT probe also asks "Who ...", so it lands here as a
        role probe by design — answer_distribution then swaps its readout via
        _probe_is_recipient. Both are role-diagnostic; they differ in polarity.
        """
        low = probe.lower()
        is_role = re.search(r"\bwho\b", low) is not None
        is_neutral = re.search(r"\bwhich\b", low) is not None or "mentioned" in low
        if is_role == is_neutral:
            raise ValueError(f"cannot classify probe as role/neutral: {probe!r}")
        return is_role

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
