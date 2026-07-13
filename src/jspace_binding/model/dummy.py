"""GPU-free synthetic backend encoding the proposal's worked example.

Exists so the full pipeline — runner, binding score, stats, plots — can be
validated against known ground truth before the real Qwen backend is loaded.
Two modes:

- ``binding``: the workspace binds fillers to roles. A REAL coordinate-swap
  moves answer mass to the swap target only when the edited slot occupies the
  probed (agent) role, so the pipeline must recover BS ~= 0.55 and clear the
  control null band.
- ``bag``: the workspace is an unordered bag of concepts. The REAL swap moves
  the same mass regardless of role, so the pipeline must recover BS ~= 0,
  inside the null band.

Stdlib only: this backend must run on any machine.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Sequence

from jspace_binding.types import EditSpec, EditType, InjectionSite

_SIGMA = 0.02  # jitter std dev
_POSITION_BIAS = 0.03  # added to P(target) when the edited slot is FIRST
_P_MIN, _P_MAX = 0.001, 0.999


class DummyModel:
    """Synthetic WorkspaceModel. Probabilities are full-softmax reads and need
    not sum to 1. "Slot" below = the sentence position holding the source
    concept, i.e. where a REAL swap lands.

    Base (P(source), P(target), P(other)) before position bias and jitter:

    ROLE probe (queries the agent):
    - NO_EDIT / NULL_NON_PARTICIPANT / RANDOM_DIRECTION, both modes:
      P(target)=0.01; whichever participant is the agent reads 0.55, the
      other 0.05 — controls must not move the readout.
    - REAL, binding mode: slot=agent -> (0.05, 0.60, 0.05);
      slot=patient -> (0.05, 0.05, 0.55). The swap shows up in the answer
      only when it hits the probed role.
    - REAL, bag mode: P(target)=0.30 regardless of slot role (role-blind
      swap); P(source)=0.05 (swapped out of the bag); P(other) keeps its
      0.55/0.05 agent/patient baseline.

    NEUTRAL probe, both modes: REAL -> (0.05, 0.55, 0.40) — the edit
    demonstrably propagates role-independently; all other edits ->
    (0.45, 0.01, 0.45), both mentioned participants read high.

    Then: +0.03 to P(target) when the slot is FIRST (a position bias the
    analysis must cancel by position-averaging); + N(0, 0.02) jitter;
    clip to [0.001, 0.999].

    Determinism: jitter is seeded from a content hash of (seed, sentence,
    probe, edit_type, site) — never from call order — so repeated and
    reordered runs produce byte-identical trial files.
    """

    def __init__(self, mode: str = "binding", seed: int = 0) -> None:
        if mode not in ("binding", "bag"):
            raise ValueError(f"unknown dummy mode {mode!r}; expected 'binding' or 'bag'")
        self.mode = mode
        self.seed = seed

    def answer_distribution(
        self,
        sentence: str,
        probe: str,
        edit: EditSpec,
        site: InjectionSite,
        answer_tokens: Sequence[str],
    ) -> dict[str, float]:
        """Synthetic probabilities per the class table.

        `answer_tokens` must follow AnswerSet.tokens order: (source, target,
        other) — with no design-cell metadata in the call, ordering is the only
        channel telling the dummy which token plays which part.
        """
        if len(answer_tokens) != 3:
            raise ValueError(
                f"DummyModel expects (source, target, other), got {len(answer_tokens)} tokens"
            )
        source, target, other = answer_tokens
        slot_first, slot_is_agent = self._locate_slot(sentence, source, target, other)
        base = self._base_probs(self._probe_is_role(probe), edit.edit_type, slot_is_agent)
        p_source, p_target, p_other = base
        if slot_first:
            p_target += _POSITION_BIAS
        jitter = self._jitter(sentence, probe, edit.edit_type, site)
        base = zip((source, target, other), (p_source, p_target, p_other), jitter, strict=True)
        return {token: min(max(p + j, _P_MIN), _P_MAX) for token, p, j in base}

    def _base_probs(
        self, probe_is_role: bool, edit_type: EditType, slot_is_agent: bool
    ) -> tuple[float, float, float]:
        """(P(source), P(target), P(other)) before position bias and jitter."""
        if not probe_is_role:
            if edit_type is EditType.REAL:
                return (0.05, 0.55, 0.40)
            return (0.45, 0.01, 0.45)
        if edit_type is EditType.REAL:
            other_baseline = 0.05 if slot_is_agent else 0.55
            if self.mode == "bag":
                return (0.05, 0.30, other_baseline)
            if slot_is_agent:
                return (0.05, 0.60, 0.05)
            return (0.05, 0.05, other_baseline)
        # NO_EDIT and both controls: sentence effectively untouched.
        if slot_is_agent:
            return (0.55, 0.01, 0.05)
        return (0.05, 0.01, 0.55)

    def _locate_slot(
        self, sentence: str, source: str, target: str, other: str
    ) -> tuple[bool, bool]:
        """Return (slot_is_first, slot_is_agent), reconstructed lexically.

        The dummy sees only the sentence string, so the design cell is
        inferred. The swap target normally is NOT in the sentence — pre-edit,
        the slot is occupied by the source concept — so the slot is located at
        the target token when present, else at the source token. Position =
        whether that slot precedes the other participant. Agent = the
        participant after a "by" phrase or a "whom" when one separates the two
        (passives, object clefts, object relatives — "X whom Y verbed" makes Y
        the agent), else the earlier participant (active order).
        """
        low = sentence.lower()
        slot_idx = self._find(low, target)
        if slot_idx is None:
            slot_idx = self._find(low, source)
        other_idx = self._find(low, other)
        if slot_idx is None or other_idx is None:
            raise ValueError(
                f"DummyModel cannot locate both participants in {sentence!r} "
                f"(slot in {source!r}/{target!r}, other {other!r})"
            )
        slot_first = slot_idx < other_idx
        for match in re.finditer(r"\b(?:by|whom)\b", low):
            slot_after, other_after = slot_idx > match.end(), other_idx > match.end()
            if slot_after != other_after:
                return slot_first, slot_after
        return slot_first, slot_first

    @staticmethod
    def _find(lowered_sentence: str, token: str) -> int | None:
        match = re.search(rf"\b{re.escape(token.lower())}\b", lowered_sentence)
        return None if match is None else match.start()

    @staticmethod
    def _probe_is_role(probe: str) -> bool:
        """Classify the probe by its draft wording (stimuli.templates): the
        role probe asks "Who ...", the neutral probe "Which ... mentioned"."""
        low = probe.lower()
        is_role = re.search(r"\bwho\b", low) is not None
        is_neutral = re.search(r"\bwhich\b", low) is not None or "mentioned" in low
        if is_role == is_neutral:
            raise ValueError(f"cannot classify probe as role/neutral: {probe!r}")
        return is_role

    def _jitter(
        self, sentence: str, probe: str, edit_type: EditType, site: InjectionSite
    ) -> tuple[float, float, float]:
        """Three N(0, 0.02) draws in fixed (source, target, other) order,
        seeded from a content hash so results are call-order independent."""
        key = "\x1f".join((str(self.seed), sentence, probe, edit_type.value, site.value))
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        return (rng.gauss(0.0, _SIGMA), rng.gauss(0.0, _SIGMA), rng.gauss(0.0, _SIGMA))
