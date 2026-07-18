"""Qwen3.6-27B + pre-fitted Jacobian-lens backend.

Implements WorkspaceModel + FittingActivationSource against the real model.
STATUS: fully shaped and importable on GPU-free machines (torch/transformers
are imported lazily inside methods), but NOT yet validated on real weights —
the one genuinely open unknown is the released lens's artifact schema, which
_load_lens documents and fails loudly about. First GPU session: download the
lens, inspect its keys, and adapt _load_lens (everything else is wired).

Layer semantics (open knob, flagged for team review): config.layer_band is an
inclusive (lo, hi) pair. Edits are applied at every layer in the band;
activations for direction fitting and J-space reads are taken at `hi` (the
assembled workspace after the band). A single-layer setup — matching the
source paper's swap experiments, the proposal's lean (a) — is layer_band:
[L, L].

Site anchoring: FINAL_TOKEN = the last token of the SENTENCE (not the probe);
ENTITY_TOKEN = the last tokenizer token of the family's target entity word.
For the NULL_NON_PARTICIPANT control the pushed direction belongs to an
absent entity but the site anchor stays the sentence's own target-entity
token — the site is a sentence position, not a property of the direction.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from jspace_binding.config import ModelConfig
from jspace_binding.directions.fit import FittedDirections, load_directions
from jspace_binding.types import EditSpec, EditType, InjectionSite, PushSign

# Baseline concept set for the "Tell me about {concept}" identity-direction
# recipe: read the J-space state at the concept token, then subtract the mean
# over this 100-concept baseline so shared prompt/format components cancel.
_BASELINE_CONCEPTS: tuple[str, ...] = (
    "apple", "river", "mountain", "book", "window", "garden", "bridge", "engine",
    "letter", "market", "forest", "island", "mirror", "bottle", "candle", "castle",
    "cloud", "desert", "diamond", "door", "dream", "father", "fire", "flower",
    "friend", "guitar", "hammer", "harbor", "horse", "hotel", "house", "hunter",
    "journey", "kitchen", "ladder", "lake", "lamp", "library", "machine", "meadow",
    "money", "moon", "morning", "mother", "museum", "music", "needle", "night",
    "ocean", "office", "orange", "painter", "paper", "pencil", "piano", "picture",
    "planet", "pocket", "police", "prince", "prison", "puzzle", "rabbit", "radio",
    "rain", "road", "rocket", "roof", "rope", "school", "shadow", "ship",
    "shoulder", "silver", "sister", "snow", "soldier", "song", "spring", "star",
    "station", "stone", "storm", "street", "summer", "sun", "table", "temple",
    "theater", "thunder", "ticket", "tiger", "tower", "train", "valley", "village",
    "water", "winter", "wolf", "zebra",
)


class LensFormatError(RuntimeError):
    """The lens artifact does not match the schema _load_lens expects."""


class QwenJLensModel:
    """WorkspaceModel + FittingActivationSource backend for the real model.

    The constructor is deliberately non-raising: it validates the config and
    collects still-open decisions into `missing_decisions`, so the runner can
    construct the backend and report everything missing in one shot when the
    first forward is attempted.
    """

    def __init__(
        self,
        config: ModelConfig,
        directions_dir: str | Path = Path("data/directions"),
        direction_variant: str = "fitted",
    ) -> None:
        if config.backend != "qwen_jlens":
            raise ValueError(f"config.backend is {config.backend!r}, expected 'qwen_jlens'")
        self.config = config
        self.directions_dir = Path(directions_dir)
        self.direction_variant = direction_variant
        self.missing_decisions: list[str] = []
        if config.layer_band is None:
            self.missing_decisions.append(
                "layer_band: workspace layer band for Qwen3.6-27B (model.layer_band in config; "
                "[L, L] matches the source paper's single-layer swaps)"
            )
        # NOTE: push_coefficient is deliberately NOT a construction-time
        # requirement — direction fitting and calibration must run before a
        # calibrated coefficient can exist, so it is enforced at push time
        # (_edit_delta) instead.
        self._model: Any = None
        self._tokenizer: Any = None
        self._lens: dict[int, tuple[Any, Any]] = {}  # layer -> (encoder, decoder) torch tensors
        self._directions: dict[InjectionSite, FittedDirections] = {}
        self._identity_mean: Any = None  # baseline-mean J-space state (identity recipe)
        self._random_subspaces: dict[int, Any] = {}  # seed -> orthonormal basis (RQ2)
        self._swap_operators: dict[tuple[str, str], Any] = {}  # (source, target) -> (V, pinv V)

    def preflight(self, sites: Sequence[InjectionSite] = ()) -> None:
        """Fail fast on everything that would block a run: open config
        decisions, missing heavy extras, the lens artifact, and (for the
        given sites) fitted directions. Scripts call this once up front so a
        mid-sweep RuntimeError (e.g. CUDA OOM) is never mistaken for a
        not-configured-yet condition."""
        self._ensure_ready()
        for site in sites:
            self._site_directions(site)

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
        """One hooked forward pass over `sentence + " " + probe`; read each
        answer token's probability from the full next-token softmax."""
        self._ensure_ready()  # before any torch import: report missing decisions first
        import torch

        if edit.edit_type is EditType.IDENTITY_SWAP:
            # Warm the swap operator BEFORE installing hooks: building it runs
            # ~101 "Tell me about {concept}" forwards, which must never happen
            # re-entrantly inside the edit hook.
            self._swap_operator(edit.swap_source, edit.swap_target)  # type: ignore[arg-type]
        text = f"{sentence} {probe}"
        ids = self._tokenizer(text, return_tensors="pt").input_ids.to(self._device())
        anchor = self._site_index(sentence, text, site, target_entity=answer_tokens[0])
        answer_ids = {tok: self._single_token_id(tok) for tok in answer_tokens}

        with torch.no_grad(), self._edit_hooks(edit, site, anchor):
            logits = self._model(ids).logits[0, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)
        return {tok: float(probs[tid]) for tok, tid in answer_ids.items()}

    # ------------------------------------------------------------------ #
    # FittingActivationSource                                            #
    # ------------------------------------------------------------------ #

    def fitting_activation(
        self, sentence: str, entity: str, site: InjectionSite
    ) -> list[float]:
        """J-space coordinates at the read layer for a fitting-corpus sentence
        (no edit applied). Site anchor: the FITTED entity's own token for
        ENTITY_TOKEN, else the sentence-final token."""
        self._ensure_ready()  # before any torch import: report missing decisions first
        anchor = self._site_index(sentence, sentence, site, target_entity=entity)
        h = self._hidden_at(sentence, anchor)
        encoder, _ = self._lens[self._read_layer()]
        coords = encoder @ h.to(encoder.dtype)
        return [float(x) for x in coords.float().cpu()]

    # ------------------------------------------------------------------ #
    # ProbeActivationSource (RQ1)                                        #
    # ------------------------------------------------------------------ #

    def probe_activation(
        self, sentence: str, entity: str, site: InjectionSite
    ) -> dict[str, list[float]]:
        """The three RQ1 sources at the read layer: "jspace" = E @ h,
        "orthogonal" = h - D @ (E @ h) (what the lens cannot reconstruct),
        "residual" = h. No edit applied."""
        self._ensure_ready()
        anchor = self._site_index(sentence, sentence, site, target_entity=entity)
        encoder, decoder = self._lens[self._read_layer()]
        h = self._hidden_at(sentence, anchor).to(encoder.dtype)
        coords = encoder @ h
        orthogonal = h - decoder @ coords
        return {
            "jspace": [float(x) for x in coords.float().cpu()],
            "orthogonal": [float(x) for x in orthogonal.float().cpu()],
            "residual": [float(x) for x in h.float().cpu()],
        }

    def _hidden_at(self, text: str, anchor: int) -> Any:
        """Residual activation at the read layer for token position `anchor`
        of an edit-free forward over `text` (the capture-hook pattern shared
        by fitting_activation / probe_activation / _concept_state)."""
        import torch

        ids = self._tokenizer(text, return_tensors="pt").input_ids.to(self._device())
        captured: dict[str, Any] = {}

        def capture(module: Any, inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured["h"] = hidden[0, anchor, :].detach()

        handle = self._decoder_layer(self._read_layer()).register_forward_hook(capture)
        try:
            with torch.no_grad():
                self._model(ids)
        finally:
            handle.remove()
        return captured["h"]

    # ------------------------------------------------------------------ #
    # Loading                                                            #
    # ------------------------------------------------------------------ #

    def _ensure_ready(self) -> None:
        if self.missing_decisions:
            raise RuntimeError(
                "QwenJLensModel cannot run; open decisions: "
                + "; ".join(self.missing_decisions)
            )
        if self._model is None:
            self._load_model()
            self._load_lens()

    def _load_model(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # torch/transformers are optional extras
            raise RuntimeError(
                "the qwen_jlens backend needs the heavy extras: pip install '.[model]'"
            ) from exc

        kwargs: dict[str, Any] = {"device_map": "auto"}
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            import torch

            kwargs["torch_dtype"] = getattr(torch, self.config.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **kwargs)
        self._model.eval()

    def _load_lens(self) -> None:
        """Load per-layer J-lens encoder/decoder matrices.

        EXPECTED SCHEMA (to be confirmed against the actual Neuronpedia/HF
        release on the first GPU session — this is the project's one open
        artifact unknown): a single .npz or .safetensors file in
        config.lens_repo (local path or HF repo id) with, per layer L:

            "layer_{L}.encoder": (k, d_model)  # residual -> J-space coords
            "layer_{L}.decoder": (d_model, k)  # J-space coords -> residual

        If the release stores different keys (or one matrix with a
        pseudoinverse convention), adapt the key-mapping below — nothing else
        in the pipeline depends on the schema.
        """
        import torch

        root = Path(self.config.lens_repo)
        if not root.exists():
            from huggingface_hub import snapshot_download

            root = Path(snapshot_download(self.config.lens_repo))
        candidates = sorted(root.glob("*.npz")) + sorted(root.glob("*.safetensors"))
        if not candidates:
            raise LensFormatError(
                f"no .npz/.safetensors lens artifact under {root}; inspect the release "
                "and adapt QwenJLensModel._load_lens"
            )
        arrays = self._read_arrays(candidates[0])
        lo, hi = self.config.layer_band  # type: ignore[misc]
        device = self._device()
        for layer in range(lo, hi + 1):
            enc_key, dec_key = f"layer_{layer}.encoder", f"layer_{layer}.decoder"
            if enc_key not in arrays or dec_key not in arrays:
                raise LensFormatError(
                    f"lens artifact {candidates[0].name} lacks {enc_key!r}/{dec_key!r}; "
                    f"available keys: {sorted(arrays)[:12]}... — adapt _load_lens's "
                    "key mapping to the actual release schema"
                )
            self._lens[layer] = (
                torch.as_tensor(arrays[enc_key], device=device),
                torch.as_tensor(arrays[dec_key], device=device),
            )

    @staticmethod
    def _read_arrays(path: Path) -> dict[str, np.ndarray]:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                return {k: data[k] for k in data.files}
        from safetensors.numpy import load_file

        return load_file(path)

    # ------------------------------------------------------------------ #
    # Edits                                                              #
    # ------------------------------------------------------------------ #

    def _edit_hooks(self, edit: EditSpec, site: InjectionSite, anchor: int) -> Any:
        """Context manager installing forward hooks over the layer band that
        apply `edit` at token position `anchor`. NO_EDIT installs nothing."""
        import contextlib

        import torch

        if edit.edit_type is EditType.NO_EDIT:
            return contextlib.nullcontext()

        lo, hi = self.config.layer_band  # type: ignore[misc]
        model = self

        @contextlib.contextmanager
        def hooks() -> Any:
            handles = []

            def make_hook(layer: int) -> Any:
                def hook(module: Any, inputs: Any, output: Any) -> Any:
                    hidden = output[0] if isinstance(output, tuple) else output
                    h = hidden[0, anchor, :]
                    delta = model._edit_delta(edit, site, layer, h)
                    hidden[0, anchor, :] = h + delta.to(h.dtype)
                    return output

                return hook

            try:
                for layer in range(lo, hi + 1):
                    handles.append(
                        model._decoder_layer(layer).register_forward_hook(make_hook(layer))
                    )
                with torch.no_grad():
                    yield
            finally:
                for handle in handles:
                    handle.remove()

        return hooks()

    def _edit_delta(self, edit: EditSpec, site: InjectionSite, layer: int, h: Any) -> Any:
        """Residual-stream delta for one edit at one layer (math per
        interventions.edits, the canonical description)."""
        import torch

        encoder, decoder = self._lens[layer]
        if edit.edit_type is EditType.ABLATE_JSPACE:
            # RQ2: remove the lens-reconstructable component at the site.
            return -(decoder @ (encoder @ h.to(encoder.dtype)))
        if edit.edit_type is EditType.ABLATE_RANDOM_SUBSPACE:
            # RQ2 capacity control: project out a seeded random orthonormal
            # subspace of the SAME dimension as the J-space (matched dim).
            basis = self._random_subspace(edit.seed or 0, encoder)
            h_cast = h.to(basis.dtype)
            return -(basis @ (basis.T @ h_cast))
        if edit.edit_type is EditType.IDENTITY_SWAP:
            # Warmed before the hooks were installed (answer_distribution);
            # this lookup must never trigger the ~101 nested baseline forwards.
            V, pinv_V = self._swap_operator(edit.swap_source, edit.swap_target)  # type: ignore[arg-type]
            h_j = encoder @ h.to(encoder.dtype)
            c = pinv_V @ h_j
            alpha = 1.0 if edit.alpha is None else float(edit.alpha)
            swapped = alpha * torch.stack([c[1], c[0]])
            return decoder @ (V @ (swapped - c))

        # Direction pushes: h += sign * coefficient * unit(decoder @ r).
        # Normalized in the RESIDUAL stream, not J-space: the decoder is not
        # orthonormal, so J-space unit vectors decode to different residual
        # norms — normalizing after decoding is what makes the random-direction
        # control genuinely norm-matched to the fitted-role push.
        if edit.sign is None:
            raise ValueError(f"{edit.edit_type.value} EditSpec lacks a PushSign")
        if edit.coefficient is None and self.config.push_coefficient is None:
            raise RuntimeError(
                "no push coefficient: set model.push_coefficient in config "
                "(scripts/calibrate.py produces it) or pass EditSpec.coefficient"
            )
        sign = 1.0 if edit.sign is PushSign.TOWARD_AGENT else -1.0
        coefficient = (
            float(edit.coefficient)
            if edit.coefficient is not None
            else float(self.config.push_coefficient)  # type: ignore[arg-type]
        )
        r = self._push_direction(edit, site, k=encoder.shape[0])
        r_t = torch.as_tensor(r, device=decoder.device, dtype=decoder.dtype)
        v = decoder @ r_t
        return sign * coefficient * (v / torch.linalg.vector_norm(v))

    def _push_direction(self, edit: EditSpec, site: InjectionSite, k: int) -> np.ndarray:
        """Unit direction in J-space coordinates for a push edit."""
        if edit.edit_type is EditType.RANDOM_DIRECTION:
            rng = np.random.default_rng(edit.seed)
            raw = rng.standard_normal(k)
            return raw / np.linalg.norm(raw)
        variant = {
            EditType.ROLE_PUSH: self.direction_variant,
            EditType.NULL_NON_PARTICIPANT: "fitted",
            EditType.SHUFFLED_LABEL_DIRECTION: "shuffled",
        }[edit.edit_type]
        directions = self._site_directions(site)
        r = directions.direction(edit.entity, variant)  # type: ignore[arg-type]
        if r.shape[0] != k:
            raise ValueError(
                f"fitted direction dim {r.shape[0]} != lens dim {k}; refit directions "
                "against this lens (scripts/fit_directions.py)"
            )
        return r

    def _site_directions(self, site: InjectionSite) -> FittedDirections:
        if site not in self._directions:
            self._directions[site] = load_directions(self.directions_dir, site)
        return self._directions[site]

    def _swap_operator(self, source: str, target: str) -> Any:
        """(V, pinv(V)) for an identity swap, cached per concept pair — the
        pseudoinverse is identical across every trial and layer, and building
        V runs the baseline-concept forwards, which must happen outside any
        active edit hook."""
        import torch

        key = (source, target)
        if key not in self._swap_operators:
            V = torch.stack(
                [self._concept_direction(source), self._concept_direction(target)], dim=1
            )  # (k, 2), J-space coords
            self._swap_operators[key] = (V, torch.linalg.pinv(V))
        return self._swap_operators[key]

    def _random_subspace(self, seed: int, encoder: Any) -> Any:
        """Seeded random orthonormal (d_model, k) basis, cached per seed."""
        import torch

        if seed not in self._random_subspaces:
            k, d_model = encoder.shape
            rng = np.random.default_rng(seed)
            raw = rng.standard_normal((d_model, k))
            q, _ = np.linalg.qr(raw)
            self._random_subspaces[seed] = torch.as_tensor(
                q, device=encoder.device, dtype=encoder.dtype
            )
        return self._random_subspaces[seed]

    # ------------------------------------------------------------------ #
    # Identity directions ("Tell me about {concept}" recipe)             #
    # ------------------------------------------------------------------ #

    @lru_cache(maxsize=256)  # noqa: B019 - the model is a long-lived singleton
    def _concept_direction(self, word: str) -> Any:
        """Lens-space identity direction for `word`: J-space state at the
        concept token of "Tell me about {word}", mean-subtracted over the
        100-concept baseline set (computed once, cached)."""
        if self._identity_mean is None:
            states = [self._concept_state(concept) for concept in _BASELINE_CONCEPTS]
            import torch

            self._identity_mean = torch.stack(states).mean(dim=0)
        return self._concept_state(word) - self._identity_mean

    def _concept_state(self, word: str) -> Any:
        text = f"Tell me about {word}"
        anchor = self._last_word_token_index(text, word)
        encoder, _ = self._lens[self._read_layer()]
        return encoder @ self._hidden_at(text, anchor).to(encoder.dtype)

    # ------------------------------------------------------------------ #
    # Indexing helpers                                                   #
    # ------------------------------------------------------------------ #

    def _site_index(
        self, sentence: str, full_text: str, site: InjectionSite, target_entity: str
    ) -> int:
        if site is InjectionSite.FINAL_TOKEN:
            return len(self._tokenizer(sentence).input_ids) - 1
        return self._last_word_token_index(full_text, target_entity)

    def _last_word_token_index(self, text: str, word: str) -> int:
        """Index of the last tokenizer token overlapping `word` in `text`
        (design guarantees exactly one occurrence per sentence)."""
        match = re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"{word!r} does not occur in {text!r}")
        encoding = self._tokenizer(text, return_offsets_mapping=True)
        last = None
        for index, (start, end) in enumerate(encoding.offset_mapping):
            if start < match.end() and end > match.start():
                last = index
        if last is None:
            raise ValueError(f"no token overlaps {word!r} in {text!r}")
        return last

    def _single_token_id(self, word: str) -> int:
        ids = self._tokenizer.encode(f" {word}", add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"answer token {word!r} is {len(ids)} tokens under this tokenizer; "
                "the vocabulary must be re-validated (stimuli.vocab.validate_single_token)"
            )
        return ids[0]

    def _read_layer(self) -> int:
        return self.config.layer_band[1]  # type: ignore[index]

    def _decoder_layer(self, layer: int) -> Any:
        return self._model.model.layers[layer]

    def _device(self) -> Any:
        return next(self._model.parameters()).device
