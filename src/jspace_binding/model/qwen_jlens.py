"""Qwen3.6-27B + pre-fitted Jacobian-lens backend.

Implements WorkspaceModel + FittingActivationSource + ProbeActivationSource
against the real model, grounded in the ACTUAL methods of Gurnee et al.
(2026) — not a guessed schema:

- The released lens artifact is a per-layer averaged Jacobian J_l, a
  (d_model x d_model) matrix mapping layer-l residual directions to their
  final-layer counterparts (their §2.1). Reading the lens is
  lens(h) = softmax(W_U norm(J_l h)), with W_U the model's own unembedding.
- The J-lens VECTOR for vocabulary token t at layer l is
  v_t = J_l^T W_U[t] — the residual-stream direction whose inner product
  with h gives t's lens score. These are the atoms every intervention uses.
- The J-space is NOT a linear subspace (their §2.3): it is the set of sparse
  nonnegative combinations of at most k J-lens vectors. The "J-space
  component" of an activation is recovered by sparse pursuit against the
  J-lens dictionary; the non-J-space component is the remainder. We
  implement a matching-pursuit approximation of their gradient pursuit
  (argmax lens score -> refit active set -> clamp negatives), k from
  config.model.jspace_k.
- The identity swap patches in lens coordinates (their §2.5):
  V = [v_s v_t], c = V^+ h, h_patched = h + V(sigma(c) - c), optional alpha.
  No auxiliary "concept vector" forwards are needed — the swap operator is
  pure linear algebra over J_l and W_U, so it is safe to build inside hooks
  (cached per pair and layer).
- J-space ABLATION zeroes the residual's projection onto the span of the
  top-k most strongly active J-lens vectors (their §3.5.2, k ~= 10; config
  model.ablate_k). Their capability evals additionally exclude tokens in the
  clean forward's top-10 to avoid ablating intended outputs — a refinement
  to consider on GPU day, not implemented here.

Layer semantics: config.layer_band is an inclusive (lo, hi) pair of RAW
layer indices. Edits apply at every layer in the band; reads (fitting, RQ1)
use `hi`. [L, L] reproduces the source paper's single-layer swaps. Their
workspace band is reindexed layers ~38-92 of 100, with single-layer analyses
typically mid-workspace (~L75 reindexed); convert to raw indices once the
model's layer count is known.

Site anchoring: FINAL_TOKEN = the last token of the SENTENCE (not the
probe); ENTITY_TOKEN = the last tokenizer token of the target entity word.
For the NULL_NON_PARTICIPANT control the pushed direction belongs to an
absent entity but the anchor stays the sentence's own target-entity token —
the site is a sentence position, not a property of the direction.

STATUS: written to the paper's spec but never run on real weights. The lens
artifact's schema is confirmed against the release (neuronpedia/jacobian-lens:
a .pt holding a nested "J" mapping of int layer -> Jacobian; see _read_arrays
and tests/test_lens_loading.py), so the remaining unknown is ordinary
first-contact bugs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from jspace_binding.config import ModelConfig
from jspace_binding.directions.fit import FittedDirections, load_directions
from jspace_binding.types import EditSpec, EditType, InjectionSite, PushSign


class LensFormatError(RuntimeError):
    """The lens artifact does not match the schema _load_lens expects."""


# Key patterns we try, in order, for layer L's Jacobian in the released
# artifact. CONFIRMED against neuronpedia/jacobian-lens (2026-07): the release
# is a torch .pt holding {"J": {int_layer: (d_model, d_model) fp16 tensor},
# "source_layers": [...], "d_model": int, "n_prompts": int}. _read_arrays
# normalises that nested mapping to flat "layer_{L}" keys, which is why the
# first pattern matches; the rest are kept for other/older spellings.
_JACOBIAN_KEY_PATTERNS: tuple[str, ...] = (
    "layer_{L}",
    "J_{L}",
    "jacobian_{L}",
    "layers.{L}.jacobian",
    "layer_{L}.jacobian",
)

# Artifact file extensions we know how to read, in preference order.
_LENS_SUFFIXES: tuple[str, ...] = (".pt", ".npz", ".safetensors")

# Seed for RQ1's capacity-control subspace (probe_activation's
# "random_subspace"). Fixed rather than drawn from experiment.seed because the
# backend holds only ModelConfig, and because the basis must be IDENTICAL for
# every example in a run — it is a readout basis, like jspace, not per-trial
# noise. Vary it only to check that a localisation result is not an artifact of
# one unlucky draw.
_RQ1_CAPACITY_SEED = 0


class QwenJLensModel:
    """WorkspaceModel backend for the real model.

    The constructor is deliberately non-raising: it validates the config and
    collects still-open decisions into `missing_decisions`, reported in one
    shot by preflight()/the first forward.
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
                "layer_band: raw workspace layer band for Qwen3.6-27B (model.layer_band; "
                "[L, L] matches the source paper's single-layer swaps — their single-layer "
                "analyses sit mid-workspace, ~reindexed L75 of 100)"
            )
        # push_coefficient is deliberately NOT a construction-time requirement:
        # direction fitting and calibration must run before it can exist, so it
        # is enforced at push time (_edit_delta).
        self._model: Any = None
        self._tokenizer: Any = None
        self._jacobians: dict[int, Any] = {}  # raw layer -> J_l (d_model x d_model)
        self._w_u: Any = None  # unembedding weight (n_vocab x d_model)
        self._directions: dict[InjectionSite, FittedDirections] = {}
        self._jlens_vectors: dict[tuple[int, int], Any] = {}  # (layer, token_id) -> v_t
        self._swap_operators: dict[tuple[str, str, int], Any] = {}  # (src, tgt, layer) -> (V, V^+)
        # (seed, rank) -> orthonormal (d_model, rank). Keyed by rank too because
        # RQ2's ablation control uses ablate_k and RQ1's capacity control uses
        # jspace_k; a seed-only key would hand back the wrong-rank basis.
        self._random_subspaces: dict[tuple[int, int], Any] = {}
        # ||delta|| / ||h|| for each applied edit, drained by the caller (RQ2).
        # An ablation removing ablate_k of d_model directions can move the
        # residual by a fraction of a percent, and a null deficit measured
        # next to an unrecorded edit cannot be distinguished from an edit that
        # never landed. One norm ratio per hook, so recording is unconditional.
        self._edit_magnitudes: list[float] = []

    def drain_edit_magnitudes(self) -> list[float]:
        """Relative residual-norm changes since the last call, then reset."""
        drained = self._edit_magnitudes
        self._edit_magnitudes = []
        return drained

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
        """The J-SPACE COMPONENT (residual-space, d_model) of the activation
        at the read layer for a fitting-corpus sentence — per the proposal,
        role directions are diff-of-means over activations projected onto
        the lens-defined workspace, and per the paper that projection is the
        sparse-pursuit component, not a matmul. No edit applied."""
        self._ensure_ready()
        anchor = self._site_index(sentence, sentence, site, target_entity=entity)
        h = self._hidden_at(sentence, anchor)
        component, _ = self._jspace_component(h, self._read_layer())
        return [float(x) for x in component.float().cpu()]

    # ------------------------------------------------------------------ #
    # ProbeActivationSource (RQ1)                                        #
    # ------------------------------------------------------------------ #

    def probe_activation(
        self, sentence: str, entity: str, site: InjectionSite
    ) -> dict[str, list[float]]:
        """The four RQ1 sources at the read layer: "jspace" = the sparse
        J-space component, "orthogonal" = h minus that component (what the
        lens cannot see), "residual" = the full activation, "random_subspace" =
        h projected onto a random subspace of the SAME rank as jspace. No edit
        applied.

        random_subspace is the capacity control (analysis.probes.PROBE_SOURCES):
        jspace has effective rank <= jspace_k while orthogonal has ~d_model, so
        jspace-vs-orthogonal confounds localisation with capacity. The basis is
        fixed per (seed, rank) rather than drawn per sentence — it is a readout
        basis like jspace, not noise.
        """
        self._ensure_ready()
        anchor = self._site_index(sentence, sentence, site, target_entity=entity)
        return self._sources_at(self._hidden_at(sentence, anchor))

    def recruitment_activation(
        self, sentence: str, probe: str, entity: str
    ) -> dict[str, list[float]]:
        """E4: the same sources, read with `probe` in context (proposal §6 E4).

        Read at the FINAL token of `sentence + probe`, not at an InjectionSite.
        The question is appended after the sentence, so under a causal mask it
        cannot affect any token inside the sentence — reading at FINAL_TOKEN
        (last token of the SENTENCE) or ENTITY_TOKEN would return identical
        activations for the role and bag questions, and E4's recruitment effect
        would be exactly zero as an artifact of where we looked. The prompt's
        last token is where the model composes its answer and the only position
        the two questions can differ at. See base.RecruitmentActivationSource.
        """
        self._ensure_ready()
        full = f"{sentence} {probe}"
        anchor = len(self._tokenizer(full).input_ids) - 1
        return self._sources_at(self._hidden_at(full, anchor))

    def _sources_at(self, h: Any) -> dict[str, list[float]]:
        """The four probe sources for one hidden state. Shared by RQ1 and E4 so
        the two cannot drift apart in what they mean by "jspace"."""
        component, _ = self._jspace_component(h, self._read_layer())
        remainder = h - component.to(h.dtype)
        basis = self._random_subspace(_RQ1_CAPACITY_SEED, rank=self.config.jspace_k)
        h_lens = h.to(basis.device, basis.dtype)
        projected = (basis @ (basis.T @ h_lens)).to(h.device, h.dtype)
        return {
            "jspace": [float(x) for x in component.float().cpu()],
            "orthogonal": [float(x) for x in remainder.float().cpu()],
            "residual": [float(x) for x in h.float().cpu()],
            "random_subspace": [float(x) for x in projected.float().cpu()],
        }

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

        # device_map="auto" routes through accelerate's sharded dispatch — right
        # for multi-GPU, but it segfaults on a CPU-only Mac. device_map=None
        # takes the classic single-device load path (config.device_map).
        kwargs: dict[str, Any] = {}
        if self.config.device_map is not None:
            kwargs["device_map"] = self.config.device_map
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            import torch

            kwargs["torch_dtype"] = getattr(torch, self.config.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **kwargs)
        self._model.eval()
        # Detached: we only READ the unembedding for lens math. Without this,
        # every J-lens score / pursuit step builds an autograd graph through
        # the (huge) unembedding weight — wasted memory that can OOM a
        # memory-tight GPU, plus a requires_grad scalar-conversion warning.
        self._w_u = self._model.get_output_embeddings().weight.detach()  # (n_vocab, d_model)

    def _load_lens(self) -> None:
        """Load the per-layer averaged Jacobians J_l for the layer band.

        Expected artifact (per the source paper's §2.1 and its companion
        release): one (d_model x d_model) matrix per layer, in a .pt / .npz /
        .safetensors file under config.lens_repo (local path or HF repo id).
        We try the key patterns in _JACOBIAN_KEY_PATTERNS; if none match,
        we fail listing the keys actually present so the mapping can be
        added in one line.

        config.lens_subpath scopes BOTH the download and the file search to
        one model's directory. The published repo holds a lens per model in
        `{model}/jlens/{corpus}/` and totals ~57 GB, so an unscoped
        snapshot_download would pull every other model's Jacobians (and an
        unscoped glob could silently load the WRONG model's lens).
        """
        import torch

        subpath = (self.config.lens_subpath or "").strip("/")
        root = Path(self.config.lens_repo)
        if not root.exists():
            from huggingface_hub import snapshot_download

            root = Path(
                snapshot_download(
                    self.config.lens_repo,
                    allow_patterns=[f"{subpath}/*"] if subpath else None,
                )
            )
        search_root = root / subpath if subpath else root
        if not search_root.exists():
            raise LensFormatError(
                f"lens_subpath {subpath!r} not found under {root}; check "
                "model.lens_subpath against the release layout"
            )
        candidates = [
            p for suffix in _LENS_SUFFIXES for p in sorted(search_root.glob(f"**/*{suffix}"))
        ]
        if not candidates:
            raise LensFormatError(
                f"no {'/'.join(_LENS_SUFFIXES)} lens artifact under {search_root}; "
                "inspect the release and adapt QwenJLensModel._load_lens"
            )
        if not subpath and len(candidates) > 1:
            raise LensFormatError(
                f"{len(candidates)} lens artifacts under {root} and no model.lens_subpath "
                f"to disambiguate: {[p.name for p in candidates[:6]]}... — set "
                "model.lens_subpath so the right model's lens is loaded"
            )
        arrays = self._read_arrays(candidates[0])
        lo, hi = self.config.layer_band  # type: ignore[misc]
        d_model = int(self._w_u.shape[1])
        artifact_d_model = arrays.pop("__d_model__", None)
        if artifact_d_model is not None and int(artifact_d_model) != d_model:
            raise LensFormatError(
                f"lens {candidates[0].name} was fitted for d_model={int(artifact_d_model)} "
                f"but {self.config.model_id} has d_model={d_model} — model.lens_subpath "
                "almost certainly points at another model's lens"
            )
        for layer in range(lo, hi + 1):
            key = next(
                (
                    pattern.format(L=layer)
                    for pattern in _JACOBIAN_KEY_PATTERNS
                    if pattern.format(L=layer) in arrays
                ),
                None,
            )
            if key is None:
                covered = sorted(
                    int(k.removeprefix("layer_"))
                    for k in arrays
                    if k.startswith("layer_") and k.removeprefix("layer_").isdigit()
                )
                if covered:
                    raise LensFormatError(
                        f"layer {layer} is outside the lens artifact's coverage "
                        f"(it holds layers {covered[0]}-{covered[-1]}); narrow "
                        "model.layer_band to that range"
                    )
                raise LensFormatError(
                    f"no Jacobian key found for layer {layer} in {candidates[0].name}; "
                    f"tried {[p.format(L=layer) for p in _JACOBIAN_KEY_PATTERNS]}; "
                    f"available keys: {sorted(arrays)[:12]}... — add the release's "
                    "naming to _JACOBIAN_KEY_PATTERNS"
                )
            matrix = arrays[key]
            if matrix.shape != (d_model, d_model):
                raise LensFormatError(
                    f"Jacobian {key!r} has shape {matrix.shape}, expected "
                    f"({d_model}, {d_model}) for this model"
                )
            self._jacobians[layer] = torch.as_tensor(
                matrix, device=self._lens_device(), dtype=self._w_u.dtype
            )

    @staticmethod
    def _read_arrays(path: Path) -> dict[str, Any]:
        """Load a lens artifact as a flat {key: matrix} mapping.

        The released .pt nests its Jacobians one level down under "J", keyed
        by INT layer index, alongside scalar metadata ("d_model",
        "source_layers", "n_prompts"). We flatten that to the flat
        "layer_{L}" spelling the key patterns expect, and pass d_model
        through as "__d_model__" so _load_lens can reject a lens fitted for
        a different model. Values stay torch tensors — torch.as_tensor
        handles both those and numpy arrays.
        """
        if path.suffix == ".pt":
            import torch

            # weights_only=True: the artifact is plain tensors + scalars, so
            # never execute pickle code from a downloaded file.
            # mmap=True: the Qwen release is 3.3 GB of Jacobians but a run
            # touches only its layer band, so map the file and let the OS page
            # in the few layers actually read. Falls back to a full load for
            # artifacts not saved in the zipfile format mmap requires.
            try:
                obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            except (RuntimeError, ValueError):
                obj = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(obj, dict):
                raise LensFormatError(
                    f"{path.name} holds a {type(obj).__name__}, expected a dict of Jacobians"
                )
            nested = obj.get("J")
            if not isinstance(nested, dict):
                raise LensFormatError(
                    f"{path.name} has no 'J' mapping of per-layer Jacobians; "
                    f"top-level keys: {sorted(map(str, obj))[:12]}"
                )
            arrays: dict[str, Any] = {f"layer_{int(layer)}": m for layer, m in nested.items()}
            if "d_model" in obj:
                arrays["__d_model__"] = obj["d_model"]
            return arrays
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                return {k: data[k] for k in data.files}
        from safetensors.numpy import load_file

        return load_file(path)

    # ------------------------------------------------------------------ #
    # J-lens primitives                                                  #
    # ------------------------------------------------------------------ #

    def _lens_device(self) -> Any:
        """Device the lens algebra runs on.

        Multi-GPU correctness. Under `device_map="auto"` accelerate spreads the
        model across GPUs, so there is no single "model device": `_w_u` follows
        the lm_head (typically the LAST GPU) while `_device()` reports the first
        parameter's device (typically the FIRST). Hidden states arrive from
        hooks on whichever GPU owns the hooked layer — a third device again.

        Pinning the Jacobians to `_device()` therefore produced
        "Expected all tensors to be on the same device" the moment the band
        landed on a different shard than the embeddings. Invisible on one GPU;
        immediate on several.

        `_w_u` is the anchor because it is the largest lens tensor (~2.5 GB)
        and already resident — everything else is moved to meet it, and hidden
        states are a few KB, so the copies are free.
        """
        return self._w_u.device

    def _lens_scores(self, h: Any, layer: int) -> Any:
        """Vocab-length lens scores of h at `layer`: W_U (J_l h). First-order
        (the paper's norm() is omitted for direction work, as is standard).

        Callers must pass an h already on `_lens_device()`; the wrappers around
        `_jspace_component` / `_edit_delta` are what guarantee that.
        """
        return self._w_u @ (self._jacobians[layer] @ h.to(self._w_u.dtype))

    def _jlens_vector(self, token_id: int, layer: int) -> Any:
        """The J-lens vector v_t = J_l^T W_U[t]: the residual-stream direction
        whose inner product with h gives token t's lens score. Cached."""
        key = (layer, token_id)
        if key not in self._jlens_vectors:
            self._jlens_vectors[key] = self._jacobians[layer].T @ self._w_u[token_id]
        return self._jlens_vectors[key]

    def _jspace_component(self, h: Any, layer: int, k: int | None = None) -> Any:
        """Sparse J-space component of h, returned on h's OWN device.

        Thin device-shim over the algebra: the lens tensors live on
        `_lens_device()` and `h` arrives from wherever its layer is sharded, so
        the computation happens on the lens device and the result comes back to
        the caller's. See `_lens_device` for why the three can differ.
        """
        component, chosen = self._jspace_component_on_lens(
            h.to(self._lens_device()), layer, k
        )
        return component.to(h.device), chosen

    def _jspace_component_on_lens(self, h: Any, layer: int, k: int | None = None) -> Any:
        """Matching-pursuit approximation of the paper's gradient pursuit
        (§2.3). Greedily selects the token whose J-lens vector scores highest
        on the current residual, refits the active set by least squares, clamps
        negative coefficients (the nonnegativity constraint), and stops at k
        atoms or when no positive score remains. Returns (component,
        active_token_ids). `h` must already be on `_lens_device()`."""
        import torch

        k = k if k is not None else self.config.jspace_k
        residual = h.to(self._w_u.dtype)
        chosen: list[int] = []
        component = torch.zeros_like(residual)
        for _ in range(k):
            scores = self._lens_scores(residual, layer)
            if chosen:
                scores[torch.tensor(chosen, device=scores.device)] = float("-inf")
            token_id = int(torch.argmax(scores))
            if float(scores[token_id]) <= 0.0:
                break
            chosen.append(token_id)
            V = torch.stack([self._jlens_vector(t, layer) for t in chosen], dim=1)
            coef = torch.linalg.lstsq(V.float(), h.float().unsqueeze(1)).solution.squeeze(1)
            coef = torch.clamp(coef, min=0.0)
            component = (V.float() @ coef).to(residual.dtype)
            residual = h.to(residual.dtype) - component
        return component, chosen

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
                    denom = float(torch.linalg.vector_norm(h.float()))
                    if denom > 0.0:
                        model._edit_magnitudes.append(
                            float(torch.linalg.vector_norm(delta.float())) / denom
                        )
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
        """Residual-stream delta for one edit at one layer, on h's OWN device.

        Device shim, as for `_jspace_component`: hooks hand us `h` on the
        hooked layer's shard, the lens/direction tensors live on
        `_lens_device()`, and the delta has to go back where it came from so
        the hook can add it. See `_lens_device`.
        """
        return self._edit_delta_on_lens(
            edit, site, layer, h.to(self._lens_device())
        ).to(h.device)

    def _edit_delta_on_lens(
        self, edit: EditSpec, site: InjectionSite, layer: int, h: Any
    ) -> Any:
        """The edit math (interventions.edits is the canonical description).
        `h` must already be on `_lens_device()`."""
        import torch

        if edit.edit_type is EditType.ABLATE_JSPACE:
            # RQ2: zero the projection onto the span of the top-k most
            # strongly active J-lens vectors (paper §3.5.2, k = ablate_k).
            _, active = self._jspace_component(h, layer, k=self.config.ablate_k)
            if not active:
                return torch.zeros_like(h)
            V = torch.stack([self._jlens_vector(t, layer) for t in active], dim=1).float()
            projection = V @ (torch.linalg.pinv(V) @ h.float())
            return -projection.to(h.dtype)
        if edit.edit_type is EditType.ABLATE_RANDOM_SUBSPACE:
            # RQ2 capacity control: project out a seeded random orthonormal
            # subspace of the SAME dimension count (ablate_k) as the J-space
            # ablation removes.
            basis = self._random_subspace(edit.seed or 0)
            h_cast = h.to(basis.dtype)
            return -(basis @ (basis.T @ h_cast)).to(h.dtype)
        if edit.edit_type is EditType.IDENTITY_SWAP:
            # Coordinate swap in lens coordinates (paper §2.5), using the
            # J-lens vectors of the two tokens directly. Pure linear algebra
            # over cached J_l and W_U — safe inside hooks.
            V, pinv_V = self._swap_operator(edit.swap_source, edit.swap_target, layer)  # type: ignore[arg-type]
            c = pinv_V @ h.float()
            alpha = 1.0 if edit.alpha is None else float(edit.alpha)
            swapped = alpha * torch.stack([c[1], c[0]])
            return (V @ (swapped - c)).to(h.dtype)

        # Direction pushes: h += sign * coefficient * unit(r). The fitted r is
        # already a residual-space unit direction (a diff-of-means of J-space
        # components); re-normalizing keeps every push — real or control —
        # perturbing the stream by exactly `coefficient`.
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
        r = self._push_direction(edit, site)
        r_t = torch.as_tensor(r, device=h.device, dtype=torch.float32)
        return (sign * coefficient * (r_t / torch.linalg.vector_norm(r_t))).to(h.dtype)

    def _push_direction(self, edit: EditSpec, site: InjectionSite) -> np.ndarray:
        """Unit direction (residual space) for a push edit."""
        if edit.edit_type is EditType.RANDOM_DIRECTION:
            rng = np.random.default_rng(edit.seed)
            raw = rng.standard_normal(int(self._w_u.shape[1]))
            return raw / np.linalg.norm(raw)
        variant = {
            EditType.ROLE_PUSH: self.direction_variant,
            EditType.NULL_NON_PARTICIPANT: "fitted",
            EditType.SHUFFLED_LABEL_DIRECTION: "shuffled",
        }[edit.edit_type]
        directions = self._site_directions(site)
        r = directions.direction(edit.entity, variant)  # type: ignore[arg-type]
        if r.shape[0] != int(self._w_u.shape[1]):
            raise ValueError(
                f"fitted direction dim {r.shape[0]} != d_model {int(self._w_u.shape[1])}; "
                "refit directions against this model (scripts/fit_directions.py)"
            )
        return r

    def _swap_operator(self, source: str, target: str, layer: int) -> Any:
        """(V, pinv(V)) for an identity swap at one layer, cached per
        (pair, layer) — the pseudoinverse is identical across trials."""
        import torch

        key = (source, target, layer)
        if key not in self._swap_operators:
            v_s = self._jlens_vector(self._single_token_id(source), layer)
            v_t = self._jlens_vector(self._single_token_id(target), layer)
            V = torch.stack([v_s, v_t], dim=1).float()
            self._swap_operators[key] = (V, torch.linalg.pinv(V))
        return self._swap_operators[key]

    def _site_directions(self, site: InjectionSite) -> FittedDirections:
        if site not in self._directions:
            self._directions[site] = load_directions(self.directions_dir, site)
        return self._directions[site]

    def _random_subspace(self, seed: int, rank: int | None = None) -> Any:
        """Seeded random orthonormal (d_model, rank) basis, cached per (seed, rank).

        `rank` defaults to ablate_k, the RQ2 ablation control's dimension. RQ1's
        capacity control passes jspace_k instead: it must match the rank of the
        subspace it is being compared against, and jspace_k != ablate_k.
        """
        import torch

        k = self.config.ablate_k if rank is None else rank
        key = (seed, k)
        if key not in self._random_subspaces:
            d_model = int(self._w_u.shape[1])
            rng = np.random.default_rng(seed)
            raw = rng.standard_normal((d_model, k))
            q, _ = np.linalg.qr(raw)
            self._random_subspaces[key] = torch.as_tensor(
                q, device=self._lens_device(), dtype=torch.float32
            )
        return self._random_subspaces[key]

    # ------------------------------------------------------------------ #
    # Forward-capture and indexing helpers                               #
    # ------------------------------------------------------------------ #

    def _hidden_at(self, text: str, anchor: int) -> Any:
        """Residual activation at the read layer for token position `anchor`
        of an edit-free forward over `text`."""
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
