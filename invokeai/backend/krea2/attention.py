"""Memory-efficient attention processor for the Krea-2 transformer.

The stock ``Krea2AttnProcessor`` calls ``scaled_dot_product_attention`` with ``enable_gqa=True`` (Krea-2 uses
grouped-query attention: 48 query heads, 12 key/value heads). PyTorch's fused flash / memory-efficient SDPA
kernels do **not** support ``enable_gqa``, so this forces the *math* backend, which materializes the full
``[heads, seq, seq]`` score matrix. At 1280x720 (3600 image tokens) that is ~5.7 GB **per attention**, and it
grows O(seq^2) — ~40 GB at 2560x1440 — so generation OOMs or the cache offloads the transformer to RAM.

This processor lets a fused SDPA kernel take the call instead; those are O(seq) in memory, and InvokeAI's
process-wide priority order (`util.sdpa_scope.DEFAULT_SDPA_PRIORITY`) decides which one runs. Measured: the same
3600-token attention drops from ~5.7 GB to ~0.19 GB.

The premise above has since become build-dependent. **cuDNN does serve grouped-query attention**, mask
included, so where it is available the K/V heads can stay at 12 and the expansion is pure waste --
``repeat_interleave`` allocates two tensors four times larger than the originals, every call. Where no
fused kernel takes GQA (the memory-efficient kernel refuses it outright, flash refuses the additive
mask, and ROCm has no cuDNN at all) the expansion is still the only thing standing between this
processor and the ~5.7 GB math path. So it is now conditional, decided per call by asking the
dispatcher rather than by assuming -- see ``_serves_grouped_query_attention``, which reads the same
backend selection the call it answers for will run under, because that is the state the answer depends on.

The math is otherwise identical to ``Krea2AttnProcessor`` (q/k RMSNorm, rotary embeddings, sigmoid output gate).
"""

import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, Protocol

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from torch.nn.attention import SDPBackend

from invokeai.backend.krea2.reference_edit import Krea2ReferenceKVCache, Krea2ReferenceKVMode
from invokeai.backend.krea2.style_reference import (
    Krea2StyleReferenceMode,
    Krea2StyleReferenceState,
    apply_style_reference,
    capture_style_reference,
)
from invokeai.backend.util.attention import sdpa_runs_fused
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.sdpa_scope import sdpa_policy

logger = InvokeAILogger.get_logger(__name__)

# Opt-in override, for measuring one backend against another and for support questions. Unset -- the
# only state a user ever sees by default -- opens no SDPA window: attention runs under the process-wide order.
KREA2_SDPA_BACKEND_ENV_VAR = "INVOKE_KREA2_SDPA_BACKEND"
# Names the process-wide order, so a benchmarking run can state which policy it used. Like unset, it opens no
# window; unlike unset, it counts as an override, which turns on the per-step timing and keeps SageAttention off.
_PROCESS_ORDER = "default"
_EXCLUSIVE_BACKENDS = {
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "flash": SDPBackend.FLASH_ATTENTION,
    "math": SDPBackend.MATH,
}


@dataclass(frozen=True)
class Krea2SdpaBackends:
    """The one SDPA backend Krea-2 attention is pinned to, or None to run under the process-wide order."""

    exclusive: SDPBackend | None = None
    override: str | None = None

    def describe(self) -> str:
        if self.exclusive is None:
            return "the process-wide SDPA priority order"
        return f"sdpa_policy([{self.exclusive.name}])"


def resolve_krea2_sdpa_backends(raw_override: str | None = None) -> Krea2SdpaBackends:
    """Resolve the SDPA backend choice, honouring KREA2_SDPA_BACKEND_ENV_VAR.

    The exclusive modes are the point of the override: an unavailable backend raises visibly instead
    of quietly degrading to math, so a run that completes proves that kernel served it. The value is
    read from the process environment, so every session in the process asks for the same window.
    """
    raw = os.environ.get(KREA2_SDPA_BACKEND_ENV_VAR) if raw_override is None else raw_override
    if raw is None or not raw.strip():
        return Krea2SdpaBackends()

    value = raw.strip().lower()
    if value == _PROCESS_ORDER:
        return Krea2SdpaBackends(override=value)
    if value in _EXCLUSIVE_BACKENDS:
        return Krea2SdpaBackends(exclusive=_EXCLUSIVE_BACKENDS[value], override=value)

    valid = ", ".join([*sorted(_EXCLUSIVE_BACKENDS), _PROCESS_ORDER])
    raise ValueError(f"{KREA2_SDPA_BACKEND_ENV_VAR}={raw!r} is not a valid value. Valid values: {valid}.")


@dataclass
class Krea2RegionalPromptingState:
    """Mutable per-forward regional attention state shared by Krea-2 transformer-block processors."""

    attention_mask: torch.Tensor | None = None

    def set_attention_mask(self, attention_mask: torch.Tensor | None) -> None:
        self.attention_mask = attention_mask


class Krea2MemoryEfficientAttnProcessor:
    """Drop-in replacement for ``Krea2AttnProcessor`` that avoids the ``enable_gqa`` math fallback."""

    def __init__(
        self,
        regional_prompting_state: Krea2RegionalPromptingState | None = None,
        sdpa_backends: Krea2SdpaBackends | None = None,
        style_reference_state: Krea2StyleReferenceState | None = None,
        block_index: int | None = None,
        reference_kv_cache: Krea2ReferenceKVCache | None = None,
    ) -> None:
        self.regional_prompting_state = regional_prompting_state
        # Resolved once per generation and handed down, not read per attention call.
        self.sdpa_backends = sdpa_backends if sdpa_backends is not None else resolve_krea2_sdpa_backends()
        # Only set on blocks selected for style reference; None everywhere else, so an inactive block costs
        # nothing beyond the identity check below.
        self.style_reference_state = style_reference_state
        self.block_index = block_index
        # Only set on the main blocks while an edit reference runs in K/V-cache mode; see reference_edit.py.
        self.reference_kv_cache = reference_kv_cache

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.regional_prompting_state is not None and self.regional_prompting_state.attention_mask is not None:
            regional_attention_mask = self.regional_prompting_state.attention_mask
            if regional_attention_mask.shape != (hidden_states.shape[1], hidden_states.shape[1]):
                raise ValueError(
                    f"Krea-2 regional attention mask shape {tuple(regional_attention_mask.shape)} does not match "
                    f"the transformer sequence length {hidden_states.shape[1]}."
                )
            attention_mask = (
                regional_attention_mask if attention_mask is None else attention_mask & regional_attention_mask
            )

        query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        gate = attn.to_gate(hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # [B, S, H, D] -> [B, H, S, D] for scaled_dot_product_attention.
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # The edit-reference K/V cache hooks in at the same point as style reference below: post-RoPE, so the
        # cached keys carry the references' own rotation, and pre GQA expansion. The two never run together.
        reference_kv = self.reference_kv_cache
        if reference_kv is not None and self.block_index is not None:
            if reference_kv.mode is Krea2ReferenceKVMode.CAPTURE:
                reference_kv.store(self.block_index, key, value)
            elif reference_kv.mode is Krea2ReferenceKVMode.APPEND:
                key, value = reference_kv.extend(self.block_index, key, value, attention_mask)

        # Style reference hooks in here: after RoPE (so the captured keys carry their rotation) but before
        # the GQA head expansion, which keeps the retained cache 4x smaller. See style_reference.py.
        style_state = self.style_reference_state
        style_mode = Krea2StyleReferenceMode.OFF if style_state is None else style_state.mode
        injection = None
        if style_state is not None and self.block_index is not None:
            if style_mode is Krea2StyleReferenceMode.CAPTURE:
                capture_style_reference(style_state, self.block_index, query, key, value)
            elif style_mode is Krea2StyleReferenceMode.INJECT:
                injection = apply_style_reference(style_state, self.block_index, query, key, value)

        # Only an exclusive override opens a window; otherwise the call runs under the process-wide order.
        # Both of these belong inside it, and in this order. `_serves_grouped_query_attention` asks the
        # dispatcher about the process-global selection the window sets, so asking outside it answers for
        # a different policy than the call will run under -- and the expansion, if needed, has to happen
        # after the answer.
        exclusive = self.sdpa_backends.exclusive
        with nullcontext() if exclusive is None else sdpa_policy([exclusive]):

            def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
                # Krea-2 has 48 query heads over 12 K/V heads. Passing them through as they are avoids
                # allocating two tensors four times larger, but only some kernels serve that shape -- so
                # ask, do not assume. Where the answer is no, expand as before: that is what keeps this
                # off the math path, which materialises the full [heads, seq, seq] score matrix.
                enable_gqa = attn.num_heads != attn.num_kv_heads and self._serves_grouped_query_attention(q, k, v, mask)
                if attn.num_heads != attn.num_kv_heads and not enable_gqa:
                    repeats = attn.num_heads // attn.num_kv_heads
                    k = k.repeat_interleave(repeats, dim=1)
                    v = v.repeat_interleave(repeats, dim=1)
                return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

            if injection is None:
                hidden_states = attend(query, key, value, attention_mask)
            else:
                # The reference keys/values are appended along the token axis, so a regional mask has to grow
                # with them.
                hidden_states = attend(
                    injection.query,
                    injection.key,
                    injection.value,
                    style_state.pad_attention_mask(attention_mask),
                )
                if injection.attention_mix < 1.0:
                    # Blend against the same (AdaIN'd) query/key attending to the target's own tokens only.
                    # At the default style_strength of 1.0 this branch is dead and the second attention is skipped.
                    sequence_length = injection.query.shape[2]
                    native = attend(
                        injection.query,
                        injection.key[:, :, :sequence_length, :],
                        injection.value[:, :, :sequence_length, :],
                        attention_mask,
                    )
                    hidden_states = native * (1.0 - injection.attention_mix) + hidden_states * injection.attention_mix

        # [B, H, S, D] -> [B, S, H, D] -> [B, S, H*D], matching Krea2AttnProcessor's output layout.
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states * torch.sigmoid(gate)
        return attn.to_out[0](hidden_states)

    def _serves_grouped_query_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> bool:
        """Whether a fused kernel will take the unexpanded K/V heads for this exact call.

        Measured on the Krea-2 shape ([1, 48q/12kv, 4608, 128], bf16): cuDNN serves it, masked and
        unmasked alike; the memory-efficient kernel refuses grouped-query attention outright, and
        flash refuses the additive mask the regional-prompting blocks pass. On a build without cuDNN
        -- ROCm -- a masked call therefore has no fused kernel left, and answering "yes" there would
        drop it onto `math` at roughly 9 GB. That is the failure this whole processor exists to
        avoid, so the answer is asked of the dispatcher rather than inferred from the platform.

        Asked as the dispatcher's own choice (`util.attention.sdpa_backend_choice`), not as "can some
        kernel serve it": which kernel the call gets depends on the enable flags *and* the priority
        order. A kernel that can serve the call but ranks below math -- cuDNN under torch's own order,
        or after a cuDNN failure demoted it -- leaves the unexpanded call on math. The choice covers
        both, whichever window is open: this processor's exclusive override, or a foreign one, which
        the call honours just the same.

        Must be called inside the window the attention call will run under, when there is one, and is
        not cached: that state is process-global and a cache key cannot honestly enumerate it. A stale
        "no" costs the expansion this function exists to skip for the rest of the generation, with
        nothing logged. The query costs microseconds (2.5us measured on CPU for this shape) against
        ~3.5ms for the attention itself, and stays silent under an exclusive override that leaves the
        call no kernel (see `sdpa_backend_choice`).
        """
        if not query.is_cuda:
            # The grouped path is measured on CUDA (ROCm included) only. Elsewhere the expansion is
            # correct and merely costs memory.
            return False
        return sdpa_runs_fused(query, key, value, attention_mask, enable_gqa=True)


class _Krea2AttentionProcessorContainer(Protocol):
    @property
    def attn_processors(self) -> dict[str, object]: ...


def build_krea2_attention_processors(
    transformer: _Krea2AttentionProcessorContainer,
    regional_prompting_state: Krea2RegionalPromptingState,
    style_reference_state: Krea2StyleReferenceState | None = None,
    style_reference_blocks: Iterable[int] | None = None,
    reference_kv_cache: Krea2ReferenceKVCache | None = None,
) -> dict[str, Krea2MemoryEfficientAttnProcessor]:
    """Build processors that apply regional masks to alternating main transformer blocks only.

    Style reference runs over its own, independent band of blocks (upstream's default is 7-27, i.e. both
    parities), so it gets a second state object rather than sharing the regional one. Leaving both style
    arguments unset reproduces the pre-style behaviour exactly.

    The edit-reference K/V cache, when given, goes to every main block: each one captures and appends its own.

    The text-fusion blocks do not match the main-block pattern, so they never receive any of these states. That
    is correct: they only ever see text tokens.
    """

    style_blocks = frozenset(style_reference_blocks or ())

    sdpa_backends = resolve_krea2_sdpa_backends()
    if sdpa_backends.override is not None:
        # Once per generation, not once per attention call.
        logger.info(
            f"Krea-2 SDPA backend override active: {KREA2_SDPA_BACKEND_ENV_VAR}={sdpa_backends.override} "
            f"-> {sdpa_backends.describe()}"
        )

    processors: dict[str, Krea2MemoryEfficientAttnProcessor] = {}
    for name in transformer.attn_processors:
        match = re.fullmatch(r"transformer_blocks\.(\d+)\.attn\.processor", name)
        block_index = int(match.group(1)) if match is not None else None
        regional = regional_prompting_state if block_index is not None and block_index % 2 == 0 else None
        style = (
            style_reference_state
            if style_reference_state is not None and block_index is not None and block_index in style_blocks
            else None
        )
        processors[name] = Krea2MemoryEfficientAttnProcessor(
            regional_prompting_state=regional,
            sdpa_backends=sdpa_backends,
            style_reference_state=style,
            block_index=block_index,
            reference_kv_cache=reference_kv_cache if block_index is not None else None,
        )
    return processors
