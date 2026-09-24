"""Training-free style reference (shared-KV reference attention) for Krea-2.

Ported from https://github.com/nkxx188/ComfyUI-Krea2-StyleTransfer (MIT). The technique transfers the
*look* of a reference image without transferring its content, and without any extra weights: the
reference image is run through the transformer alongside the target, and in a band of transformer blocks
the target queries additionally attend to the reference's image-token keys/values.

InvokeAI runs this as **two passes** rather than upstream's doubled batch:

1. ``CAPTURE`` -- the reference latent goes through the transformer alone. The processors of the styled
   blocks stash the post-RoPE image-token K/V (plus Q/K token statistics).
2. ``INJECT`` -- the target pass splices those K/V onto its own.

This is mathematically identical: upstream's reference rows never attend to the target (its ``out_ref``
uses only reference q/k/v), the cross-batch AdaIN writes only into the target rows, and no reduction
spans both batch halves. Splitting the passes also sidesteps a real problem -- ``krea2_denoise`` *strips*
padded text tokens instead of masking them, so the target and reference prompts have different sequence
lengths and could not be batch-concatenated without re-padding both.

Two deliberate deviations from upstream, both verified equivalent:

* K/V are captured **before** the GQA head expansion (12 kv heads, not 48). ``repeat_interleave``
  duplicates each head, so per-``(head, dim)`` token statistics are identical across a group, and the
  frequency scale vector is per-``dim`` and therefore head-invariant. This is 4x smaller -- at 2560x1440
  it is the difference between 1.7 GiB and 6.9 GiB of retained cache.
* The RoPE axis dims come from ``transformer.config.axes_dims_rope`` rather than being re-derived from
  the head dim by a heuristic. For Krea-2 both give ``(32, 48, 48)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Sequence

import torch

# Krea-2's transformer has 28 main blocks. Styling the earliest ones destroys structure, so upstream's
# default band starts at 7.
KREA2_NUM_BLOCKS = 28
KREA2_DEFAULT_STYLE_BLOCKS = "7-27"

Krea2StyleValueMode = Literal["target", "raw_reference", "ref_mean", "target_adain", "target_adain_plus_ref"]

_ADAIN_EPS = 1e-6


class Krea2StyleReferenceMode(Enum):
    """Which side of the two-pass scheme the shared state is currently driving."""

    OFF = "off"
    CAPTURE = "capture"
    INJECT = "inject"


@dataclass(frozen=True)
class Krea2StyleReferenceSettings:
    """Upstream's ``recommended`` preset, the only combination its README claims is stable.

    ``style_strength`` is a master knob rather than a plain mix: upstream pulls ``high_scale_start``,
    ``low_scale_end`` and ``adain_strength`` toward neutral as it drops, *and* uses it as the
    native/styled attention mix. See :func:`resolve_effective_settings`.

    Note that at the recommended values ``value_adain_strength`` has no effect: ``value_mode`` is
    ``target_adain_plus_ref`` and ``ref_value_mix`` is 1.0, so the reference value path returns the raw
    reference values and discards the AdaIN'd blend it would otherwise mix in. It stays exposed because it
    becomes live as soon as ``ref_value_mix`` is lowered.
    """

    style_strength: float = 1.0
    ref_k_strength: float = 1.06
    adain_strength: float = 0.85
    value_mode: Krea2StyleValueMode = "target_adain_plus_ref"
    value_adain_strength: float = 0.65
    ref_value_mix: float = 1.0
    high_scale_start: float = 1.04
    high_scale_end: float = 0.0
    low_scale_start: float = 1.0
    low_scale_end: float = 1.10
    beta: float = 2.5


@dataclass(frozen=True)
class Krea2StyleReferenceEffectiveSettings:
    """:class:`Krea2StyleReferenceSettings` after ``style_strength`` has been folded in."""

    ref_k_strength: float
    adain_strength: float
    value_mode: Krea2StyleValueMode
    value_adain_strength: float
    ref_value_mix: float
    high_scale_start: float
    high_scale_end: float
    low_scale_start: float
    low_scale_end: float
    beta: float
    attention_mix: float


def resolve_effective_settings(settings: Krea2StyleReferenceSettings) -> Krea2StyleReferenceEffectiveSettings:
    """Fold ``style_strength`` into the parameters it modulates.

    Mirrors upstream exactly, including its asymmetry: only ``high_scale_start`` and ``low_scale_end`` are
    modulated (not their ``*_end`` / ``*_start`` counterparts), and each factor saturates at a different
    point. At ``style_strength == 1.0`` every effective value equals its configured value.
    """
    strength = max(0.0, float(settings.style_strength))
    return Krea2StyleReferenceEffectiveSettings(
        ref_k_strength=max(0.0, float(settings.ref_k_strength)),
        adain_strength=max(0.0, min(1.0, float(settings.adain_strength) * min(strength, 1.25))),
        value_mode=settings.value_mode,
        value_adain_strength=max(0.0, min(1.5, float(settings.value_adain_strength))),
        ref_value_mix=max(0.0, min(1.0, float(settings.ref_value_mix))),
        high_scale_start=1.0 + (float(settings.high_scale_start) - 1.0) * min(strength, 1.5),
        high_scale_end=float(settings.high_scale_end),
        low_scale_start=float(settings.low_scale_start),
        low_scale_end=1.0 + (float(settings.low_scale_end) - 1.0) * strength,
        beta=float(settings.beta),
        attention_mix=max(0.0, min(1.0, strength)),
    )


def parse_block_spec(spec: str, num_blocks: int = KREA2_NUM_BLOCKS) -> frozenset[int]:
    """Parse a block selection like ``"7-27"``, ``"7-27,3"`` or ``"5"`` into block indices.

    Unlike upstream this validates against the real block count, so a typo fails at graph time instead of
    silently styling nothing.
    """
    active: set[int] = set()
    for raw_part in str(spec or "").replace(";", ",").split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, _, end_str = part.partition("-")
            try:
                start, end = int(start_str.strip()), int(end_str.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid Krea-2 style block range {part!r}.") from exc
            if end < start:
                raise ValueError(f"Invalid Krea-2 style block range {part!r}: end is before start.")
            active.update(range(start, end + 1))
        else:
            try:
                active.add(int(part))
            except ValueError as exc:
                raise ValueError(f"Invalid Krea-2 style block index {part!r}.") from exc

    if not active:
        raise ValueError(f"Krea-2 style block spec {spec!r} selects no blocks.")
    out_of_range = sorted(index for index in active if index < 0 or index >= num_blocks)
    if out_of_range:
        raise ValueError(
            f"Krea-2 style block spec {spec!r} selects blocks {out_of_range}, but the transformer only has "
            f"{num_blocks} blocks (0-{num_blocks - 1})."
        )
    return frozenset(active)


def lerp_scales(settings: Krea2StyleReferenceEffectiveSettings, progress: float) -> tuple[float, float]:
    """Interpolate the high/low frequency scales for a point in the sampling schedule."""
    progress = max(0.0, min(1.0, float(progress)))
    high = settings.high_scale_start + (settings.high_scale_end - settings.high_scale_start) * progress
    low = settings.low_scale_start + (settings.low_scale_end - settings.low_scale_start) * progress
    return high, low


def build_rope_scale_vector(
    axes_dims: Sequence[int],
    high_scale: float,
    low_scale: float,
    beta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Per-head-dim multiplier applied to the reference keys, shaped by the RoPE frequency layout.

    ``Krea2RotaryPosEmbed`` builds its embedding by concatenating one ``get_1d_rotary_pos_embed`` block
    per position axis with ``repeat_interleave_real=True``. Within an axis the *first* pair is therefore
    the highest frequency, and each frequency occupies two consecutive dims -- hence the curve over pair
    index from ``high_scale`` to ``low_scale``, and the ``repeat_interleave(2)``.

    Axis 0 is the temporal axis. Every Krea-2 token sits at t=0, so its rotation is the identity and there
    is no frequency structure to shape; it is held flat at ``low_scale``.

    With the default ``high_scale_end=0.0`` the highest-frequency bands of the reference key decay to zero
    across the schedule, which is what makes the reference contribute position-agnostic style rather than
    spatially-located content.
    """
    head_dim = int(sum(int(dim) for dim in axes_dims))
    if head_dim <= 0:
        raise ValueError(f"axes_dims must sum to a positive head dim, got {list(axes_dims)}.")

    def curve(pairs: int) -> torch.Tensor:
        if pairs <= 1:
            x = torch.zeros(max(pairs, 1), device=device, dtype=torch.float32)
        else:
            x = torch.linspace(0.0, 1.0, pairs, device=device, dtype=torch.float32)
        return float(high_scale) + (float(low_scale) - float(high_scale)) * x.pow(float(beta))

    pieces: list[torch.Tensor] = []
    has_temporal_axis = len(axes_dims) >= 2
    for axis_index, axis_dim in enumerate(int(dim) for dim in axes_dims):
        pairs = axis_dim // 2
        if pairs <= 0:
            pieces.append(torch.ones(axis_dim, device=device, dtype=dtype))
            continue
        if has_temporal_axis and axis_index == 0:
            pair_scales = torch.full((pairs,), float(low_scale), device=device, dtype=torch.float32)
        else:
            pair_scales = curve(pairs)
        pieces.append(pair_scales.to(dtype=dtype).repeat_interleave(2))
        if axis_dim % 2:
            pieces.append(torch.ones(1, device=device, dtype=dtype))

    return torch.cat(pieces, dim=0)[:head_dim]


def _token_mean_std(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean/std over the token axis of a ``[B, H, L, D]`` tensor, with the variance accumulated in fp32."""
    mean = x.mean(dim=2, keepdim=True)
    std = x.float().var(dim=2, keepdim=True, unbiased=False).add(_ADAIN_EPS).sqrt().to(x.dtype)
    return mean, std


def _adain_to_stats(
    target: torch.Tensor, style_mean: torch.Tensor, style_std: torch.Tensor, strength: float
) -> torch.Tensor:
    """Blend ``target`` toward the given per-``(head, dim)`` style statistics over the token axis."""
    alpha = max(0.0, min(1.0, float(strength)))
    if alpha <= 0.0:
        return target
    target_mean, target_std = _token_mean_std(target)
    styled = (target - target_mean) / target_std * style_std + style_mean
    if alpha >= 1.0:
        return styled
    return target * (1.0 - alpha) + styled * alpha


def _build_reference_value(
    target_value: torch.Tensor,
    reference_value: torch.Tensor,
    settings: Krea2StyleReferenceEffectiveSettings,
) -> torch.Tensor:
    """Construct the value vectors the reference keys are paired with.

    At the recommended settings (``target_adain_plus_ref`` with ``ref_value_mix=1.0``) this is exactly
    ``reference_value``; the other modes exist for tuning.
    """
    mode = settings.value_mode
    if mode == "raw_reference":
        return reference_value
    if mode == "target":
        base = target_value
    elif mode == "ref_mean":
        base = reference_value.mean(dim=2, keepdim=True).expand_as(reference_value)
    else:
        reference_mean, reference_std = _token_mean_std(reference_value)
        base = _adain_to_stats(target_value, reference_mean, reference_std, settings.value_adain_strength)
    if mode == "target_adain_plus_ref":
        mix = settings.ref_value_mix
        return base * (1.0 - mix) + reference_value * mix
    return base


@dataclass
class Krea2StyleReferenceBlockCache:
    """One styled block's captured reference tensors.

    ``reference_key`` / ``reference_value`` are ``[B, kv_heads, image_seq_len, head_dim]`` and are the
    dominant memory cost. The statistics are ``[B, heads, 1, head_dim]`` and negligible.
    """

    reference_key: torch.Tensor
    reference_value: torch.Tensor
    query_mean: torch.Tensor
    query_std: torch.Tensor
    key_mean: torch.Tensor
    key_std: torch.Tensor


@dataclass
class Krea2StyleInjection:
    """What the processor should attend with, and how to blend it with the unstyled result."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention_mix: float


@dataclass
class Krea2StyleReferenceState:
    """Mutable state shared between the denoise loop and the styled blocks' attention processors.

    It lives on the attention processors, which stay installed on the *cached* transformer after the
    invocation ends -- so :meth:`clear` must be wired to the denoise node's exit stack.
    """

    settings: Krea2StyleReferenceEffectiveSettings
    image_seq_len: int
    axes_dims_rope: tuple[int, ...]
    mode: Krea2StyleReferenceMode = Krea2StyleReferenceMode.OFF
    progress: float = 0.0
    _cache: dict[int, Krea2StyleReferenceBlockCache] = field(default_factory=dict, repr=False)
    _scale_vector: torch.Tensor | None = field(default=None, repr=False)
    _padded_masks: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list, repr=False)

    def begin_capture(self) -> None:
        self._cache.clear()
        self.mode = Krea2StyleReferenceMode.CAPTURE

    def begin_inject(self, progress: float) -> None:
        if not self._cache:
            raise RuntimeError("Krea-2 style reference: inject requested before any reference pass was captured.")
        self.progress = max(0.0, min(1.0, float(progress)))
        self._scale_vector = None
        self.mode = Krea2StyleReferenceMode.INJECT

    def disable(self) -> None:
        self.mode = Krea2StyleReferenceMode.OFF

    def clear(self) -> None:
        """Drop every retained tensor. Wired to the denoise node's exit stack."""
        self.mode = Krea2StyleReferenceMode.OFF
        self._cache.clear()
        self._scale_vector = None
        self._padded_masks.clear()

    def store(self, block_index: int, cache: Krea2StyleReferenceBlockCache) -> None:
        self._cache[block_index] = cache

    def get(self, block_index: int) -> Krea2StyleReferenceBlockCache:
        try:
            return self._cache[block_index]
        except KeyError:
            raise RuntimeError(
                f"Krea-2 style reference: block {block_index} was not captured during the reference pass. "
                "The reference and target passes must run over the same set of styled blocks."
            ) from None

    def scale_vector(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """The frequency scale vector for the current ``progress``, built once per injected step."""
        if self._scale_vector is None or self._scale_vector.device != device or self._scale_vector.dtype != dtype:
            high, low = lerp_scales(self.settings, self.progress)
            self._scale_vector = build_rope_scale_vector(
                self.axes_dims_rope, high, low, self.settings.beta, device, dtype
            )
        return self._scale_vector

    def pad_attention_mask(self, attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        """Widen a ``(S, S)`` regional mask to ``(S, S + image_seq_len)`` for the appended reference keys.

        The appended columns are all ``True``: every target query -- text and image alike, in every region
        -- may see the reference. Upstream sidesteps this by skipping style entirely whenever a mask is
        present; padding instead lets regional prompting and style reference coexist.

        Cached per source mask, because at high resolution this tensor is large enough that rebuilding it
        for each of the ~21 styled blocks would be a real cost.
        """
        if attention_mask is None:
            return None
        for source, padded in self._padded_masks:
            if source is attention_mask:
                return padded
        pad = attention_mask.new_ones((attention_mask.shape[0], self.image_seq_len))
        padded = torch.cat([attention_mask, pad], dim=-1)
        # Only the conditional and unconditional masks are ever live at the same time.
        if len(self._padded_masks) >= 2:
            self._padded_masks.pop(0)
        self._padded_masks.append((attention_mask, padded))
        return padded


def _image_token_start(state: Krea2StyleReferenceState, seq_len: int) -> int:
    """Krea-2 concatenates ``[text, image]``, so the image tokens are the tail of the sequence."""
    start = int(seq_len) - int(state.image_seq_len)
    if start < 0:
        raise ValueError(
            f"Krea-2 style reference: image_seq_len={state.image_seq_len} exceeds the transformer sequence "
            f"length {seq_len}."
        )
    return start


def capture_style_reference(
    state: Krea2StyleReferenceState,
    block_index: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Stash one block's reference image-token K/V and Q/K statistics.

    All tensors are ``[B, heads, seq, head_dim]``, post-RoPE and **pre** GQA head expansion. The slices
    are cloned so the (much larger) full-sequence tensors can be freed with the rest of the reference pass.
    """
    image_start = _image_token_start(state, query.shape[2])
    reference_key = key[:, :, image_start:, :].clone()
    reference_value = value[:, :, image_start:, :].clone()
    query_mean, query_std = _token_mean_std(query[:, :, image_start:, :])
    key_mean, key_std = _token_mean_std(reference_key)
    state.store(
        block_index,
        Krea2StyleReferenceBlockCache(
            reference_key=reference_key,
            reference_value=reference_value,
            query_mean=query_mean,
            query_std=query_std,
            key_mean=key_mean,
            key_std=key_std,
        ),
    )


def apply_style_reference(
    state: Krea2StyleReferenceState,
    block_index: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> Krea2StyleInjection:
    """Build the styled Q/K/V for one block of the target pass.

    Steps, in upstream's order: AdaIN the target's image-token Q and K toward the reference statistics,
    scale the reference keys by the frequency vector and ``ref_k_strength``, build the paired reference
    values, then append both to the target's own keys/values.

    The AdaIN happens *before* the native/styled split, so the returned ``query`` is used for both
    branches -- the ``attention_mix`` blend is between "AdaIN'd, target-only K/V" and "AdaIN'd,
    reference-appended K/V", not between styled and untouched.
    """
    cache = state.get(block_index)
    settings = state.settings
    image_start = _image_token_start(state, query.shape[2])

    if cache.reference_key.shape[2] != state.image_seq_len:
        raise RuntimeError(
            f"Krea-2 style reference: the captured reference has {cache.reference_key.shape[2]} image tokens "
            f"but the target pass has {state.image_seq_len}. The reference must be encoded at the target size."
        )

    if settings.adain_strength > 0.0:
        query = query.clone()
        key = key.clone()
        query[:, :, image_start:, :] = _adain_to_stats(
            query[:, :, image_start:, :], cache.query_mean, cache.query_std, settings.adain_strength
        )
        key[:, :, image_start:, :] = _adain_to_stats(
            key[:, :, image_start:, :], cache.key_mean, cache.key_std, settings.adain_strength
        )

    scale_vector = state.scale_vector(key.device, key.dtype).view(1, 1, 1, -1)
    reference_key = cache.reference_key * scale_vector * settings.ref_k_strength
    reference_value = _build_reference_value(value[:, :, image_start:, :], cache.reference_value, settings)

    return Krea2StyleInjection(
        query=query,
        key=torch.cat([key, reference_key], dim=2),
        value=torch.cat([value, reference_value], dim=2),
        attention_mix=settings.attention_mix,
    )
