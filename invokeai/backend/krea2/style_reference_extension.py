"""Orchestration for Krea-2 style reference: latents in, capture/inject lifecycle out.

Keeps the denoise node free of style-reference bookkeeping. The attention-side math lives in
``style_reference.py`` and the noising schedule in ``style_reference_rf.py``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import torch

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.fields import Krea2StyleReferenceField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.krea2.sampling_utils import pack_latents
from invokeai.backend.krea2.style_reference import (
    KREA2_NUM_BLOCKS,
    Krea2StyleReferenceSettings,
    Krea2StyleReferenceState,
    parse_block_spec,
    resolve_effective_settings,
)
from invokeai.backend.krea2.style_reference_rf import build_linear_reference_latents

# Krea-2 latent channels (Qwen-Image VAE z_dim); mirrors KREA2_LATENT_CHANNELS in krea2_denoise.py.
_KREA2_LATENT_CHANNELS = 16

# Krea2Transformer2DModel defaults. Needed for the working-memory estimate, which runs *before* the model
# is on device and therefore cannot read transformer.config. Asserted against the real config in
# build_state() once the transformer is available.
KREA2_NUM_KV_HEADS = 12
KREA2_HEAD_DIM = 128
KREA2_AXES_DIMS_ROPE = (32, 48, 48)


class Krea2StyleReferenceExtension:
    """Holds the reference latents and drives the two-pass capture/inject lifecycle."""

    def __init__(
        self,
        reference_latents: torch.Tensor,
        settings: Krea2StyleReferenceSettings,
        block_indices: frozenset[int],
        image_seq_len: int,
    ) -> None:
        self._reference_latents = reference_latents
        self._settings = settings
        self._block_indices = block_indices
        self._image_seq_len = image_seq_len
        self._schedule: list[torch.Tensor] | None = None
        self._state: Krea2StyleReferenceState | None = None

    @property
    def block_indices(self) -> frozenset[int]:
        return self._block_indices

    @property
    def state(self) -> Krea2StyleReferenceState:
        if self._state is None:
            raise RuntimeError("Krea-2 style reference: build_state() must be called before the state is used.")
        return self._state

    @classmethod
    def from_field(
        cls,
        context: InvocationContext,
        field: Krea2StyleReferenceField,
        *,
        denoise_width: int,
        denoise_height: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "Krea2StyleReferenceExtension":
        """Load and pack the reference latents, validating them against the denoise resolution.

        The reference has to occupy exactly as many image tokens as the target -- its keys are appended to
        the target's along the token axis and share the target's rotary embedding. Checking here, against
        the dims the encoder recorded, gives a message that names both sides instead of a shape error deep
        inside attention.
        """
        if field.width != denoise_width or field.height != denoise_height:
            raise ValueError(
                f"Krea-2 style reference was encoded at {field.width}x{field.height} but denoise is set to "
                f"{denoise_width}x{denoise_height}. Set the same width and height on both nodes."
            )

        latents = context.tensors.load(field.reference_latents_name).to(device=device, dtype=dtype)
        # The Qwen-Image VAE emits (B, C, frames, H, W); style reference is a single frame.
        if latents.dim() == 5:
            latents = latents.squeeze(2)
        if latents.dim() != 4:
            raise ValueError(f"Krea-2 style reference latents must be 4D or 5D, got shape {tuple(latents.shape)}.")

        latent_height = denoise_height // LATENT_SCALE_FACTOR
        latent_width = denoise_width // LATENT_SCALE_FACTOR
        if latents.shape[-2:] != (latent_height, latent_width):
            raise ValueError(
                f"Krea-2 style reference latents are {tuple(latents.shape[-2:])} but denoise expects "
                f"({latent_height}, {latent_width}). Re-encode the reference at the denoise resolution."
            )

        settings = Krea2StyleReferenceSettings(
            style_strength=field.style_strength,
            ref_k_strength=field.ref_k_strength,
            adain_strength=field.adain_strength,
            value_mode=field.value_mode,
            value_adain_strength=field.value_adain_strength,
            ref_value_mix=field.ref_value_mix,
            high_scale_start=field.high_scale_start,
            high_scale_end=field.high_scale_end,
            low_scale_start=field.low_scale_start,
            low_scale_end=field.low_scale_end,
            beta=field.beta,
        )
        block_indices = parse_block_spec(field.blocks, KREA2_NUM_BLOCKS)

        packed = pack_latents(latents, 1, _KREA2_LATENT_CHANNELS, latent_height, latent_width)
        return cls(
            reference_latents=packed,
            settings=settings,
            block_indices=block_indices,
            image_seq_len=packed.shape[1],
        )

    def build_state(self, transformer: torch.nn.Module) -> Krea2StyleReferenceState:
        """Create the shared state, taking the RoPE axis layout from the real transformer config."""
        config = getattr(transformer, "config", None)
        axes_dims = tuple(int(dim) for dim in getattr(config, "axes_dims_rope", KREA2_AXES_DIMS_ROPE))
        num_blocks = int(getattr(config, "num_layers", KREA2_NUM_BLOCKS))
        out_of_range = sorted(index for index in self._block_indices if index >= num_blocks)
        if out_of_range:
            raise ValueError(
                f"Krea-2 style reference targets blocks {out_of_range}, but this transformer has {num_blocks} blocks."
            )
        self._state = Krea2StyleReferenceState(
            settings=resolve_effective_settings(self._settings),
            image_seq_len=self._image_seq_len,
            axes_dims_rope=axes_dims,
        )
        return self._state

    def prepare(self, sigmas: Sequence[float]) -> None:
        """Build the reference's noise trajectory over the active sigma schedule."""
        self._schedule = build_linear_reference_latents(self._reference_latents, sigmas)

    def reference_latents_for_step(self, step_index: int) -> torch.Tensor:
        if self._schedule is None:
            raise RuntimeError("Krea-2 style reference: prepare() must be called before the denoise loop.")
        return self._schedule[step_index]

    @staticmethod
    def progress_for_step(step_index: int, total_steps: int) -> float:
        """Position in the schedule, used to interpolate the frequency scales.

        Defined over the *active* window, so an img2img run that starts at ``denoising_start > 0`` sweeps
        the full 0..1 curve across the steps it actually takes rather than starting mid-curve.
        """
        return step_index / max(total_steps - 1, 1)

    def kv_cache_bytes(self, dtype: torch.dtype) -> int:
        """Bytes of reference K/V retained across the target pass, for the working-memory estimate.

        One key and one value per styled block, at the pre-expansion KV head count.
        """
        element_size = torch.empty((), dtype=dtype).element_size()
        return len(self._block_indices) * 2 * KREA2_NUM_KV_HEADS * self._image_seq_len * KREA2_HEAD_DIM * element_size

    @contextmanager
    def capture(self) -> Iterator[None]:
        """Run the enclosed reference forward in capture mode."""
        self.state.begin_capture()
        try:
            yield
        finally:
            self.state.disable()

    @contextmanager
    def inject(self, progress: float) -> Iterator[None]:
        """Run the enclosed target forward(s) with the captured reference spliced in."""
        self.state.begin_inject(progress)
        try:
            yield
        finally:
            self.state.disable()
