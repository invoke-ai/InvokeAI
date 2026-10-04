"""AdaLN-pruned ("adaln curves") variant of the MiniMax H3 transformer.

MiniMax's inference-only "pruned" checkpoints (e.g. the Comfy-Org
``minimax_h3_fl2va_pruned_*`` single files) delete the 33B model's largest
inference-constant component: the timestep MLP and the full-width
(``time_embed_dim=2688``) AdaLN input projections, ~13B parameters in total.
They are replaced by a precomputed rank-8 basis of the time-embedding curve:

- a global ``adaln_t_table`` float32 buffer of shape ``[grid, curve_dim]``
  (``[1025, 8]`` in the released checkpoints), sampled over ``t in [0, 1]``;
- per-block AdaLN projections whose input dim is ``curve_dim`` instead of
  ``time_embed_dim``.

At inference, the timestep embedding is a linear interpolation of the two
neighbouring table rows (semantics verified against ComfyUI's implementation;
reimplemented here):

    pos  = clamp(t, 0, 1) * (grid - 1)
    i0   = floor(pos).clamp(max=grid - 2)
    temb = lerp(table[i0], table[i0 + 1], pos - i0)

The table stores the curve *post-activation*, so unlike the full model the AdaLN
projections must NOT apply SiLU to ``temb``. The AdaLN projections run float32
(their outputs are cast to the block stack's dtype afterwards, matching the
reference, which casts per use).

This subclass deliberately lives outside the vendored ``transformer_minimax_h3``
module so the vendored file stays byte-identical to upstream diffusers for
future re-vendoring.
"""

import torch
from diffusers.configuration_utils import register_to_config
from diffusers.utils import apply_lora_scale

from invokeai.backend.minimax_h3.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3AdaLayerNormModulation,
    MiniMaxH3AdaLayerNormOut,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerOutput,
)


class MiniMaxH3AdaLayerNormModulationCurve(MiniMaxH3AdaLayerNormModulation):
    """AdaLN modulation fed by the precomputed time-embedding curve: no SiLU on ``temb``,
    float32 projection, outputs cast to the block stack's dtype."""

    def __init__(self, time_embed_dim: int, hidden_size: int, output_dtype: torch.dtype = torch.bfloat16):
        super().__init__(time_embed_dim=time_embed_dim, hidden_size=hidden_size)
        self.output_dtype = output_dtype

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        temb = self.linear(temb.to(self.linear.weight.dtype))
        # The vendored block multiplies these into the (bf16) packed stream without casting,
        # so cast here — one cast of the small modulation table instead of per-row upcasting
        # of the whole sequence. Same numerics as the reference's per-use casts.
        temb = temb.view(-1, 6 * self.hidden_size).to(self.output_dtype)
        return temb.chunk(6, dim=-1)


class MiniMaxH3AdaLayerNormOutCurve(MiniMaxH3AdaLayerNormOut):
    """Output AdaLN fed by the precomputed time-embedding curve: no SiLU on ``temb``."""

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, timestep_indices: torch.Tensor) -> torch.Tensor:
        shift, scale = self.linear(temb.to(self.linear.weight.dtype)).chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        # float32 modulation promotes the result to float32; the caller casts to the output
        # heads' dtype (float32) immediately after, so nothing is lost or duplicated. ComfyUI's
        # reference casts shift/scale down to the stream dtype (bf16) before modulating; keeping
        # them float32 here is a deliberate, strictly-more-precise divergence - the very next op
        # is the float32 output heads, so there is no downstream dtype cost.
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices)


class MiniMaxH3PrunedTransformer3DModel(MiniMaxH3Transformer3DModel):
    r"""
    The AdaLN-pruned MiniMax H3 transformer.

    Construction mirrors [`MiniMaxH3Transformer3DModel`] with ``time_embed_dim`` replaced by
    ``adaln_curve_dim``; the timestep MLP is deleted and the ``adaln_t_table`` buffer plus
    SiLU-free float32 AdaLN modules take its place.

    Args:
        adaln_curve_grid (`int`, defaults to `1025`):
            Number of grid rows of the precomputed time-embedding curve over ``t in [0, 1]``.
        adaln_curve_dim (`int`, defaults to `8`):
            Rank of the curve basis, i.e. the input dimension of every AdaLN projection.
    """

    # The pruned checkpoint's mixed-precision islands: patch projections and output heads stay
    # float32 like the full model; the AdaLN curve table and projections are float32 as well
    # (the reference runs them float32 and casts per use). Entries match as substrings.
    _keep_in_fp32_modules = [
        "proj_in",
        "audio_proj_in",
        "proj_out",
        "audio_proj_out",
        "rope",
        "adaln_proj",
        "norm_out",
        "adaln_t_table",
    ]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        adaln_curve_grid: int = 1025,
        adaln_curve_dim: int = 8,
    ) -> None:
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            ffn_dim=ffn_dim,
            in_channels=in_channels,
            audio_in_channels=audio_in_channels,
            patch_size=patch_size,
            text_dim=text_dim,
            time_embed_dim=adaln_curve_dim,
            rope_freq_dim=rope_freq_dim,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
        )

        # The timestep MLP does not exist in pruned checkpoints — the curve table replaces it.
        del self.time_proj
        del self.time_embedder
        self.register_buffer(
            "adaln_t_table", torch.zeros(adaln_curve_grid, adaln_curve_dim, dtype=torch.float32), persistent=True
        )

        # Swap the SiLU-applying AdaLN modules built by the parent for their curve variants.
        # The parent built them with in_features == adaln_curve_dim already, so shapes and
        # state-dict keys are identical; only the forward semantics change.
        for block in self.transformer_blocks:
            block.adaln_proj = MiniMaxH3AdaLayerNormModulationCurve(
                time_embed_dim=adaln_curve_dim, hidden_size=hidden_size
            )
        self.norm_out = MiniMaxH3AdaLayerNormOutCurve(
            hidden_size=hidden_size, time_embed_dim=adaln_curve_dim, eps=final_norm_eps
        )

    def _curve_temb(self, timestep: torch.Tensor) -> torch.Tensor:
        """Interpolate the precomputed time-embedding curve at the given timesteps in [0, 1]."""
        table = self.adaln_t_table
        pos = timestep.to(device=table.device, dtype=torch.float32).clamp(0.0, 1.0) * (table.shape[0] - 1)
        # max-clamp keeps t=1.0 on the last interval instead of reading past the table.
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))

    # The body below mirrors MiniMaxH3Transformer3DModel.forward verbatim except for step 2
    # (the curve lookup replaces the timestep MLP). The vendored forward offers no hook for the
    # timestep-embedding path and must not be modified, so the duplication is deliberate.
    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_kwargs: dict | None = None,
        return_dict: bool = True,
    ) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}.")
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError(
                "`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors matching `position_ids`, got "
                f"{list(token_tags.shape)} and {list(timestep_indices.shape)} for seq_len={sequence_length}."
            )

        rotary_emb = self.rope(position_ids)

        # 1. Project each modality and scatter the rows into the packed sequence buffer.
        video_embeds = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.weight.dtype))
        text_embeds = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))
        text_embeds = self.token_refiner(text_embeds)

        hidden_states = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
        hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
        hidden_states = hidden_states.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        hidden_states = hidden_states.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        # 2. One curve-interpolated timestep embedding per distinct noise level.
        temb = self._curve_temb(timestep)

        # 3. Row -> AdaLN table row.
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)

        # 4. Padding rows (tag `-1`) form their own attention document.
        attention_mask = None
        is_pad = token_tags < 0
        if bool(is_pad.any()):
            attention_mask = is_pad[None, :] == is_pad[:, None]

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, temb, adaln_indices, rotary_emb, attention_mask
                )
            else:
                hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)

        # 5. Both heads run over every row, then the rows of each modality are selected.
        hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(self.proj_out.weight.dtype)
        video_output = self.proj_out(hidden_states).index_select(1, video_indices)
        audio_output = self.audio_proj_out(hidden_states).index_select(1, audio_indices)

        if not return_dict:
            return (video_output, audio_output)
        return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)


def set_curve_modulation_dtype(model: MiniMaxH3PrunedTransformer3DModel, dtype: torch.dtype) -> None:
    """Point every curve AdaLN modulation at the block stack's compute dtype."""
    for block in model.transformer_blocks:
        adaln = block.adaln_proj
        assert isinstance(adaln, MiniMaxH3AdaLayerNormModulationCurve)
        adaln.output_dtype = dtype
