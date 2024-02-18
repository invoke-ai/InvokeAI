from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.utils import USE_PEFT_BACKEND


@dataclass
class Range:
    start: int
    end: int


@dataclass
class RegionalPromptData:
    # The region masks for each prompt.
    # shape: (batch_size, num_prompts, height, width)
    # dtype: float*
    # The mask is set to 1.0 in regions where the prompt should be applied, and 0.0 elsewhere.
    masks: torch.Tensor

    # The embedding ranges for each prompt.
    # The i'th mask is applied to the embeddings in:
    # encoder_hidden_states[:, embedding_ranges[i].start:embedding_ranges[i].end, :]
    embedding_ranges: list[Range]


class RegionalPromptAttnProcessor2_0(AttnProcessor2_0):
    """An attention processor that supports regional prompt attention for PyTorch 2.0."""

    def _prepare_regional_prompt_attention_mask(
        self,
        regional_prompt_data: RegionalPromptData,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        orig_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Infer the current spatial dimensions from the shape of `hidden_states`.
        _, query_seq_len, _ = hidden_states.shape
        per_prompt_query_masks = regional_prompt_data.masks
        _, _, h, w = per_prompt_query_masks.shape

        # Downsample by factors of 2 until the spatial dimensions match the current query sequence length.
        scale_factor = 1
        while h * w > query_seq_len:
            scale_factor *= 2
            h //= 2
            w //= 2
        assert h * w == query_seq_len

        # Apply max-pooling to resize the masks to the target spatial dimensions.
        # TODO(ryand): We should be able to pre-compute all of the mask sizes. There's a lot of redundant computation
        # here.
        per_prompt_query_masks = F.max_pool2d(per_prompt_query_masks, kernel_size=scale_factor, stride=scale_factor)
        batch_size, num_prompts, resized_h, resized_w = per_prompt_query_masks.shape
        assert resized_h == h and resized_w == w

        # Flatten the spatial dimensions of the masks.
        # Shape after reshape: (batch_size, num_prompts, query_seq_len)
        per_prompt_query_masks = per_prompt_query_masks.reshape((batch_size, num_prompts, -1, 1))

        # Create a cross-attention mask for each prompt that selects the corresponding embeddings from
        # `encoder_hidden_states`.

        # attn_mask shape: (batch_size, query_seq_len, key_seq_len)
        _, key_seq_len, _ = encoder_hidden_states.shape
        # HACK(ryand): We are assuming the batch size.
        attn_mask = torch.zeros((2, query_seq_len, key_seq_len), device=hidden_states.device)

        for i, embedding_range in enumerate(regional_prompt_data.embedding_ranges):
            # HACK(ryand): We are assuming that batch 0 is unconditioned and batch 1 is conditioned. This is too fragile
            # to merge.
            attn_mask[1, :, embedding_range.start : embedding_range.end] = per_prompt_query_masks[:, i, :, :]

        # HACK(ryand): We are assuming that batch 0 is unconditioned and batch 1 is conditioned. We are also assuming
        # the intent of attn_mask. And we shouldn't have to do this awkward mask type conversion.
        orig_mask = torch.zeros_like(orig_attn_mask[0, ...])
        orig_mask[orig_attn_mask[0, ...] > -0.5] = 1.0
        attn_mask[0, ...] = orig_mask

        return attn_mask > 0.5

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        regional_prompt_data: Optional[RegionalPromptData] = None,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if encoder_hidden_states is not None:
            assert regional_prompt_data is not None
            assert attention_mask is not None
            attention_mask = self._prepare_regional_prompt_attention_mask(
                regional_prompt_data, hidden_states, encoder_hidden_states, attention_mask
            )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


@contextmanager
def apply_regional_prompt_attn(unet: UNet2DConditionModel):
    """A context manager that patches `unet` with RegionalPromptAttnProcessor2_0 attention processors."""

    orig_attn_processors = unet.attn_processors

    try:
        unet.set_attn_processor(RegionalPromptAttnProcessor2_0())
        yield None
    finally:
        unet.set_attn_processor(orig_attn_processors)
