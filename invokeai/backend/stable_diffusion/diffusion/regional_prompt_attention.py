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


class RegionalPromptData:
    def __init__(self, attn_masks_by_seq_len: dict[int, torch.Tensor]):
        self._attn_masks_by_seq_len = attn_masks_by_seq_len

    @classmethod
    def from_masks_and_ranges(
        cls,
        masks: list[torch.Tensor],
        embedding_ranges: list[list[Range]],
        key_seq_len: int,
        # TODO(ryand): Pass in a list of downscale factors?
        max_downscale_factor: int = 8,
    ):
        """Construct a `RegionalPromptData` object.

        Args:
            masks (list[torch.Tensor]): masks[i] contains the regions masks for the i'th sample in the batch.
                The shape of masks[i] is (num_prompts, height, width). The mask is set to 1.0 in regions where the
                prompt should be applied, and 0.0 elsewhere.

            embedding_ranges (list[list[Range]]): embedding_ranges[i][j] contains the embedding range for the j'th
                prompt in the i'th batch sample. masks[i][j, ...] is applied to the embeddings in:
                encoder_hidden_states[i, embedding_ranges[j].start:embedding_ranges[j].end, :].

            key_seq_len (int): The sequence length of the expected prompt embeddings (which act as the key in the
                cross-attention layers). This is most likely equal to the max embedding range end, but we pass it
                explicitly to be sure.
        """
        attn_masks_by_seq_len = {}

        # batch_attn_mask_by_seq_len[b][s] contains the attention mask for the b'th batch sample with a query sequence
        # length of s.
        batch_attn_masks_by_seq_len: list[dict[int, torch.Tensor]] = []
        for batch_masks, batch_ranges in zip(masks, embedding_ranges, strict=True):
            batch_attn_masks_by_seq_len.append({})

            # Convert the bool masks to float masks so that max pooling can be applied.
            batch_masks = batch_masks.to(dtype=torch.float32)

            # Downsample the spatial dimensions by factors of 2 until max_downscale_factor is reached.
            downscale_factor = 1
            while downscale_factor <= max_downscale_factor:
                _, num_prompts, h, w = batch_masks.shape
                query_seq_len = h * w

                # Flatten the spatial dimensions of the mask by reshaping to (1, num_prompts, query_seq_len, 1).
                batch_query_masks = batch_masks.reshape((1, num_prompts, -1, 1))

                # Create a cross-attention mask for each prompt that selects the corresponding embeddings from
                # `encoder_hidden_states`.
                # attn_mask shape: (batch_size, query_seq_len, key_seq_len)
                # TODO(ryand): What device / dtype should this be?
                attn_mask = torch.zeros((1, query_seq_len, key_seq_len))

                for prompt_idx, embedding_range in enumerate(batch_ranges):
                    attn_mask[0, :, embedding_range.start : embedding_range.end] = batch_query_masks[
                        :, prompt_idx, :, :
                    ]

                batch_attn_masks_by_seq_len[-1][query_seq_len] = attn_mask

                downscale_factor *= 2
                if downscale_factor <= max_downscale_factor:
                    # We use max pooling because we downscale to a pretty low resolution, so we don't want small prompt
                    # regions to be lost entirely.
                    # TODO(ryand): In the future, we may want to experiment with other downsampling methods, and could
                    # potentially use a weighted mask rather than a binary mask.
                    batch_masks = F.max_pool2d(batch_masks, kernel_size=2, stride=2)

        # Merge the batch_attn_masks_by_seq_len into a single attn_masks_by_seq_len.
        for query_seq_len in batch_attn_masks_by_seq_len[0].keys():
            attn_masks_by_seq_len[query_seq_len] = torch.cat(
                [batch_attn_masks_by_seq_len[i][query_seq_len] for i in range(len(batch_attn_masks_by_seq_len))]
            )

        return cls(attn_masks_by_seq_len)

    def get_attn_mask(self, query_seq_len: int) -> torch.Tensor:
        """Get the attention mask for the given query sequence length (i.e. downscaling level).

        This is called during cross-attention, where query_seq_len is the length of the flattened spatial features, so
        it changes at each downscaling level in the model.

        key_seq_len is the length of the expected prompt embeddings.

        Returns:
            torch.Tensor: The masks.
                shape: (batch_size, query_seq_len, key_seq_len).
                dtype: float
                The mask is a binary mask with values of 0.0 and 1.0.
        """
        return self._attn_masks_by_seq_len[query_seq_len]


class RegionalPromptAttnProcessor2_0(AttnProcessor2_0):
    """An attention processor that supports regional prompt attention for PyTorch 2.0."""

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
            _, query_seq_len, _ = hidden_states.shape
            prompt_region_attention_mask = regional_prompt_data.get_attn_mask(query_seq_len)
            # TODO(ryand): Avoid redundant type/device conversion here.
            prompt_region_attention_mask = prompt_region_attention_mask.to(
                dtype=attention_mask.dtype, device=attention_mask.device
            )
            prompt_region_attention_mask[prompt_region_attention_mask < 0.5] = -10000.0
            prompt_region_attention_mask[prompt_region_attention_mask >= 0.5] = 0.0

            attention_mask = prompt_region_attention_mask + attention_mask

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
