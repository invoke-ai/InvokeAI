# adapted from bloc97's CrossAttentionControl colab
# https://github.com/bloc97/CrossAttentionControl


import enum
from dataclasses import dataclass, field
from typing import Optional

import torch
from compel.cross_attention_control import Arguments
from diffusers.models.attention_processor import Attention, SlicedAttnProcessor
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from ...util import torch_dtype


class CrossAttentionType(enum.Enum):
    SELF = 1
    TOKENS = 2


class CrossAttnControlContext:
    def __init__(self, arguments: Arguments):
        """
        :param arguments: Arguments for the cross-attention control process
        """
        self.cross_attention_mask: Optional[torch.Tensor] = None
        self.cross_attention_index_map: Optional[torch.Tensor] = None
        self.arguments = arguments

    def get_active_cross_attention_control_types_for_step(
        self, percent_through: float = None
    ) -> list[CrossAttentionType]:
        """
        Should cross-attention control be applied on the given step?
        :param percent_through: How far through the step sequence are we (0.0=pure noise, 1.0=completely denoised image). Expected range 0.0..<1.0.
        :return: A list of attention types that cross-attention control should be performed for on the given step. May be [].
        """
        if percent_through is None:
            return [CrossAttentionType.SELF, CrossAttentionType.TOKENS]

        opts = self.arguments.edit_options
        to_control = []
        if opts["s_start"] <= percent_through < opts["s_end"]:
            to_control.append(CrossAttentionType.SELF)
        if opts["t_start"] <= percent_through < opts["t_end"]:
            to_control.append(CrossAttentionType.TOKENS)
        return to_control


def setup_cross_attention_control_attention_processors(unet: UNet2DConditionModel, context: CrossAttnControlContext):
    """
    Inject attention parameters and functions into the passed in model to enable cross attention editing.

    :param model: The unet model to inject into.
    :return: None
    """

    # adapted from init_attention_edit
    device = context.arguments.edited_conditioning.device

    # urgh. should this be hardcoded?
    max_length = 77
    # mask=1 means use base prompt attention, mask=0 means use edited prompt attention
    mask = torch.zeros(max_length, dtype=torch_dtype(device))
    indices_target = torch.arange(max_length, dtype=torch.long)
    indices = torch.arange(max_length, dtype=torch.long)
    for name, a0, a1, b0, b1 in context.arguments.edit_opcodes:
        if b0 < max_length:
            if name == "equal":  # or (name == "replace" and a1 - a0 == b1 - b0):
                # these tokens have not been edited
                indices[b0:b1] = indices_target[a0:a1]
                mask[b0:b1] = 1

    context.cross_attention_mask = mask.to(device)
    context.cross_attention_index_map = indices.to(device)
    old_attn_processors = unet.attn_processors
    if torch.backends.mps.is_available():
        # see note in StableDiffusionGeneratorPipeline.__init__ about borked slicing on MPS
        unet.set_attn_processor(SwapCrossAttnProcessor())
    else:
        # try to re-use an existing slice size
        default_slice_size = 4
        slice_size = next(
            (p.slice_size for p in old_attn_processors.values() if type(p) is SlicedAttnProcessor), default_slice_size
        )
        unet.set_attn_processor(SlicedSwapCrossAttnProcesser(slice_size=slice_size))


@dataclass
class SwapCrossAttnContext:
    modified_text_embeddings: torch.Tensor
    index_map: torch.Tensor  # maps from original prompt token indices to the equivalent tokens in the modified prompt
    mask: torch.Tensor  # in the target space of the index_map
    cross_attention_types_to_do: list[CrossAttentionType] = field(default_factory=list)

    def wants_cross_attention_control(self, attn_type: CrossAttentionType) -> bool:
        return attn_type in self.cross_attention_types_to_do

    @classmethod
    def make_mask_and_index_map(
        cls, edit_opcodes: list[tuple[str, int, int, int, int]], max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # mask=1 means use original prompt attention, mask=0 means use modified prompt attention
        mask = torch.zeros(max_length)
        indices_target = torch.arange(max_length, dtype=torch.long)
        indices = torch.arange(max_length, dtype=torch.long)
        for name, a0, a1, b0, b1 in edit_opcodes:
            if b0 < max_length:
                if name == "equal":
                    # these tokens remain the same as in the original prompt
                    indices[b0:b1] = indices_target[a0:a1]
                    mask[b0:b1] = 1

        return mask, indices


class SlicedSwapCrossAttnProcesser(SlicedAttnProcessor):
    # TODO: dynamically pick slice size based on memory conditions

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        # kwargs
        swap_cross_attn_context: SwapCrossAttnContext = None,
        **kwargs,
    ):
        attention_type = CrossAttentionType.SELF if encoder_hidden_states is None else CrossAttentionType.TOKENS

        # if cross-attention control is not in play, just call through to the base implementation.
        if (
            attention_type is CrossAttentionType.SELF
            or swap_cross_attn_context is None
            or not swap_cross_attn_context.wants_cross_attention_control(attention_type)
        ):
            # print(f"SwapCrossAttnContext for {attention_type} not active - passing request to superclass")
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask)
        # else:
        #    print(f"SwapCrossAttnContext for {attention_type} active")

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask=attention_mask,
            target_length=sequence_length,
            batch_size=batch_size,
        )

        query = attn.to_q(hidden_states)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)

        original_text_embeddings = encoder_hidden_states
        modified_text_embeddings = swap_cross_attn_context.modified_text_embeddings
        original_text_key = attn.to_k(original_text_embeddings)
        modified_text_key = attn.to_k(modified_text_embeddings)
        original_value = attn.to_v(original_text_embeddings)
        modified_value = attn.to_v(modified_text_embeddings)

        original_text_key = attn.head_to_batch_dim(original_text_key)
        modified_text_key = attn.head_to_batch_dim(modified_text_key)
        original_value = attn.head_to_batch_dim(original_value)
        modified_value = attn.head_to_batch_dim(modified_value)

        # compute slices and prepare output tensor
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // attn.heads),
            device=query.device,
            dtype=query.dtype,
        )

        # do slices
        for i in range(max(1, hidden_states.shape[0] // self.slice_size)):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            query_slice = query[start_idx:end_idx]
            original_key_slice = original_text_key[start_idx:end_idx]
            modified_key_slice = modified_text_key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            original_attn_slice = attn.get_attention_scores(query_slice, original_key_slice, attn_mask_slice)
            modified_attn_slice = attn.get_attention_scores(query_slice, modified_key_slice, attn_mask_slice)

            # because the prompt modifications may result in token sequences shifted forwards or backwards,
            # the original attention probabilities must be remapped to account for token index changes in the
            # modified prompt
            remapped_original_attn_slice = torch.index_select(
                original_attn_slice, -1, swap_cross_attn_context.index_map
            )

            # only some tokens taken from the original attention probabilities. this is controlled by the mask.
            mask = swap_cross_attn_context.mask
            inverse_mask = 1 - mask
            attn_slice = remapped_original_attn_slice * mask + modified_attn_slice * inverse_mask

            del remapped_original_attn_slice, modified_attn_slice

            attn_slice = torch.bmm(attn_slice, modified_value[start_idx:end_idx])
            hidden_states[start_idx:end_idx] = attn_slice

        # done
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class SwapCrossAttnProcessor(SlicedSwapCrossAttnProcesser):
    def __init__(self):
        super(SwapCrossAttnProcessor, self).__init__(slice_size=int(1e9))  # massive slice size = don't slice
