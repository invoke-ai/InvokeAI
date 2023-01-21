
"""
# base implementation

class CrossAttnProcessor:
    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

"""
import enum
from dataclasses import field, dataclass

import torch

from diffusers.models.cross_attention import CrossAttention, CrossAttnProcessor

class AttentionType(enum.Enum):
    SELF = 1
    TOKENS = 2

@dataclass
class SwapCrossAttnContext:

    cross_attention_types_to_do: list[AttentionType]
    modified_text_embeddings: torch.Tensor
    index_map: torch.Tensor # maps from original prompt token indices to the equivalent tokens in the modified prompt
    mask: torch.Tensor # in the target space of the index_map

    def __int__(self,
                cac_types_to_do: [AttentionType],
                modified_text_embeddings: torch.Tensor,
                index_map: torch.Tensor,
                mask: torch.Tensor):
        self.cross_attention_types_to_do = cac_types_to_do
        self.modified_text_embeddings = modified_text_embeddings
        self.index_map = index_map
        self.mask = mask

    def wants_cross_attention_control(self, attn_type: AttentionType) -> bool:
        return attn_type in self.cross_attention_types_to_do


class SwapCrossAttnProcessor(CrossAttnProcessor):
    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 # kwargs
                 cross_attention_swap_context_provider: SwapCrossAttnContext=None):

        if cross_attention_swap_context_provider is None:
            raise RuntimeError("a SwapCrossAttnContext instance must be passed via attention processor kwargs")

        attention_type = AttentionType.SELF if encoder_hidden_states is None else AttentionType.TOKENS
        # if cross-attention control is not in play, just call through to the base implementation.
        if not cross_attention_swap_context_provider.wants_cross_attention_control(attention_type):
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask)

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        # helper function
        def get_attention_probs(embeddings):
            this_key = attn.to_k(embeddings)
            this_key = attn.head_to_batch_dim(this_key)
            return attn.get_attention_scores(query, this_key, attention_mask)

        if attention_type == AttentionType.SELF:
            # self attention has no remapping, it just bluntly copies the whole tensor
            attention_probs = get_attention_probs(hidden_states)
            value = attn.to_v(hidden_states)
        else:
            # tokens (cross) attention
            # first, find attention probabilities for the "original" prompt
            original_text_embeddings = encoder_hidden_states
            original_attention_probs = get_attention_probs(original_text_embeddings)

            # then, find attention probabilities for the "modified" prompt
            modified_text_embeddings = cross_attention_swap_context_provider.modified_text_embeddings
            modified_attention_probs = get_attention_probs(modified_text_embeddings)

            # because the prompt modifications may result in token sequences shifted forwards or backwards,
            # the original attention probabilities must be remapped to account for token index changes in the
            # modified prompt
            remapped_original_attention_probs = torch.index_select(original_attention_probs, -1,
                                                                   cross_attention_swap_context_provider.index_map)

            # only some tokens taken from the original attention probabilities. this is controlled by the mask.
            mask = cross_attention_swap_context_provider.mask
            inverse_mask = 1 - mask
            attention_probs = \
                remapped_original_attention_probs * mask + \
                modified_attention_probs * inverse_mask

            # for the "value" just use the modified text embeddings.
            value = attn.to_v(modified_text_embeddings)

        value = attn.head_to_batch_dim(value)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class P2PCrossAttentionProc:

    def __init__(self, head_size, upcast_attention, attn_maps_reweight):
        super().__init__(head_size=head_size, upcast_attention=upcast_attention)
        self.attn_maps_reweight = attn_maps_reweight

    def __call__(self, hidden_states, query_proj, key_proj, value_proj, encoder_hidden_states, modified_text_embeddings):
        batch_size, sequence_length, _ = hidden_states.shape
        query = query_proj(hidden_states)

        context = context if context is not None else hidden_states
        attention_probs = []
        original_text_embeddings = encoder_hidden_states
        for context in [original_text_embeddings, modified_text_embeddings]:
            key = key_proj(original_text_embeddings)
            value = self.value_proj(original_text_embeddings)

            query = self.head_to_batch_dim(query, self.head_size)
            key = self.head_to_batch_dim(key, self.head_size)
            value = self.head_to_batch_dim(value, self.head_size)

            attention_probs.append(self.get_attention_scores(query, key))

        merged_probs = self.attn_maps_reweight * torch.cat(attention_probs)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)
        return hidden_states

proc = P2PCrossAttentionProc(unet.config.head_size, unet.config.upcast_attention, 0.6)
