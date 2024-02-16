# copied from https://github.com/tencent-ailab/IP-Adapter (Apache License 2.0)
#   and modified as needed

# tencent-ailab comment:
# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0 as DiffusersAttnProcessor2_0

from invokeai.backend.ip_adapter.ip_attention_weights import IPAttentionProcessorWeights


# Create a version of AttnProcessor2_0 that is a sub-class of nn.Module. This is required for IP-Adapter state_dict
# loading.
class AttnProcessor2_0(DiffusersAttnProcessor2_0, nn.Module):
    def __init__(self):
        DiffusersAttnProcessor2_0.__init__(self)
        nn.Module.__init__(self)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        ip_adapter_image_prompt_embeds=None,
    ):
        """Re-definition of DiffusersAttnProcessor2_0.__call__(...) that accepts and ignores the
        ip_adapter_image_prompt_embeds parameter.
        """
        return DiffusersAttnProcessor2_0.__call__(
            self, attn, hidden_states, encoder_hidden_states, attention_mask, temb
        )


class IPAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, weights: list[IPAttentionProcessorWeights], scales: list[float]):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        assert len(weights) == len(scales)

        self._weights = weights
        self._scales = scales

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        ip_adapter_image_prompt_embeds=None,
    ):
        """Apply IP-Adapter attention.

        Args:
            ip_adapter_image_prompt_embeds (torch.Tensor): The image prompt embeddings.
                Shape: (batch_size, num_ip_images, seq_len, ip_embedding_len).
        """
        # If true, we are doing cross-attention, if false we are doing self-attention.
        is_cross_attention = encoder_hidden_states is not None

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

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

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

        if is_cross_attention:
            # If encoder_hidden_states is not None, then we are doing cross-attention, not self-attention. In this case,
            # we will apply IP-Adapter conditioning. We validate the inputs for IP-Adapter conditioning here.
            assert ip_adapter_image_prompt_embeds is not None
            assert len(ip_adapter_image_prompt_embeds) == len(self._weights)

            for ipa_embed, ipa_weights, scale in zip(
                ip_adapter_image_prompt_embeds, self._weights, self._scales, strict=True
            ):
                # The batch dimensions should match.
                assert ipa_embed.shape[0] == encoder_hidden_states.shape[0]
                # The token_len dimensions should match.
                assert ipa_embed.shape[-1] == encoder_hidden_states.shape[-1]

                ip_hidden_states = ipa_embed

                # Expected ip_hidden_state shape: (batch_size, num_ip_images, ip_seq_len, ip_image_embedding)

                ip_key = ipa_weights.to_k_ip(ip_hidden_states)
                ip_value = ipa_weights.to_v_ip(ip_hidden_states)

                # Expected ip_key and ip_value shape: (batch_size, num_ip_images, ip_seq_len, head_dim * num_heads)

                ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # Expected ip_key and ip_value shape: (batch_size, num_heads, num_ip_images * ip_seq_len, head_dim)

                # TODO: add support for attn.scale when we move to Torch 2.1
                ip_hidden_states = F.scaled_dot_product_attention(
                    query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                )

                # Expected ip_hidden_states shape: (batch_size, num_heads, query_seq_len, head_dim)

                ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                ip_hidden_states = ip_hidden_states.to(query.dtype)

                # Expected ip_hidden_states shape: (batch_size, query_seq_len, num_heads * head_dim)

                hidden_states = hidden_states + scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
