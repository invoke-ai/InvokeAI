from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, cast

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from invokeai.backend.stable_diffusion.extensions.ip_adapter import IPAdapterExt

if TYPE_CHECKING:
    from invokeai.backend.stable_diffusion.diffusion.regional_ip_data import RegionalIPData
    from invokeai.backend.stable_diffusion.diffusion.regional_prompt_data import RegionalPromptData
    from invokeai.backend.stable_diffusion.extensions.ip_adapter import IPAdapterAttentionWeights


class CustomAttnProcessor2_0(AttnProcessor2_0):
    """A custom implementation of AttnProcessor2_0 that supports additional Invoke features.
    This implementation is based on
    https://github.com/huggingface/diffusers/blame/fcfa270fbd1dc294e2f3a505bae6bcb791d721c3/src/diffusers/models/attention_processor.py#L1204
    Supported custom features:
    - IP-Adapter
    - Regional prompt attention
    """

    def __init__(
        self,
        ip_adapter_attention_weights: Optional[List[IPAdapterAttentionWeights]] = None,
    ):
        """Initialize a CustomAttnProcessor2_0.
        Note: Arguments that are the same for all attention layers are passed to __call__(). Arguments that are
        layer-specific are passed to __init__().
        Args:
            ip_adapter_weights: The IP-Adapter attention weights. ip_adapter_weights[i] contains the attention weights
                for the i'th IP-Adapter.
        """
        super().__init__()
        self._ip_adapter_attention_weights = ip_adapter_attention_weights

    def add_ip_adapter(self, ip_adapter_attention_weight: IPAdapterAttentionWeights):
        if self._ip_adapter_attention_weights is None:
            self._ip_adapter_attention_weights = []
        self._ip_adapter_attention_weights.append(ip_adapter_attention_weight)

    def remove_ip_adapter(self, weight: IPAdapterAttentionWeights):
        self._ip_adapter_attention_weights.remove(weight)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        # For Regional Prompting:
        regional_prompt_data: Optional[RegionalPromptData] = None,
        percent_through: Optional[torch.Tensor] = None,
        # For IP-Adapter:
        regional_ip_data: Optional[RegionalIPData] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        """Apply attention.
        Args:
            regional_prompt_data: The regional prompt data for the current batch. If not None, this will be used to
                apply regional prompt masking.
            regional_ip_data: The IP-Adapter data for the current batch.
        """
        # If true, we are doing cross-attention, if false we are doing self-attention.
        is_cross_attention = encoder_hidden_states is not None

        # Start unmodified block from AttnProcessor2_0.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
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
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # End unmodified block from AttnProcessor2_0.

        if is_cross_attention:
            attention_mask = self.apply_regional_prompt_mask(
                attention_mask,
                hidden_states,
                encoder_hidden_states,
                regional_prompt_data,
                percent_through,
            )

        # Start unmodified block from AttnProcessor2_0.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
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
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # End unmodified block from AttnProcessor2_0.

        # Apply IP-Adapter conditioning.
        if is_cross_attention:
            hidden_states = IPAdapterExt.run_adapters(
                self, attn, query, hidden_states, encoder_hidden_states, regional_ip_data
            )

        # Start unmodified block from AttnProcessor2_0.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # End of unmodified block from AttnProcessor2_0

        # casting torch.Tensor to torch.FloatTensor to avoid type issues
        return cast(torch.FloatTensor, hidden_states)

    def apply_regional_prompt_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        regional_prompt_data: Optional[RegionalPromptData],
        percent_through: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if regional_prompt_data is None:
            return attention_mask

        assert percent_through is not None
        query_seq_len = hidden_states.shape[1]
        sequence_length = encoder_hidden_states.shape[1]

        # 0/-10000
        prompt_region_attention_mask = regional_prompt_data.get_cross_attn_mask(
            query_seq_len=query_seq_len, key_seq_len=sequence_length
        )

        if attention_mask is None:
            attention_mask = prompt_region_attention_mask
        else:
            attention_mask = prompt_region_attention_mask + attention_mask

        return attention_mask
