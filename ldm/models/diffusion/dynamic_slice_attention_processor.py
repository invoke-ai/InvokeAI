import math

import psutil
import torch
from diffusers.models.cross_attention import CrossAttention, SlicedAttnProcessor


def estimate_free_memory_bytes(device: torch.device) -> int:
    if device.type == 'cpu' or device.type == 'mps':
        return psutil.virtual_memory().free
    elif device.type == 'cuda':
        stats = torch.cuda.memory_stats(device)
        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(device)
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch
        return mem_free_total
    else:
        raise ValueError(f"unrecognized device {device}")


class DynamicSlicedAttnProcessor(SlicedAttnProcessor):
    def __init__(self):
        very_large_slice_size = 1024*1024
        super(DynamicSlicedAttnProcessor, self).__init__(slice_size=very_large_slice_size)

    def get_free_memory_bytes(self, device: torch.device) -> int:
        safety_factor = 3.3 / 4.0  # magic numbers pulled from old invoke code
        free_memory_bytes = int(estimate_free_memory_bytes(device) * safety_factor)
        return free_memory_bytes

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None,
                 attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)

        is_self_attention = (encoder_hidden_states is None)
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        batch_size_attention = query.shape[0]
        # self attention needs much more memory
        if is_self_attention:
            bytes_per_element = hidden_states.element_size()
            required_size_for_baddbmm_bytes = batch_size_attention * sequence_length * sequence_length * bytes_per_element

            free_memory_bytes = self.get_free_memory_bytes(hidden_states.device)
            slice_count = math.ceil(required_size_for_baddbmm_bytes / free_memory_bytes)
            slice_size = math.ceil(batch_size_attention / slice_count)
            #one_gb = 1024*1024*1024
            #print(f"free_memory gb: {free_memory_bytes/one_gb}, required: {required_size_for_baddbmm_bytes/one_gb}")
            #if slice_count > 1:
            #    print(f"dynamically slicing into {slice_count} pieces")
        else:
            # effectively: do not slice
            slice_size = batch_size_attention


        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // attn.heads), device=query.device, dtype=query.dtype
        )

        assert(slice_size > 0)
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)

            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
