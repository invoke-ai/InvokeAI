import math
from dataclasses import dataclass
from typing import List, Optional

import psutil
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.utils.import_utils import is_xformers_available

import invokeai.backend.util.logging as logger
from invokeai.app.services.config.config_default import get_config
from invokeai.backend.ip_adapter.ip_attention_weights import IPAttentionProcessorWeights
from invokeai.backend.stable_diffusion.diffusion.regional_ip_data import RegionalIPData
from invokeai.backend.stable_diffusion.diffusion.regional_prompt_data import RegionalPromptData
from invokeai.backend.util.devices import TorchDevice

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


@dataclass
class IPAdapterAttentionWeights:
    ip_adapter_weights: IPAttentionProcessorWeights
    skip: bool


class CustomAttnProcessor:
    """A custom implementation of attention processor that supports additional Invoke features.
    This implementation is based on
    AttnProcessor (https://github.com/huggingface/diffusers/blob/fcfa270fbd1dc294e2f3a505bae6bcb791d721c3/src/diffusers/models/attention_processor.py#L732)
    SlicedAttnProcessor (https://github.com/huggingface/diffusers/blob/fcfa270fbd1dc294e2f3a505bae6bcb791d721c3/src/diffusers/models/attention_processor.py#L1616)
    XFormersAttnProcessor (https://github.com/huggingface/diffusers/blob/fcfa270fbd1dc294e2f3a505bae6bcb791d721c3/src/diffusers/models/attention_processor.py#L1113)
    AttnProcessor2_0 (https://github.com/huggingface/diffusers/blob/fcfa270fbd1dc294e2f3a505bae6bcb791d721c3/src/diffusers/models/attention_processor.py#L1204)
    Supported custom features:
    - IP-Adapter
    - Regional prompt attention
    """

    def __init__(
        self,
        ip_adapter_attention_weights: Optional[List[IPAdapterAttentionWeights]] = None,
    ):
        """Initialize a CustomAttnProcessor.
        Note: Arguments that are the same for all attention layers are passed to __call__(). Arguments that are
        layer-specific are passed to __init__().
        Args:
            ip_adapter_weights: The IP-Adapter attention weights. ip_adapter_weights[i] contains the attention weights
                for the i'th IP-Adapter.
        """

        self._ip_adapter_attention_weights = ip_adapter_attention_weights

        device = TorchDevice.choose_torch_device()
        self.is_old_cuda = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8

        config = get_config()
        self.attention_type = config.attention_type
        if self.attention_type == "auto":
            self.attention_type = self._select_attention_type()

        self.slice_size = config.attention_slice_size
        if self.slice_size == "auto":
            self.slice_size = self._select_slice_size()

        if self.attention_type == "xformers" and xformers is None:
            raise ImportError("xformers attention requires xformers module to be installed.")

    def _select_attention_type(self) -> str:
        device = TorchDevice.choose_torch_device()
        # On some mps system normal attention still faster than torch-sdp, on others - on par
        # Results torch-sdp vs normal attention
        # gogurt: 67.993s vs 67.729s
        # Adreitz: 260.868s vs 226.638s
        if device.type == "mps":
            return "normal"
        elif device.type == "cuda":
            # In testing on a Tesla P40 (Pascal architecture), torch-sdp is much slower than xformers
            # (8.84 s/it vs. 1.81 s/it for SDXL). We have not tested extensively to find the precise GPU architecture or
            # compute capability where this performance gap begins.
            # Flash Attention is supported from sm80 compute capability onwards in PyTorch
            # (https://pytorch.org/blog/accelerated-pytorch-2/). For now, we use this as the cutoff for selecting
            # between xformers and torch-sdp.
            if self.is_old_cuda:
                if xformers is not None:
                    return "xformers"
                logger.warning(
                    f"xFormers is not installed, but is recommended for best performance with GPU {torch.cuda.get_device_properties(device).name}"
                )

            return "torch-sdp"
        else:  # cpu
            return "torch-sdp"

    def _select_slice_size(self) -> str:
        device = TorchDevice.choose_torch_device()
        if device.type in ["cpu", "mps"]:
            total_ram_gb = math.ceil(psutil.virtual_memory().total / 2**30)
            if total_ram_gb <= 16:
                return "max"
            if total_ram_gb <= 32:
                return "balanced"
            return "none"
        elif device.type == "cuda":
            total_vram_gb = math.ceil(torch.cuda.get_device_properties(device).total_memory / 2**30)
            if total_vram_gb <= 4:
                return "max"
            if total_vram_gb <= 6:
                return "balanced"
            return "none"
        else:
            raise ValueError(f"Unknown device: {device.type}")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        # For Regional Prompting:
        regional_prompt_data: Optional[RegionalPromptData] = None,
        # For IP-Adapter:
        regional_ip_data: Optional[RegionalIPData] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # If true, we are doing cross-attention, if false we are doing self-attention.
        is_cross_attention = encoder_hidden_states is not None

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        query_length = hidden_states.shape[1]

        # Regional Prompt Attention Mask
        if regional_prompt_data is not None and is_cross_attention:
            prompt_region_attention_mask = regional_prompt_data.get_cross_attn_mask(
                query_seq_len=query_length, key_seq_len=key_length
            )

            if attention_mask is None:
                attention_mask = prompt_region_attention_mask
            else:
                attention_mask = prompt_region_attention_mask + attention_mask

        attention_mask = attn.prepare_attention_mask(attention_mask, key_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        hidden_states = self.run_attention(
            attn=attn,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
        )

        if is_cross_attention:
            hidden_states = self.run_ip_adapters(
                attn=attn,
                hidden_states=hidden_states,
                regional_ip_data=regional_ip_data,
                query_length=query_length,
                query=query,
            )

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

    def run_ip_adapters(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        regional_ip_data: Optional[RegionalIPData],
        query_length: int,  # TODO: just read from query?
        query: torch.Tensor,
    ) -> torch.Tensor:
        if self._ip_adapter_attention_weights is None:
            # If IP-Adapter is not enabled, then regional_ip_data should not be passed in.
            assert regional_ip_data is None
            return hidden_states

        assert regional_ip_data is not None
        ip_masks = regional_ip_data.get_masks(query_seq_len=query_length)

        assert (
            len(regional_ip_data.image_prompt_embeds)
            == len(self._ip_adapter_attention_weights)
            == len(regional_ip_data.scales)
            == ip_masks.shape[1]
        )

        for ipa_index, ip_hidden_states in enumerate(regional_ip_data.image_prompt_embeds):
            # The batch dimensions should match.
            # assert ip_hidden_states.shape[0] == encoder_hidden_states.shape[0]
            # The token_len dimensions should match.
            # assert ip_hidden_states.shape[-1] == encoder_hidden_states.shape[-1]

            if self._ip_adapter_attention_weights[ipa_index].skip:
                continue

            ipa_weights = self._ip_adapter_attention_weights[ipa_index].ip_adapter_weights
            ipa_scale = regional_ip_data.scales[ipa_index]
            ip_mask = ip_masks[0, ipa_index, ...]

            # Expected ip_hidden_state shape: (batch_size, num_ip_images, ip_seq_len, ip_image_embedding)
            ip_key = ipa_weights.to_k_ip(ip_hidden_states)
            ip_value = ipa_weights.to_v_ip(ip_hidden_states)

            ip_hidden_states = self.run_attention(
                attn=attn,
                query=query,
                key=ip_key,
                value=ip_value,
                attention_mask=None,
            )

            # Expected ip_hidden_states shape: (batch_size, query_seq_len, num_heads * head_dim)
            hidden_states = hidden_states + ipa_scale * ip_hidden_states * ip_mask

        return hidden_states

    def _get_slice_size(self, attn: Attention) -> Optional[int]:
        if self.slice_size == "none":
            return None
        if isinstance(self.slice_size, int):
            return self.slice_size

        if self.slice_size == "max":
            return 1
        if self.slice_size == "balanced":
            return max(1, attn.sliceable_head_dim // 2)

        raise ValueError(f"Incorrect slice_size value: {self.slice_size}")

    def run_attention(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        no_sliced: bool = False,
    ) -> torch.Tensor:
        slice_size = self._get_slice_size(attn)
        if not no_sliced and slice_size is not None:
            return self.run_attention_sliced(
                attn=attn,
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                slice_size=slice_size,
            )

        if self.attention_type == "torch-sdp":
            attn_call = self.run_attention_sdp
        elif self.attention_type == "normal":
            attn_call = self.run_attention_normal
        elif self.attention_type == "xformers":
            attn_call = self.run_attention_xformers
        else:
            raise Exception(f"Unknown attention type: {self.attention_type}")

        return attn_call(
            attn=attn,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
        )

    @staticmethod
    def _align_attention_mask_memory(attention_mask: torch.Tensor, alignment: int = 8) -> torch.Tensor:
        if attention_mask.stride(-2) % alignment == 0 and attention_mask.stride(-2) != 0:
            return attention_mask

        last_mask_dim = attention_mask.shape[-1]
        new_last_mask_dim = last_mask_dim + (alignment - (last_mask_dim % alignment))
        attention_mask_mem = torch.empty(
            attention_mask.shape[:-1] + (new_last_mask_dim,),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        attention_mask_mem[..., :last_mask_dim] = attention_mask
        return attention_mask_mem[..., :last_mask_dim]

    @staticmethod
    def _head_to_batch_dim(tensor: torch.Tensor, head_dim: int) -> torch.Tensor:
        # [B, S, H*He] -> [B, S, H, He]
        tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], -1, head_dim)
        # [B, S, H, He] -> [B, H, S, He]
        tensor = tensor.permute(0, 2, 1, 3)
        # [B, H, S, He] -> [B*H, S, He]
        tensor = tensor.reshape(-1, tensor.shape[2], head_dim)
        return tensor

    @staticmethod
    def _batch_to_head_dim(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        # [B*H, S, He] -> [B, H, S, He]
        tensor = tensor.reshape(batch_size, -1, tensor.shape[1], tensor.shape[2])
        # [B, H, S, He] -> [B, S, H, He]
        tensor = tensor.permute(0, 2, 1, 3)
        # [B, S, H, He] -> [B, S, H*He]
        tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        return tensor

    def run_attention_normal(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        head_dim = attn.to_q.weight.shape[0] // attn.heads

        query = self._head_to_batch_dim(query, head_dim)
        key = self._head_to_batch_dim(key, head_dim)
        value = self._head_to_batch_dim(value, head_dim)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)

        hidden_states = self._batch_to_head_dim(hidden_states, batch_size)
        return hidden_states

    def run_attention_xformers(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        multihead: Optional[bool] = None,
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        head_dim = attn.to_q.weight.shape[0] // attn.heads

        # batched execution on xformers slightly faster for small heads count
        # 8 heads, fp16 (100000 attention calls):
        # xformers(dim3): 20.155955553054810s vram: 16483328b
        # xformers(dim4): 17.558132648468018s vram: 16483328b
        # 1 head, fp16 (100000 attention calls):
        # xformers(dim3):  5.660739183425903s vram:  9516032b
        # xformers(dim4):  6.114191055297852s vram:  9516032b
        if multihead is None:
            heads_count = query.shape[2] // head_dim
            multihead = heads_count >= 4

        if multihead:
            # [B, S, H*He] -> [B, S, H, He]
            query = query.view(batch_size, query.shape[1], -1, head_dim)
            key = key.view(batch_size, key.shape[1], -1, head_dim)
            value = value.view(batch_size, value.shape[1], -1, head_dim)

            if attention_mask is not None:
                # [B*H, 1, S_key] -> [B, H, 1, S_key]
                attention_mask = attention_mask.view(batch_size, -1, attention_mask.shape[1], attention_mask.shape[2])
                # expand our mask's singleton query dimension:
                #   [B, H,       1, S_key] ->
                #   [B, H, S_query, S_key]
                # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
                attention_mask = attention_mask.expand(-1, -1, query.shape[1], -1)
                # xformers requires mask memory to be aligned to 8
                attention_mask = self._align_attention_mask_memory(attention_mask)

            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, op=None, scale=attn.scale
            )
            # [B, S_query, H, He] -> [B, S_query, H*He]
            hidden_states = hidden_states.reshape(hidden_states.shape[:-2] + (-1,))
            hidden_states = hidden_states.to(query.dtype)

        else:
            # contiguous inputs slightly faster in batched execution
            # [B, S, H*He] -> [B*H, S, He]
            query = self._head_to_batch_dim(query, head_dim).contiguous()
            key = self._head_to_batch_dim(key, head_dim).contiguous()
            value = self._head_to_batch_dim(value, head_dim).contiguous()

            if attention_mask is not None:
                # expand our mask's singleton query dimension:
                #   [B*H,       1, S_key] ->
                #   [B*H, S_query, S_key]
                # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
                attention_mask = attention_mask.expand(-1, query.shape[1], -1)
                # xformers requires mask memory to be aligned to 8
                attention_mask = self._align_attention_mask_memory(attention_mask)

            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, op=None, scale=attn.scale
            )
            hidden_states = hidden_states.to(query.dtype)
            # [B*H, S_query, He] -> [B, S_query, H*He]
            hidden_states = self._batch_to_head_dim(hidden_states, batch_size)

        return hidden_states

    def run_attention_sdp(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        multihead: Optional[bool] = None,
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        head_dim = attn.to_q.weight.shape[0] // attn.heads

        if multihead is None:
            # multihead extremely slow on old cuda gpu:
            # fp16 (100000 attention calls):
            # torch-sdp(dim3): 30.07543110847473s vram: 23954432b
            # torch-sdp(dim4): 299.3908393383026s vram: 13861888b
            multihead = not self.is_old_cuda

        if multihead:
            # [B, S, H*He] -> [B, H, S, He]
            query = query.view(batch_size, query.shape[1], -1, head_dim).transpose(1, 2)
            key = key.view(batch_size, key.shape[1], -1, head_dim).transpose(1, 2)
            value = value.view(batch_size, value.shape[1], -1, head_dim).transpose(1, 2)

            if attention_mask is not None:
                # [B*H, 1, S_key] -> [B, H, 1, S_key]
                attention_mask = attention_mask.view(batch_size, -1, attention_mask.shape[1], attention_mask.shape[2])
                # mask alignment to 8 decreases memory consumption and increases speed
                # fp16 (100000 attention calls):
                # torch-sdp(dim4, mask):          6.1701478958129880s vram:  7864320b
                # torch-sdp(dim4, aligned mask):  3.3127212524414062s vram:  2621440b
                # fp32 (100000 attention calls):
                # torch-sdp(dim4, mask):         23.0943229198455800s vram: 16121856b
                # torch-sdp(dim4, aligned mask): 17.3104763031005860s vram:  5636096b
                attention_mask = self._align_attention_mask_memory(attention_mask)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=attn.scale
            )

            # [B, H, S_query, He] -> [B, S_query, H, He]
            hidden_states = hidden_states.transpose(1, 2)
            # [B, S_query, H, He] -> [B, S_query, H*He]
            hidden_states = hidden_states.reshape(hidden_states.shape[:-2] + (-1,))
            hidden_states = hidden_states.to(query.dtype)
        else:
            # [B, S, H*He] -> [B*H, S, He]
            query = self._head_to_batch_dim(query, head_dim)
            key = self._head_to_batch_dim(key, head_dim)
            value = self._head_to_batch_dim(value, head_dim)

            # attention mask already in shape [B*H, 1, S_key]/[B*H, S_query, S_key]
            # and there no noticable changes from memory alignment in batched run:
            # fp16 (100000 attention calls):
            # torch-sdp(dim3, mask):          9.7391905784606930s vram: 12713984b
            # torch-sdp(dim3, aligned mask): 10.0090200901031500s vram: 12713984b

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=attn.scale
            )

            hidden_states = hidden_states.to(query.dtype)
            # [B*H, S_query, He] -> [B, S_query, H*He]
            hidden_states = self._batch_to_head_dim(hidden_states, batch_size)

        return hidden_states

    def run_attention_sliced(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        slice_size: int,
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        head_dim = attn.to_q.weight.shape[0] // attn.heads
        heads_count = query.shape[2] // head_dim

        # [B, S, H*He] -> [B, H, S, He]
        query = query.reshape(query.shape[0], query.shape[1], -1, head_dim).transpose(1, 2)
        key = key.reshape(key.shape[0], key.shape[1], -1, head_dim).transpose(1, 2)
        value = value.reshape(value.shape[0], value.shape[1], -1, head_dim).transpose(1, 2)
        # [B*H, S_query/1, S_key] -> [B, H, S_query/1, S_key]
        if attention_mask is not None:
            attention_mask = attention_mask.reshape(batch_size, -1, attention_mask.shape[1], attention_mask.shape[2])

        # [B, H, S_query, He]
        hidden_states = torch.empty(query.shape, device=query.device, dtype=query.dtype)

        for i in range((heads_count - 1) // slice_size + 1):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size

            # [B, H_s, S, He] -> [B, S, H_s*He]
            query_slice = query[:, start_idx:end_idx, :, :].transpose(1, 2).reshape(batch_size, query.shape[2], -1)
            key_slice = key[:, start_idx:end_idx, :, :].transpose(1, 2).reshape(batch_size, key.shape[2], -1)
            value_slice = value[:, start_idx:end_idx, :, :].transpose(1, 2).reshape(batch_size, value.shape[2], -1)

            # [B, H_s, S_query/1, S_key] -> [B*H_s, S_query/1, S_key]
            attn_mask_slice = None
            if attention_mask is not None:
                attn_mask_slice = attention_mask[:, start_idx:end_idx, :, :].reshape((-1,) + attention_mask.shape[-2:])

            # [B, S_query, H_s*He]
            hidden_states_slice = self.run_attention(
                attn=attn,
                query=query_slice,
                key=key_slice,
                value=value_slice,
                attention_mask=attn_mask_slice,
                no_sliced=True,
            )

            # [B, S_query, H_s*He] -> [B, H_s, S_query, He]
            hidden_states[:, start_idx:end_idx] = hidden_states_slice.reshape(
                hidden_states_slice.shape[:-1] + (-1, head_dim)
            ).transpose(1, 2)

        # [B, H_s, S_query, He] -> [B, S_query, H_s*He]
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states.reshape(hidden_states.shape[:-2] + (-1,))
