from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

import torch
import torch.nn.functional as F
import torchvision
from transformers import CLIPVisionModelWithProjection

from invokeai.backend.ip_adapter.ip_adapter import IPAdapter
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningMode
from invokeai.backend.stable_diffusion.diffusion.regional_ip_data import RegionalIPData
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback
from invokeai.backend.util.mask import to_standard_float_mask

if TYPE_CHECKING:
    from diffusers import UNet2DConditionModel
    from diffusers.models.attention_processor import Attention

    from invokeai.app.invocations.model import ModelIdentifierField
    from invokeai.app.invocations.primitives import ImageField
    from invokeai.app.services.shared.invocation_context import InvocationContext
    from invokeai.backend.ip_adapter.ip_attention_weights import IPAttentionProcessorWeights
    from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
    from invokeai.backend.util.original_weights_storage import OriginalWeightsStorage

# TODO: refactor a little, when no longer restricted by old backend logic


@dataclass
class IPAdapterAttentionWeights:
    ip_adapter_weights: IPAttentionProcessorWeights
    skip: bool
    negative: bool


class RegionalIPDataNew:
    def __init__(
        self,
        cond_mode: ConditioningMode,
        device: torch.device,
        dtype: torch.dtype,
        max_downscale_factor: int = 8,
    ):
        self.cond_mode = cond_mode
        self.device = device
        self.dtype = dtype
        self.max_downscale_factor = max_downscale_factor
        self.image_prompt_embeds = []
        self.masks = []
        self.scales = []
        self.seq_masks = None

    def add(self, uncond: torch.Tensor, cond: torch.Tensor, mask: Optional[torch.Tensor]) -> int:
        assert len(self.image_prompt_embeds) == len(self.masks)
        self.image_prompt_embeds.append(torch.stack([uncond, cond]))
        self.masks.append(mask)
        self.scales.append(1.0)
        return len(self.masks) - 1

    def update_scale(self, id: int, value: float):
        self.scales[id] = value

    def build_masks(self):
        if self.seq_masks is None:
            self.seq_masks = RegionalIPData._prepare_masks(
                self.masks, self.max_downscale_factor, self.device, self.dtype
            )
            self.masks = None

    def get_masks(self, query_seq_len: int) -> torch.Tensor:
        """Get the mask for the given query sequence length."""
        return self.seq_masks[query_seq_len]


class IPAdapterExt(ExtensionBase):
    def __init__(
        self,
        node_context: InvocationContext,
        model_id: ModelIdentifierField,
        image_encoder_id: ModelIdentifierField,
        images: ImageField | List[ImageField],
        weight: Union[float, List[float]],
        begin_step_percent: float,
        end_step_percent: float,
        target_blocks: List[str],
        method: str,  # TODO: enum/literal/...
        mask: Optional[torch.Tensor],
    ):
        super().__init__()
        self._node_context = node_context
        self._model_id = model_id
        self._image_encoder_id = image_encoder_id
        self._images = images if isinstance(images, list) else [images]
        self._weight = weight
        self._begin_step_percent = begin_step_percent
        self._end_step_percent = end_step_percent
        self._target_blocks = target_blocks
        self._method = method
        self._mask = mask

        self._model: IPAdapter | None = None

    # collect all ip adapters info
    @callback(ExtensionCallbackType.SETUP)
    def setup(self, ctx: DenoiseContext):
        images = [self._node_context.images.get_pil(image.image_name, mode="RGB") for image in self._images]

        self._model = ctx.exit_stack.enter_context(self._node_context.models.load(self._model_id))
        assert isinstance(self._model, IPAdapter)

        with self._node_context.models.load(self._image_encoder_id) as image_encoder_model:
            assert isinstance(image_encoder_model, CLIPVisionModelWithProjection)
            # Get image embeddings from CLIP and ImageProjModel.
            image_prompt_embeds, image_prompt_embeds_uncond = self._model.get_image_embeds(images, image_encoder_model)

        # HACK: unload image projection model, as it no longer needed
        # self._model._image_proj_model.to('meta')

        _, _, latent_height, latent_width = ctx.inputs.orig_latents.shape
        mask = self._node_context.tensors.load(self._mask.tensor_name) if self._mask is not None else None
        mask = self._preprocess_mask(mask, latent_height, latent_width, dtype=ctx.dtype)

        if "IPAdapterExt_IPData" not in ctx.extra:  # TODO: cond mode to attention kwargs
            ctx.extra["IPAdapterExt_IPData"] = RegionalIPDataNew(ConditioningMode.Both, ctx.device, ctx.dtype)

        self._data_id = ctx.extra["IPAdapterExt_IPData"].add(
            image_prompt_embeds_uncond,
            image_prompt_embeds,
            mask,
        )

    @staticmethod
    def _preprocess_mask(
        mask: Optional[torch.Tensor], target_height: int, target_width: int, dtype: torch.dtype
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones((1, 1, target_height, target_width), dtype=dtype)

        mask = to_standard_float_mask(mask, out_dtype=dtype)

        tf = torchvision.transforms.Resize(
            (target_height, target_width), interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )

        # Add a batch dimension to the mask, because torchvision expects shape (batch, channels, h, w).
        mask = mask.unsqueeze(0)  # Shape: (1, h, w) -> (1, 1, h, w)
        resized_mask = tf(mask)
        return resized_mask

    # pass ip adapter weights to attention processors
    @contextmanager
    def patch_unet(self, unet: UNet2DConditionModel, original_weights: OriginalWeightsStorage):
        # TODO: somewhere store/cache attn_processors dict
        attn_processors = unet.attn_processors
        to_remove = {}

        try:
            for idx, (name, attn_proc) in enumerate(attn_processors.items()):
                # "attn1" processors do not use IP-Adapters.
                if name.endswith("attn1.processor"):
                    continue

                ip_adapter_weights = self._model.attn_weights.get_attention_processor_weights(idx)
                skip = True
                negative = False
                for block in self._target_blocks:
                    if block in name:
                        skip = False
                        # TODO: check
                        negative = self._method == "style_precise" and (
                            block == "down_blocks.2.attentions.1" or block == "down_blocks.2" or block == "mid_block"
                        )
                        break
                ip_adapter_attention_weights: IPAdapterAttentionWeights = IPAdapterAttentionWeights(
                    ip_adapter_weights=ip_adapter_weights, skip=skip, negative=negative
                )
                attn_proc.add_ip_adapter(ip_adapter_attention_weights)
                to_remove[name] = ip_adapter_attention_weights

            yield
        finally:
            for name, value in to_remove.items():
                attn_processors[name].remove_ip_adapter(value)

    # before starting denoise, building masks tensors
    @callback(ExtensionCallbackType.PRE_DENOISE_LOOP)
    def build_masks(self, ctx: DenoiseContext):
        ctx.extra["IPAdapterExt_IPData"].build_masks()

    # update weights/scales and pass info to attention args
    @callback(ExtensionCallbackType.PRE_UNET)
    def set_attn_kwargs(self, ctx: DenoiseContext):
        ctx.extra["IPAdapterExt_IPData"].update_scale(
            self._data_id,
            self.scale_for_step(ctx.step_index, len(ctx.inputs.timesteps)),
        )

        if ctx.unet_kwargs.cross_attention_kwargs is None:
            ctx.unet_kwargs.cross_attention_kwargs = {}
        if "regional_ip_data" not in ctx.unet_kwargs.cross_attention_kwargs:
            ctx.unet_kwargs.cross_attention_kwargs.update(
                regional_ip_data=ctx.extra["IPAdapterExt_IPData"],
            )
            ctx.extra["IPAdapterExt_IPData"].cond_mode = ctx.conditioning_mode

    def scale_for_step(self, step_index: int, total_steps: int) -> float:
        first_adapter_step = math.floor(self._begin_step_percent * total_steps)
        last_adapter_step = math.ceil(self._end_step_percent * total_steps)
        weight = self._weight[step_index] if isinstance(self._weight, list) else self._weight
        if step_index >= first_adapter_step and step_index <= last_adapter_step:
            # Only apply this IP-Adapter if the current step is within the IP-Adapter's begin/end step range.
            return weight
        # Otherwise, set the IP-Adapter's scale to 0, so it has no effect.
        return 0.0

    # run adapters in attention processor
    @staticmethod
    def run_adapters(
        attn_processor,
        attn: Attention,
        query: torch.Tensor,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        regional_ip_data: Optional[RegionalIPData],  # RegionalIPDataNew
    ) -> torch.Tensor:
        batch_size, _, query_seq_len, head_dim = query.shape
        token_len = encoder_hidden_states.shape[-1]

        if attn_processor._ip_adapter_attention_weights:
            assert regional_ip_data is not None
            ip_masks = regional_ip_data.get_masks(query_seq_len=query_seq_len)

            assert (
                len(regional_ip_data.image_prompt_embeds)
                == len(attn_processor._ip_adapter_attention_weights)
                == len(regional_ip_data.scales)
                == ip_masks.shape[1]
            )

            for ipa_index, ipa_embed in enumerate(regional_ip_data.image_prompt_embeds):
                ipa_weights = attn_processor._ip_adapter_attention_weights[ipa_index].ip_adapter_weights
                ipa_scale = regional_ip_data.scales[ipa_index]
                ip_mask = ip_masks[0, ipa_index, ...]

                # ipa_embed[0] - uncond (neg)
                # ipa_embed[1] - cond (pos)
                assert ipa_embed.shape[0] == 2
                # The token_len dimensions should match.
                assert ipa_embed.shape[-1] == token_len

                # Expected ip_hidden_state shape: (batch_size, num_ip_images, ip_seq_len, ip_image_embedding)

                if not attn_processor._ip_adapter_attention_weights[ipa_index].skip:
                    # apply the IP-Adapter weights to the negative embeds
                    if attn_processor._ip_adapter_attention_weights[ipa_index].negative:
                        ipa_embed = torch.cat([ipa_embed[1:2], ipa_embed[0:1] * 0], dim=0)

                    if regional_ip_data.cond_mode == ConditioningMode.Positive:
                        ip_hidden_states = ipa_embed[1:2]  # [1]
                    elif regional_ip_data.cond_mode == ConditioningMode.Negative:
                        ip_hidden_states = ipa_embed[0:1]  # [0]
                    else:
                        ip_hidden_states = ipa_embed

                    ip_key = ipa_weights.to_k_ip(ip_hidden_states)
                    ip_value = ipa_weights.to_v_ip(ip_hidden_states)

                    # Expected ip_key and ip_value shape:
                    # (batch_size, num_ip_images, ip_seq_len, head_dim * num_heads)

                    ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                    # Expected ip_key and ip_value shape:
                    # (batch_size, num_heads, num_ip_images * ip_seq_len, head_dim)

                    # TODO: add support for attn.scale when we move to Torch 2.1
                    ip_hidden_states = F.scaled_dot_product_attention(
                        query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                    )

                    # Expected ip_hidden_states shape: (batch_size, num_heads, query_seq_len, head_dim)
                    ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

                    ip_hidden_states = ip_hidden_states.to(query.dtype)

                    # Expected ip_hidden_states shape: (batch_size, query_seq_len, num_heads * head_dim)
                    hidden_states = hidden_states + ipa_scale * ip_hidden_states * ip_mask
        else:
            # If IP-Adapter is not enabled, then regional_ip_data should not be passed in.
            assert regional_ip_data is None

        return hidden_states
