# type: ignore
# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.controlnet import zero_module
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.outputs import BaseOutput

from invokeai.backend.bria.transformer_bria import (
    EmbedND,
    FluxSingleTransformerBlock,
    FluxTransformerBlock,
    TimestepProjEmbeddings,
)

logger = logging.get_logger(__name__)


BRIA_CONTROL_MODES = Literal["depth", "canny", "colorgrid", "recolor", "tile", "pose"]


class BriaControlModes(Enum):
    depth = 0
    canny = 1
    colorgrid = 2
    recolor = 3
    tile = 4
    pose = 5


@dataclass
class BriaControlNetOutput(BaseOutput):
    controlnet_block_samples: Tuple[torch.Tensor]
    controlnet_single_block_samples: Tuple[torch.Tensor]


class BriaControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Optional[List[int]] = None,
        num_mode: int = None,
        rope_theta: int = 10000,
        time_theta: int = 10000,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)
        axes_dims_rope = [16, 56, 56] if axes_dims_rope is None else axes_dims_rope
        self.pos_embed = EmbedND(theta=rope_theta, axes_dim=axes_dims_rope)

        # text_time_guidance_cls = (
        # CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        # )
        # self.time_text_embed = text_time_guidance_cls(
        # embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        # )
        self.time_embed = TimestepProjEmbeddings(embedding_dim=self.inner_dim, time_theta=time_theta)

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = torch.nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for i in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for i in range(num_single_layers)
            ]
        )

        # controlnet_blocks
        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(len(self.transformer_blocks)):
            self.controlnet_blocks.append(zero_module(nn.Linear(self.inner_dim, self.inner_dim)))

        self.controlnet_single_blocks = nn.ModuleList([])
        for _ in range(len(self.single_transformer_blocks)):
            self.controlnet_single_blocks.append(zero_module(nn.Linear(self.inner_dim, self.inner_dim)))

        self.union = num_mode is not None and num_mode > 0
        if self.union:
            self.controlnet_mode_embedder = nn.Embedding(num_mode, self.inner_dim)

        self.controlnet_x_embedder = zero_module(torch.nn.Linear(in_channels, self.inner_dim))

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self):
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.
        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.
                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.
        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @classmethod
    def from_transformer(
        cls,
        transformer,
        num_layers: int = 4,
        num_single_layers: int = 10,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        load_weights_from_transformer=True,
    ):
        config = transformer.config
        config["num_layers"] = num_layers
        config["num_single_layers"] = num_single_layers
        config["attention_head_dim"] = attention_head_dim
        config["num_attention_heads"] = num_attention_heads

        controlnet = cls(**config)

        if load_weights_from_transformer:
            controlnet.pos_embed.load_state_dict(transformer.pos_embed.state_dict())
            controlnet.time_text_embed.load_state_dict(transformer.time_text_embed.state_dict())
            controlnet.context_embedder.load_state_dict(transformer.context_embedder.state_dict())
            controlnet.x_embedder.load_state_dict(transformer.x_embedder.state_dict())
            controlnet.transformer_blocks.load_state_dict(transformer.transformer_blocks.state_dict(), strict=False)
            controlnet.single_transformer_blocks.load_state_dict(
                transformer.single_transformer_blocks.state_dict(), strict=False
            )

            controlnet.controlnet_x_embedder = zero_module(controlnet.controlnet_x_embedder)

        return controlnet

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        controlnet_mode: torch.Tensor = None,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            controlnet_cond (`torch.Tensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            controlnet_mode (`torch.Tensor`):
                The mode tensor of shape `(batch_size, 1)`.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for ControlNet outputs.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if guidance is not None:
            print("guidance is not supported in BriaControlNetModel")
        if pooled_projections is not None:
            print("pooled_projections is not supported in BriaControlNetModel")
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        hidden_states = self.x_embedder(hidden_states)

        # Convert controlnet_cond to the same dtype as the model weights
        controlnet_cond = controlnet_cond.to(dtype=self.controlnet_x_embedder.weight.dtype)

        # add
        hidden_states = hidden_states + self.controlnet_x_embedder(controlnet_cond)

        timestep = timestep.to(hidden_states.dtype)  # Original code was * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype)  # Original code was * 1000
        else:
            guidance = None

        temb = self.time_embed(timestep, dtype=hidden_states.dtype)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        if self.union:
            # union mode
            if controlnet_mode is None:
                raise ValueError("`controlnet_mode` cannot be `None` when applying ControlNet-Union")

            # Validate controlnet_mode values are within the valid range
            if torch.any(controlnet_mode < 0) or torch.any(controlnet_mode >= self.num_mode):
                raise ValueError(
                    f"`controlnet_mode` values must be in range [0, {self.num_mode - 1}], but got values outside this range"
                )

            # union mode emb
            controlnet_mode_emb = self.controlnet_mode_embedder(controlnet_mode)
            if controlnet_mode_emb.shape[0] < encoder_hidden_states.shape[0]:  # duplicate mode emb for each batch
                controlnet_mode_emb = controlnet_mode_emb.expand(
                    encoder_hidden_states.shape[0], 1, encoder_hidden_states.shape[2]
                )
            encoder_hidden_states = torch.cat([controlnet_mode_emb, encoder_hidden_states], dim=1)

        txt_ids = torch.cat((txt_ids[0:1, :], txt_ids), dim=0)
        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        block_samples = ()
        for _, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )
            block_samples = block_samples + (hidden_states,)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        single_block_samples = ()
        for _, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )
            single_block_samples = single_block_samples + (hidden_states[:, encoder_hidden_states.shape[1] :],)

        # controlnet block
        controlnet_block_samples = ()
        for block_sample, controlnet_block in zip(block_samples, self.controlnet_blocks, strict=False):
            block_sample = controlnet_block(block_sample)
            controlnet_block_samples = controlnet_block_samples + (block_sample,)

        controlnet_single_block_samples = ()
        for single_block_sample, controlnet_block in zip(
            single_block_samples, self.controlnet_single_blocks, strict=False
        ):
            single_block_sample = controlnet_block(single_block_sample)
            controlnet_single_block_samples = controlnet_single_block_samples + (single_block_sample,)

        # scaling
        controlnet_block_samples = [sample * conditioning_scale for sample in controlnet_block_samples]
        controlnet_single_block_samples = [sample * conditioning_scale for sample in controlnet_single_block_samples]

        controlnet_block_samples = None if len(controlnet_block_samples) == 0 else controlnet_block_samples
        controlnet_single_block_samples = (
            None if len(controlnet_single_block_samples) == 0 else controlnet_single_block_samples
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (controlnet_block_samples, controlnet_single_block_samples)

        return BriaControlNetOutput(
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
        )


class BriaMultiControlNetModel(ModelMixin):
    r"""
    `BriaMultiControlNetModel` wrapper class for Multi-BriaControlNetModel
    This module is a wrapper for multiple instances of the `BriaControlNetModel`. The `forward()` API is designed to be
    compatible with `BriaControlNetModel`.
    Args:
        controlnets (`List[BriaControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. You must set multiple
            `BriaControlNetModel` as a list.
    """

    def __init__(self, controlnets):
        super().__init__()
        self.nets = nn.ModuleList(controlnets)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        controlnet_cond: List[torch.tensor],
        controlnet_mode: List[torch.tensor],
        conditioning_scale: List[float],
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[BriaControlNetOutput, Tuple]:
        # ControlNet-Union with multiple conditions
        # only load one ControlNet for saving memories
        if len(self.nets) == 1 and self.nets[0].union:
            controlnet = self.nets[0]

            for i, (image, mode, scale) in enumerate(
                zip(controlnet_cond, controlnet_mode, conditioning_scale, strict=False)
            ):
                block_samples, single_block_samples = controlnet(
                    hidden_states=hidden_states,
                    controlnet_cond=image,
                    controlnet_mode=mode[:, None],
                    conditioning_scale=scale,
                    timestep=timestep,
                    guidance=guidance,
                    pooled_projections=pooled_projections,
                    encoder_hidden_states=encoder_hidden_states,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=return_dict,
                )

                # merge samples
                if i == 0:
                    control_block_samples = block_samples
                    control_single_block_samples = single_block_samples
                else:
                    control_block_samples = [
                        control_block_sample + block_sample
                        for control_block_sample, block_sample in zip(
                            control_block_samples, block_samples, strict=False
                        )
                    ]

                    control_single_block_samples = [
                        control_single_block_sample + block_sample
                        for control_single_block_sample, block_sample in zip(
                            control_single_block_samples, single_block_samples, strict=False
                        )
                    ]

        # Regular Multi-ControlNets
        # load all ControlNets into memories
        else:
            for i, (image, mode, scale, controlnet) in enumerate(
                zip(controlnet_cond, controlnet_mode, conditioning_scale, self.nets, strict=False)
            ):
                block_samples, single_block_samples = controlnet(
                    hidden_states=hidden_states,
                    controlnet_cond=image,
                    controlnet_mode=mode[:, None],
                    conditioning_scale=scale,
                    timestep=timestep,
                    guidance=guidance,
                    pooled_projections=pooled_projections,
                    encoder_hidden_states=encoder_hidden_states,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=return_dict,
                )

                # merge samples
                if i == 0:
                    control_block_samples = block_samples
                    control_single_block_samples = single_block_samples
                else:
                    if block_samples is not None and control_block_samples is not None:
                        control_block_samples = [
                            control_block_sample + block_sample
                            for control_block_sample, block_sample in zip(
                                control_block_samples, block_samples, strict=False
                            )
                        ]
                    if single_block_samples is not None and control_single_block_samples is not None:
                        control_single_block_samples = [
                            control_single_block_sample + block_sample
                            for control_single_block_sample, block_sample in zip(
                                control_single_block_samples, single_block_samples, strict=False
                            )
                        ]

        return control_block_samples, control_single_block_samples
