"""The loader for Qwen-Image ControlNets, in either design the config records: alibaba-pai's Fun-ControlNet-Union
branch or InstantX's ControlNet-Union."""

from collections.abc import Mapping
from typing import Optional, assert_never

import accelerate
import torch
from diffusers.models.controlnets.controlnet_qwenimage import QwenImageControlNetModel

from invokeai.backend.fun_controlnet.qwen_image import QwenImageFunControlBranch
from invokeai.backend.model_manager.configs.controlnet import ControlNet_Checkpoint_QwenImage_Config
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.fp8_capability import Unimplemented
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.fun_controlnet import read_dense_checkpoint
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.util.devices import TorchDevice

# Qwen-Image's RoPE splits a 128-wide head into frame, height and width axes. The ControlNet runs its own RoPE over the
# same positions as the transformer and the file does not record the split, so it is a constant of the family.
QWEN_IMAGE_AXES_DIMS_ROPE = (16, 56, 56)


def load_qwen_image_instantx_controlnet(
    sd: Mapping[str, torch.Tensor], *, dtype: torch.dtype
) -> QwenImageControlNetModel:
    """Build the `QwenImageControlNetModel` the checkpoint describes and load it strictly, cast to ``dtype``.

    The geometry is read off the tensors: the width and latent features from `img_in`, the text width from `txt_in`,
    the head width from the query norm, the block count from the highest `controlnet_blocks.N` index. A control input
    wider than the latent is the inpainting layout, whose extra mask features nothing here produces, so it is refused.
    """
    required = (
        "img_in.weight",
        "txt_in.weight",
        "controlnet_x_embedder.weight",
        "transformer_blocks.0.attn.norm_q.weight",
    )
    absent = [key for key in required if key not in sd]
    if absent:
        raise ValueError(f"Not an InstantX Qwen Image ControlNet checkpoint: {', '.join(absent)} missing.")
    inner_dim, in_channels = sd["img_in.weight"].shape
    control_features = sd["controlnet_x_embedder.weight"].shape[1]
    if control_features != in_channels:
        raise ValueError(
            f"This InstantX Qwen Image ControlNet takes {control_features} control features per token, {in_channels} "
            "for the latent plus a mask: the inpainting layout, which is not supported yet."
        )
    head_dim = sd["transformer_blocks.0.attn.norm_q.weight"].shape[0]
    if head_dim != sum(QWEN_IMAGE_AXES_DIMS_ROPE) or inner_dim % head_dim:
        raise ValueError(
            f"A {inner_dim}-wide ControlNet with {head_dim}-wide attention heads is not Qwen-Image's geometry "
            f"({sum(QWEN_IMAGE_AXES_DIMS_ROPE)}-wide heads)."
        )
    block_indices = {
        int(index) for key in sd if key.startswith("controlnet_blocks.") and (index := key.split(".")[1]).isdigit()
    }
    num_layers = max(block_indices, default=-1) + 1
    if num_layers == 0:
        raise ValueError("Not an InstantX Qwen Image ControlNet checkpoint: it has no `controlnet_blocks`.")

    with accelerate.init_empty_weights():
        model = QwenImageControlNetModel(
            in_channels=in_channels,
            num_layers=num_layers,
            attention_head_dim=head_dim,
            num_attention_heads=inner_dim // head_dim,
            joint_attention_dim=sd["txt_in.weight"].shape[1],
            axes_dims_rope=QWEN_IMAGE_AXES_DIMS_ROPE,
        )
    missing, unexpected = model.load_state_dict(dict(sd), strict=False, assign=True)
    if missing or unexpected:
        raise ValueError(
            "This checkpoint does not match the InstantX Qwen Image ControlNet layout: "
            f"missing {sorted(missing)[:5]}, unexpected {sorted(unexpected)[:5]}."
        )
    return model.to(dtype=dtype).eval()


@ModelLoaderRegistry.register(
    base=BaseModelType.QwenImage,
    type=ModelType.ControlNet,
    format=ModelFormat.Checkpoint,
    fp8_storage=Unimplemented(
        "both designs are cast to the compute dtype at load and run as their own module trees, so the layerwise cast "
        "would fit; at 3.5 GB in bf16 each it is worth closing"
    ),
)
class QwenImageControlNetLoader(ModelLoader):
    """Loads a Qwen-Image ControlNet from its safetensors file, in the design its record names.

    Neither file carries a config: the geometry is read off the tensors, and the Fun branch's base-layer map is the
    family's constant (`QWEN_IMAGE_FUN_CONTROL_LAYERS`).
    """

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        if not isinstance(config, ControlNet_Checkpoint_QwenImage_Config):
            raise ValueError("Only Qwen Image ControlNet checkpoint configs are supported here.")
        sd = read_dense_checkpoint(config, "Qwen Image")
        dtype = TorchDevice.choose_bfloat16_safe_dtype(self._torch_device)
        match config.control_architecture:
            case "fun_union":
                return QwenImageFunControlBranch.from_state_dict(sd, dtype=dtype)
            case "instantx_union":
                return load_qwen_image_instantx_controlnet(sd, dtype=dtype)
            case _:
                assert_never(config.control_architecture)
