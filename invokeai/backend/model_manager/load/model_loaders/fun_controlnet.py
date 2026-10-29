"""The FLUX.2 [dev] Fun-ControlNet loader (alibaba-pai VideoX-Fun), and the dense-checkpoint read every ControlNet
branch loader shares. The Qwen-Image registration, which also loads InstantX's design, is `qwen_image_controlnet.py`."""

from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file

from invokeai.backend.fun_controlnet.flux2 import Flux2FunControlBranch
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.fp8_capability import Unimplemented
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.quantization.fp8_scaled import reject_quantized_side_channel


def read_dense_checkpoint(config: AnyModelConfig, label: str) -> dict[str, torch.Tensor]:
    """The ControlNet's tensors, refused if any carries a quantization side channel.

    Refused before the geometry probe: the shapes it reads mean nothing on a packed weight, and the loaders assign the
    file's tensors as they are, so no scale would be folded in. alibaba-pai and Comfy-Org publish dense bf16 files
    only.
    """
    if not isinstance(config, Checkpoint_Config_Base):
        raise ValueError("Only checkpoint configs are supported here.")
    path = Path(config.path)
    sd = load_file(path)
    reject_quantized_side_channel(sd, f"{label} ControlNet checkpoint {path.name}")
    return sd


@ModelLoaderRegistry.register(
    base=BaseModelType.Flux2,
    type=ModelType.ControlNet,
    format=ModelFormat.Checkpoint,
    fp8_storage=Unimplemented(
        "the branch is cast to bfloat16 at load and runs as its own module tree, so the layerwise cast would fit; at "
        "8.2 GB in bf16, resident beside a 32B transformer, it is worth closing"
    ),
)
class Flux2FunControlNetLoader(ModelLoader):
    """Loads the FLUX.2 [dev] Fun-ControlNet-Union branch from its safetensors file.

    The file carries no config: the branch's geometry is read off its tensors and the base-layer map is the family's
    constant (`FLUX2_FUN_CONTROL_LAYERS`). Cast to bfloat16, the dtype every FLUX.2 transformer loader forces and
    `flux2_denoise` runs in.
    """

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        return Flux2FunControlBranch.from_state_dict(read_dense_checkpoint(config, "FLUX.2"), dtype=torch.bfloat16)
