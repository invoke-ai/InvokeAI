"""Loaders for Fun-ControlNet branches (alibaba-pai VideoX-Fun); today the Qwen-Image Fun-ControlNet-Union."""

from pathlib import Path
from typing import Optional

from safetensors.torch import load_file

from invokeai.backend.fun_controlnet.qwen_image import QwenImageFunControlBranch
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.fp8_capability import Unimplemented
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.quantization.fp8_scaled import reject_quantized_side_channel
from invokeai.backend.util.devices import TorchDevice


@ModelLoaderRegistry.register(
    base=BaseModelType.QwenImage,
    type=ModelType.ControlNet,
    format=ModelFormat.Checkpoint,
    fp8_storage=Unimplemented(
        "the branch is cast to the compute dtype at load and runs as its own module tree, so the layerwise cast would "
        "fit; at 3.5 GB in bf16 it is worth closing"
    ),
)
class QwenImageFunControlNetLoader(ModelLoader):
    """Loads the Qwen-Image Fun-ControlNet-Union branch from its safetensors file.

    The file carries no config: the branch's geometry is read off its tensors and the base-layer map is the family's
    constant (`QWEN_IMAGE_FUN_CONTROL_LAYERS`).
    """

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        path = Path(config.path)
        if not isinstance(config, Checkpoint_Config_Base):
            raise ValueError("Only checkpoint configs are supported here.")
        sd = load_file(path)
        # Before the geometry probe: the shapes it reads mean nothing on a packed weight, and no scale is folded here.
        reject_quantized_side_channel(sd, f"Qwen Image ControlNet checkpoint {path.name}")
        return QwenImageFunControlBranch.from_state_dict(
            sd, dtype=TorchDevice.choose_bfloat16_safe_dtype(self._torch_device)
        )
