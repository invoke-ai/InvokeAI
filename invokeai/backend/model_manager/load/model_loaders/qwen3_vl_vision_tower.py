from pathlib import Path
from typing import Optional

import accelerate
import torch

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.qwen3_vl_encoder import Qwen3VLVisionTower_GGUF_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.model_manager.util.qwen3_vl_mmproj import mmproj_to_vision_tower_state_dict
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader
from invokeai.backend.qwen3_vl.qwen3_vl_assets import load_bundled_qwen3_vl_config_dict
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.state_dict_loading import load_state_dict_ignoring_extras, reject_incomplete_load


@ModelLoaderRegistry.register(
    base=BaseModelType.Any, type=ModelType.Qwen3VLVisionTower, format=ModelFormat.GGUFQuantized
)
class Qwen3VLVisionTowerGGUFLoader(ModelLoader):
    """Loads a llama.cpp ``mmproj`` GGUF as the HuggingFace ``Qwen3VLVisionModel`` the other Qwen3-VL encoders' towers are.

    The model is the tower itself, not a submodel: an edit attaches it to a GGUF encoder (`attached_vision_tower`)
    exactly as it does the tower of a safetensors encoder or a Diffusers pipeline. Weights are dequantized to the
    compute dtype at load, whatever the file stores (F16 for the released mmprojs, Q8_0 for the smaller ones).
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        from transformers import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        if not isinstance(config, Qwen3VLVisionTower_GGUF_Config):
            raise ValueError("Only Qwen3VLVisionTower_GGUF_Config models are supported here.")
        if submodel_type is not None:
            raise ValueError("A Qwen3-VL vision tower has no submodels.")

        dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())
        vision_config = Qwen3VLConfig.from_dict(load_bundled_qwen3_vl_config_dict(config.variant)).vision_config

        raw = gguf_sd_loader(Path(config.path), compute_dtype=dtype)
        tensors: dict[str, torch.Tensor] = {}
        for name in list(raw):
            tensor = raw.pop(name)
            tensors[name] = (tensor.get_dequantized_tensor() if isinstance(tensor, GGMLTensor) else tensor).to(dtype)
        state_dict = mmproj_to_vision_tower_state_dict(tensors, vision_config.deepstack_visual_indexes)

        what = f"Qwen3-VL vision tower '{config.name}'"
        with accelerate.init_empty_weights():
            tower = Qwen3VLVisionModel._from_config(vision_config, torch_dtype=dtype)
        load_state_dict_ignoring_extras(tower, state_dict, source=what, assign=True, allow_missing=True)
        reject_incomplete_load(tower, what=what)
        return tower.eval()
