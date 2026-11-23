"""A Krea-2 Control LoRA file goes from identification through its registered loader to the Control LoRA patch."""

from logging import Logger
from unittest.mock import MagicMock

import torch
from safetensors.torch import save_file

from invokeai.backend.krea2.control_lora import Krea2ControlLoRAPatch
from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.lora import ControlLoRA_LyCORIS_Krea2_Config
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.patches.lora_conversions.krea2_lora_constants import KREA2_LORA_TRANSFORMER_PREFIX


def test_a_control_lora_file_loads_as_a_krea2_control_lora_patch(tmp_path) -> None:
    path = tmp_path / "depth-control-lora.safetensors"
    weight, bias = torch.randn(6144, 128), torch.randn(6144)
    save_file(
        {
            "blocks.0.attn.wq.A": torch.randn(2, 8),
            "blocks.0.attn.wq.B": torch.randn(8, 2),
            "first.weight": weight,
            "first.bias": bias,
        },
        str(path),
    )
    config = ModelConfigFactory.from_model_on_disk(path, allow_unknown=False).config
    assert isinstance(config, ControlLoRA_LyCORIS_Krea2_Config)
    implementation, _, _ = ModelLoaderRegistry.get_implementation(config, None)
    loader = implementation.__new__(implementation)
    loader._logger = MagicMock(spec=Logger)
    loader._model_base = config.base
    loader._torch_dtype = torch.float32

    patch = loader._load_model(config)

    assert isinstance(patch, Krea2ControlLoRAPatch)
    assert list(patch.layers) == [f"{KREA2_LORA_TRANSFORMER_PREFIX}transformer_blocks.0.attn.to_q"]
    assert torch.equal(patch.input_projection_weight, weight)
    assert torch.equal(patch.input_projection_bias, bias)
