"""Identification of the Anima IP-Adapter and of its SigLIP2 encoder.

Config classes are tried in no fixed order, so the IP-Adapter must be claimed by exactly one of them. Its bundled
LoRA keys (`lora.base_model.model.blocks.N.cross_attn...`) look enough like an Anima PEFT LoRA to be worth proving
the LoRA probes, the generic IP-Adapter checkpoints and everything else leave it alone -- with the real key names
and shapes, which no single test file of a few megabytes could carry, so they are served as meta tensors.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.ip_adapter import IPAdapter_Checkpoint_Anima_Config
from invokeai.backend.model_manager.configs.siglip import SigLIP_Diffusers_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from tests.backend.patches.lora_conversions.lora_state_dicts.anima_ip_adapter_format import (
    metadata,
    state_dict_keys,
)


class _CheckpointOnDisk(ModelOnDisk):
    """A real file on disk (for hashing and path checks) whose contents are the released checkpoint's layout."""

    def __init__(self, path: Path, state_dict: dict[str, torch.Tensor], file_metadata: dict[str, str]):
        super().__init__(path)
        self._sd = state_dict
        self._file_metadata = file_metadata

    def load_state_dict(self, path=None):
        return self._sd

    def metadata(self, path=None):
        return self._file_metadata


def _identify(tmp_path: Path, state_dict: dict[str, torch.Tensor]):
    path = tmp_path / "ip_adapter-Character_Reference-10.safetensors"
    save_file({"placeholder": torch.zeros(1)}, path)
    return ModelConfigFactory.from_model_on_disk(_CheckpointOnDisk(path, state_dict, metadata), allow_unknown=False)


def _released_checkpoint() -> dict[str, torch.Tensor]:
    return {key: torch.empty(shape, device="meta") for key, shape in state_dict_keys.items()}


def test_the_released_checkpoint_is_claimed_by_the_anima_ip_adapter_config_alone(tmp_path: Path) -> None:
    result = _identify(tmp_path, _released_checkpoint())

    matches = [name for name, outcome in result.details.items() if isinstance(outcome, Config_Base)]
    assert matches == [IPAdapter_Checkpoint_Anima_Config.__name__]
    assert isinstance(result.config, IPAdapter_Checkpoint_Anima_Config)
    assert (result.config.base, result.config.type, result.config.format) == (
        BaseModelType.Anima,
        ModelType.IPAdapter,
        ModelFormat.Checkpoint,
    )


def test_an_ip_adapter_for_another_width_is_not_an_anima_one(tmp_path: Path) -> None:
    state_dict = _released_checkpoint()
    state_dict["blocks.0.ip_k_proj.weight"] = torch.empty(3072, 768, device="meta")

    result = _identify(tmp_path, state_dict)

    assert not isinstance(result.details[IPAdapter_Checkpoint_Anima_Config.__name__], Config_Base)


def _siglip_dir(tmp_path: Path, config: dict) -> Path:
    model_dir = tmp_path / "siglip"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config))
    save_file({"placeholder": torch.zeros(1)}, model_dir / "model.safetensors")
    return model_dir


# google/siglip2-base-patch16-512's config.json, verbatim: a model type, no class name.
_SIGLIP2_CONFIG = {
    "initializer_factor": 1.0,
    "model_type": "siglip",
    "text_config": {"model_type": "siglip_text_model", "vocab_size": 256000},
    "transformers_version": "4.49.0.dev0",
    "vision_config": {"image_size": 512, "model_type": "siglip_vision_model"},
}


def test_a_siglip2_config_without_a_class_name_is_a_siglip_model(tmp_path: Path) -> None:
    result = ModelConfigFactory.from_model_on_disk(_siglip_dir(tmp_path, _SIGLIP2_CONFIG), allow_unknown=False)

    assert isinstance(result.config, SigLIP_Diffusers_Config)


def test_a_config_that_names_another_class_is_still_refused(tmp_path: Path) -> None:
    config = {**_SIGLIP2_CONFIG, "architectures": ["SiglipForImageClassification"]}

    result = ModelConfigFactory.from_model_on_disk(_siglip_dir(tmp_path, config), allow_unknown=True)

    assert not isinstance(result.config, SigLIP_Diffusers_Config)
