import json
from pathlib import Path

from invokeai.backend.model_manager.configs.controlnet import ControlNet_Diffusers_SDXL_Config
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import ModelRepoVariant


def test_controlnet_diffusers_fp16_repo_variant(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    config_path = model_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "_class_name": "ControlNetModel",
                "cross_attention_dim": 2048,
            }
        )
    )

    (model_dir / "diffusion_pytorch_model.fp16.safetensors").touch()

    mod = ModelOnDisk(model_dir)
    config = ControlNet_Diffusers_SDXL_Config.from_model_on_disk(mod, {})

    assert config.repo_variant is ModelRepoVariant.FP16
