"""Tests that an SDNQ-quantized ZImagePipeline folder identifies as the SDNQ config.

A full ZImagePipeline folder whose ``transformer/quantization_config.json`` has
``quant_method: "sdnq"`` must classify as ``Main_SDNQ_Diffusers_ZImage_Config``, not as the plain
``Main_Diffusers_ZImage_Config``. Both configs accept the same ZImagePipeline class name, and the
factory only breaks ties by model type, so without an explicit SDNQ-rejection guard on the plain
config the pipeline could be classified as plain diffusers — which would then mis-read the packed
uint8 weights and, via the self-contained path in the Z-Image model loader, force the user to
select separate VAE/Qwen3 sources for a single installed pipeline.

This mirrors the SDNQ-rejection guards already present on the FLUX and FLUX.2 diffusers configs.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.main import (
    Main_Diffusers_ZImage_Config,
    Main_SDNQ_Diffusers_ZImage_Config,
)

_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/z-image-sdnq",
    "file_size": 1000,
    "name": "z-image-sdnq",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def _make_sdnq_zimage_pipeline_folder(root: Path) -> Path:
    """Create a minimal SDNQ-quantized ZImagePipeline folder on disk."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "ZImagePipeline",
                "transformer": ["diffusers", "ZImageTransformer2DModel"],
                "text_encoder": ["transformers", "Qwen3ForCausalLM"],
                "tokenizer": ["transformers", "Qwen2Tokenizer"],
                "vae": ["diffusers", "AutoencoderKL"],
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            }
        ),
        encoding="utf-8",
    )

    transformer_dir = root / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        json.dumps({"_class_name": "ZImageTransformer2DModel"}), encoding="utf-8"
    )
    # The SDNQ marker: quant_method == "sdnq".
    (transformer_dir / "quantization_config.json").write_text(
        json.dumps({"quant_method": "sdnq", "weights_dtype": "uint4", "group_size": 128}), encoding="utf-8"
    )

    scheduler_dir = root / "scheduler"
    scheduler_dir.mkdir()
    (scheduler_dir / "scheduler_config.json").write_text(json.dumps({"shift": 3.0}), encoding="utf-8")

    return root


class TestZImageSDNQIdentification:
    def test_plain_diffusers_config_rejects_sdnq_transformer(self, tmp_path: Path):
        """Main_Diffusers_ZImage_Config must NOT accept an SDNQ ZImagePipeline folder."""
        root = _make_sdnq_zimage_pipeline_folder(tmp_path / "z-image-sdnq")
        mod = MagicMock()
        mod.path = root

        with pytest.raises(NotAMatchError):
            Main_Diffusers_ZImage_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})

    def test_sdnq_config_accepts_sdnq_transformer(self, tmp_path: Path):
        """Main_SDNQ_Diffusers_ZImage_Config must accept the same folder and expose submodels."""
        root = _make_sdnq_zimage_pipeline_folder(tmp_path / "z-image-sdnq")
        mod = MagicMock()
        mod.path = root

        config = Main_SDNQ_Diffusers_ZImage_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
        assert config.format.value == "sdnq_quantized"
        assert config.base.value == "z-image"
        assert config.submodels

    def test_factory_resolves_to_sdnq_config(self, tmp_path: Path):
        """End-to-end: the factory classifies the folder as the SDNQ config, not plain diffusers."""
        root = _make_sdnq_zimage_pipeline_folder(tmp_path / "z-image-sdnq")

        result = ModelConfigFactory.from_model_on_disk(root, allow_unknown=True)

        assert result.config is not None
        assert isinstance(result.config, Main_SDNQ_Diffusers_ZImage_Config)
        # The plain diffusers config must not be among the matches.
        assert not any(isinstance(m, Main_Diffusers_ZImage_Config) for m in result.all_matches)
