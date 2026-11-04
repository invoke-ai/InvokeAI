"""Identifying the FLUX.3 prompt upsampler -- a Qwen3.5-9B vision-language model -- as a Text LLM.

Two containers: the release folder (`black-forest-labs/flux-3-image-klein-pup`) and ComfyUI's single files. The
single files are their captured headers written back without data (`write_header_only_safetensors`), with the int8
build's `.comfy_quant` markers as the only bytes: identification reads nothing else, and it runs through the whole
config factory, so a second config claiming a file shows up as a second match. What separates the upsampler from a
Qwen3.5 text-encoder export is its LM head alone, so both sides of that line are pinned here, on the real header.
"""

import shutil
from pathlib import Path
from typing import Any

import pytest

import invokeai.backend.qwen3_5 as qwen3_5_package
from invokeai.backend.model_manager.configs.factory import ModelClassificationResult, ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError
from invokeai.backend.model_manager.configs.qwen3_5_encoder import Qwen35Encoder_Checkpoint_Config
from invokeai.backend.model_manager.configs.qwen3_vl_encoder import Qwen3VLEncoder_Checkpoint_Config
from invokeai.backend.model_manager.configs.text_llm import TextLLM_Checkpoint_Config, TextLLM_Diffusers_Config
from invokeai.backend.model_manager.taxonomy import ModelFormat, ModelType, Qwen35VariantType
from tests.backend.model_manager.load.state_dicts import (
    flux3_pup_bf16_keys,
    flux3_pup_int8_convrot_keys,
    flux3_pup_w4a8_keys,
    qwen3_vl_4b_comfy_keys,
)
from tests.backend.model_manager.load.state_dicts.utils import write_header_only_safetensors

Keys = dict[str, tuple[list[int], str]]

# The release's `config.json`, as vendored verbatim beside the Qwen3.5 assets.
_RELEASE_CONFIG = Path(qwen3_5_package.__file__).parent / "qwen3_5_9b_vl_config.json"

_BF16: Keys = flux3_pup_bf16_keys.state_dict_keys
_INT8: Keys = flux3_pup_int8_convrot_keys.state_dict_keys

# A layer every build quantizes or keeps as one unit; the quantized variants below are written onto it.
_LAYER = "model.language_model.layers.0.mlp.gate_proj"


def _identify(path: Path, override_fields: dict[str, Any] | None = None) -> ModelClassificationResult:
    return ModelConfigFactory.from_model_on_disk(path, override_fields, allow_unknown=False)


def _matches(result: ModelClassificationResult) -> list[str]:
    return [type(config).__name__ for config in result.all_matches]


def _release_folder(root: Path, *, processor: bool = True) -> Path:
    root.mkdir()
    shutil.copyfile(_RELEASE_CONFIG, root / "config.json")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (root / name).write_text("{}")
    if processor:
        for name in ("processor_config.json", "preprocessor_config.json"):
            (root / name).write_text("{}")
    return root


class TestTheReleaseFolder:
    def test_it_is_a_text_llm_that_reads_images_and_nothing_else(self, tmp_path: Path) -> None:
        result = _identify(_release_folder(tmp_path / "flux-3-image-klein-pup"))

        assert _matches(result) == [TextLLM_Diffusers_Config.__name__]
        assert isinstance(result.config, TextLLM_Diffusers_Config)
        assert result.config.supports_images is True

    def test_without_its_processor_files_it_reads_text_only(self, tmp_path: Path) -> None:
        result = _identify(_release_folder(tmp_path / "flux-3-image-klein-pup", processor=False))

        assert isinstance(result.config, TextLLM_Diffusers_Config)
        assert result.config.supports_images is False


def _with_layer(keys: Keys, **tensors: tuple[list[int], str]) -> Keys:
    """`keys` with `_LAYER`'s tensors replaced: `weight=...`, `weight_scale=...` and so on."""
    replaced = {key: value for key, value in keys.items() if not key.startswith(f"{_LAYER}.")}
    return replaced | {f"{_LAYER}.{name}": value for name, value in tensors.items()}


class TestComfyUISingleFiles:
    @pytest.mark.parametrize(
        ("keys", "markers"),
        [(_BF16, None), (_INT8, flux3_pup_int8_convrot_keys.markers)],
        ids=["bf16", "int8_convrot"],
    )
    def test_each_build_is_a_text_llm_that_reads_images_and_nothing_else(
        self, tmp_path: Path, keys: Keys, markers: dict | None
    ) -> None:
        result = _identify(write_header_only_safetensors(tmp_path / "upsampler.safetensors", keys, markers))

        assert _matches(result) == [TextLLM_Checkpoint_Config.__name__]
        assert isinstance(result.config, TextLLM_Checkpoint_Config)
        assert result.config.format is ModelFormat.Checkpoint
        assert result.config.supports_images is True

    @pytest.mark.parametrize(
        ("keys", "scheme"),
        [
            (flux3_pup_w4a8_keys.state_dict_keys, "asym_w4a8_int8"),
            (_with_layer(_BF16, weight=([12288, 4096], "F8_E4M3"), weight_scale=([], "F32")), "scaled fp8"),
            (_with_layer(_BF16, weight=([12288, 4096], "F8_E4M3"), weight_scale=([12288, 128], "U8")), "mxfp8"),
            (
                _with_layer(
                    _BF16,
                    weight=([12288, 2048], "U8"),
                    weight_scale=([12288, 256], "F8_E4M3"),
                    weight_scale_2=([], "F32"),
                    input_scale=([], "F32"),
                ),
                "nvfp4",
            ),
            # An int8 build whose markers are gone: nothing says how to decode it.
            (
                {key: value for key, value in _INT8.items() if not key.endswith(".comfy_quant")},
                "int8 without an int8_tensorwise marker",
            ),
        ],
        ids=["w4a8", "scaled fp8", "mxfp8", "nvfp4", "unmarked int8"],
    )
    def test_a_build_the_loader_cannot_decode_is_refused_naming_its_scheme(
        self, tmp_path: Path, keys: Keys, scheme: str
    ) -> None:
        result = ModelConfigFactory.from_model_on_disk(
            write_header_only_safetensors(tmp_path / "upsampler.safetensors", keys), allow_unknown=True
        )

        # Final, not "unknown": nothing is written to the database.
        assert result.config is None
        refusal = result.details[TextLLM_Checkpoint_Config.__name__]
        assert isinstance(refusal, InvalidMatchError)
        assert scheme in str(refusal)


class TestTheLineBetweenTheUpsamplerAndAQwen3_5Encoder:
    """The upsampler is a Qwen3.5 encoder export plus `lm_head.weight`; MOSS-Image's ComfyUI export is the same model
    with the vision tower and no LM head. Each config has to claim its side of that line and only its side, and the
    encoder keeps every file the Text LLM loader could not build."""

    def test_without_its_lm_head_the_upsampler_is_an_encoder_export(self, tmp_path: Path) -> None:
        keys = {key: value for key, value in _BF16.items() if key != "lm_head.weight"}

        result = _identify(write_header_only_safetensors(tmp_path / "qwen35_9b.safetensors", keys))

        assert _matches(result) == [Qwen35Encoder_Checkpoint_Config.__name__]
        assert isinstance(result.config, Qwen35Encoder_Checkpoint_Config)
        assert result.config.variant is Qwen35VariantType.Qwen35_9B

    def test_an_explicit_encoder_type_still_installs_the_upsampler_as_one(self, tmp_path: Path) -> None:
        path = write_header_only_safetensors(tmp_path / "upsampler.safetensors", _BF16)

        result = _identify(path, {"type": ModelType.Qwen35Encoder})

        assert _matches(result) == [Qwen35Encoder_Checkpoint_Config.__name__]

    def test_a_size_without_a_vendored_config_stays_a_qwen3_5_encoder(self, tmp_path: Path) -> None:
        """The Text LLM loader builds the 9B only. A 4B with its tower and LM head -- the embedding width decides --
        is still the encoder Anima-3.8B reads."""
        keys = dict(_BF16)
        keys["model.language_model.embed_tokens.weight"] = ([248320, 2560], "BF16")

        result = _identify(write_header_only_safetensors(tmp_path / "qwen35_4b.safetensors", keys))

        assert _matches(result) == [Qwen35Encoder_Checkpoint_Config.__name__]
        assert isinstance(result.config, Qwen35Encoder_Checkpoint_Config)
        assert result.config.variant is Qwen35VariantType.Qwen35_4B

    def test_a_text_only_export_with_an_lm_head_stays_an_encoder(self, tmp_path: Path) -> None:
        """No vision tower, nothing the Text LLM loader builds: an LM head alone does not take a file from the
        encoder config."""
        keys = {key: value for key, value in _BF16.items() if not key.startswith("model.visual.")}

        result = _identify(write_header_only_safetensors(tmp_path / "qwen35_9b_causal_lm.safetensors", keys))

        assert _matches(result) == [Qwen35Encoder_Checkpoint_Config.__name__]

    def test_a_qwen3_vl_export_of_the_same_outer_layout_stays_the_qwen3_vl_encoder(self, tmp_path: Path) -> None:
        path = tmp_path / "qwen3vl_4b_bf16.safetensors"

        result = _identify(write_header_only_safetensors(path, qwen3_vl_4b_comfy_keys.state_dict_keys))

        assert _matches(result) == [Qwen3VLEncoder_Checkpoint_Config.__name__]
