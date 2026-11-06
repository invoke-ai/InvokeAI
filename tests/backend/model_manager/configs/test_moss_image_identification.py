"""Identifying MOSS-Image's transformer, VAE and pipeline folder, through the whole config factory.

The single files are captured headers written back without their data (`write_header_only_safetensors`):
identification reads nothing else, so every config class sees the real key names, shapes and dtypes, and a second
config claiming the file shows up as a second match.
"""

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import gguf
import numpy as np
import pytest
import torch

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_MossImage_Config,
    Main_Diffusers_MossImage_Config,
    Main_GGUF_MossImage_Config,
)
from invokeai.backend.model_manager.configs.vae import VAE_Checkpoint_Flux2_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from tests.backend.model_manager.load.moss_image_fixtures import fuse_qkv_keys, int8_convrot_keys
from tests.backend.model_manager.load.state_dicts import moss_image_dit_keys, moss_image_vae_keys
from tests.backend.model_manager.load.state_dicts.utils import write_header_only_safetensors
from tests.fixtures.quantized_payloads import q8_cr_marker, quantize_convrot, write_gguf

Keys = dict[str, tuple[list[int], str]]


def _identify(path: Path) -> Any:
    return ModelConfigFactory.from_model_on_disk(path, allow_unknown=False)


def _dit() -> Keys:
    return dict(moss_image_dit_keys.state_dict_keys)


def test_the_transformer_is_a_moss_image_checkpoint_and_nothing_else(tmp_path: Path) -> None:
    result = _identify(write_header_only_safetensors(tmp_path / "moss-image-dit.safetensors", _dit()))

    assert type(result.config) is Main_Checkpoint_MossImage_Config
    assert result.config.base is BaseModelType.MossImage
    assert result.config.format is ModelFormat.Checkpoint
    assert result.match_count == 1


def test_comfy_orgs_fused_bf16_export_is_identified_as_the_same_checkpoint(tmp_path: Path) -> None:
    keys = fuse_qkv_keys(_dit())
    # Header facts of moss_image_base_bf16.safetensors: 463 keys, minus the two projections each of 42 blocks fused.
    assert len(keys) == 379
    assert keys["layers.0.proj.qkv.weight"] == ([9216, 3072], "BF16")

    result = _identify(write_header_only_safetensors(tmp_path / "moss_image_base_bf16.safetensors", keys))

    assert type(result.config) is Main_Checkpoint_MossImage_Config
    assert result.match_count == 1


def test_comfy_orgs_int8_convrot_export_is_identified_as_a_checkpoint(tmp_path: Path) -> None:
    keys, markers = int8_convrot_keys(_dit())
    # Header facts of moss_image_base_int8_convrot.safetensors: 379 tensors plus, per block, four scales, four
    # markers and an attention-kernel record.
    assert len(keys) + len(markers) == 757
    assert keys["layers.0.proj.qkv.weight_scale"] == ([9216, 1], "F32")

    result = _identify(
        write_header_only_safetensors(tmp_path / "moss_image_base_int8_convrot.safetensors", keys, markers)
    )

    assert type(result.config) is Main_Checkpoint_MossImage_Config
    assert result.config.format is ModelFormat.Checkpoint
    assert result.match_count == 1


def _write_header_only_gguf(path: Path, keys: Keys, architecture: str | None) -> Path:
    """A GGUF that has a layout's real tensor names and one element each. Identification reads names and types."""
    writer = gguf.GGUFWriter(str(path), architecture or "unused")
    if architecture is None:
        # The writer always declares an architecture; leejet's files have no key/value pairs at all.
        writer.kv_data = [{}]
    for key in keys:
        writer.add_tensor(key, np.zeros(1, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def test_unsloths_gguf_is_identified_by_its_keys_and_its_architecture_is_not_needed(tmp_path: Path) -> None:
    keys = _dit()
    assert len(keys) == 463

    result = _identify(_write_header_only_gguf(tmp_path / "MOSS-Image-Base-Q4_K_M.gguf", keys, "moss_image"))

    assert type(result.config) is Main_GGUF_MossImage_Config
    assert result.config.format is ModelFormat.GGUFQuantized
    assert result.match_count == 1


def test_leejets_gguf_without_any_metadata_is_identified_by_its_fused_keys(tmp_path: Path) -> None:
    keys = fuse_qkv_keys(_dit())
    assert len(keys) == 379

    path = _write_header_only_gguf(tmp_path / "MOSS-Image-SFT-Q4_K_M.gguf", keys, None)
    assert [name for name in gguf.GGUFReader(str(path)).fields if not name.startswith("GGUF.")] == []
    result = _identify(path)

    assert type(result.config) is Main_GGUF_MossImage_Config
    assert result.match_count == 1


def test_a_comfyui_gguf_convrot_build_is_refused_with_its_reason(tmp_path: Path) -> None:
    int8 = quantize_convrot(torch.randn(4, 256))
    path = tmp_path / "MOSS-Image-Base-Q8_CR.gguf"
    write_gguf(
        path,
        {
            "shared_block_modulation.weight": torch.ones(8, 4),
            "final_linear_video.weight": torch.ones(8, 4),
            "text_embedder.blocks.0.qkv.weight": torch.ones(12, 4),
            "layers.0.proj.o.weight": int8.codes,
            "layers.0.proj.o.weight_scale": int8.scale,
        },
        quant={"layers.0.proj.o.weight": q8_cr_marker()},
        architecture="moss_image",
    )

    result = ModelConfigFactory.from_model_on_disk(path, allow_unknown=True)

    assert result.config is None
    assert any("Q8_CR" in str(reason) for reason in result.invalid_matches)


def test_a_comfyui_prefixed_transformer_is_still_identified(tmp_path: Path) -> None:
    keys = {f"model.diffusion_model.{key}": value for key, value in _dit().items()}

    result = _identify(write_header_only_safetensors(tmp_path / "moss.safetensors", keys))

    assert type(result.config) is Main_Checkpoint_MossImage_Config
    assert result.match_count == 1


def test_the_vae_is_one_flux2_vae(tmp_path: Path) -> None:
    keys = dict(moss_image_vae_keys.state_dict_keys)

    result = _identify(write_header_only_safetensors(tmp_path / "moss-image-vae.safetensors", keys))

    assert type(result.config) is VAE_Checkpoint_Flux2_Config
    assert result.match_count == 1


def test_a_lora_for_the_transformer_is_not_a_main_model(tmp_path: Path) -> None:
    # A LoRA addresses the same modules, so the fingerprint's module names are all there.
    keys: Keys = {}
    for module in ("shared_block_modulation", "final_linear_video", "text_embedder.blocks.0.qkv", "layers.0.proj.q"):
        keys[f"diffusion_model.{module}.lora_A.weight"] = ([16, 3072], "BF16")
        keys[f"diffusion_model.{module}.lora_B.weight"] = ([3072, 16], "BF16")
        keys[f"diffusion_model.{module}.weight"] = ([3072, 3072], "BF16")

    result = ModelConfigFactory.from_model_on_disk(
        write_header_only_safetensors(tmp_path / "moss-lora.safetensors", keys), allow_unknown=True
    )

    assert isinstance(result.details[Main_Checkpoint_MossImage_Config.__name__], NotAMatchError)


def _torchao_fp8(
    keys: Keys, *, scale_shape: list[int] | None = None, zero_point: bool = False, drop_scale_of: str | None = None
) -> None:
    """Store every block projection as unsloth's torchao build does: `_weight_qdata` plus a per-row `_weight_scale`
    in place of `.weight`. `zero_point` makes the payload int8 with a zero point, as the INT8 build is."""
    for key in [k for k in keys if re.fullmatch(r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)\.weight", k)]:
        shape, _ = keys.pop(key)
        layer = key.removesuffix(".weight")
        keys[f"{layer}._weight_qdata"] = (shape, "I8" if zero_point else "F8_E4M3")
        keys[f"{layer}._weight_scale"] = (scale_shape or [shape[0], 1], "F32")
        if layer == drop_scale_of:
            del keys[f"{layer}._weight_scale"]
        if zero_point:
            keys[f"{layer}._weight_zero_point"] = ([shape[0], 1], "I8")


def test_unsloths_torchao_fp8_build_installs_with_fp8_storage_on(tmp_path: Path) -> None:
    keys = _dit()
    _torchao_fp8(keys)
    # Header facts of MOSS-Image-Base-FP8.safetensors: 252 quantized layers, the embedders and heads dense bf16.
    assert sum(k.endswith("._weight_qdata") for k in keys) == 252

    result = _identify(write_header_only_safetensors(tmp_path / "MOSS-Image-Base-FP8.safetensors", keys))

    assert type(result.config) is Main_Checkpoint_MossImage_Config
    assert result.match_count == 1
    # Identification turns FP8 Storage on for a float8 denoiser, and the loader keeps the file's fp8 layers for it.
    assert result.config.default_settings is not None
    assert result.config.default_settings.fp8_storage is True


@pytest.mark.parametrize(
    ("quantize", "message"),
    [
        pytest.param(
            lambda keys: keys.update({"layers.0.proj.q.weight_scale": ([], "F32")}),
            "side channel on a weight that is not int8",
            id="scale-side-channel",
        ),
        pytest.param(
            lambda keys: keys.update({"layers.0.proj.q.weight": ([3072, 3072], "F8_E4M3")}),
            "plain or ComfyUI scaled fp8 weight",
            id="raw-fp8",
        ),
        pytest.param(
            lambda keys: keys.update(
                {
                    "layers.0.proj.q.weight": ([3072, 3072], "F8_E4M3"),
                    "layers.0.proj.q.weight_scale": ([3072, 1], "F32"),
                }
            ),
            "plain or ComfyUI scaled fp8 weight",
            id="comfy-scaled-fp8",
        ),
        pytest.param(
            lambda keys: keys.update({"layers.0.proj.q.comfy_quant": ([32], "U8")}),
            "side channel on a weight that is not int8",
            id="comfy-quant-marker",
        ),
        pytest.param(
            lambda keys: keys.update(
                {"layers.0.proj.q.weight": ([3072, 1536], "U8"), "layers.0.proj.q.weight_scale_2": ([], "F32")}
            ),
            "this is an nvfp4-quantized MOSS-Image transformer",
            id="nvfp4",
        ),
        pytest.param(
            lambda keys: keys.update({"layers.0.proj.q.weight": ([3072, 1536], "U8")}),
            "packed 4-bit codes",
            id="packed-uint8-weights",
        ),
        pytest.param(
            lambda keys: _torchao_fp8(keys, zero_point=True),
            "carries a zero point",
            id="unsloth-int8-torchao",
        ),
        pytest.param(
            lambda keys: _torchao_fp8(keys, scale_shape=[3072, 24]),
            "block-wise scales",
            id="torchao-fp8-blockwise-scale",
        ),
        pytest.param(
            lambda keys: _torchao_fp8(keys, drop_scale_of="layers.0.proj.q"),
            "'layers.0.proj.q' has a quantized weight but no torchao scale",
            id="torchao-half-pair",
        ),
        pytest.param(
            lambda keys: keys.update(
                {"layers.0.proj.q.weight": ([3072, 3072], "I8"), "layers.0.proj.q.weight_scale": ([3072, 1], "F32")}
            ),
            "carry no readable marker",
            id="int8-without-marker",
        ),
    ],
)
def test_an_unsupported_quantized_transformer_is_refused_not_registered(
    tmp_path: Path, quantize: Any, message: str
) -> None:
    keys = _dit()
    quantize(keys)

    result = ModelConfigFactory.from_model_on_disk(
        write_header_only_safetensors(tmp_path / "moss-quantized.safetensors", keys), allow_unknown=True
    )

    # Final, not "unknown": nothing is written to the database.
    assert result.config is None
    refusal = result.details[Main_Checkpoint_MossImage_Config.__name__]
    assert isinstance(refusal, InvalidMatchError)
    assert message in str(refusal)


def test_a_w4a8_build_is_refused_naming_what_does_load(tmp_path: Path) -> None:
    """Comfy's W4A8 stores I8 codes beside a codebook and marks each layer `w4a8`: int8 weights, but not ours."""
    keys = _dit()
    keys.update(
        {
            "layers.0.proj.q.weight": ([3072, 3072], "I8"),
            "layers.0.proj.q.weight_scale": ([3072, 1], "F32"),
            "layers.0.proj.q.weight_codebook": ([16], "F32"),
        }
    )

    result = ModelConfigFactory.from_model_on_disk(
        write_header_only_safetensors(
            tmp_path / "moss-w4a8.safetensors", keys, {"layers.0.proj.q": {"format": "w4a8"}}
        ),
        allow_unknown=True,
    )

    assert result.config is None
    refusal = str(result.details[Main_Checkpoint_MossImage_Config.__name__])
    assert "'w4a8'" in refusal
    assert "bf16, unsloth torchao FP8 or Comfy-Org int8_convrot build, or a GGUF" in refusal
    assert "Only Comfy-Org" not in refusal


def test_an_int8_build_of_another_format_than_int8_tensorwise_is_refused(tmp_path: Path) -> None:
    keys, markers = int8_convrot_keys(_dit())
    markers["layers.0.proj.o"] = {"format": "int4_tensorwise", "convrot": True, "convrot_groupsize": 256}

    result = ModelConfigFactory.from_model_on_disk(
        write_header_only_safetensors(tmp_path / "moss-int8-mixed.safetensors", keys, markers), allow_unknown=True
    )

    assert result.config is None
    refusal = result.details[Main_Checkpoint_MossImage_Config.__name__]
    assert isinstance(refusal, InvalidMatchError)
    assert "int4_tensorwise" in str(refusal)


def test_a_gguf_state_dict_is_not_a_safetensors_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "moss-image-Q4_0.gguf"
    path.touch()
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = {
        key: GGMLTensor(
            data=torch.zeros((1,), dtype=torch.uint8),
            ggml_quantization_type=gguf.GGMLQuantizationType.Q4_0,
            tensor_shape=torch.Size(shape),
            compute_dtype=torch.float32,
        )
        for key, (shape, _) in _dit().items()
    }

    with pytest.raises(NotAMatchError, match="GGUF"):
        Main_Checkpoint_MossImage_Config.from_model_on_disk(mod, {})


def _pipeline_folder(root: Path, *, transformer_class: str = "MOSSImageTransformer", skip: str | None = None) -> Path:
    """A MOSS-Image pipeline folder with the configs identification reads and an empty shard per component."""
    root.mkdir()
    (root / "model_index.json").write_text(json.dumps({"_class_name": "MOSSImagePipeline"}))
    files = {
        "transformer/config.json": json.dumps({"_class_name": transformer_class, "num_layers": 42}),
        "transformer/diffusion_pytorch_model-00001-of-00001.safetensors": None,
        "text_encoder/config.json": json.dumps({"architectures": ["Qwen3_5Model"], "model_type": "qwen3_5"}),
        "text_encoder/model-00001-of-00001.safetensors": None,
        "tokenizer/tokenizer_config.json": json.dumps({"tokenizer_class": "Qwen2Tokenizer"}),
        "tokenizer/tokenizer.json": json.dumps({"version": "1.0"}),
        "tokenizer/chat_template.jinja": "{{ messages }}",
        "vae/config.json": json.dumps({"_class_name": "AutoencoderKLFlux2"}),
        "vae/diffusion_pytorch_model.safetensors": None,
    }
    for relative, content in files.items():
        if skip is not None and (relative == skip or relative.startswith(f"{skip}/")):
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            write_header_only_safetensors(path, {"weight": ([1], "BF16")})
        else:
            path.write_text(content)
    return root


def test_the_pipeline_folder_is_a_moss_image_diffusers_model(tmp_path: Path) -> None:
    result = _identify(_pipeline_folder(tmp_path / "MOSS-Image-Base"))

    assert type(result.config) is Main_Diffusers_MossImage_Config
    assert result.config.format is ModelFormat.Diffusers
    assert result.match_count == 1


@pytest.mark.parametrize(
    "missing",
    [
        "transformer",
        "text_encoder",
        "vae",
        "tokenizer/tokenizer_config.json",
        "tokenizer/tokenizer.json",
        # MOSS-Image ships its chat template as a file of its own, not as a `tokenizer_config.json` key, and the
        # prompt encoding cannot run without it.
        "tokenizer/chat_template.jinja",
    ],
)
def test_an_incomplete_pipeline_folder_names_what_is_missing(tmp_path: Path, missing: str) -> None:
    result = ModelConfigFactory.from_model_on_disk(
        _pipeline_folder(tmp_path / "MOSS-Image-Base", skip=missing), allow_unknown=True
    )

    assert result.config is None
    refusal = result.details[Main_Diffusers_MossImage_Config.__name__]
    assert isinstance(refusal, InvalidMatchError)
    assert missing in str(refusal)


def test_a_pipeline_folder_with_another_transformer_is_refused(tmp_path: Path) -> None:
    result = ModelConfigFactory.from_model_on_disk(
        _pipeline_folder(tmp_path / "MOSS-Image-Base", transformer_class="SomethingElse"), allow_unknown=True
    )

    refusal = result.details[Main_Diffusers_MossImage_Config.__name__]
    assert isinstance(refusal, InvalidMatchError)
    assert "SomethingElse" in str(refusal)
