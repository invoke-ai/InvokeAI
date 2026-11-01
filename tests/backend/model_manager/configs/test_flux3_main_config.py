"""Identification of FLUX.3 Image models, end to end through `ModelConfigFactory`.

Run over the whole candidate set, because the question is that exactly one config claims each release:
FLUX.3 GGUFs call themselves `general.architecture = "flux"`, and the FLUX.1 and FLUX.2 detectors read some
of the same key names (`txt_in`, `time_in`, `single_blocks`).
"""

import json
from pathlib import Path

import pytest

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_Flux3_Config,
    Main_Diffusers_Flux3_Config,
    Main_GGUF_Flux3_Config,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import Flux3VariantType, ModelFormat
from tests.backend.flux3.flux3_fixture import tiny_flux3_model, to_native, write_flux3_gguf
from tests.backend.model_manager.load.state_dicts import (
    flux3_klein_base_keys,
    flux3_klein_fp8r_keys,
    flux3_klein_int8_convrot_keys,
    flux3_klein_keys,
    flux3_klein_nvfp4_bfl_keys,
    flux3_klein_nvfp4_comfy_keys,
)
from tests.backend.model_manager.load.state_dicts.utils import write_header_only_safetensors, write_shrunk_safetensors

# The released folder's JSON files (black-forest-labs/flux-3-image-klein, 2026-10-15), shortened where nothing
# reads the rest. The root `config.json` is BFL's own package config, which sits beside the pipeline.
_MODULAR_INDEX = {
    "_blocks_class_name": "Flux3DistilledAutoBlocks",
    "_class_name": "Flux3DistilledModularPipeline",
    "_diffusers_version": "0.41.0.dev0",
    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration", {"subfolder": "text_encoder"}],
    "tokenizer": ["transformers", "Qwen3VLProcessor", {"subfolder": "text_encoder"}],
    "transformer": ["diffusers", "Flux3Transformer2DModel", {"subfolder": "transformer"}],
    "vae": ["diffusers", "AutoencoderKLFlux2", {"subfolder": "vae"}],
}
_BFL_PACKAGE_CONFIG = {
    "format_version": 1,
    "model": "flux-3-image-klein",
    "quantization": None,
    "dit": {"flow_map": True},
}
_TRANSFORMER_CONFIG = {
    "_class_name": "Flux3Transformer2DModel",
    "attention_head_dim": 128,
    "joint_attention_dim": 20480,
    "num_layers": 5,
    "num_single_layers": 28,
}


def _identify(path: Path):
    return ModelConfigFactory.from_model_on_disk(ModelOnDisk(path, "random"), {}, allow_unknown=True)


def _write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


def _write_folder(root: Path, *, flow_map: bool, base: bool = False) -> Path:
    index = dict(_MODULAR_INDEX)
    if base:
        index["_class_name"] = "Flux3ModularPipeline"
    _write_json(root / "modular_model_index.json", index)
    _write_json(root / "config.json", _BFL_PACKAGE_CONFIG)
    _write_json(root / "transformer" / "config.json", {**_TRANSFORMER_CONFIG, "flow_map": flow_map})
    (root / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"")
    _write_json(root / "vae" / "config.json", {"_class_name": "AutoencoderKLFlux2", "latent_channels": 32})
    _write_json(root / "text_encoder" / "config.json", {"architectures": ["Qwen3VLForConditionalGeneration"]})
    (root / "text_encoder" / "tokenizer.json").write_text("{}")
    return root


@pytest.mark.parametrize(
    ("flow_map", "base", "variant"),
    [(True, False, Flux3VariantType.Klein), (False, True, Flux3VariantType.KleinBase)],
    ids=["klein", "base"],
)
def test_a_diffusers_folder_installs_with_the_variant_its_transformer_declares(
    tmp_path: Path, flow_map: bool, base: bool, variant: Flux3VariantType
) -> None:
    result = _identify(_write_folder(tmp_path / "flux-3-image-klein", flow_map=flow_map, base=base))

    assert result.match_count == 1, [type(match).__name__ for match in result.all_matches]
    assert isinstance(result.config, Main_Diffusers_Flux3_Config)
    assert result.config.variant is variant
    assert result.config.default_settings is not None
    assert result.config.default_settings.steps == (16 if flow_map else 50)


@pytest.mark.parametrize("missing", ["vae/config.json", "text_encoder/config.json", "text_encoder/tokenizer.json"])
def test_a_partial_diffusers_folder_is_refused_by_name(tmp_path: Path, missing: str) -> None:
    root = _write_folder(tmp_path / "flux3", flow_map=True)
    (root / missing).unlink()

    with pytest.raises(InvalidMatchError):
        Main_Diffusers_Flux3_Config.from_model_on_disk(ModelOnDisk(root, "random"), {})


@pytest.mark.parametrize(
    ("fixture", "variant"),
    [(flux3_klein_keys, Flux3VariantType.Klein), (flux3_klein_base_keys, Flux3VariantType.KleinBase)],
    ids=["klein", "base"],
)
def test_a_released_single_file_installs_as_flux3(tmp_path: Path, fixture, variant: Flux3VariantType) -> None:
    path = write_header_only_safetensors(tmp_path / "flux3_image_klein.safetensors", fixture.state_dict_keys)

    result = _identify(path)

    assert result.match_count == 1, [type(match).__name__ for match in result.all_matches]
    assert isinstance(result.config, Main_Checkpoint_Flux3_Config)
    assert result.config.variant is variant


_RELEASED_QUANTIZED = {
    "int8-convrot": flux3_klein_int8_convrot_keys,
    "nvfp4-comfy": flux3_klein_nvfp4_comfy_keys,
    "nvfp4-bfl": flux3_klein_nvfp4_bfl_keys,
    "fp8r": flux3_klein_fp8r_keys,
}


def _write_release(path: Path, fixture, *, base: bool = False) -> Path:
    """A captured release as a small complete file: identification reads its markers' bytes and its metadata, the
    `_quantization_metadata` block included, beyond the tensor entries. Klein Base's builds are Klein's layout
    without the flow-map embedder `ref_time_in` (checked on their headers)."""
    metadata = dict(fixture.metadata)
    if layer_hints := getattr(fixture, "layer_hints", None):
        metadata["_quantization_metadata"] = json.dumps({"layers": layer_hints})
    keys = {
        key: entry for key, entry in fixture.state_dict_keys.items() if not (base and key.startswith("ref_time_in."))
    }
    return write_shrunk_safetensors(path, keys, markers=getattr(fixture, "markers", None), metadata=metadata)


@pytest.mark.parametrize("base", [False, True], ids=["klein", "base"])
@pytest.mark.parametrize("name", _RELEASED_QUANTIZED)
def test_a_released_quantized_single_file_installs_as_flux3(tmp_path: Path, name: str, base: bool) -> None:
    result = _identify(_write_release(tmp_path / "flux3.safetensors", _RELEASED_QUANTIZED[name], base=base))

    assert result.match_count == 1, [type(match).__name__ for match in result.all_matches]
    assert isinstance(result.config, Main_Checkpoint_Flux3_Config)
    assert result.config.variant is (Flux3VariantType.KleinBase if base else Flux3VariantType.Klein)
    # FP8 Storage comes on at install for the fp8 build, which then stays fp8; an int8 build has no fp8 to keep.
    # (The nvfp4 builds get it too, from their fp8 block scales, and the loader skips it: a known issue of the shared
    # install default, documented on the FLUX.3 page.)
    fp8_storage = getattr(result.config.default_settings, "fp8_storage", None)
    if name == "fp8r":
        assert fp8_storage is True
    elif name == "int8-convrot":
        assert fp8_storage is not True


_INT8 = {"weight": ([3072, 3072], "I8"), "weight_scale": ([3072, 1], "F32"), "comfy_quant": ([72], "U8")}
_INT8_MARKER = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
_NVFP4 = {"weight": ([3072, 1536], "U8"), "weight_scale": ([3072, 192], "F8_E4M3"), "weight_scale_2": ([], "F32")}
_FP8_SCALED = {"weight": ([3072, 3072], "F8_E4M3"), "weight_scale": ([], "F32")}


def _header_naming(*layers: str) -> dict[str, str]:
    return {"_quantization_metadata": json.dumps({"layers": {layer: {"format": "nvfp4"} for layer in layers}})}


def _with(layers: dict[str, dict[str, tuple[list[int], str]]]) -> dict[str, tuple[list[int], str]]:
    """The released Klein layout with each of `layers` stored as its entries instead."""
    keys = dict(flux3_klein_keys.state_dict_keys)
    for layer, entries in layers.items():
        keys = {key: entry for key, entry in keys.items() if not key.startswith(f"{layer}.")}
        keys.update({f"{layer}.{suffix}": entry for suffix, entry in entries.items()})
    return keys


@pytest.mark.parametrize(
    ("layers", "markers"),
    [
        # A plain fp8 cast, no scale.
        ({"single_blocks.0.q_proj": {"weight": ([3072, 3072], "F8_E4M3")}}, None),
        # Per-tensor scaled fp8, as ComfyUI writes it.
        ({"single_blocks.0.q_proj": _FP8_SCALED}, None),
        # nvfp4 named by ComfyUI's per-layer marker rather than its header.
        (
            {"single_blocks.0.q_proj": {**_NVFP4, "comfy_quant": ([18], "U8")}},
            {"single_blocks.0.q_proj": {"format": "nvfp4"}},
        ),
    ],
    ids=["fp8-raw", "fp8-per-tensor", "nvfp4-comfy-marker"],
)
def test_the_other_loadable_quantizations_install_as_flux3(tmp_path: Path, layers, markers) -> None:
    result = _identify(write_shrunk_safetensors(tmp_path / "flux3.safetensors", _with(layers), markers=markers))

    assert result.match_count == 1, [type(match).__name__ for match in result.all_matches]
    assert isinstance(result.config, Main_Checkpoint_Flux3_Config)


_Q, _K = "single_blocks.0.q_proj", "single_blocks.0.k_proj"

# What the loader cannot build, each on a layer or two of the released layout: the layers, their markers, the file's
# metadata, and the reason the refusal must give.
_REFUSED = {
    "int8-unmarked": (
        {_Q: {"weight": ([3072, 3072], "I8"), "weight_scale": ([3072, 1], "F32")}},
        {},
        {},
        "no `comfy_quant` marker",
    ),
    "int8-group-size": ({_Q: _INT8}, {_Q: {**_INT8_MARKER, "convrot_groupsize": 512}}, {}, "groups of 512"),
    "torchao": (
        {_Q: {"_weight_qdata": ([3072, 3072], "F8_E4M3"), "_weight_scale": ([3072, 1], "F32")}},
        {},
        {},
        "torchao",
    ),
    # bitsandbytes NF4 flattens the packed weight and keeps its state beside it.
    "bitsandbytes": (
        {_Q: {"weight": ([4718592, 1], "U8"), "weight.absmax": ([147456], "F32"), "weight.quant_map": ([16], "F32")}},
        {},
        {},
        "bitsandbytes 4-bit quantization state",
    ),
    "packed-without-global-scale": ({_Q: {"weight": ([3072, 1536], "U8")}}, {}, {}, "without an nvfp4 global scale"),
    "awq": ({_Q: {**_NVFP4, "pre_quant_scale": ([3072], "F32")}}, {}, _header_naming(_Q), "AWQ"),
    "fp8-unparsed-scale": (
        {_Q: {"weight": ([3072, 3072], "F8_E4M3"), "weight_scale_inv": ([3072], "F32")}},
        {},
        {},
        "a spelling InvokeAI does not read",
    ),
    "nvfp4-unnamed": ({_Q: _NVFP4}, {}, {}, "no ComfyUI `comfy_quant` marker"),
    # 100 rows are no whole 128-row scale tile.
    "nvfp4-misaligned": (
        {_Q: {"weight": ([100, 1536], "U8"), "weight_scale": ([100, 192], "F8_E4M3"), "weight_scale_2": ([], "F32")}},
        {},
        _header_naming(_Q),
        "tile layout",
    ),
    "int8-with-nvfp4": ({_Q: _INT8, _K: _NVFP4}, {_Q: _INT8_MARKER}, _header_naming(_K), "mixes int8 and nvfp4"),
    "int8-with-scaled-fp8": ({_Q: _INT8, _K: _FP8_SCALED}, {_Q: _INT8_MARKER}, {}, "scaled fp8"),
    "packed-embedder": (
        {
            "txt_in": {
                "weight": ([3072, 10240], "U8"),
                "weight_scale": ([3072, 1280], "F8_E4M3"),
                "weight_scale_2": ([], "F32"),
            }
        },
        {},
        _header_naming("txt_in"),
        "stores 'txt_in.weight' packed",
    ),
}


@pytest.mark.parametrize("layout", sorted(_REFUSED))
def test_a_quantized_single_file_the_loader_cannot_build_is_refused(tmp_path: Path, layout: str) -> None:
    layers, markers, metadata, reason = _REFUSED[layout]
    path = write_shrunk_safetensors(tmp_path / "flux3.safetensors", _with(layers), markers=markers, metadata=metadata)

    result = _identify(path)

    # Refused outright: neither installed as FLUX.3 nor taken by another config, nor registered as unknown.
    assert result.config is None
    assert any(reason in str(error) for error in result.invalid_matches), result.invalid_matches


def test_a_torchao_export_is_refused_rather_than_installed_as_unknown(tmp_path: Path) -> None:
    """torchao quantizes every linear, the modulation the fingerprint looks at included, and stores each one without
    a `.weight`; the file must still be recognized as FLUX.3 for its refusal to be reached."""
    keys: dict[str, tuple[list[int], str]] = {}
    for key, (shape, dtype) in flux3_klein_keys.state_dict_keys.items():
        if key.endswith(".weight") and len(shape) == 2:
            layer = key[: -len(".weight")]
            keys[f"{layer}._weight_qdata"] = (shape, "F8_E4M3")
            keys[f"{layer}._weight_scale"] = ([shape[0], 1], "F32")
        else:
            keys[key] = (shape, dtype)

    result = _identify(write_shrunk_safetensors(tmp_path / "flux3.safetensors", keys))

    assert result.config is None
    assert any("torchao" in str(error) for error in result.invalid_matches), result.all_matches


def test_a_lora_with_flux3_names_is_not_a_main_model(tmp_path: Path) -> None:
    keys = {
        f"{name[: -len('.weight')]}.lora_A.weight": ([4, 3072], "BF16")
        for name in flux3_klein_keys.state_dict_keys
        if name.endswith(".weight")
    }
    path = write_header_only_safetensors(tmp_path / "flux3_lora.safetensors", keys)

    with pytest.raises(NotAMatchError):
        Main_Checkpoint_Flux3_Config.from_model_on_disk(ModelOnDisk(path, "random"), {})


@pytest.mark.parametrize("prefix", ["", "model.diffusion_model."], ids=["bare", "comfy-prefixed"])
def test_a_gguf_installs_as_flux3_and_not_as_flux1(tmp_path: Path, prefix: str) -> None:
    path = tmp_path / "flux3_image_klein_sd_Q8_0.gguf"
    write_flux3_gguf(path, to_native(tiny_flux3_model(flow_map=True)), prefix=prefix)

    result = _identify(path)

    assert result.match_count == 1, [type(match).__name__ for match in result.all_matches]
    assert isinstance(result.config, Main_GGUF_Flux3_Config)
    assert result.config.format is ModelFormat.GGUFQuantized
    assert result.config.variant is Flux3VariantType.Klein


def test_a_full_repo_download_sizes_the_tokenizer_without_the_root_package(tmp_path: Path) -> None:
    """The tokenizer has no folder of its own; sizing it by the folder root counted BFL's 11 GB single file there.

    The model cache reserves that size before a load, so a few-MB tokenizer evicted the transformer.
    """
    from unittest.mock import MagicMock

    from invokeai.backend.model_manager.load.model_loaders.flux3 import Flux3DiffusersModel
    from invokeai.backend.model_manager.taxonomy import SubModelType

    root = _write_folder(tmp_path / "flux-3-image-klein", flow_map=True)
    (root / "model.safetensors").write_bytes(b"\0" * 2_000_000)
    (root / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"\0" * 1_000)
    config = _identify(root).config
    assert isinstance(config, Main_Diffusers_Flux3_Config)
    loader = Flux3DiffusersModel(app_config=MagicMock(), logger=MagicMock(), ram_cache=MagicMock())

    assert loader.get_size_fs(config, root, SubModelType.Tokenizer) == 0
    assert 1_000 <= loader.get_size_fs(config, root, SubModelType.Transformer) < 2_000_000
