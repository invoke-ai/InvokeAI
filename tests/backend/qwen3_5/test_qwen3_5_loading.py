"""`load_qwen3_5_text_encoder` reads a checkpoint's language model and nothing else, from a file or shards.

Written against small safetensors files in the real key layouts: a transformers `Qwen3_5Model` save as
MOSS-Image ships it (`language_model.*` beside a `visual.*` tower, final norm included) and Anima-3.8B's
encoder (bare keys, a projection head under `norm.*`, the last layer without its MLP).
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from invokeai.backend.model_manager.configs.qwen3_5_encoder import Qwen35Encoder_Checkpoint_Config
from invokeai.backend.model_manager.load.model_loaders import qwen3_5_encoder as loader_module
from invokeai.backend.model_manager.load.model_loaders.qwen3_5_encoder import (
    Qwen35EncoderCheckpointLoader,
    load_qwen3_5_text_encoder,
)
from invokeai.backend.model_manager.taxonomy import Qwen35VariantType, SubModelType
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear
from invokeai.backend.qwen3_5.qwen3_5_encoder import Qwen35Encoder, qwen3_5_text_config_from_config_json
from tests.fixtures.quantized_payloads import comfy_quant_marker, quantize_convrot

_LAST = 3
_PROJECTION_HEAD = {"norm.0.weight": [8, 32], "norm.0.bias": [8], "norm.1.weight": [8], "norm.3.weight": [8, 8]}
_VISION_TOWER = {"visual.patch_embed.proj.weight": [16, 3, 2, 2], "visual.blocks.0.attn.qkv.weight": [48, 16]}


def _config() -> Qwen3_5TextConfig:
    return qwen3_5_text_config_from_config_json(
        {
            "text_config": {
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 48,
                "num_hidden_layers": _LAST + 1,  # three linear-attention layers, then full attention
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 0.25,
                    "mrope_section": [1, 1, 0],
                    "mrope_interleaved": True,
                },
            }
        }
    )


def _reference(final_norm: bool) -> Qwen35Encoder:
    torch.manual_seed(0)
    reference = Qwen35Encoder(_config(), final_norm=final_norm).eval()
    for parameter in reference.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    return reference


def _random(shapes: dict[str, list[int]]) -> dict[str, torch.Tensor]:
    return {key: torch.randn(shape) for key, shape in shapes.items()}


def _bytes(tensors: dict[str, torch.Tensor]) -> int:
    """What the load must reserve: every tensor at float32, the dtype the fixture loads at."""
    return sum(tensor.numel() * 4 for tensor in tensors.values())


@pytest.fixture
def loader(monkeypatch: pytest.MonkeyPatch) -> Qwen35EncoderCheckpointLoader:
    monkeypatch.setattr(loader_module.TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(
        loader_module.TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _device: torch.float32)
    )
    loader = object.__new__(Qwen35EncoderCheckpointLoader)
    loader._ram_cache = MagicMock()
    loader._logger = logging.getLogger(__name__)
    return loader


@pytest.fixture
def tensors_read(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every tensor the loader reads from disk, by checkpoint key."""
    read: list[str] = []
    real_safe_open = loader_module.safe_open

    class _Recorder:
        def __init__(self, file) -> None:
            self._file = file

        def keys(self) -> list[str]:
            return self._file.keys()

        def get_tensor(self, key: str) -> torch.Tensor:
            read.append(key)
            return self._file.get_tensor(key)

    @contextmanager
    def recording_safe_open(path, **kwargs) -> Iterator[_Recorder]:
        with real_safe_open(path, **kwargs) as file:
            yield _Recorder(file)

    monkeypatch.setattr(loader_module, "safe_open", recording_safe_open)
    return read


def _assert_loaded(model: Qwen35Encoder, expected: dict[str, torch.Tensor]) -> None:
    loaded = model.state_dict()
    assert loaded.keys() == expected.keys()
    for name, tensor in expected.items():
        assert torch.equal(loaded[name], tensor), name


def test_a_multimodal_export_loads_its_language_model_and_never_reads_the_vision_tower(
    tmp_path: Path, loader, tensors_read: list[str]
) -> None:
    weights = _reference(final_norm=True).state_dict()
    path = tmp_path / "qwen3_5_9b.safetensors"
    save_file({**{f"language_model.{k}": v for k, v in weights.items()}, **_random(_VISION_TOWER)}, path)

    model = load_qwen3_5_text_encoder(loader, [path], _config(), final_norm=True, keep_fp8=False, what="test")

    _assert_loaded(model, weights)
    assert [key for key in tensors_read if not key.startswith("language_model.")] == []
    loader._ram_cache.make_room.assert_called_once_with(_bytes(weights))


def test_shards_load_as_one_checkpoint(tmp_path: Path, loader, tensors_read: list[str]) -> None:
    # The embedding and layer 0's linear attention -- what locates the language model -- sit in
    # different shards, so neither shard's keys alone identify it.
    weights = _reference(final_norm=True).state_dict()
    keys = {f"language_model.{k}": v for k, v in weights.items()}
    first = {k: v for k, v in keys.items() if k == "language_model.embed_tokens.weight" or ".layers.3." in k}
    second = {k: v for k, v in keys.items() if k not in first} | _random(_VISION_TOWER)
    save_file(first, tmp_path / "model-00001-of-00002.safetensors")
    save_file(second, tmp_path / "model-00002-of-00002.safetensors")
    shards = sorted(tmp_path.glob("*.safetensors"))

    model = load_qwen3_5_text_encoder(loader, shards, _config(), final_norm=True, keep_fp8=False, what="test")

    _assert_loaded(model, weights)
    assert [key for key in tensors_read if key.startswith("visual.")] == []
    loader._ram_cache.make_room.assert_called_once_with(_bytes(weights))


def test_a_checkpoint_without_its_final_norm_is_refused_when_the_norm_is_read(tmp_path: Path, loader) -> None:
    weights = _reference(final_norm=True).state_dict()
    del weights["norm.weight"]
    path = tmp_path / "qwen3_5_9b.safetensors"
    save_file({f"language_model.{k}": v for k, v in weights.items()}, path)

    with pytest.raises(RuntimeError, match="norm.weight"):
        load_qwen3_5_text_encoder(loader, [path], _config(), final_norm=True, keep_fp8=False, what="test")


def test_a_layer_without_its_mlp_is_refused_when_the_last_hidden_state_is_read(tmp_path: Path, loader) -> None:
    # Only a reader of intermediate layers may run a layer attention-only; the last hidden state runs it whole.
    weights = {k: v for k, v in _reference(final_norm=True).state_dict().items() if f"layers.{_LAST}.mlp." not in k}
    path = tmp_path / "qwen3_5_9b.safetensors"
    save_file({f"language_model.{k}": v for k, v in weights.items()}, path)

    with pytest.raises(RuntimeError, match=f"layers.{_LAST}.mlp"):
        load_qwen3_5_text_encoder(loader, [path], _config(), final_norm=True, keep_fp8=False, what="test")


def test_anima_3_8b_s_encoder_runs_its_last_layer_attention_only(tmp_path: Path, loader) -> None:
    """Bare keys, no MLP (or its norm) on the last layer, and a projection head where the final norm would be."""
    reference = _reference(final_norm=False)
    weights = {
        k: v
        for k, v in reference.state_dict().items()
        if not k.startswith((f"layers.{_LAST}.mlp.", f"layers.{_LAST}.post_attention_layernorm."))
    }
    path = tmp_path / "qwen35_4b.safetensors"
    save_file(weights | _random(_PROJECTION_HEAD), path)

    model = load_qwen3_5_text_encoder(loader, [path], _config(), final_norm=False, keep_fp8=False, what="test")

    _assert_loaded(model, weights)
    assert model.norm is None
    input_ids = torch.randint(0, 64, (1, 9))
    (expected,) = reference(input_ids, (_LAST,), last_layer_attention_only=True)
    (output,) = model(input_ids, (_LAST,))
    torch.testing.assert_close(output, expected)
    loader._ram_cache.make_room.assert_called_once_with(_bytes(weights))


@pytest.mark.parametrize(
    ("variant", "loads_the_norm"), [(Qwen35VariantType.Qwen35_4B, False), (Qwen35VariantType.Qwen35_9B, True)]
)
def test_the_variant_decides_whether_a_single_file_loads_the_final_norm(
    tmp_path: Path, loader, monkeypatch: pytest.MonkeyPatch, variant: Qwen35VariantType, loads_the_norm: bool
) -> None:
    """MOSS-Image reads the 9B after its final norm; Anima-3.8B reads the 4B's intermediate layers."""
    monkeypatch.setattr("invokeai.backend.qwen3_5.qwen3_5_encoder.qwen3_5_text_config", lambda _variant: _config())
    loader._keep_fp8_weights = lambda _config, _submodel: False
    loader._apply_fp8_layerwise_casting = lambda model, _config, _submodel: model
    weights = _reference(final_norm=True).state_dict()
    path = tmp_path / "encoder.safetensors"
    save_file({f"language_model.{k}": v for k, v in weights.items()}, path)
    config = Qwen35Encoder_Checkpoint_Config.model_construct(path=str(path), name="encoder", variant=variant)

    model = loader._load_model(config, SubModelType.TextEncoder)

    assert isinstance(model, Qwen35Encoder)
    assert ("norm.weight" in model.state_dict()) is loads_the_norm


# -- ComfyUI's int8_tensorwise build: the whole vision-language model, of which the encoder reads the language model. --

_PREFIX = "model.language_model."
_GROUP = 16
# What the build quantizes in the language model: every decoder Linear but the Gated DeltaNet's `in_proj_a/b`, and the
# embedding, which is not a Linear.
_DENSE_LINEARS = ("in_proj_a", "in_proj_b")


def _quantizable(reference: Qwen35Encoder) -> list[str]:
    return ["embed_tokens"] + [
        name
        for name, module in reference.named_modules()
        if name.startswith("layers.") and isinstance(module, torch.nn.Linear) and not name.endswith(_DENSE_LINEARS)
    ]


def _int8_whole_model_file(reference: Qwen35Encoder) -> dict[str, torch.Tensor]:
    """The reference as the int8 build stores it, with the vision tower, LM head and MTP head beside it.

    The reference is left holding the round trip, which is what a correct load computes.
    """
    sd = {f"{_PREFIX}{key}": value.detach().clone() for key, value in reference.state_dict().items()}
    for path in _quantizable(reference):
        payload = quantize_convrot(sd[f"{_PREFIX}{path}.weight"], group_size=_GROUP)
        sd[f"{_PREFIX}{path}.weight"], sd[f"{_PREFIX}{path}.weight_scale"] = payload.codes, payload.scale
        sd[f"{_PREFIX}{path}.comfy_quant"] = _marker()
        # W8A8 builds ship an activation scale the dequantizing layers have nothing to apply to.
        sd[f"{_PREFIX}{path}.input_scale"] = torch.tensor(0.5)
        with torch.no_grad():
            reference.get_submodule(path).weight.copy_(payload.dequantized)
    for path, shape in {"model.visual.blocks.0.attn.qkv": (48, 16), "lm_head": (64, 32), "mtp.fc": (32, 64)}.items():
        payload = quantize_convrot(torch.randn(shape), group_size=_GROUP)
        sd[f"{path}.weight"], sd[f"{path}.weight_scale"] = payload.codes, payload.scale
        sd[f"{path}.comfy_quant"] = _marker()
    return sd


def _marker() -> torch.Tensor:
    return comfy_quant_marker({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": _GROUP})


@pytest.mark.parametrize("final_norm", [True, False], ids=["9b, final norm", "4b, intermediate layers"])
def test_an_int8_whole_model_file_computes_what_its_dequantized_weights_compute(
    tmp_path: Path, loader, tensors_read: list[str], final_norm: bool
) -> None:
    reference = _reference(final_norm=final_norm)
    path = tmp_path / "qwen3.5_int8_convrot.safetensors"
    save_file(_int8_whole_model_file(reference), path)

    model = load_qwen3_5_text_encoder(loader, [path], _config(), final_norm=final_norm, keep_fp8=False, what="test")

    kept = {name for name, module in model.named_modules() if isinstance(module, Int8ConvrotLinear)}
    assert kept == set(_quantizable(reference)) - {"embed_tokens"}
    # The embedding is widened: `Qwen35Encoder.dtype` reads it, and an int8 table would hand out codes.
    assert model.dtype is torch.float32 and not isinstance(model.embed_tokens, Int8ConvrotLinear)
    # Dense where the file is dense.
    assert type(model.layers[0].linear_attn.in_proj_a) is torch.nn.Linear
    assert model.layers[0].linear_attn.conv1d.weight.dtype is torch.float32
    # Neither the vision tower, the LM head nor the MTP head is read.
    assert all(key.startswith(_PREFIX) for key in tensors_read)

    input_ids = torch.randint(0, 64, (1, 9))
    if final_norm:
        expected = reference.last_hidden_state(input_ids)
        output = model.last_hidden_state(input_ids)
    else:
        (expected,) = reference(input_ids, (_LAST,))
        (output,) = model(input_ids, (_LAST,))
    assert expected.abs().max() > 0.1
    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
    # One reservation, for what the load ends with: marked Linears stay one byte per weight, and the activation
    # scales are not loaded.
    loader._ram_cache.make_room.assert_called_once_with(
        sum(t.nelement() * t.element_size() for t in model.state_dict().values())
    )


def test_a_dense_whole_model_file_loads_the_same_language_model_as_an_encoder_export(
    tmp_path: Path, loader, tensors_read: list[str]
) -> None:
    weights = _reference(final_norm=True).state_dict()
    path = tmp_path / "qwen3.5_9b_bf16.safetensors"
    save_file(
        {f"{_PREFIX}{k}": v for k, v in weights.items()}
        | _random(
            {"lm_head.weight": [64, 32], "mtp.fc.weight": [32, 64], "model.visual.blocks.0.attn.qkv.weight": [48, 16]}
        ),
        path,
    )

    model = load_qwen3_5_text_encoder(loader, [path], _config(), final_norm=True, keep_fp8=False, what="test")

    _assert_loaded(model, weights)
    assert all(key.startswith(_PREFIX) for key in tensors_read)


def test_an_int8_layer_without_an_int8_tensorwise_marker_is_refused(tmp_path: Path, loader) -> None:
    reference = _reference(final_norm=True)
    file = _int8_whole_model_file(reference)
    del file[f"{_PREFIX}layers.0.mlp.gate_proj.comfy_quant"]
    path = tmp_path / "qwen3.5_9b_int8_convrot.safetensors"
    save_file(file, path)

    with pytest.raises(ValueError, match="no readable int8_tensorwise marker"):
        load_qwen3_5_text_encoder(loader, [path], _config(), final_norm=True, keep_fp8=False, what="test")
    loader._ram_cache.make_room.assert_not_called()


def test_a_w4a8_layer_is_refused_naming_its_format(tmp_path: Path, loader) -> None:
    reference = _reference(final_norm=True)
    file = _int8_whole_model_file(reference)
    file[f"{_PREFIX}layers.0.mlp.gate_proj.comfy_quant"] = comfy_quant_marker({"format": "asym_w4a8_int8"})
    path = tmp_path / "qwen3.5_9b_w4a8.safetensors"
    save_file(file, path)

    with pytest.raises(ValueError, match="asym_w4a8_int8"):
        load_qwen3_5_text_encoder(loader, [path], _config(), final_norm=True, keep_fp8=False, what="test")
    loader._ram_cache.make_room.assert_not_called()


def test_a_scaled_fp8_layer_is_refused_before_room_is_reserved(tmp_path: Path, loader) -> None:
    weights = {f"{_PREFIX}{k}": v for k, v in _reference(final_norm=True).state_dict().items()}
    gate = f"{_PREFIX}layers.0.mlp.gate_proj"
    weights[f"{gate}.weight"] = weights[f"{gate}.weight"].to(torch.float8_e4m3fn)
    weights[f"{gate}.weight_scale"] = torch.tensor(0.01)
    path = tmp_path / "qwen3.5_9b_fp8_scaled.safetensors"
    save_file(weights, path)

    with pytest.raises(ValueError, match="quantization side channel"):
        load_qwen3_5_text_encoder(loader, [path], _config(), final_norm=True, keep_fp8=False, what="test")
    loader._ram_cache.make_room.assert_not_called()
