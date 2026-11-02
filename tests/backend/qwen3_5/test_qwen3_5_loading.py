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
from invokeai.backend.qwen3_5.qwen3_5_encoder import Qwen35Encoder, qwen3_5_text_config_from_config_json

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
