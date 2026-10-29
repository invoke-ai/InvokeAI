"""Loading a T5 encoder from one safetensors file in the layouts ComfyUI distributes T5-XXL in.

A tiny T5 v1.1 stands in for the 4.7B encoder: the loader reads the architecture from tensor shapes, so the
same code path builds both. Each file is written the way ComfyUI writes it -- the token embedding twice, and
for the fp8 builds either every tensor in float8 (``t5xxl_fp8_e4m3fn``) or the Linear weights in float8 with a
per-tensor ``scale_weight`` and a ``scaled_fp8`` marker beside fp16 norms (``t5xxl_fp8_e4m3fn_scaled``).

Forwards run in float32: CPU bf16 matmuls fault on part of the hosted Windows fleet.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file
from transformers import T5Config, T5EncoderModel

import invokeai.backend.model_manager.load.model_loaders.flux as loader_module
from invokeai.backend.model_manager.configs.t5_encoder import T5Encoder_Checkpoint_Config
from invokeai.backend.model_manager.load.load_default import put_in_eval_mode
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders.flux import T5EncoderSingleFileLoader
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.util.logging import InvokeAILogger

FP8 = torch.float8_e4m3fn
LINEAR_SUFFIXES = (
    "SelfAttention.q.weight",
    "SelfAttention.k.weight",
    "SelfAttention.v.weight",
    "SelfAttention.o.weight",
    "DenseReluDense.wi_0.weight",
    "DenseReluDense.wi_1.weight",
    "DenseReluDense.wo.weight",
)


def _is_linear_weight(key: str) -> bool:
    return key.endswith(LINEAR_SUFFIXES)


def _tiny_reference() -> T5EncoderModel:
    torch.manual_seed(0)
    config = T5Config(
        vocab_size=32,
        d_model=16,
        d_kv=4,
        d_ff=32,
        num_layers=2,
        num_heads=4,
        feed_forward_proj="gated-gelu",
        is_gated_act=True,
        dense_act_fn="gelu_new",
        use_cache=False,
    )
    model = T5EncoderModel(config).eval()
    # T5 initializes some weights near zero; spread them so fp8 rounding and a lost scale are both visible.
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn_like(param) * 0.5)
    return model


def _comfy_state_dict(reference: T5EncoderModel) -> dict[str, torch.Tensor]:
    """The reference's state dict as ComfyUI saves it: `shared` and `encoder.embed_tokens` both present."""
    sd = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    sd["encoder.embed_tokens.weight"] = sd["shared.weight"].clone()
    return sd


def _write_fp16(reference: T5EncoderModel, path: Path) -> dict[str, torch.Tensor]:
    sd = {key: value.half() for key, value in _comfy_state_dict(reference).items()}
    save_file(sd, str(path))
    return {key: value.float() for key, value in sd.items()}


def _write_fp16_embed_tokens_only(reference: T5EncoderModel, path: Path) -> dict[str, torch.Tensor]:
    """A file that carries the token embedding only as the encoder's copy."""
    sd = {key: value.half() for key, value in _comfy_state_dict(reference).items() if key != "shared.weight"}
    save_file(sd, str(path))
    dequantized = {key: value.float() for key, value in sd.items()}
    dequantized["shared.weight"] = dequantized["encoder.embed_tokens.weight"]
    return dequantized


def _write_raw_fp8(reference: T5EncoderModel, path: Path) -> dict[str, torch.Tensor]:
    sd = {key: value.to(FP8) for key, value in _comfy_state_dict(reference).items()}
    save_file(sd, str(path))
    return {key: value.float() for key, value in sd.items()}


def _write_scaled_fp8(reference: T5EncoderModel, path: Path) -> dict[str, torch.Tensor]:
    file_sd: dict[str, torch.Tensor] = {"scaled_fp8": torch.empty(0, dtype=FP8)}
    dequantized: dict[str, torch.Tensor] = {}
    for key, value in _comfy_state_dict(reference).items():
        if _is_linear_weight(key):
            scale = (value.abs().max() / 448.0).float()
            quantized = (value / scale).to(FP8)
            file_sd[key] = quantized
            file_sd[key[: -len(".weight")] + ".scale_weight"] = scale
            dequantized[key] = quantized.float() * scale
        else:
            file_sd[key] = value.half()
            dequantized[key] = file_sd[key].float()
    save_file(file_sd, str(path))
    return dequantized


@pytest.fixture
def loader(monkeypatch: pytest.MonkeyPatch) -> T5EncoderSingleFileLoader:
    loader = object.__new__(T5EncoderSingleFileLoader)
    loader._ram_cache = MagicMock()
    loader._logger = InvokeAILogger.get_logger("test")
    loader._torch_device = torch.device("cpu")
    monkeypatch.setattr(loader_module.TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _: torch.float32))
    return loader


def _keep_fp8(monkeypatch: pytest.MonkeyPatch, keep: bool) -> None:
    monkeypatch.setattr(loader_module, "should_keep_fp8_weights", lambda _device: keep)
    monkeypatch.setattr(loader_module, "_device_supports_fp8_storage", lambda _device, _logger=None: keep)


def _load_encoder(loader: T5EncoderSingleFileLoader, path: Path) -> T5EncoderModel:
    # `put_in_eval_mode` is what `ModelLoader.load_model` wraps every `_load_model` in.
    return put_in_eval_mode(loader._load_model(_config(path), SubModelType.TextEncoder2))


def _config(path: Path) -> T5Encoder_Checkpoint_Config:
    return T5Encoder_Checkpoint_Config.model_construct(path=str(path), name=path.stem, cpu_only=None)


def _expected_output(dequantized: dict[str, torch.Tensor], reference: T5EncoderModel) -> torch.Tensor:
    """The encoder's output with the file's weights as they decode, built without the loader."""
    expected = T5EncoderModel(reference.config).eval()
    expected.load_state_dict(dequantized, strict=False)
    return _encode(expected)


def _encode(model: torch.nn.Module) -> torch.Tensor:
    input_ids = torch.tensor([[3, 7, 1, 30, 12, 0]])
    with torch.no_grad():
        return model(input_ids=input_ids).last_hidden_state


@pytest.mark.parametrize("writer", [_write_fp16, _write_fp16_embed_tokens_only, _write_raw_fp8, _write_scaled_fp8])
def test_a_folded_load_reproduces_the_weights_the_file_decodes_to(
    loader: T5EncoderSingleFileLoader, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, writer
) -> None:
    _keep_fp8(monkeypatch, False)
    reference = _tiny_reference()
    path = tmp_path / "t5xxl.safetensors"
    dequantized = writer(reference, path)

    model = _load_encoder(loader, path)

    assert [name for name, p in model.named_parameters() if p.is_meta] == []
    assert all(p.dtype == torch.float32 for p in model.parameters())
    # One embedding, tied: the duplicate the file carries is not held twice.
    assert model.encoder.embed_tokens.weight is model.shared.weight
    for name, param in model.state_dict().items():
        assert torch.equal(param, dequantized[name]), name
    torch.testing.assert_close(_encode(model), _expected_output(dequantized, reference))


@pytest.mark.parametrize("writer", [_write_raw_fp8, _write_scaled_fp8])
def test_a_kept_load_holds_linear_weights_in_fp8_and_encodes_the_same(
    loader: T5EncoderSingleFileLoader, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, writer
) -> None:
    _keep_fp8(monkeypatch, True)
    reference = _tiny_reference()
    path = tmp_path / "t5xxl.safetensors"
    dequantized = writer(reference, path)

    model = _load_encoder(loader, path)

    for name, param in model.named_parameters():
        # Only Linear weights may stay quantized: an fp8 norm or embedding would hand fp8 to the next layer.
        assert param.dtype == (FP8 if _is_linear_weight(name) else torch.float32), name
    assert model.encoder.embed_tokens.weight is model.shared.weight
    # Kept for storage only: the fp8 matmul would also quantize T5's outlier-heavy activations.
    fp8_linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear) and m.weight.dtype == FP8]
    assert fp8_linears and all(m._fp8_full_precision_matmul for m in fp8_linears)
    # `wo` reads its own weight's dtype; in fp8 the activations would be rounded to fp8 without the patch.
    apply_custom_layers_to_model(model)
    torch.testing.assert_close(_encode(model), _expected_output(dequantized, reference))


def test_keeping_fp8_reserves_less_than_folding(
    loader: T5EncoderSingleFileLoader, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "t5xxl.safetensors"
    _write_scaled_fp8(_tiny_reference(), path)

    reserved = {}
    for keep in (False, True):
        _keep_fp8(monkeypatch, keep)
        loader._ram_cache = MagicMock()
        _load_encoder(loader, path)
        loader._ram_cache.make_room.assert_called_once()
        reserved[keep] = loader._ram_cache.make_room.call_args.args[0]

    assert 0 < reserved[True] < reserved[False]


def test_the_tokenizer_is_the_bundled_one_and_reserves_nothing(
    loader: T5EncoderSingleFileLoader, tmp_path: Path
) -> None:
    from invokeai.backend.t5.t5_tokenizer import load_bundled_t5_tokenizer

    path = tmp_path / "t5xxl.safetensors"
    config = _config(path)

    assert loader._load_model(config, SubModelType.Tokenizer2) is load_bundled_t5_tokenizer()
    assert loader.get_size_fs(config, path, SubModelType.Tokenizer3) == 0
