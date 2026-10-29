"""Loader-level tests for single-file Qwen3 encoders in the two quantized layouts that reach them.

Drives `Qwen3EncoderCheckpointLoader._load_from_singlefile` on a real, tiny safetensors file laid out like
`Comfy-Org/z_image`'s `qwen_3_4b_fp4_mixed`: nvfp4 projections named by `comfy_quant` markers -- or, as another
producer might write them, only in the `_quantization_metadata` header -- one scaled-fp8 projection, dense tensors
for the rest. What it pins is order inside the loader: room is reserved before the scaled-fp8 fold widens anything;
the nvfp4 layers leave the state dict before that fold, which would pair their block scales with the packed codes,
and come back only after the blanket cast, which would widen them; and an nvfp4 `lm_head` is dropped rather than
packed, since the loader ties `lm_head` to the embeddings. And that the returned model encodes exactly like a dense
Qwen3 holding the same weights.

The second is `int8_convrot`, which `supermind/int8_convrot_models` publishes for both encoder sizes and
Comfy-Org publishes for neither. Everything the fp4 path does wrong to an int8 file is silent: the scaled-fp8
fold pairs each `weight_scale` with its weight and widens it, and the blanket cast turns what is left into bf16
integers. So what is pinned there is that the loader commits to the int8 branch before either runs.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from invokeai.backend.model_manager.configs.qwen3_encoder import Qwen3Encoder_Checkpoint_Config
from invokeai.backend.model_manager.load.model_loaders import z_image
from invokeai.backend.model_manager.load.model_loaders.z_image import Qwen3EncoderCheckpointLoader
from invokeai.backend.model_manager.taxonomy import Qwen3VariantType
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear
from invokeai.backend.quantization.nvfp4 import NVFP4Linear
from tests.fixtures.quantized_payloads import comfy_quant_marker, nvfp4_signed_tensors, quantize_convrot

# Not a known Qwen3 size, so the loader reads the head counts off the projections at its fixed head_dim of 128:
# a one-head model.
HIDDEN = 128
INTERMEDIATE = 256
# A multiple of 128, so an nvfp4 `lm_head` over it is a valid block-scale grid.
VOCAB = 128

NVFP4_PROJECTIONS = {
    "self_attn.q_proj": (HIDDEN, HIDDEN),
    "self_attn.k_proj": (HIDDEN, HIDDEN),
    "self_attn.o_proj": (HIDDEN, HIDDEN),
    "mlp.gate_proj": (INTERMEDIATE, HIDDEN),
    "mlp.up_proj": (INTERMEDIATE, HIDDEN),
    "mlp.down_proj": (HIDDEN, INTERMEDIATE),
}


def _nvfp4_layer(path: str, positive: torch.Tensor, evidence: str) -> dict[str, torch.Tensor]:
    """`evidence` picks the transport: the per-tensor marker, or the header entry the caller writes."""
    layer, _ = nvfp4_signed_tensors(path, positive)
    if evidence == "marker":
        layer[f"{path}.comfy_quant"] = comfy_quant_marker({"format": "nvfp4"})
    return layer


def _write_checkpoint(tmp_path: Path, evidence: str) -> tuple[Path, dict[str, torch.Tensor]]:
    """Returns the file and the dense weights it has to load as."""
    torch.manual_seed(0)
    tensors: dict[str, torch.Tensor] = {}
    dense: dict[str, torch.Tensor] = {}
    nvfp4_paths = [f"model.layers.0.{name}" for name in NVFP4_PROJECTIONS] + ["lm_head"]
    for path, shape in zip(nvfp4_paths, [*NVFP4_PROJECTIONS.values(), (VOCAB, HIDDEN)], strict=True):
        positive = torch.randint(0, 2, shape, dtype=torch.bool)
        tensors.update(_nvfp4_layer(path, positive, evidence))
        if path != "lm_head":
            dense[f"{path}.weight"] = torch.where(positive, 0.5, -0.5)

    fp8_values = torch.randint(-8, 9, (HIDDEN, HIDDEN)).float()
    tensors["model.layers.0.self_attn.v_proj.weight"] = fp8_values.to(torch.float8_e4m3fn)
    tensors["model.layers.0.self_attn.v_proj.weight_scale"] = torch.tensor(0.5)
    tensors["model.layers.0.self_attn.v_proj.comfy_quant"] = comfy_quant_marker({"format": "float8_e4m3fn"})
    dense["model.layers.0.self_attn.v_proj.weight"] = fp8_values * 0.5

    for key, tensor in {
        "model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN),
        "model.layers.0.input_layernorm.weight": torch.rand(HIDDEN) + 0.5,
        "model.layers.0.post_attention_layernorm.weight": torch.rand(HIDDEN) + 0.5,
        "model.layers.0.self_attn.q_norm.weight": torch.rand(128) + 0.5,
        "model.layers.0.self_attn.k_norm.weight": torch.rand(128) + 0.5,
        "model.norm.weight": torch.rand(HIDDEN) + 0.5,
    }.items():
        tensors[key] = tensor
        dense[key] = tensor

    checkpoint = tmp_path / "qwen_3_4b_fp4_mixed.safetensors"
    header = {path: {"format": "nvfp4"} for path in nvfp4_paths}
    metadata = {"_quantization_metadata": json.dumps({"layers": header})} if evidence == "header" else None
    save_file(tensors, checkpoint, metadata=metadata)
    return checkpoint, dense


def _dense_reference(dense: dict[str, torch.Tensor]) -> Qwen3ForCausalLM:
    """The same encoder as a plain Qwen3 holding the decoded weights, configured the way the loader does it."""
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=40960,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        use_sliding_window=False,
        attention_bias=False,
        attention_dropout=0.0,
    )
    model = Qwen3ForCausalLM(config)
    model.load_state_dict(dense, strict=False)
    model.tie_weights()
    return model.eval()


@pytest.mark.parametrize("evidence", ["marker", "header"])
def test_an_fp4_mixed_encoder_loads_packed_and_encodes_like_its_dense_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, evidence: str
) -> None:
    checkpoint, dense = _write_checkpoint(tmp_path, evidence)
    monkeypatch.setattr(z_image.TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(z_image.TorchDevice, "choose_bfloat16_safe_dtype", lambda _device: torch.float32)
    loader = object.__new__(Qwen3EncoderCheckpointLoader)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    reserved_before_fold: list[bool] = []
    fold = z_image._fold_comfy_scaled_weights

    def recording_fold(sd: dict, dtype: torch.dtype) -> int:
        reserved_before_fold.append(loader._ram_cache.make_room.called)
        return fold(sd, dtype)

    monkeypatch.setattr(z_image, "_fold_comfy_scaled_weights", recording_fold)
    config = Qwen3Encoder_Checkpoint_Config.model_construct(path=str(checkpoint), variant=Qwen3VariantType.Qwen3_4B)

    model = loader._load_from_singlefile(config)

    layer = model.model.layers[0]
    for name in NVFP4_PROJECTIONS:
        module = layer.get_submodule(name)
        assert isinstance(module, NVFP4Linear), name
        # Still as stored: installed before the cast, the buffers would be widened.
        assert module.weight.dtype is torch.uint8 and module.weight_scale.dtype is torch.uint8, name
    assert type(layer.self_attn.v_proj) is torch.nn.Linear
    assert type(model.lm_head) is torch.nn.Linear
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert reserved_before_fold == [True]

    input_ids = torch.tensor([[1, 5, 7, 2, 30]])
    with torch.no_grad():
        encoded = model(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
        expected = _dense_reference(dense)(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
    assert torch.equal(encoded, expected)

    # One reservation: the rest of the state dict at float32 plus the packed projections as stored. The dropped
    # `lm_head` costs nothing, and the packed layers are not charged their decoded size.
    packed_paths = {f"model.layers.0.{name}.weight" for name in NVFP4_PROJECTIONS}
    rest = sum(tensor.nelement() * 4 for key, tensor in dense.items() if key not in packed_paths)
    packed = sum(rows * columns // 2 + rows * columns // 16 + 4 for rows, columns in NVFP4_PROJECTIONS.values())
    loader._ram_cache.make_room.assert_called_once()
    (reserved,), _ = loader._ram_cache.make_room.call_args
    assert rest + packed <= reserved < rest + packed + 1024


# The real repacks rotate over 256-wide groups, which this 128-wide toy encoder cannot hold. The marker
# carries the group size for exactly this reason -- a repack derotated with the wrong width runs and
# generates noise -- and 64-wide repacks exist, so reading it is what the loader has to do.
_INT8_GROUP = 64
_INT8_MARKER = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": _INT8_GROUP}


def _write_int8_checkpoint(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    """The int8 layout: codes, a per-output-row scale, a marker. Norms and embeddings stay dense,
    exactly as the 8B repack ships them."""
    torch.manual_seed(0)
    tensors: dict[str, torch.Tensor] = {}
    dense: dict[str, torch.Tensor] = {}
    projections = {**NVFP4_PROJECTIONS, "self_attn.v_proj": (HIDDEN, HIDDEN)}
    for name, shape in projections.items():
        path = f"model.layers.0.{name}"
        codes, scale, restored = quantize_convrot(torch.randn(shape) * 0.05, group_size=_INT8_GROUP)
        tensors[f"{path}.weight"] = codes
        tensors[f"{path}.weight_scale"] = scale
        tensors[f"{path}.comfy_quant"] = comfy_quant_marker(_INT8_MARKER)
        # W8A8 activation scales. This path dequantizes the weight and computes in the compute
        # dtype, so there is nothing to apply them to; one Qwen3-VL repack ships 337 of them.
        tensors[f"{path}.input_scale"] = torch.ones(shape[1])
        dense[f"{path}.weight"] = restored

    for key, tensor in {
        "model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN),
        "model.layers.0.input_layernorm.weight": torch.rand(HIDDEN) + 0.5,
        "model.layers.0.post_attention_layernorm.weight": torch.rand(HIDDEN) + 0.5,
        "model.layers.0.self_attn.q_norm.weight": torch.rand(128) + 0.5,
        "model.layers.0.self_attn.k_norm.weight": torch.rand(128) + 0.5,
        "model.norm.weight": torch.rand(HIDDEN) + 0.5,
    }.items():
        tensors[key] = tensor
        dense[key] = tensor

    checkpoint = tmp_path / "qwen_3_8b_int8_convrot.safetensors"
    save_file(tensors, checkpoint)
    return checkpoint, dense


def test_an_int8_convrot_encoder_stays_int8_resident_and_encodes_like_its_dense_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint, dense = _write_int8_checkpoint(tmp_path)
    monkeypatch.setattr(z_image.TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(z_image.TorchDevice, "choose_bfloat16_safe_dtype", lambda _device: torch.float32)
    loader = object.__new__(Qwen3EncoderCheckpointLoader)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())

    def refuse(*_args, **_kwargs) -> int:
        raise AssertionError("the scaled-fp8 fold must not see an int8 checkpoint")

    monkeypatch.setattr(z_image, "_fold_comfy_scaled_weights", refuse)
    config = Qwen3Encoder_Checkpoint_Config.model_construct(path=str(checkpoint), variant=Qwen3VariantType.Qwen3_8B)

    model = loader._load_from_singlefile(config)

    layer = model.model.layers[0]
    for name in (*NVFP4_PROJECTIONS, "self_attn.v_proj"):
        module = layer.get_submodule(name)
        assert isinstance(module, Int8ConvrotLinear), name
        # As stored. Cast to the compute dtype these would be bf16 integers, which loads and encodes noise.
        assert module.weight.dtype is torch.int8, name
    assert model.lm_head.weight is model.model.embed_tokens.weight

    input_ids = torch.tensor([[1, 5, 7, 2, 30]])
    with torch.no_grad():
        encoded = model(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
        expected = _dense_reference(dense)(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
    assert torch.allclose(encoded, expected, atol=1e-4)

    # One reservation, made before anything widened. Spelled out rather than bounded: the int8
    # payloads charged one byte per code and their float32 scale column, the dense remainder charged
    # float32, and the `.input_scale` activation scales charged nothing -- this path has nothing to
    # apply them to, and left in the dict they would each cost `in_features` floats here. Reserving
    # the decoded size instead would ask the cache to free four times what this load uses -- on the
    # real 8B encoder, ~30GB for ~8.
    shapes = [*NVFP4_PROJECTIONS.values(), (HIDDEN, HIDDEN)]
    quantized_weights = {f"model.layers.0.{name}.weight" for name in (*NVFP4_PROJECTIONS, "self_attn.v_proj")}
    codes_and_scales = sum(rows * columns + rows * 4 for rows, columns in shapes)
    dense_bytes = sum(tensor.nelement() * 4 for key, tensor in dense.items() if key not in quantized_weights)
    loader._ram_cache.make_room.assert_called_once()
    (reserved,), _ = loader._ram_cache.make_room.call_args
    assert reserved == codes_and_scales + dense_bytes


def test_an_nvfp4_layer_missing_its_global_scale_is_refused_before_the_cache_is_evicted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The degraded nvfp4 half-state, at this seam rather than at the detector.

    A packed uint8 weight with a block-scale grid and no `weight_scale_2` is the state a guard keyed
    on `weight_scale_2` -- the key the decode keys on -- lets straight through. `_find_nvfp4_layers`
    refuses it and `test_nvfp4.py` pins that; what only a seam can answer is whether this loader
    reaches the detector before it asks the cache for room. Asserted through the cache double this
    file already uses, in the same idiom as the cells above, because this loader reads its header
    from the real path rather than from a served state dict.
    """
    tensors = {
        "model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN),
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(HIDDEN, HIDDEN // 2, dtype=torch.uint8),
        "model.layers.0.self_attn.q_proj.weight_scale": torch.zeros(HIDDEN, HIDDEN // 16).to(torch.float8_e4m3fn),
    }
    checkpoint = tmp_path / "qwen_3_4b_fp4_half.safetensors"
    save_file(tensors, checkpoint)
    monkeypatch.setattr(z_image.TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(z_image.TorchDevice, "choose_bfloat16_safe_dtype", lambda _device: torch.float32)
    loader = object.__new__(Qwen3EncoderCheckpointLoader)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    config = Qwen3Encoder_Checkpoint_Config.model_construct(path=str(checkpoint), variant=Qwen3VariantType.Qwen3_4B)

    with pytest.raises(ValueError, match="with a weight_scale but no weight_scale_2"):
        loader._load_from_singlefile(config)

    loader._ram_cache.make_room.assert_not_called()
