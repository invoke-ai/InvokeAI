"""The single-file Qwen3-VL encoder loader keeps Comfy's nvfp4 builds packed.

`qwen3vl_8b_nvfp4.safetensors` (Comfy-Org, Ideogram 4) packs all seven projections of every layer, names them only in
`.comfy_quant` markers, and keeps the token embedding and the LM head in scaled fp8. Community 4B repacks follow
either that marker convention or name their layers in the safetensors header, one of the latter packing the visual
tower too. Both reach the loader shared by Krea-2 and Ideogram 4. What these tests pin is the order: the packed
layers leave the state dict before either side-channel branch reads a scale, come back packed under the transformers
paths of a real `Qwen3VLModel`, are counted in the one reservation as they are held, and a folded scaled embedding is
not re-quantized without its scale -- while the Linears an MXFP8 build folds still go to fp8 storage as before. The
compute dtype is bf16, as in production: the packed global scale must not be cast with it.
"""

import json

import pytest
import safetensors.torch
import torch
from transformers import Qwen3VLConfig, Qwen3VLModel

from invokeai.backend.model_manager.configs.qwen3_vl_encoder import Qwen3VLEncoder_Checkpoint_Config
from invokeai.backend.model_manager.load.model_loaders import krea2
from invokeai.backend.model_manager.load.model_loaders.krea2 import Qwen3VLEncoderCheckpointLoader
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear
from invokeai.backend.quantization.nvfp4 import NVFP4Linear
from tests.fixtures.loader_seams import Seam, prepare
from tests.fixtures.quantized_payloads import (
    MX_BLOCK_SIZE,
    comfy_quant_marker,
    mxfp8_marker,
    mxfp8_tensors,
    nvfp4_signed_tensors,
    quantize_scaled_fp8,
)

COMPUTE_DTYPE = torch.bfloat16
HIDDEN = 128
INTERMEDIATE = 256
VOCAB = 32
HEAD_DIM = 32

# Every projection, as the official build packs them: (layer path in the file, [out, in]). The widths are whole
# 128-row scale tiles, the smallest an nvfp4 layer can be.
PROJECTIONS = {
    "model.layers.0.self_attn.q_proj": (HIDDEN, HIDDEN),
    "model.layers.0.self_attn.k_proj": (HIDDEN, HIDDEN),
    "model.layers.0.self_attn.v_proj": (HIDDEN, HIDDEN),
    "model.layers.0.self_attn.o_proj": (HIDDEN, HIDDEN),
    "model.layers.0.mlp.gate_proj": (INTERMEDIATE, HIDDEN),
    "model.layers.0.mlp.up_proj": (INTERMEDIATE, HIDDEN),
    "model.layers.0.mlp.down_proj": (HIDDEN, INTERMEDIATE),
}
# A packed visual layer, as the header-named 4B repack ships one. The tower is dropped before anything reads it.
VISUAL = "model.visual.blocks.0.mlp.linear_fc1"
# The bf16 the folded embedding and the norms (two per layer, q/k norms, the final one) are held in.
DENSE_ELEMENTS = VOCAB * HIDDEN + 2 * HIDDEN + 2 * HEAD_DIM + HIDDEN

SEAM = Seam(
    loader=Qwen3VLEncoderCheckpointLoader,
    module=krea2,
    entry="_load_text_encoder",
    # The loader imports `load_file` inside the method, so the name it resolves is the package's.
    load_file_host=safetensors.torch,
    compute_dtype=COMPUTE_DTYPE,
    patches_device=True,
    sets_torch_dtype=False,
    # This entry calls `_apply_fp8_to_nn_module` itself, and the storage pass is part of what is tested.
    casts_fp8_storage=False,
)


def _te_config() -> Qwen3VLConfig:
    """Qwen3-VL at toy width: one text layer whose projections all fit nvfp4's tiles, and a one-block tower."""
    return Qwen3VLConfig(
        text_config={
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": HEAD_DIM,
            "vocab_size": VOCAB,
            "rope_scaling": {"rope_type": "default", "mrope_section": [8, 4, 4], "mrope_interleaved": True},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 2,
            "out_hidden_size": HIDDEN,
            "deepstack_visual_indexes": [],
        },
    )


def _transformers_path(path: str) -> str:
    """Where `Qwen3VLModel` holds a language-model layer the file spells `model.X`."""
    return path.replace("model.", "language_model.", 1)


def _packed_bytes(rows: int, columns: int) -> int:
    return rows * columns // 2 + rows * (columns // 16) + 4


def _dense_checkpoint() -> dict[str, torch.Tensor]:
    """The toy model's own state dict in bf16, spelled the way a Comfy export spells it.

    Starting from the real model is what makes every key the loader has to fill be there, and the norms carry
    real values.
    """
    torch.manual_seed(0)
    reference = Qwen3VLModel(_te_config()).state_dict()
    return {
        f"model.{key.removeprefix('language_model.')}": value.to(torch.bfloat16) for key, value in reference.items()
    }


def _checkpoint(naming: str) -> tuple[dict[str, torch.Tensor], dict[str, str] | None, dict[str, torch.Tensor]]:
    """The official build's layout at toy width: (state dict, header, expected weight per transformers path)."""
    state_dict = _dense_checkpoint()
    header: dict[str, dict[str, str]] = {}
    expected: dict[str, torch.Tensor] = {}
    for path, (rows, columns) in PROJECTIONS.items():
        del state_dict[f"{path}.weight"]
        tensors, weight = nvfp4_signed_tensors(path, torch.randint(0, 2, (rows, columns), dtype=torch.bool))
        state_dict.update(tensors)
        expected[_transformers_path(path)] = weight
        if naming == "marker":
            state_dict[f"{path}.comfy_quant"] = comfy_quant_marker({"format": "nvfp4"})
        else:
            header[path] = {"format": "nvfp4"}
    if naming == "header":
        del state_dict[f"{VISUAL}.weight"]
        tensors, _ = nvfp4_signed_tensors(VISUAL, torch.ones(128, 64, dtype=torch.bool))
        state_dict.update(tensors)
        header[VISUAL] = {"format": "nvfp4"}

    # The official build's scaled fp8 embedding and LM head.
    for path, payload in (
        ("model.embed_tokens", quantize_scaled_fp8(torch.randn(VOCAB, HIDDEN))),
        ("lm_head", quantize_scaled_fp8(torch.randn(VOCAB, HIDDEN))),
    ):
        state_dict[f"{path}.weight"] = payload.codes
        state_dict[f"{path}.weight_scale"] = payload.scale
        state_dict[f"{path}.comfy_quant"] = comfy_quant_marker(
            {"format": "float8_e4m3fn", "full_precision_matrix_mult": False}
        )
        expected[f"{path}.dequantized"] = payload.dequantized
    metadata = {"_quantization_metadata": json.dumps({"layers": header})} if header else None
    return state_dict, metadata, expected


def _prepare(monkeypatch: pytest.MonkeyPatch, state_dict, metadata, *, matmul: bool = False, storage: bool = False):
    run = prepare(
        SEAM,
        monkeypatch,
        state_dict=state_dict,
        metadata=metadata,
        observe=("dequantize_fp8_scaled", "split_fp8_scaled_layers"),
    )
    run.loader._load_te_config = lambda _config: _te_config()
    monkeypatch.setattr(krea2, "should_keep_fp8_weights", lambda _device: matmul)
    monkeypatch.setattr(krea2, "_device_supports_fp8_storage", lambda *_args: storage)
    return run


def _config() -> Qwen3VLEncoder_Checkpoint_Config:
    return Qwen3VLEncoder_Checkpoint_Config.model_construct(path="qwen3vl_8b_nvfp4.safetensors", name="encoder")


def _logged(run) -> list[str]:
    return [str(call.args[0]) for call in run.loader._logger.info.call_args_list]


def _assert_packed(model: torch.nn.Module, path: str, weight: torch.Tensor) -> None:
    module = model.get_submodule(_transformers_path(path))
    assert isinstance(module, NVFP4Linear), path
    assert module.weight.dtype is torch.uint8, path
    assert module.weight_scale_2.dtype is torch.float32, path
    # A float32 activation, not the compute dtype: CPU bf16 GEMM faults on part of the windows runner fleet.
    # Every decoded weight is +-0.5, exact in either width.
    x = torch.randn(3, module.in_features, dtype=torch.float32)
    torch.testing.assert_close(module(x), torch.nn.functional.linear(x, weight))


@pytest.mark.parametrize("mode", ["fold", "fp8_compute", "fp8_storage"])
@pytest.mark.parametrize("naming", ["marker", "header"])
def test_named_nvfp4_layers_stay_packed_beside_a_folded_embedding(
    monkeypatch: pytest.MonkeyPatch, naming: str, mode: str
) -> None:
    state_dict, metadata, expected = _checkpoint(naming)
    run = _prepare(monkeypatch, state_dict, metadata, matmul=mode == "fp8_compute", storage=mode == "fp8_storage")

    model = run.load(_config())

    for path in PROJECTIONS:
        _assert_packed(model, path, expected[_transformers_path(path)])

    # Comfy's embedding is scaled fp8, and no matmul takes an embedding: it is folded with its scale. Under fp8
    # storage it must stay folded -- cast again it would be re-quantized without the scale. With every Linear
    # packed nothing else is left to cast, and the log must not claim otherwise.
    embedding = model.language_model.embed_tokens.weight
    assert embedding.dtype is COMPUTE_DTYPE
    assert torch.equal(embedding, expected["model.embed_tokens.dequantized"].to(COMPUTE_DTYPE))
    assert not any("FP8 layerwise casting enabled" in message for message in _logged(run))

    # One reservation, before the fold and the split widen anything. Counted by hand: the packed layers as held,
    # the folded embedding and the norms at bf16, and the embedding's scalar scale. The LM head is in none of it:
    # this model has none, and folding it first would only be paid for and thrown away.
    packed = sum(_packed_bytes(rows, columns) for rows, columns in PROJECTIONS.values())
    assert run.reserved == [packed + DENSE_ELEMENTS * COMPUTE_DTYPE.itemsize + 4]
    assert run.order and all(reserved == 1 for _step, reserved in run.order), run.order


@pytest.mark.parametrize("naming", ["marker", "header"])
def test_an_int8_neighbour_leaves_the_nvfp4_layers_packed_and_reserved(
    monkeypatch: pytest.MonkeyPatch, naming: str
) -> None:
    """The int8 branch strips every marker it leaves, nvfp4's included, so the packed layers must be out first; and
    its reservation is the only one, so it has to count them."""
    state_dict, metadata, expected = _checkpoint(naming)
    int8 = "model.layers.0.mlp.down_proj"
    # An int8 file carries no scaled fp8 beside it: that branch refuses foreign float8 scales.
    for path in ("model.embed_tokens", "lm_head"):
        state_dict[f"{path}.weight"] = expected[f"{path}.dequantized"].to(torch.bfloat16)
        del state_dict[f"{path}.weight_scale"], state_dict[f"{path}.comfy_quant"]
    for suffix in (".weight", ".weight_scale", ".weight_scale_2", ".comfy_quant"):
        state_dict.pop(f"{int8}{suffix}", None)
    if metadata is not None:
        header = json.loads(metadata["_quantization_metadata"])
        del header["layers"][int8]
        metadata = {"_quantization_metadata": json.dumps(header)}
    codes = torch.randint(-127, 128, (HIDDEN, INTERMEDIATE), dtype=torch.int8)
    scale = torch.rand(HIDDEN, 1) + 0.5
    state_dict[f"{int8}.weight"] = codes
    state_dict[f"{int8}.weight_scale"] = scale
    state_dict[f"{int8}.comfy_quant"] = comfy_quant_marker({"format": "int8_tensorwise", "convrot": False})
    run = _prepare(monkeypatch, state_dict, metadata)

    model = run.load(_config())

    for path in PROJECTIONS:
        if path != int8:
            _assert_packed(model, path, expected[_transformers_path(path)])
    int8_module = model.get_submodule(_transformers_path(int8))
    assert isinstance(int8_module, Int8ConvrotLinear)
    torch.testing.assert_close(
        int8_module._dequantized_weight(torch.device("cpu"), torch.float32), codes.float() * scale
    )

    packed = sum(_packed_bytes(rows, columns) for path, (rows, columns) in PROJECTIONS.items() if path != int8)
    held_int8 = HIDDEN * INTERMEDIATE + HIDDEN * scale.element_size()
    assert run.reserved == [packed + held_int8 + DENSE_ELEMENTS * COMPUTE_DTYPE.itemsize]


def test_an_mxfp8_build_keeps_its_language_model_in_fp8_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every MXFP8 Linear is folded -- `_scaled_mm` cannot apply a block-wise scale -- so its scaled layers reach the
    storage-only pass, which must still hold them in fp8: only a folded *embedding* stays wide there. Kept at the
    compute dtype instead, the language model would be resident at twice the size it had before."""
    state_dict = _dense_checkpoint()
    expected: dict[str, torch.Tensor] = {}
    torch.manual_seed(1)
    for path, (rows, columns) in PROJECTIONS.items():
        # Exponent bytes around the bias: block scales from 0.5 to 4, exact in e4m3 as in bf16.
        tensors, block_scales = mxfp8_tensors(path, torch.randint(126, 130, (rows, columns // MX_BLOCK_SIZE)))
        state_dict.update(tensors)
        state_dict[f"{path}.comfy_quant"] = comfy_quant_marker(mxfp8_marker())
        expected[_transformers_path(path)] = block_scales.repeat_interleave(MX_BLOCK_SIZE, dim=1)
    run = _prepare(monkeypatch, state_dict, None, storage=True)

    model = run.load(_config())

    for path, weight in expected.items():
        stored = model.get_submodule(path).weight
        assert stored.dtype is torch.float8_e4m3fn, path
        assert torch.equal(stored.float(), weight), path
    assert any("FP8 layerwise casting enabled" in message for message in _logged(run))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda sd: sd.pop("model.layers.0.self_attn.q_proj.comfy_quant"),
            "no ComfyUI `comfy_quant` marker",
            id="unnamed",
        ),
        pytest.param(
            lambda sd: sd.pop("model.layers.0.self_attn.q_proj.weight_scale_2"),
            "no weight_scale_2",
            id="no-global-scale",
        ),
        pytest.param(
            lambda sd: sd.update({"model.layers.0.self_attn.q_proj.pre_quant_scale": torch.ones(HIDDEN)}),
            "AWQ",
            id="awq",
        ),
    ],
)
def test_a_layer_the_decode_would_misread_is_refused_before_the_loader_reserves_room(
    monkeypatch: pytest.MonkeyPatch, mutate, message: str
) -> None:
    state_dict, metadata, _ = _checkpoint("marker")
    mutate(state_dict)
    run = _prepare(monkeypatch, state_dict, metadata)

    with pytest.raises(ValueError, match=message):
        run.load(_config())

    assert run.reserved == []
