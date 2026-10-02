"""MiniMax H3's int8_tensorwise single-file path, driven through the loader over a real file.

H3 is where this scheme entered the codebase, and it kept its own copy of the call site: the
marker's ``convrot_groupsize`` was never read and the scale layout was never checked. Both are
things a checkpoint can vary and neither fails loudly, so they are pinned here against the loader
rather than against the shared helper they now go through.

The fixture is a real safetensors file because the loader reads the markers twice by two different
routes -- once from the header before committing to the tensor read, once from the tensors -- and
the header route is only exercised by an actual file.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.minimax_h3.transformer_minimax_h3 import MiniMaxH3AttnProcessor
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_MiniMaxH3_Config
from invokeai.backend.model_manager.load.model_loaders.minimax_h3 import MiniMaxH3CheckpointModel
from invokeai.backend.model_manager.taxonomy import MiniMaxH3VariantType
from invokeai.backend.quantization.int8_convrot import (
    Int8ConvrotLinear,
    read_comfy_quant_markers,
)
from tests.fixtures.quantized_payloads import comfy_quant_marker, quantize_convrot


@pytest.fixture(autouse=True)
def _restore_attention_processor():
    """The loader installs H3's contiguous-QKV attention patch, which rebinds
    `MiniMaxH3AttnProcessor.__call__` for the whole process.

    `test_contiguous_attention.py` compares the patched call against the *unpatched* one, so a
    worker that ran this file first would have it comparing the patch to itself -- passing, and
    checking nothing. Under `--dist loadfile` which file a worker gets is arbitrary, so restore.
    """
    original = MiniMaxH3AttnProcessor.__call__
    was_marked = "_invokeai_contiguous_qkv" in vars(MiniMaxH3AttnProcessor)
    yield
    MiniMaxH3AttnProcessor.__call__ = original
    if not was_marked and "_invokeai_contiguous_qkv" in vars(MiniMaxH3AttnProcessor):
        delattr(MiniMaxH3AttnProcessor, "_invokeai_contiguous_qkv")


# Hidden/inner/ffn all 64 so every quantized in_features is divisible by the 64-wide group the
# marker below declares. Curve grid 5 x 3, one block and one refiner block: the AdaLN-pruned
# layout, whose converted keys are known to match the model exactly.
HIDDEN = INNER = FFN = 64
HEAD_DIM = 32
CURVE_DIM = 3
GROUP_SIZE = 64

QUANTIZED_SOURCE_KEY = "blocks.0.attn.out_proj.weight"
QUANTIZED_MODULE_PATH = ("transformer_blocks", 0, "attn", "to_out", 0)


def _tiny_remote_code_state_dict() -> dict[str, torch.Tensor]:
    """A minimal AdaLN-pruned H3 checkpoint in MiniMax's remote-code key layout."""
    sd: dict[str, torch.Tensor] = {
        "video_patch_proj.weight": torch.randn(HIDDEN, 4),
        "video_patch_proj.bias": torch.randn(HIDDEN),
        "audio_patch_proj.weight": torch.randn(HIDDEN, 4),
        "audio_patch_proj.bias": torch.randn(HIDDEN),
        "condition_proj.weight": torch.randn(HIDDEN, 6),
        "condition_proj.bias": torch.randn(HIDDEN),
        "adaln_t_table": torch.randn(5, CURVE_DIM),
        "rope.inv_freq": torch.randn(2),
        "token_refiner.final_norm.weight": torch.randn(HIDDEN),
        "final_layer.norm.weight": torch.randn(HIDDEN),
        "final_layer.adaln_proj.linear.weight": torch.randn(2 * HIDDEN, CURVE_DIM),
        "final_layer.adaln_proj.linear.bias": torch.randn(2 * HIDDEN),
        "final_layer.video_out.weight": torch.randn(4, HIDDEN),
        "final_layer.video_out.bias": torch.randn(4),
        "final_layer.audio_out.weight": torch.randn(4, HIDDEN),
        "final_layer.audio_out.bias": torch.randn(4),
    }
    for prefix in ("token_refiner.blocks.0.", "blocks.0."):
        sd[prefix + "norm1.weight"] = torch.randn(HIDDEN)
        sd[prefix + "norm2.weight"] = torch.randn(HIDDEN)
        sd[prefix + "attn.qkv_proj.weight"] = torch.randn(3 * INNER, HIDDEN)
        sd[prefix + "attn.q_norm.weight"] = torch.randn(HEAD_DIM)
        sd[prefix + "attn.k_norm.weight"] = torch.randn(HEAD_DIM)
        sd[prefix + "attn.out_proj.weight"] = torch.randn(HIDDEN, INNER)
        sd[prefix + "mlp.fc1.weight"] = torch.randn(2 * FFN, HIDDEN)
        sd[prefix + "mlp.fc2.weight"] = torch.randn(HIDDEN, FFN)
    sd["blocks.0.adaln_proj.linear.weight"] = torch.randn(6 * HIDDEN * 3, CURVE_DIM)
    sd["blocks.0.adaln_proj.linear.bias"] = torch.randn(6 * HIDDEN * 3)
    return sd


def _write_checkpoint(
    tmp_path: Path, *, scale: torch.Tensor | None = None, marker: dict | None = None, pad: int = 16
) -> tuple[Path, torch.Tensor]:
    """A tiny pruned H3 checkpoint with `blocks.0.attn.out_proj` quantized; returns its true weight."""
    torch.manual_seed(0)
    sd = _tiny_remote_code_state_dict()
    original = sd[QUANTIZED_SOURCE_KEY]
    quantized, derived_scale, _restored = quantize_convrot(original, group_size=GROUP_SIZE)
    sd[QUANTIZED_SOURCE_KEY] = quantized
    sd["blocks.0.attn.out_proj.weight_scale"] = derived_scale if scale is None else scale
    sd["blocks.0.attn.out_proj.comfy_quant"] = comfy_quant_marker(
        marker or {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP_SIZE}, pad=pad
    )
    path = tmp_path / "minimax_h3_int8_convrot.safetensors"
    save_file(sd, str(path))
    return path, original


def _load(path: Path) -> torch.nn.Module:
    config = Main_Checkpoint_MiniMaxH3_Config.model_construct(
        path=str(path), variant=MiniMaxH3VariantType.FL2VA, pruned=True, name="h3"
    )
    loader = object.__new__(MiniMaxH3CheckpointModel)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    loader._logger = MagicMock()
    return loader._load_transformer_from_singlefile(config)


def _quantized_module(model: torch.nn.Module) -> torch.nn.Module:
    module: torch.nn.Module = model
    for step in QUANTIZED_MODULE_PATH:
        module = module[step] if isinstance(step, int) else getattr(module, step)
    return module


def test_the_marker_group_size_reaches_the_module(tmp_path) -> None:
    """The rotation width is per tensor, and H3 assumed it. A 64-wide repack derotated with a
    256-wide Hadamard is not a subtle error -- the shapes do not even divide -- but a producer
    varying it the other way (a wider group on a wide layer) would run and generate noise."""
    path, original = _write_checkpoint(tmp_path)

    model = _load(path)

    module = _quantized_module(model)
    assert isinstance(module, Int8ConvrotLinear)
    assert module.group_size == GROUP_SIZE
    # And the weight it computes is the un-rotated original, not the still-rotated one.
    dequantized = module._dequantized_weight(torch.device("cpu"), torch.float32).flatten()
    assert torch.corrcoef(torch.stack([dequantized, original.flatten()]))[0, 1] > 0.999


def test_a_blockwise_scale_grid_is_refused_by_name(tmp_path) -> None:
    """H3 constructed its modules without the scale-layout check, so a block grid broadcast
    against the weight instead of raising: for a square weight that is silent."""
    path, _ = _write_checkpoint(tmp_path, scale=torch.ones(HIDDEN // 8, INNER // 8))

    with pytest.raises(ValueError, match=r"Blockwise scale grids"):
        _load(path)


def test_a_nul_padded_header_marker_is_read(tmp_path) -> None:
    """The header reader runs before the tensor read, so its strict parse was what a padded marker
    actually hit -- a JSONDecodeError out of the middle of a load, naming neither file nor key.
    The tolerant parser the state-dict readers use was written for exactly this."""
    path, _ = _write_checkpoint(tmp_path, pad=16)

    markers = read_comfy_quant_markers(path)

    assert markers == {
        "blocks.0.attn.out_proj": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP_SIZE}
    }


def _write_fp8_checkpoint(tmp_path: Path, *, declared_in_header: bool) -> Path:
    """A tiny pruned H3 checkpoint in ComfyUI scaled fp8: e4m3 codes plus a per-tensor scale on the
    fused qkv and the out projection, the two layers the converter handles differently."""
    import json

    from tests.fixtures.quantized_payloads import quantize_scaled_fp8

    torch.manual_seed(0)
    sd = _tiny_remote_code_state_dict()
    layers = {}
    for path in ("blocks.0.attn.qkv_proj", "blocks.0.attn.out_proj"):
        payload = quantize_scaled_fp8(sd[f"{path}.weight"])
        sd[f"{path}.weight"], sd[f"{path}.weight_scale"] = payload.codes, payload.scale.reshape(())
        layers[path] = {"format": "float8_e4m3fn"}
    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": layers})}
    path = tmp_path / "minimax_h3_fp8_scaled.safetensors"
    save_file(sd, str(path), metadata=metadata if declared_in_header else None)
    return path


def test_a_header_only_fp8_build_is_refused_before_the_tensor_read(tmp_path, monkeypatch) -> None:
    """The gate read per-tensor `.comfy_quant` markers and nothing else, but the producer tool decides
    the transport: FLUX.2's Comfy-Org fp8 build names every layer in `_quantization_metadata` and
    carries no marker at all. Such a file passed the gate, paid for the full tensor read, and died in
    the fused-qkv split on a per-tensor scale -- measured as a bare `IndexError: tuple index out of
    range`, which names neither fp8 nor the format. The header says what it is; read it there.
    """
    import safetensors.torch

    path = _write_fp8_checkpoint(tmp_path, declared_in_header=True)
    monkeypatch.setattr(
        safetensors.torch, "load_file", lambda _path: pytest.fail("the tensors were read for a file refused by name")
    )

    with pytest.raises(ValueError, match=r"float8_e4m3fn.*MiniMax H3 checkpoint"):
        _load(path)


def test_an_undeclared_fp8_build_is_refused_by_name(tmp_path) -> None:
    """The older ComfyUI shape: fp8 weights and a `weight_scale` beside them, declared nowhere --
    neither marker nor header. Nothing can be refused before the read, but it can be before the
    converter, which is where it failed: the qkv split on the scale, or `load_state_dict` on the
    orphaned out-projection scale as an "unexpected key"."""
    path = _write_fp8_checkpoint(tmp_path, declared_in_header=False)

    with pytest.raises(ValueError, match=r"float8 weight"):
        _load(path)


def test_the_text_encoder_reads_the_header_the_same_way(tmp_path, monkeypatch) -> None:
    """The encoder carried its own copy of the marker-only gate, so it had the same blind spot. Both
    now go through one helper; this pins that the encoder is wired to it *before* its tensor read.
    What the call buys is the ~25 GiB not spent: with it gone, a float8 file is still refused, by
    name, one step later by the dtype check. Only a declared format that is not float8 -- nvfp4,
    say -- would be misdiagnosed without it."""
    import json

    import safetensors.torch

    from invokeai.backend.model_manager.load.model_loaders.minimax_h3 import MiniMaxH3TextEncoderCheckpointModel

    layers = {"model.layers.0.mlp.down_proj": {"format": "float8_e4m3fn"}}
    path = tmp_path / "qwen3vl_32b_minimax_h3_fp8_scaled.safetensors"
    save_file(
        {"model.layers.0.mlp.down_proj.weight": torch.zeros(4, 4).to(torch.float8_e4m3fn)},
        str(path),
        metadata={"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": layers})},
    )
    monkeypatch.setattr(
        safetensors.torch, "load_file", lambda _path: pytest.fail("the tensors were read for a file refused by name")
    )
    loader = object.__new__(MiniMaxH3TextEncoderCheckpointModel)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    loader._logger = MagicMock()

    with pytest.raises(ValueError, match=r"float8_e4m3fn.*MiniMax H3 text encoder"):
        loader._load_text_encoder_from_singlefile(SimpleNamespace(path=str(path)))


def _encoder_loader():
    from invokeai.backend.model_manager.load.model_loaders.minimax_h3 import MiniMaxH3TextEncoderCheckpointModel

    loader = object.__new__(MiniMaxH3TextEncoderCheckpointModel)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    loader._logger = MagicMock()
    return loader


def test_the_text_encoder_refuses_undeclared_fp8_after_the_read(tmp_path) -> None:
    """The encoder's dtype check, which the header cell above cannot see: this file declares nothing,
    so the gate lets it through and the refusal has to come from the tensors themselves."""
    path = tmp_path / "qwen3vl_32b_minimax_h3_fp8_scaled.safetensors"
    save_file(
        {
            "model.layers.0.mlp.down_proj.weight": torch.zeros(4, 4).to(torch.float8_e4m3fn),
            "model.layers.0.mlp.down_proj.weight_scale": torch.tensor(1.0),
        },
        str(path),
    )

    with pytest.raises(ValueError, match=r"MiniMax H3 text encoder .* float8 weight"):
        _encoder_loader()._load_text_encoder_from_singlefile(SimpleNamespace(path=str(path)))


def test_a_raw_fp8_file_with_no_scale_is_refused_too(tmp_path) -> None:
    """The dtype check keys on the dtype, not on a scale key, and that is what this pins: a raw
    float8 file carries no side channel at all, and this loader assigns without casting, so it would
    load fp8 parameters that fail at the first matmul. e5m2 rather than e4m3fn, so a check narrowed
    to one float8 flavour would let it through."""
    torch.manual_seed(0)
    sd = _tiny_remote_code_state_dict()
    sd[QUANTIZED_SOURCE_KEY] = sd[QUANTIZED_SOURCE_KEY].to(torch.float8_e5m2)
    path = tmp_path / "minimax_h3_fp8_e5m2.safetensors"
    save_file(sd, str(path))

    with pytest.raises(ValueError, match=r"float8 weight"):
        _load(path)


def test_an_int8_build_that_also_writes_the_header_still_loads(tmp_path) -> None:
    """The other half of reading the header. No published H3 int8 build writes
    `_quantization_metadata` today, so nothing else in the tree would notice a gate that refused one:
    an `int8_tensorwise` entry has to pass, and so does an entry carrying only a per-layer flag,
    which declares no scheme at all."""
    import json

    torch.manual_seed(0)
    sd = _tiny_remote_code_state_dict()
    quantized, scale, _restored = quantize_convrot(sd[QUANTIZED_SOURCE_KEY], group_size=GROUP_SIZE)
    sd[QUANTIZED_SOURCE_KEY] = quantized
    sd["blocks.0.attn.out_proj.weight_scale"] = scale
    sd["blocks.0.attn.out_proj.comfy_quant"] = comfy_quant_marker(
        {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP_SIZE}
    )
    layers = {
        "blocks.0.attn.out_proj": {"format": "int8_tensorwise"},
        "blocks.0.mlp.fc1": {"full_precision_matrix_mult": True},
    }
    path = tmp_path / "minimax_h3_int8_convrot_with_header.safetensors"
    save_file(
        sd, str(path), metadata={"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": layers})}
    )

    model = _load(path)

    assert isinstance(_quantized_module(model), Int8ConvrotLinear)


def test_a_per_tensor_marker_declaring_another_format_is_refused(tmp_path) -> None:
    """The header gate's other half. Every refusal cell above goes through `_quantization_metadata`
    or the float8 dtype check; nothing pinned the per-tensor route the gate was originally written
    for, so a change loosening it -- it already skips header entries without a `format` -- would
    reopen it unnoticed."""
    path, _ = _write_checkpoint(tmp_path, marker={"format": "float8_e4m3fn"})

    with pytest.raises(ValueError, match=r"float8_e4m3fn.*MiniMax H3 checkpoint"):
        _load(path)


def test_a_marker_the_converter_cannot_parse_is_refused_rather_than_defaulted(tmp_path) -> None:
    """The two marker readers decode the same blob by different routes, and can disagree.

    `read_comfy_quant_markers` decodes raw file bytes; `parse_comfy_quant_marker` goes through
    `tensor.numpy()`, which returns `{}` for a dtype numpy has no equivalent for -- bfloat16 and the
    float8s, while int8, float16 and float32 decode the same bytes fine. A marker stored as one of
    those satisfies the header gate and reaches the swap empty -- and an empty marker is not refused
    there, it is *defaulted*: `convrot` off and a 256-wide group. A
    64-wide repack derotated with a 256-wide Hadamard loads and renders noise, measured at
    correlation 0.14 to the true weight.

    ComfyUI writes uint8, so this is hardening rather than a live defect -- which is exactly why it
    needs a cell: the check reads as redundant with the header gate, and was briefly deleted as such.
    """
    torch.manual_seed(0)
    sd = _tiny_remote_code_state_dict()
    quantized, scale, _restored = quantize_convrot(sd[QUANTIZED_SOURCE_KEY], group_size=GROUP_SIZE)
    sd[QUANTIZED_SOURCE_KEY] = quantized
    sd["blocks.0.attn.out_proj.weight_scale"] = scale
    marker = comfy_quant_marker({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP_SIZE}, pad=16)
    if marker.numel() % 2:  # a byte view needs an even length to reinterpret as a 2-byte dtype
        marker = torch.cat([marker, torch.zeros(1, dtype=torch.uint8)])
    # Same bytes, a dtype `numpy()` refuses.
    sd["blocks.0.attn.out_proj.comfy_quant"] = marker.view(torch.bfloat16)
    path = tmp_path / "minimax_h3_int8_convrot_bf16_marker.safetensors"
    save_file(sd, str(path))

    with pytest.raises(ValueError, match="Unsupported comfy_quant format"):
        _load(path)
