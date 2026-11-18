"""`QwenImageCheckpointModel` keeps Comfy's `int8_tensorwise` builds int8, rotated or not.

An int8 layer ships a `.weight_scale` exactly like a scaled fp8 layer, and the fp8 extraction pops every one of them,
so the int8 layers have to be claimed first. What these tests pin is the decode (each layer reproduces the weight its
codes and scale stand for, rotated and not), the single reservation before anything widens, and the refusals: a weight
no marker explains, a file that also holds scaled fp8, and a layer the model needs dense.

The geometry is the released module names at toy width, with a rotation group of 64 instead of 256.
"""

import json
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from invokeai.backend.model_manager.configs.main import Main_Checkpoint_QwenImage_Config
from invokeai.backend.model_manager.load import load_default
from invokeai.backend.model_manager.load.model_loaders import qwen_image
from invokeai.backend.model_manager.load.model_loaders.qwen_image import QwenImageCheckpointModel
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear
from invokeai.backend.quantization.nvfp4 import NVFP4Linear
from tests.fixtures.loader_seams import Seam, SeamRun, prepare
from tests.fixtures.quantized_payloads import (
    INT8_LEVELS,
    comfy_quant_marker,
    nvfp4_signed_tensors,
    quantize_convrot,
    quantize_scaled_fp8,
)

COMPUTE_DTYPE = torch.bfloat16
GROUP = 64
ROTATED = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}
UNROTATED = {"format": "int8_tensorwise"}

#: `<path>: (out, in)`. Every input width is a whole number of rotation groups.
LAYERS = {
    "transformer_blocks.0.attn.to_q": (128, 64),
    "transformer_blocks.0.img_mlp.net.0.proj": (256, 64),
    "transformer_blocks.0.txt_mlp.net.2": (128, 64),
}


class _TinyQwenImageTransformer(torch.nn.Module):
    """The diffusers module names Comfy's Qwen-Image build uses, at toy width."""

    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.to_q = torch.nn.Linear(64, 128)
        block.img_mlp = torch.nn.Module()
        block.img_mlp.net = torch.nn.ModuleList([torch.nn.Module()])
        block.img_mlp.net[0].proj = torch.nn.Linear(64, 256)
        block.txt_mlp = torch.nn.Module()
        block.txt_mlp.net = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity(), torch.nn.Linear(64, 128)])
        self.transformer_blocks = torch.nn.ModuleList([block])
        self.img_in = torch.nn.Linear(64, 128)
        self.norm_out = torch.nn.Module()
        self.norm_out.linear = torch.nn.Linear(64, 128)


def _unrotated(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """As the unrotated builds store it: one 0-dim scale for the whole tensor. Returns codes, scale, round trip."""
    scale = (weight.abs().max() / INT8_LEVELS).to(torch.float32)
    codes = torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8)
    return codes, scale, codes.float() * scale


def _state_dict(rotated: bool = True, marked: bool = True) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A checkpoint with every `LAYERS` entry int8, and the weight each must decode to."""
    torch.manual_seed(0)
    state_dict: dict[str, torch.Tensor] = {}
    expected: dict[str, torch.Tensor] = {}
    for path, shape in LAYERS.items():
        weight = torch.randn(shape)
        if rotated:
            payload = quantize_convrot(weight, group_size=GROUP)
            codes, scale, expected[path] = payload.codes, payload.scale, payload.dequantized
        else:
            codes, scale, expected[path] = _unrotated(weight)
        state_dict[f"{path}.weight"] = codes
        state_dict[f"{path}.weight_scale"] = scale
        state_dict[f"{path}.bias"] = torch.randn(shape[0])
        if not rotated:
            # The unrotated builds ship an activation scale beside each layer, which has nothing to attach to.
            state_dict[f"{path}.input_scale"] = torch.tensor(1.0)
        if marked:
            state_dict[f"{path}.comfy_quant"] = comfy_quant_marker(ROTATED if rotated else UNROTATED)
    state_dict["img_in.weight"] = torch.randn(128, 64)
    state_dict["img_in.bias"] = torch.randn(128)
    state_dict["norm_out.linear.weight"] = torch.randn(128, 64)
    state_dict["norm_out.linear.bias"] = torch.randn(128)
    # Edit 2511 carries this: dotless, no elements, and no module to land in.
    state_dict["__index_timestep_zero__"] = torch.zeros(0)
    return state_dict, expected


SEAM = Seam(
    loader=QwenImageCheckpointModel,
    module=qwen_image,
    entry="_load_from_singlefile",
    load_file_host=safetensors.torch,
    compute_dtype=COMPUTE_DTYPE,
    patches_device=True,
    # `_torch_dtype` is read nowhere in `qwen_image`; the storage cast is driven through `_load_model` below.
    sets_torch_dtype=False,
    casts_fp8_storage=False,
)


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    state_dict: dict,
    *,
    header: dict | None = None,
    fp8_storage: bool = False,
) -> tuple[SeamRun, Main_Checkpoint_QwenImage_Config]:
    import diffusers

    checkpoint = tmp_path / "qwen_image_edit_2511_int8_convrot.safetensors"
    checkpoint.touch()
    config = Main_Checkpoint_QwenImage_Config.model_construct(
        path=str(checkpoint), name="qwen_image_int8", default_settings=SimpleNamespace(fp8_storage=fp8_storage)
    )
    metadata = None if header is None else {"_quantization_metadata": json.dumps({"layers": header})}
    run = prepare(
        SEAM,
        monkeypatch,
        state_dict=state_dict,
        metadata=metadata,
        geometry=lambda patch: patch.setattr(
            diffusers, "QwenImageTransformer2DModel", _TinyQwenImageTransformer, raising=False
        ),
    )
    return run, config


def _load(monkeypatch, tmp_path, state_dict, **kwargs):
    run, config = _run(monkeypatch, tmp_path, state_dict, **kwargs)
    return run.load(config), run


def _assert_decodes(model: torch.nn.Module, expected: dict[str, torch.Tensor], source: dict) -> None:
    """Each layer reproduces `expected`; its bias is the source tensor's, rounded to the compute dtype by the load."""
    for path, weight in expected.items():
        layer = model.get_submodule(path)
        assert isinstance(layer, Int8ConvrotLinear), path
        assert layer.weight.dtype is torch.int8, path
        assert layer.weight_scale.dtype is torch.float32, path
        # A float32 activation, not the compute dtype: torch's CPU bf16 GEMM faults on part of the windows runner fleet.
        x = torch.randn(3, layer.in_features)
        bias = source[f"{path}.bias"].to(COMPUTE_DTYPE).float()
        torch.testing.assert_close(layer(x), torch.nn.functional.linear(x, weight, bias), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("rotated", [True, False], ids=["rotated_column_scale", "unrotated_scalar_scale"])
def test_the_released_builds_decode_to_the_weights_their_codes_stand_for(monkeypatch, tmp_path, rotated) -> None:
    """The scale is `[out, 1]` in a rotated build and 0-dim in an unrotated one, and the rotated one also has to be
    un-rotated. The expected weights come from the payload builders, not from the loader's own decode."""
    state_dict, expected = _state_dict(rotated=rotated)

    model, _ = _load(monkeypatch, tmp_path, state_dict)

    _assert_decodes(model, expected, state_dict)
    assert all(model.get_submodule(path).convrot is rotated for path in expected)
    assert not isinstance(model.img_in, Int8ConvrotLinear)
    assert model.img_in.weight.dtype is COMPUTE_DTYPE


@pytest.mark.parametrize("rotated", [True, False], ids=["rotated", "unrotated"])
def test_a_build_that_names_its_layers_only_in_the_header_loads(monkeypatch, tmp_path, rotated) -> None:
    """The header has to say whether the codes were rotated; an entry without `convrot` is refused as ambiguous."""
    state_dict, expected = _state_dict(rotated=rotated, marked=False)
    header = {path: dict(ROTATED if rotated else {**UNROTATED, "convrot": False}) for path in LAYERS}

    model, _ = _load(monkeypatch, tmp_path, state_dict, header=header)

    _assert_decodes(model, expected, state_dict)
    assert all(model.get_submodule(path).convrot is rotated for path in expected)


def test_a_header_entry_that_does_not_say_whether_the_weight_was_rotated_is_refused(monkeypatch, tmp_path) -> None:
    state_dict, _ = _state_dict(marked=False)
    run, config = _run(monkeypatch, tmp_path, state_dict, header={path: dict(UNROTATED) for path in LAYERS})

    with pytest.raises(ValueError, match="does not say whether the weight was rotated"):
        run.load(config)

    assert run.reserved == []


@pytest.mark.parametrize("rotated", [True, False], ids=["rotated", "unrotated"])
def test_the_reservation_is_one_call_made_before_the_split_and_the_cast_widen_anything(
    monkeypatch, tmp_path, rotated
) -> None:
    from invokeai.backend.quantization import int8_convrot

    state_dict, _ = _state_dict(rotated=rotated)
    run, config = _run(monkeypatch, tmp_path, state_dict)
    widening_steps: list[tuple[str, int]] = []
    for name in ("split_int8_convrot_layers", "cast_unquantized"):

        def step(*args, _name=name, _original=getattr(int8_convrot, name), **kwargs):
            widening_steps.append((_name, len(run.reserved)))
            return _original(*args, **kwargs)

        monkeypatch.setattr(int8_convrot, name, step)

    run.load(config)

    # Counted by hand: the int8 weights as stored (one byte); the fp32 scales, `[out, 1]` or one scalar per layer; and
    # the dense remainder at the compute width (the biases, `img_in` and `norm_out.linear`). The markers, the
    # activation scales and the zero-element key are dropped and cost nothing.
    int8 = sum(out * inp for out, inp in LAYERS.values())
    scales = (sum(out for out, _ in LAYERS.values()) if rotated else len(LAYERS)) * 4
    biases = sum(out for out, _ in LAYERS.values())
    dense = biases + (128 * 64 + 128) + (128 * 64 + 128)
    assert run.reserved == [int8 + scales + dense * COMPUTE_DTYPE.itemsize]
    assert widening_steps == [("split_int8_convrot_layers", 1), ("cast_unquantized", 1)]


def test_fp8_storage_leaves_the_int8_layers_int8(monkeypatch, tmp_path) -> None:
    """The model-level storage cast walks the dense remainder; the int8 buffers are not layers it casts."""
    state_dict, expected = _state_dict()
    run, config = _run(monkeypatch, tmp_path, state_dict, fp8_storage=True)
    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", lambda *_args: True)
    run.loader._torch_dtype = COMPUTE_DTYPE

    model = run.loader._load_model(config, SubModelType.Transformer)

    for path in expected:
        layer = model.get_submodule(path)
        assert isinstance(layer, Int8ConvrotLinear), path
        assert layer.weight.dtype is torch.int8, path
        assert layer.weight_scale.dtype is torch.float32, path
    assert model.img_in.weight.dtype is torch.float8_e4m3fn


def test_a_marked_layer_the_model_needs_dense_is_widened_with_its_scale_applied(monkeypatch, tmp_path) -> None:
    """`norm_out.linear` matches the model's own "norm" skip pattern. A repack that quantizes it must still come
    back dense and correct, rather than as codes cast without their scale."""
    state_dict, expected = _state_dict()
    payload = quantize_convrot(torch.randn(128, 64), group_size=GROUP)
    state_dict["norm_out.linear.weight"] = payload.codes
    state_dict["norm_out.linear.weight_scale"] = payload.scale
    state_dict["norm_out.linear.comfy_quant"] = comfy_quant_marker(ROTATED)

    model, _ = _load(monkeypatch, tmp_path, state_dict)

    widened = model.norm_out.linear
    assert not isinstance(widened, Int8ConvrotLinear)
    assert widened.weight.dtype is COMPUTE_DTYPE
    torch.testing.assert_close(widened.weight.float(), payload.dequantized, rtol=0.02, atol=0.02)
    _assert_decodes(model, expected, state_dict)


def test_an_int8_weight_without_a_marker_is_refused_before_the_cache_is_evicted(monkeypatch, tmp_path) -> None:
    state_dict, _ = _state_dict(marked=False)
    run, config = _run(monkeypatch, tmp_path, state_dict)

    with pytest.raises(ValueError, match="no readable int8_tensorwise marker"):
        run.load(config)

    assert run.reserved == []


def test_a_marker_without_a_format_is_refused(monkeypatch, tmp_path) -> None:
    """Files marked only `{"convrot": true}` say nothing about how their codes are scaled."""
    state_dict, _ = _state_dict()
    for path in LAYERS:
        state_dict[f"{path}.comfy_quant"] = comfy_quant_marker({"convrot": True})
    run, config = _run(monkeypatch, tmp_path, state_dict)

    with pytest.raises(ValueError, match=r"int8 weight\(s\) with no readable int8_tensorwise marker"):
        run.load(config)

    assert run.reserved == []


def test_a_file_mixing_int8_with_scaled_fp8_is_refused(monkeypatch, tmp_path) -> None:
    """Inside the int8 branch the fp8 pipeline is skipped, so an fp8 layer would be cast without its scale."""
    state_dict, _ = _state_dict()
    payload = quantize_scaled_fp8(torch.randn(128, 64))
    state_dict["img_in.weight"] = payload.codes
    state_dict["img_in.weight_scale"] = payload.scale

    with pytest.raises(ValueError, match="float8 weight scale"):
        _load(monkeypatch, tmp_path, state_dict)


def test_a_layer_whose_tensors_are_missing_is_refused(monkeypatch, tmp_path) -> None:
    """The load tolerates missing keys; a dense layer lost from an int8 file must not survive as a meta tensor."""
    state_dict, _ = _state_dict()
    del state_dict["img_in.weight"]

    with pytest.raises(RuntimeError, match="meta"):
        _load(monkeypatch, tmp_path, state_dict)


def test_int8_and_nvfp4_layers_in_one_file_both_load_and_one_reservation_covers_both(monkeypatch, tmp_path) -> None:
    state_dict, expected = _state_dict()
    nvfp4_path = "transformer_blocks.0.attn.to_q"
    for suffix in ("weight", "weight_scale"):
        del state_dict[f"{nvfp4_path}.{suffix}"]
    del state_dict[f"{nvfp4_path}.comfy_quant"]
    packed, weight = nvfp4_signed_tensors(nvfp4_path, torch.randint(0, 2, (128, 64), dtype=torch.bool))
    state_dict.update(packed)
    del expected[nvfp4_path]
    run, config = _run(monkeypatch, tmp_path, state_dict, header={nvfp4_path: {"format": "nvfp4"}})

    model = run.load(config)

    _assert_decodes(model, expected, state_dict)
    layer = model.get_submodule(nvfp4_path)
    assert isinstance(layer, NVFP4Linear)
    x = torch.randn(3, 64)
    torch.testing.assert_close(
        layer(x), torch.nn.functional.linear(x, weight, state_dict[f"{nvfp4_path}.bias"].to(COMPUTE_DTYPE).float())
    )
    # The packed layer as stored (codes, block-scale grid, global scale), beside what the int8 case counts without it.
    packed_bytes = 128 * 64 // 2 + 128 * (64 // 16) + 4
    int8 = sum(out * inp for path, (out, inp) in LAYERS.items() if path != nvfp4_path)
    scales = sum(out for path, (out, _) in LAYERS.items() if path != nvfp4_path) * 4
    dense = sum(out for out, _ in LAYERS.values()) + 2 * (128 * 64 + 128)
    assert run.reserved == [packed_bytes + int8 + scales + dense * COMPUTE_DTYPE.itemsize]
