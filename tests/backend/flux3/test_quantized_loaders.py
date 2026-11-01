"""The FLUX.3 single-file loader over the quantized releases, driven through `Flux3CheckpointModel`.

Each build is a tiny native checkpoint written the way its release writes it (`quantize_flux3_native`), and what
is asserted is the forward: a loaded model must predict what a dense model of the values the build means predicts.
That covers the rename of the quantized layers and of their side channels -- the int8 build's fused `qkv` above all,
whose codes and per-row scales split three ways -- the output adaLN's row swap carried onto its quantized forms, and
BFL's ModelOpt nvfp4 layout re-laid out into ComfyUI's. The forwards run in float32, since a CPU bf16 GEMM faults on
part of the hosted Windows fleet.
"""

import json

import pytest
import safetensors.torch
import torch

from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Flux3_Config
from invokeai.backend.model_manager.load import load_default
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders import flux3 as flux3_loaders
from invokeai.backend.model_manager.load.model_loaders.flux3 import Flux3CheckpointModel
from invokeai.backend.model_manager.taxonomy import Flux3VariantType, SubModelType
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear
from invokeai.backend.quantization.nvfp4 import NVFP4Linear
from tests.backend.flux3.flux3_fixture import (
    QUANTIZED_SCHEMES,
    QuantizedFlux3,
    flux3_model_from_native,
    forward,
    quantize_flux3_native,
    tiny_flux3_model,
    to_native,
)
from tests.fixtures.loader_seams import Seam, SeamRun, prepare
from tests.fixtures.quantized_payloads import nvfp4_codes, nvfp4_tensors

SEAM = Seam(
    loader=Flux3CheckpointModel,
    module=flux3_loaders,
    entry="_load_model",
    load_file_host=safetensors.torch,
    patches_device=True,
)

HIDDEN = 256
"""The tiny model's width: two heads of 128."""

# The diffusers modules each release keeps quantized, spelled out rather than derived from the converter.
_BLOCKS = {
    "int8_convrot": (
        "mode_blocks.image.0",
        "mode_blocks.image_cond.0",
        "txt_mode_blocks.0",
        "single_transformer_blocks.0",
        "single_transformer_blocks.1",
    ),
    "nvfp4": ("mode_blocks.image_cond.0", "single_transformer_blocks.0"),
    "fp8r": (
        "mode_blocks.image.0",
        "mode_blocks.image_cond.0",
        "single_transformer_blocks.0",
        "single_transformer_blocks.1",
    ),
}
_LINEARS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.linear_in", "ff.linear_out")
_MODULATIONS = tuple(
    f"{stage}.{mode}.linear"
    for stage in ("early_stream_modulations", "single_stream_modulations")
    for mode in ("image", "image_cond", "txt")
)


def _build(scheme: str, *, flow_map: bool = True, **options) -> QuantizedFlux3:
    return quantize_flux3_native(to_native(tiny_flux3_model(flow_map=flow_map)), scheme, **options)


def _config(flow_map: bool = True) -> Main_Checkpoint_Flux3_Config:
    variant = Flux3VariantType.Klein if flow_map else Flux3VariantType.KleinBase
    return Main_Checkpoint_Flux3_Config.model_construct(path="flux3.safetensors", name="flux3", variant=variant)


def _load(
    monkeypatch: pytest.MonkeyPatch, build: QuantizedFlux3, observe: tuple[str, ...] = (), *, flow_map: bool = True
) -> tuple[SeamRun, torch.nn.Module]:
    run = prepare(SEAM, monkeypatch, state_dict=build.state_dict, metadata=build.metadata, observe=observe)
    return run, run.load(_config(flow_map), SubModelType.Transformer)


def _refused(monkeypatch: pytest.MonkeyPatch, build: QuantizedFlux3, match: str) -> SeamRun:
    run = prepare(SEAM, monkeypatch, state_dict=build.state_dict, metadata=build.metadata)
    with pytest.raises(ValueError, match=match):
        run.load(_config(), SubModelType.Transformer)
    return run


def _assert_predicts_what_it_means(
    model: torch.nn.Module, build: QuantizedFlux3, *, exact: bool, flow_map: bool = True
) -> None:
    reference = forward(flux3_model_from_native(build.meant, flow_map=flow_map), flow_map=flow_map)
    assert reference.abs().max() > 0, "a forward that is zero everywhere would compare equal to anything"
    loaded = forward(model, flow_map=flow_map)
    if exact:
        # nvfp4 and fp8 decode to exactly the meant weights, so the forward is the reference's to the bit.
        assert torch.equal(loaded, reference)
    else:
        # int8 derotates a fused qkv per projection, the reference all three at once: BLAS may sum differently.
        torch.testing.assert_close(loaded, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("scheme", QUANTIZED_SCHEMES)
def test_a_quantized_release_loads_as_the_model_it_means(monkeypatch: pytest.MonkeyPatch, scheme: str) -> None:
    build = _build(scheme)

    run, model = _load(monkeypatch, build)

    _assert_predicts_what_it_means(model, build, exact=scheme != "int8_convrot")
    assert len(run.reserved) == 1, "one reservation, before anything widens"
    if scheme == "fp8r":
        # Neither the fp8 matmul nor FP8 Storage asked for them here, so the scales are folded in.
        assert not any(p.dtype is torch.float8_e4m3fn for p in model.parameters())
    else:
        packed = Int8ConvrotLinear if scheme == "int8_convrot" else NVFP4Linear
        blocks = _BLOCKS["int8_convrot" if scheme == "int8_convrot" else "nvfp4"]
        for name in (f"{block}.{linear}" for block in blocks for linear in _LINEARS):
            assert isinstance(model.get_submodule(name), packed), name
    if scheme.startswith("nvfp4"):
        # The blocks the release leaves in bf16 stay plain linears.
        for block in ("mode_blocks.image.0", "txt_mode_blocks.0", "single_transformer_blocks.1"):
            assert type(model.get_submodule(f"{block}.attn.to_q")) is torch.nn.Linear
    # The FP8 Storage pass runs only where no layer is kept quantized by the build itself.
    assert run.casting_calls == (1 if scheme == "fp8r" else 0)


@pytest.mark.parametrize("scheme", QUANTIZED_SCHEMES)
def test_a_quantized_klein_base_release_loads_as_klein_base(monkeypatch: pytest.MonkeyPatch, scheme: str) -> None:
    """Klein Base's builds are Klein's without the flow-map embedder; nothing in the quantized path may assume it."""
    build = _build(scheme, flow_map=False)

    _, model = _load(monkeypatch, build, flow_map=False)

    assert model.config.flow_map is False
    _assert_predicts_what_it_means(model, build, exact=scheme != "int8_convrot", flow_map=False)


def test_one_reservation_holds_the_int8_codes_at_one_byte(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counted by hand: codes at one byte, their float32 scales as stored, every dense tensor at the compute dtype
    (float32 here). The markers and the attention-kernel hints are dropped before the load and cost nothing."""
    build = _build("int8_convrot")
    codes = sum(t.numel() for t in build.state_dict.values() if t.dtype is torch.int8)
    scales = sum(t.numel() for key, t in build.state_dict.items() if key.endswith(".weight_scale"))
    dense = sum(
        t.numel() for key, t in build.state_dict.items() if t.is_floating_point() and not key.endswith(".weight_scale")
    )

    run, _ = _load(monkeypatch, build)

    assert run.reserved == [codes + 4 * scales + 4 * dense]


@pytest.mark.parametrize("scheme", ["nvfp4_comfy", "nvfp4_bfl"])
def test_one_reservation_holds_the_nvfp4_layers_as_stored(monkeypatch: pytest.MonkeyPatch, scheme: str) -> None:
    """Counted by hand: each packed layer at its stored size (codes, grid, float32 global scale), every dense tensor at
    the compute dtype (float32 here). The activation scales are dropped and cost nothing. BFL's layers are re-laid out
    after the reservation, into copies of the same size."""
    sd = _build(scheme).state_dict
    layers = [key[: -len(".weight_scale_2")] for key in sd if key.endswith(".weight_scale_2")]
    side = {
        f"{layer}{part}" for layer in layers for part in (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")
    }
    packed = sum(sd[f"{layer}.weight"].numel() + sd[f"{layer}.weight_scale"].numel() + 4 for layer in layers)
    dense = sum(t.numel() for key, t in sd.items() if key not in side)

    run, _ = _load(monkeypatch, _build(scheme))

    assert run.reserved == [packed + 4 * dense]


def test_one_reservation_holds_a_folded_fp8r_build_at_the_compute_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counted by hand: with no consumer for fp8, every weight is folded to the compute dtype (float32 here), and the
    per-row scales are held beside the weights until the fold has used them."""
    build = _build("fp8r")

    run, _ = _load(monkeypatch, build)

    assert run.reserved == [4 * sum(t.numel() for t in build.state_dict.values())]


def test_one_reservation_holds_kept_fp8r_weights_at_one_byte(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counted by hand: kept for the fp8 matmul, every quantized weight stays one byte per element, its per-row scale
    float32, and the rest is at the compute dtype (float32 here)."""
    monkeypatch.setattr(load_default, "should_keep_fp8_weights", lambda _device: True)
    sd = _build("fp8r").state_dict
    fp8 = sum(t.numel() for t in sd.values() if t.dtype is torch.float8_e4m3fn)
    scales = sum(t.numel() for key, t in sd.items() if key.endswith(".weight_scale"))
    dense = sum(t.numel() for key, t in sd.items() if t.dtype is torch.float32 and not key.endswith(".weight_scale"))

    run, _ = _load(monkeypatch, _build("fp8r"))

    assert run.reserved == [fp8 + 4 * scales + 4 * dense]


@pytest.mark.parametrize("scheme", QUANTIZED_SCHEMES)
def test_a_quantized_output_adaln_is_decoded_with_its_halves_swapped(
    monkeypatch: pytest.MonkeyPatch, scheme: str
) -> None:
    """No release quantizes it, and a build that did would load wrong without a word: BFL stores [shift, scale]
    rows, the per-row scales of int8 and fp8r beside them, and nvfp4's grid in 128-row tiles. `norm_out.linear` sits
    under the model's `norm` pattern, so it is decoded rather than kept quantized."""
    build = _build(scheme, adaln=True)

    _, model = _load(monkeypatch, build)

    adaln = model.norm_out.linear
    assert type(adaln) is torch.nn.Linear and adaln.weight.dtype is torch.float32
    _assert_predicts_what_it_means(model, build, exact=scheme != "int8_convrot")


def test_a_fused_nvfp4_qkv_splits_on_whole_scale_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    build = _build("nvfp4_comfy", fuse_qkv=True)

    _, model = _load(monkeypatch, build)

    for block in _BLOCKS["nvfp4"]:
        for projection in ("to_q", "to_k", "to_v"):
            module = model.get_submodule(f"{block}.attn.{projection}")
            assert isinstance(module, NVFP4Linear) and module.out_features == HIDDEN
    _assert_predicts_what_it_means(model, build, exact=True)


def test_a_comfy_nvfp4_file_is_never_re_laid_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ModelOpt layout is keyed on BFL's metadata *and* on the absence of ComfyUI's names. A ComfyUI file that also
    says `quantization: nvfp4` is read as ComfyUI's; re-laid out, every nibble pair and scale tile would be swapped."""
    build = _build("nvfp4_comfy")
    build.metadata["quantization"] = "nvfp4"

    _, model = _load(monkeypatch, build)

    _assert_predicts_what_it_means(model, build, exact=True)


def test_the_ordered_steps_run_after_the_one_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fold and the cast widen weights, BFL's nvfp4 re-layout copies every packed layer, and the packed layers go
    in last; all of it inside the room."""
    run, _ = _load(monkeypatch, _build("fp8r"), observe=("dequantize_fp8_scaled", "cast_state_dict"))
    assert run.order == [("dequantize_fp8_scaled", 1), ("cast_state_dict", 1)]

    run, _ = _load(monkeypatch, _build("nvfp4_comfy"), observe=("cast_state_dict", "install_nvfp4_layers"))
    assert run.order == [("cast_state_dict", 1), ("install_nvfp4_layers", 1)]

    observe = ("relayout_modelopt_nvfp4_payloads", "install_nvfp4_layers")
    run, _ = _load(monkeypatch, _build("nvfp4_bfl"), observe=observe)
    assert run.order == [("relayout_modelopt_nvfp4_payloads", 1), ("install_nvfp4_layers", 1)]


def test_kept_fp8r_weights_carry_their_per_row_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the fp8 matmul on, the weights stay e4m3 and the `[N]` scale goes on as a buffer, which `scaled_mm`
    applies to the output columns. The text blocks BFL leaves in bf16 and the output adaLN, under the `norm`
    pattern, are not fp8."""
    monkeypatch.setattr(load_default, "should_keep_fp8_weights", lambda _device: True)
    build = _build("fp8r", adaln=True)

    _, model = _load(monkeypatch, build)

    for name in (*(f"{block}.{linear}" for block in _BLOCKS["fp8r"] for linear in _LINEARS), *_MODULATIONS):
        module = model.get_submodule(name)
        assert module.weight.dtype is torch.float8_e4m3fn, name
        assert module.weight_scale.shape == (module.out_features,), name
    assert model.get_submodule("txt_mode_blocks.0.attn.to_q").weight.dtype is torch.float32
    assert model.norm_out.linear.weight.dtype is torch.float32
    apply_custom_layers_to_model(model)
    _assert_predicts_what_it_means(model, build, exact=True)


def test_an_fp8_scale_the_conversion_does_not_carry_is_refused_before_room_is_reserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`weight_scale_inv` is no spelling the key conversion moves: renamed, the weight would leave it behind under the
    old name and load unscaled. Asked on the native names, before anything is renamed."""
    build = _build("fp8r")
    sd = dict(build.state_dict)
    sd["single_blocks.0.q_proj.weight_scale_inv"] = sd.pop("single_blocks.0.q_proj.weight_scale")

    run = _refused(monkeypatch, build._replace(state_dict=sd), "a spelling InvokeAI does not read")

    assert run.reserved == []


def test_int8_weights_without_a_marker_are_refused_before_room_is_reserved(monkeypatch: pytest.MonkeyPatch) -> None:
    build = _build("int8_convrot")
    unmarked = {key: value for key, value in build.state_dict.items() if not key.endswith(".comfy_quant")}

    run = _refused(monkeypatch, build._replace(state_dict=unmarked), "no `comfy_quant` marker")

    assert run.reserved == []


def test_nvfp4_layers_named_by_no_producer_are_refused_before_room_is_reserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """ComfyUI's layout with its header gone: another producer's file under the same key names, whose bytes
    nothing here can vouch for."""
    build = _build("nvfp4_comfy")
    metadata = {key: value for key, value in build.metadata.items() if key != "_quantization_metadata"}

    run = _refused(monkeypatch, build._replace(metadata=metadata), "no ComfyUI `comfy_quant` marker")

    assert run.reserved == []


def test_a_packed_input_embedder_is_refused_before_room_is_reserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model's input widths are read off the embedders' weights, which nvfp4 stores at half width. Named in the
    header like the rest, so what refuses it is the width, not the naming."""
    build = _build("nvfp4_comfy")
    positive = torch.randint(0, 2, (HIDDEN, HIDDEN), dtype=torch.bool)
    del build.state_dict["txt_in.weight"]
    build.state_dict.update(nvfp4_tensors("txt_in", nvfp4_codes(positive), block_scale=1.0, global_scale=1.0))
    header = json.loads(build.metadata["_quantization_metadata"])
    header["layers"]["txt_in"] = {"format": "nvfp4"}
    build.metadata["_quantization_metadata"] = json.dumps(header)

    run = _refused(monkeypatch, build, "stores 'txt_in.weight' packed")

    assert run.reserved == []
