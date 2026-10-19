"""Loader-level tests for the `int8_tensorwise` FLUX.2 path.

These drive `Flux2CheckpointModel._load_from_singlefile` and assert the end state of the module:
which layers keep their codes, what the reconstructed weight is, and what the cache was asked to
free. The geometry is the released Klein 9B layout scaled down by 16 -- same module tree and the
same two fusions, 4 tensors per block instead of 4096-wide ones.

The branch this covers is one decision: an int8 layer ships a `.weight_scale` exactly like a scaled
fp8 layer, so a loader that probes for scales first takes the whole checkpoint down the fp8 path.
There the codes are scaled but never dequantized to a dtype torch computes in.
"""

import json

import pytest
import torch

from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Flux2_Config
from invokeai.backend.model_manager.load.model_loaders import flux
from invokeai.backend.model_manager.load.model_loaders.flux import Flux2CheckpointModel
from invokeai.backend.model_manager.taxonomy import Flux2VariantType, SubModelType
from invokeai.backend.quantization.fp8_scaled import iter_weight_scale_pairs
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear
from tests.fixtures.loader_seams import Seam, prepare
from tests.fixtures.quantized_payloads import comfy_quant_marker, quantize_scaled_fp8

MARKER = {"format": "int8_tensorwise"}

H = 256  # hidden size: two 128-wide attention heads, the smallest the architecture allows
JOINT = 768  # context_embedder input; only its presence and rank matter here
IN_CHANNELS = 128  # FLUX.2's 32 latent channels x 4 packing -- what identification keys on

# `<bfl path>: (out, in)`, at the released ratios. The two fused layers are what make FLUX.2 the
# hard case: one int8 tensor and one scalar scale become three (or two) diffusers projections.
QUANTIZED: dict[str, tuple[int, int]] = {
    "double_blocks.0.img_attn.qkv": (3 * H, H),
    "double_blocks.0.img_attn.proj": (H, H),
    "double_blocks.0.img_mlp.0": (6 * H, H),
    "double_blocks.0.img_mlp.2": (H, 3 * H),
    "double_blocks.0.txt_attn.qkv": (3 * H, H),
    "double_blocks.0.txt_attn.proj": (H, H),
    "double_blocks.0.txt_mlp.0": (6 * H, H),
    "double_blocks.0.txt_mlp.2": (H, 3 * H),
    "single_blocks.0.linear1": (9 * H, H),
    "single_blocks.0.linear2": (H, 4 * H),
}

DENSE: dict[str, tuple[int, ...]] = {
    "img_in.weight": (H, IN_CHANNELS),
    "txt_in.weight": (H, JOINT),
    "time_in.in_layer.weight": (H, 256),
    "time_in.out_layer.weight": (H, H),
    "double_stream_modulation_img.lin.weight": (6 * H, H),
    "double_stream_modulation_txt.lin.weight": (6 * H, H),
    "single_stream_modulation.lin.weight": (3 * H, H),
    "final_layer.adaLN_modulation.1.weight": (2 * H, H),
    "final_layer.linear.weight": (IN_CHANNELS, H),
    "double_blocks.0.img_attn.norm.key_norm.scale": (128,),
    "double_blocks.0.img_attn.norm.query_norm.scale": (128,),
    "double_blocks.0.txt_attn.norm.key_norm.scale": (128,),
    "double_blocks.0.txt_attn.norm.query_norm.scale": (128,),
    "single_blocks.0.norm.key_norm.scale": (128,),
    "single_blocks.0.norm.query_norm.scale": (128,),
}


def _quantize_tensorwise(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """As the repack stores it: one scalar scale for the whole tensor, no rotation."""
    scale = weight.abs().max() / 127.0
    return torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8), scale.to(torch.float32)


# Stored fp32 rather than bf16, because a repack that mixes precisions in its dense remainder is
# what makes `cast_unquantized` observable at all: `load_state_dict_ignoring_extras(assign=True)`
# assigns storage dtypes straight through, so without the cast this layer lands as fp32 inside a
# bf16 model -- and the reservation, which charged it two bytes, comes up short by its size.
STORED_FP32 = "img_in.weight"

# W8A8 metadata for a mode this path does not implement. The load is not strict, but the cast and
# the reservation would both pay for it.
ACTIVATION_SCALE = "double_blocks.0.img_attn.proj.input_scale"


def _checkpoint(marker: dict | None = MARKER) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A BFL-format state dict with every block linear stored int8, plus the float originals."""
    torch.manual_seed(0)
    sd: dict[str, torch.Tensor] = {
        key: torch.randn(*shape, dtype=torch.float32 if key == STORED_FP32 else torch.bfloat16)
        for key, shape in DENSE.items()
    }
    sd[ACTIVATION_SCALE] = torch.ones((), dtype=torch.float32)

    originals: dict[str, torch.Tensor] = {}
    for path, (out_features, in_features) in QUANTIZED.items():
        weight = torch.randn(out_features, in_features, dtype=torch.float32)
        originals[path] = weight
        sd[f"{path}.weight"], sd[f"{path}.weight_scale"] = _quantize_tensorwise(weight)
        if marker is not None:
            sd[f"{path}.comfy_quant"] = comfy_quant_marker(marker)
    return sd, originals


def _expected_reservation() -> int:
    """What the load occupies, with the four widths spelled out rather than recomputed.

    A payload that stays int8 is one byte, whatever the compute dtype. Its scalar scale is *copied*
    to each projection the fused layer becomes and stays fp32 there, so that count follows the
    destinations and not the checkpoint's ten layers. Everything else arrives at the compute width,
    including the layer the checkpoint stored as fp32 -- charged what it becomes, not what it was.
    The marker and the activation scale are dropped before the load and charged nothing.

    Charging a kept int8 payload two bytes asks the cache to evict roughly twice what an 8.8 GiB
    file needs; charging none lets the split's transient land on an unreserved cache.
    """
    int8_bytes = sum(out * inp for out, inp in QUANTIZED.values())
    # How many modules the quantized layers become: one each, except the two fused qkv that become
    # three. Counted here rather than read off the conversion, so that a wrong mapping shifts the
    # measurement without shifting what it is measured against.
    destinations = len(QUANTIZED) + 2 * 2
    scale_bytes = destinations * torch.float32.itemsize
    dense_bytes = sum(torch.Size(shape).numel() for shape in DENSE.values()) * torch.bfloat16.itemsize
    # Klein has no guidance embedder, so the loader fills one with zeros shaped like the timestep
    # embedder's two linears. It is part of what the load occupies and is reserved for.
    guidance_bytes = (H * 256 + H * H) * torch.bfloat16.itemsize
    return int8_bytes + scale_bytes + dense_bytes + guidance_bytes


SEAM = Seam(
    loader=Flux2CheckpointModel,
    module=flux,
    compute_dtype=torch.bfloat16,
    # `_load_from_singlefile` never reaches the FP8 Storage pass, and the tests that do reach it
    # through `_load_model` install their own counter. Nothing here would read this stub, and a
    # stub nothing reads is one more thing to keep true.
    casts_fp8_storage=False,
)


def _driver(monkeypatch, tmp_path, state_dict: dict, header: dict | None = None, observe: tuple[str, ...] = ()):
    checkpoint = tmp_path / "flux-2-klein-9b-int8-convrot.safetensors"
    checkpoint.touch()
    config = Main_Checkpoint_Flux2_Config.model_construct(
        path=str(checkpoint), name="klein-9b-int8", variant=Flux2VariantType.Klein9B
    )

    def geometry(patch):
        # Both spellings of the same decision: the module-level helper, and the loader method that
        # supersedes it once FP8 Storage counts as a consumer too. `raising=False` tolerates the
        # module symbol being gone.
        patch.setattr(flux, "should_keep_fp8_weights", lambda _device: False, raising=False)

    run = prepare(SEAM, monkeypatch, state_dict=state_dict, metadata=header, geometry=geometry, observe=observe)
    # An instance attribute shadows the method, which is what keeps this working either way.
    run.loader._keep_fp8_weights = lambda _config, _submodel=None: False
    return run, config


def _load(monkeypatch, tmp_path, state_dict, header: dict | None = None):
    run, config = _driver(monkeypatch, tmp_path, state_dict, header)
    return run.load(config), run


def test_an_int8_checkpoint_stays_int8_resident(monkeypatch, tmp_path) -> None:
    """The point of the build: the file's size is the resident size.

    `Int8ConvrotLinear` holds the stored codes and dequantizes per forward, so the weights must
    still be int8 after the load -- on a path whose every other branch casts to bf16.
    """
    state_dict, _ = _checkpoint()

    model, _ = _load(monkeypatch, tmp_path, state_dict)

    swapped = [name for name, module in model.named_modules() if isinstance(module, Int8ConvrotLinear)]
    assert swapped, "no layer kept its int8 storage"
    assert all(model.get_submodule(name).weight.dtype is torch.int8 for name in swapped)
    # This repack declares no rotation, and inventing one would derotate a weight that was never
    # rotated -- which loads cleanly and generates noise.
    assert all(model.get_submodule(name).convrot is False for name in swapped)


def test_both_fusions_are_swapped_and_reconstruct_their_source(monkeypatch, tmp_path) -> None:
    """The fused layers are where a marker or a scale goes missing silently.

    Each projection is compared against its slice of the *original float* tensor, so a scale left
    on the fused path -- the failure that has no error attached to it -- fails here.
    """
    state_dict, originals = _checkpoint()

    model, _ = _load(monkeypatch, tmp_path, state_dict)

    for fused, parts in (
        ("double_blocks.0.img_attn.qkv", ("transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_k")),
        ("double_blocks.0.txt_attn.qkv", ("transformer_blocks.0.attn.add_q_proj",)),
    ):
        source = originals[fused]
        for index, path in enumerate(parts):
            layer = model.get_submodule(path)
            assert isinstance(layer, Int8ConvrotLinear), path
            reconstructed = layer._dequantized_weight(torch.device("cpu"), torch.float32)
            expected = source[index * H : (index + 1) * H]
            assert torch.allclose(reconstructed, expected, rtol=0.05, atol=0.05 * source.abs().max())


def test_the_dense_remainder_is_cast_to_the_compute_dtype(monkeypatch, tmp_path) -> None:
    """The layers the repack left alone still have to arrive in the compute dtype.

    `assign=True` hands the checkpoint's storage dtype straight to the module, so a repack whose
    dense remainder is fp32 -- the module docstring names real ones that mix precisions -- would
    otherwise put an fp32 `x_embedder` inside a bf16 model. The fixture stores exactly one layer
    that way, which is what makes this assertion able to fail.
    """
    state_dict, _ = _checkpoint()
    assert state_dict[STORED_FP32].dtype is torch.float32, "fixture no longer exercises the cast"

    model, _ = _load(monkeypatch, tmp_path, state_dict)

    for path in ("x_embedder", "context_embedder", "proj_out"):
        layer = model.get_submodule(path)
        assert not isinstance(layer, Int8ConvrotLinear), path
        assert layer.weight.dtype is torch.bfloat16, path


def test_an_unusable_activation_scale_is_dropped(monkeypatch, tmp_path) -> None:
    """W8A8 sidecars are metadata for a mode this path does not implement.

    Left in place they are cast to the compute dtype and charged against the reservation for
    nothing, and on a strict loader they would fail the load outright.
    """
    with_scale, without_scale = _checkpoint()[0], _checkpoint()[0]
    del without_scale[ACTIVATION_SCALE]

    model, with_run = _load(monkeypatch, tmp_path, with_scale)
    _, without_run = _load(monkeypatch, tmp_path, without_scale)

    assert isinstance(model.get_submodule("transformer_blocks.0.attn.to_out.0"), Int8ConvrotLinear)
    # Compared against the same file without the sidecar rather than against a computed total: what
    # has to hold is that carrying one costs nothing, and a dropped `drop_unconsumed_...` shows up
    # as the two branches disagreeing.
    assert with_run.reserved[-1] == without_run.reserved[-1] == _expected_reservation()


def test_the_reservation_charges_the_int8_payloads_one_byte(monkeypatch, tmp_path) -> None:
    """This path asked the cache for nothing at all before int8 arrived.

    The number matters in both directions: charging bf16's two bytes asks the cache to evict about
    twice what an 8.8 GiB file needs, and charging none lets the split's transient land on an
    unreserved cache.
    """
    state_dict, _ = _checkpoint()

    _, run = _load(monkeypatch, tmp_path, state_dict)

    assert run.reserved[-1] == _expected_reservation()


def test_an_int8_weight_without_a_marker_is_refused(monkeypatch, tmp_path) -> None:
    """An unclaimed int8 weight would be cast to bf16 as raw codes: a model of small integers.

    It is refused rather than loaded, because nothing downstream can tell the difference.
    """
    state_dict, _ = _checkpoint(marker=None)

    with pytest.raises(ValueError, match="int8"):
        _load(monkeypatch, tmp_path, state_dict)


def test_the_fp8_storage_pass_is_not_offered_an_int8_model(monkeypatch, tmp_path) -> None:
    """FP8 Storage is documented as a no-op on an already-quantized checkpoint, and here it has to
    be made one: the cast walks `model.parameters()` and only converts the handful of layers this
    path left dense -- `Int8ConvrotLinear` holds buffers and is not a layer type it casts at all.
    So it would spend the walk for no saving on the 144 layers that matter, and it is skipped.
    """
    state_dict, _ = _checkpoint()
    calls: list[str] = []

    run, config = _driver(monkeypatch, tmp_path, state_dict)
    run.loader._apply_fp8_layerwise_casting = lambda model, *_args: calls.append("cast") or model

    # `_load_model`, not the seam's `_load_from_singlefile`: the FP8 Storage pass sits one level up.
    model = run.loader._load_model(config, SubModelType.Transformer)

    assert calls == []
    assert any(isinstance(module, Int8ConvrotLinear) for module in model.modules())


def test_a_dense_checkpoint_still_reaches_the_fp8_storage_pass(monkeypatch, tmp_path) -> None:
    # The other half: skipping for int8 must not skip for everything else, or the FP8 Storage
    # setting quietly stops working for ordinary FLUX.2 checkpoints.
    state_dict, originals = _checkpoint()
    for path in QUANTIZED:
        del state_dict[f"{path}.comfy_quant"]
        del state_dict[f"{path}.weight_scale"]
        state_dict[f"{path}.weight"] = originals[path].to(torch.bfloat16)
    calls: list[str] = []

    run, config = _driver(monkeypatch, tmp_path, state_dict)
    run.loader._apply_fp8_layerwise_casting = lambda model, *_args: calls.append("cast") or model

    # `_load_model`, not the seam's `_load_from_singlefile`: the FP8 Storage pass sits one level up.
    model = run.loader._load_model(config, SubModelType.Transformer)

    assert calls == ["cast"]
    assert not any(isinstance(module, Int8ConvrotLinear) for module in model.modules())


def test_a_file_mixing_int8_with_scaled_fp8_is_refused(monkeypatch, tmp_path) -> None:
    """Inside the int8 branch the fp8 pipeline is skipped entirely.

    An fp8 layer that came along would be cast to the compute dtype without its scale -- off by
    `1/weight_scale` -- while the non-strict load drops the orphaned scale and says nothing. The
    detection is file-wide, so a merged export is exactly the shape that reaches this.
    """
    state_dict, _ = _checkpoint()
    fp8 = quantize_scaled_fp8(state_dict["final_layer.linear.weight"].float())
    state_dict["final_layer.linear.weight"] = fp8.codes
    state_dict["final_layer.linear.weight_scale"] = fp8.scale

    with pytest.raises(ValueError, match="mixing int8_tensorwise with scaled fp8"):
        _load(monkeypatch, tmp_path, state_dict)


def test_a_repack_that_quantizes_a_skip_patterned_layer_is_dequantized(monkeypatch, tmp_path) -> None:
    """`Flux2Transformer2DModel` declares `('pos_embed', 'norm')`, and the final adaLN projection
    lands on `norm_out.linear` after the rename -- the one quantizable path those patterns match.

    A repack that claims it must still come back dense, and the reservation must charge it the
    width it is about to become rather than the one byte its marker implies.
    """
    state_dict, originals = _checkpoint()
    path = "final_layer.adaLN_modulation.1"
    weight = state_dict[f"{path}.weight"].float()
    originals[path] = weight
    state_dict[f"{path}.weight"], state_dict[f"{path}.weight_scale"] = _quantize_tensorwise(weight)
    state_dict[f"{path}.comfy_quant"] = comfy_quant_marker(MARKER)

    model, _ = _load(monkeypatch, tmp_path, state_dict)

    skipped = model.get_submodule("norm_out.linear")
    assert not isinstance(skipped, Int8ConvrotLinear)
    assert skipped.weight.dtype is torch.bfloat16
    # The block linears it was mixed with still keep their codes.
    assert isinstance(model.get_submodule("transformer_blocks.0.attn.to_q"), Int8ConvrotLinear)


def test_a_checkpoint_that_declares_int8_only_in_its_header_still_loads(monkeypatch, tmp_path) -> None:
    """Per-layer markers are one of the two forms ComfyUI writes, and not the one it writes here.

    Comfy-Org's own fp8 build of Klein 9B carries no `.comfy_quant` tensors at all and names every
    layer in the safetensors header instead. A repack from that tooling would otherwise be refused
    for a marker it never had to write -- and refused by the guard whose message says the file
    would generate noise, which is the wrong story to tell about a perfectly valid checkpoint.

    The header names layers in the BFL scheme, so it needs the same one-to-many rename as the
    scales: a fused `qkv` becomes three diffusers modules.
    """
    state_dict, _ = _checkpoint(marker=None)
    header = {
        "_quantization_metadata": json.dumps(
            {"format_version": "1.0", "layers": {path: {"format": "int8_tensorwise"} for path in QUANTIZED}}
        )
    }

    model, _ = _load(monkeypatch, tmp_path, state_dict, header=header)

    for path in (
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_v",
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj",
    ):
        assert isinstance(model.get_submodule(path), Int8ConvrotLinear), path


def test_a_per_layer_marker_wins_over_the_header(monkeypatch, tmp_path) -> None:
    # Only the per-layer marker can carry `convrot` and the group size, so where both are present
    # it is the one that decides how the weight is decoded.
    state_dict, _ = _checkpoint(marker={**MARKER, "convrot": False})
    header = {
        "_quantization_metadata": json.dumps(
            {"layers": {path: {"format": "int8_tensorwise", "convrot": True} for path in QUANTIZED}}
        )
    }

    model, _ = _load(monkeypatch, tmp_path, state_dict, header=header)

    assert model.get_submodule("transformer_blocks.0.attn.to_q").convrot is False


def _scaled_fp8_checkpoint() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The other build this loader takes: ComfyUI scaled fp8, one scale per tensor."""
    state_dict, originals = _checkpoint(marker=None)
    for path in QUANTIZED:
        fp8 = quantize_scaled_fp8(originals[path])
        state_dict[f"{path}.weight"] = fp8.codes
        state_dict[f"{path}.weight_scale"] = fp8.scale
    return state_dict, originals


def test_the_extraction_runs_before_the_fold_safety_net(monkeypatch, tmp_path) -> None:
    """`_dequantize_fp8_weights` still carries a scale fold, described as a safety net. It cannot
    fire, because `extract_fp8_scaled_layers` runs first and takes every scale key it is shown --
    and that ordering is the whole of the reason. Swap the two calls and the fold goes live over a
    state dict whose scales are already gone, which is a silent no-op on exactly the layers it was
    meant to rescue.

    The ordering is held today by the order the two lines happen to be written in, so it is asserted
    on what the method is actually handed rather than on the lines.
    """
    state_dict, _ = _scaled_fp8_checkpoint()
    assert list(iter_weight_scale_pairs(state_dict)), "the fixture carries no pair to observe"
    handed: list[list[tuple[str, str]]] = []
    original = Flux2CheckpointModel._dequantize_fp8_weights

    def recording(self, sd, keep_fp8=False):
        handed.append(list(iter_weight_scale_pairs(sd)))
        return original(self, sd, keep_fp8=keep_fp8)

    monkeypatch.setattr(Flux2CheckpointModel, "_dequantize_fp8_weights", recording)

    _load(monkeypatch, tmp_path, state_dict)

    assert handed == [[]], f"a weight/scale pair reached the safety net: {handed}"


def test_a_scaled_fp8_checkpoint_still_folds_its_scales(monkeypatch, tmp_path) -> None:
    """The fp8 pipeline moved wholesale into the `else` branch, and nothing drove it end to end.

    Without the fp8 matmul the scales are folded into the weights, so what has to come out is a
    plain bf16 Linear whose values are the original ones -- a dropped scale is off by
    `1/weight_scale`, i.e. orders of magnitude, not a rounding difference.
    """
    state_dict, originals = _scaled_fp8_checkpoint()

    model, _ = _load(monkeypatch, tmp_path, state_dict)

    assert not any(isinstance(module, Int8ConvrotLinear) for module in model.modules())
    for path, source in (
        ("transformer_blocks.0.attn.to_q", originals["double_blocks.0.img_attn.qkv"][:H]),
        ("transformer_blocks.0.ff.linear_out", originals["double_blocks.0.img_mlp.2"]),
    ):
        layer = model.get_submodule(path)
        assert layer.weight.dtype is torch.bfloat16, path
        assert not hasattr(layer, "weight_scale"), path
        # fp8 keeps ~2 decimal digits, so this is a "the scale was applied" check, not a bit compare.
        assert torch.allclose(layer.weight.float(), source, rtol=0.1, atol=0.1 * source.abs().max())


def test_a_comfyui_prefixed_checkpoint_loads_the_same_as_a_bare_one(monkeypatch, tmp_path) -> None:
    """Not about int8 -- about the prefix strip this loader performs before it reads any side
    channel, which was eight lines inline here and which no test noticed. A redistribution wrapping
    every key in `model.diffusion_model.` passes the config probes and reaches this method; with the
    strip gone, the markers and scales are read from keys the model does not have. Asserted against
    the bare load, so the two cannot drift apart.
    """
    state_dict, _ = _checkpoint()
    prefixed = {f"model.diffusion_model.{key}": value for key, value in state_dict.items()}

    run, config = _driver(monkeypatch, tmp_path, prefixed)
    from_prefixed = run.loader._load_model(config, SubModelType.Transformer).state_dict()
    run, config = _driver(monkeypatch, tmp_path, state_dict)
    from_bare = run.loader._load_model(config, SubModelType.Transformer).state_dict()

    assert set(from_prefixed) == set(from_bare)
    for key, value in from_bare.items():
        assert torch.equal(from_prefixed[key].to(torch.float32), value.to(torch.float32)), key


def test_a_header_hint_reaches_every_projection_the_fused_qkv_became(monkeypatch, tmp_path) -> None:
    """`full_precision_matrix_mult` names a BFL layer; the modules it has to reach are the three a
    fused qkv becomes. Un-renamed it matches nothing and is ignored in silence -- the layer then runs
    on the fp8 tensor cores the producer measured as unsafe, with nothing logged.

    This is the fp8 half of the header re-key. Its int8 twin is covered two cells up; this one was
    the gap the mutation sweep found, and closing it costs one fixture that already existed.
    """
    from invokeai.backend.quantization import fp8_scaled

    state_dict, _ = _scaled_fp8_checkpoint()
    fused = next(path for path in QUANTIZED if path.endswith("img_attn.qkv"))
    header = {
        "_quantization_metadata": json.dumps(
            {
                "format_version": "1.0",
                "layers": {
                    path: {"format": "float8_e4m3fn", "full_precision_matrix_mult": path == fused} for path in QUANTIZED
                },
            }
        )
    }
    # Process-wide, so set through monkeypatch rather than the setter: the flag must not leak to
    # whichever test this worker runs next.
    monkeypatch.setattr(fp8_scaled, "_full_precision_hints_override", True)

    run, config = _driver(monkeypatch, tmp_path, state_dict, header)
    # The scales only stay attached where something keeps the weights quantized.
    run.loader._keep_fp8_weights = lambda _config, _submodel=None: True
    model = run.load(config)

    for projection in ("to_q", "to_k", "to_v"):
        marked = model.get_submodule(f"transformer_blocks.0.attn.{projection}")
        assert marked._fp8_full_precision_matmul is True, projection
    assert model.get_submodule("transformer_blocks.0.attn.to_out.0")._fp8_full_precision_matmul is False


def test_room_is_reserved_before_the_scales_are_folded(monkeypatch, tmp_path) -> None:
    """This loader cannot reach its main reservation before it widens, so it needs an earlier one.

    Its architecture is read off the converted keys, so no model exists at fold time to size a
    prediction against -- unlike every peer in this series, which builds an empty model first and
    reserves once. Left alone, the fold and `_dequantize_fp8_weights` both widen weights against a
    cache that `_load_and_cache` sized for the file alone: roughly another file size, unreserved, on
    the way to a 9B transformer.

    The model-less prediction is exact here because it is only taken when nothing stays quantized,
    and that is the only case in which either step widens anything.

    It also has to cover the scales themselves. `extract_fp8_scaled_layers` recovers them eleven lines
    above this reservation and the fold consumes them immediately below, so they are resident across
    it and are no longer in the dict any prediction walks -- on an MXFP8 build that is ~0.125 B per
    weight element of decoded exponent grid. Asserted by value, because this is the one reservation in
    the series whose amount changed rather than only its position.
    """
    state_dict, _ = _scaled_fp8_checkpoint()
    run, config = _driver(monkeypatch, tmp_path, state_dict, observe=("dequantize_fp8_scaled",))

    run.load(config)

    assert [step for step, _ in run.order] == ["dequantize_fp8_scaled"], run.order
    assert all(reserved for _step, reserved in run.order), run.order

    # Counted by destination, not by source key: the conversion runs before the extraction, so a
    # fused `qkv`'s one scalar scale is recovered against each of the three projections it becomes and
    # is charged once per destination. That replication is why the count follows `destinations` here
    # exactly as it does in `_expected_reservation`, and it is a slight over-charge of the same tensor
    # -- conservative, which is the safe direction for a reservation.
    destinations = len(QUANTIZED) + 2 * 2
    recovered = destinations * torch.float32.itemsize
    widened = sum(
        tensor.nelement() * torch.bfloat16.itemsize
        for key, tensor in state_dict.items()
        if not (key.endswith(".weight_scale") or key.endswith(".input_scale"))
    )
    assert recovered > 0, "otherwise this cell could not notice the term being dropped"
    assert run.reserved[0] == widened + recovered
