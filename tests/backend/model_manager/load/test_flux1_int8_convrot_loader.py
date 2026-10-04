"""Loader-level tests for the `int8_tensorwise` FLUX.1 path.

These drive `FluxCheckpointModel._load_from_singlefile` and assert the end state of the module:
which layers keep their codes, what weight they reconstruct, and what the cache was asked to free.

FLUX.1 is the simple shape of this scheme next to FLUX.2: no key conversion stands between a
marker and its module, and `qkv` stays one fused Linear. What it does have, and Klein 9B does not,
is the scheme's full form -- `convrot` at group size 256 with per-output-channel scales -- so a
loader that ignored the rotation would produce a weight that loads cleanly and generates noise.
"""

import json

import pytest
import torch

from invokeai.backend.flux.model import FluxParams
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_FLUX_Config
from invokeai.backend.model_manager.load.model_loaders import flux
from invokeai.backend.model_manager.load.model_loaders.flux import FluxCheckpointModel
from invokeai.backend.model_manager.taxonomy import FluxVariantType
from invokeai.backend.quantization.int8_convrot import (
    CONVROT_GROUP_SIZE,
    Int8ConvrotLinear,
)
from tests.fixtures.loader_seams import Seam, prepare
from tests.fixtures.quantized_payloads import comfy_quant_marker, quantize_convrot, quantize_scaled_fp8

# As the file carries it: rotated along the input dim, per-output-channel scales.
MARKER = {"format": "int8_tensorwise", "per_row": True, "convrot": True, "convrot_groupsize": CONVROT_GROUP_SIZE}

# The released geometry scaled down: convrot works in groups of 256 along the input dim, so the
# hidden size cannot go below one group. One double and one single block is enough -- the file
# quantizes the same six layer kinds in every block.
TINY = FluxParams(
    in_channels=64,
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=CONVROT_GROUP_SIZE,
    mlp_ratio=4.0,
    num_heads=2,
    depth=1,
    depth_single_blocks=1,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    guidance_embed=True,
)

# What this repack quantizes: both `qkv`, both MLPs, every modulation `lin` -- and, notably, not
# `attn.proj`. `adaln`/embedders/final layer stay dense.
QUANTIZED = (
    "double_blocks.0.img_attn.qkv",
    "double_blocks.0.img_mlp.0",
    "double_blocks.0.img_mlp.2",
    "double_blocks.0.img_mod.lin",
    "double_blocks.0.txt_attn.qkv",
    "single_blocks.0.linear1",
    "single_blocks.0.linear2",
    "single_blocks.0.modulation.lin",
)


# Stored fp32 rather than bf16, because a repack that mixes precisions in its dense remainder is
# what makes `cast_unquantized` observable at all: `load_state_dict(assign=True)` assigns storage
# dtypes straight through, so without the cast this layer lands as fp32 inside a bf16 model -- and
# the reservation, which charged it the compute width, comes up short by its size.
STORED_FP32 = "img_in.weight"

# W8A8 metadata for a mode this path does not implement. The load tolerates extras, but the cast
# and the reservation would both pay for it.
ACTIVATION_SCALE = "double_blocks.0.img_attn.qkv.input_scale"


def _checkpoint(marker: dict | None = MARKER) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A full state dict for the tiny geometry with `QUANTIZED` stored int8_tensorwise+convrot."""
    torch.manual_seed(0)
    with torch.device("cpu"):
        from invokeai.backend.flux.model import Flux

        reference = Flux(TINY)
    state_dict = {key: value.clone().to(torch.bfloat16) for key, value in reference.state_dict().items()}
    state_dict[STORED_FP32] = state_dict[STORED_FP32].float()
    state_dict[ACTIVATION_SCALE] = torch.ones((), dtype=torch.float32)

    originals: dict[str, torch.Tensor] = {}
    for path in QUANTIZED:
        key = f"{path}.weight"
        weight = state_dict[key].float()
        originals[path] = weight
        payload = quantize_convrot(weight)
        state_dict[key], state_dict[f"{path}.weight_scale"] = payload.codes, payload.scale
        if marker is not None:
            state_dict[f"{path}.comfy_quant"] = comfy_quant_marker(marker)
    return state_dict, originals


SEAM = Seam(
    loader=FluxCheckpointModel,
    module=flux,
    compute_dtype=torch.bfloat16,
    # `_load_from_singlefile` never reaches the FP8 Storage pass, and the tests that do reach it
    # through `_load_model` install their own counter. Nothing here would read this stub, and a
    # stub nothing reads is one more thing to keep true.
    casts_fp8_storage=False,
)


def _driver(monkeypatch, tmp_path, state_dict: dict, header: dict | None = None):
    checkpoint = tmp_path / "flux1-dev-int8-convrot.safetensors"
    checkpoint.touch()
    config = Main_Checkpoint_FLUX_Config.model_construct(
        path=str(checkpoint), name="flux1-dev-int8", variant=FluxVariantType.Dev
    )

    def geometry(patch):
        patch.setattr(flux, "get_flux_transformers_params", lambda _variant: TINY)
        # Both spellings of the same decision: the module-level helper, and the loader method that
        # supersedes it once FP8 Storage counts as a consumer too. `raising=False` tolerates the
        # module symbol being gone.
        patch.setattr(flux, "should_keep_fp8_weights", lambda _device: False, raising=False)

    run = prepare(SEAM, monkeypatch, state_dict=state_dict, metadata=header, geometry=geometry)
    # An instance attribute shadows the method, which is what keeps this working either way.
    run.loader._keep_fp8_weights = lambda _config, _submodel=None: False
    return run, config


def _load(monkeypatch, tmp_path, state_dict, header: dict | None = None):
    run, config = _driver(monkeypatch, tmp_path, state_dict, header)
    return run.load(config), run


def test_an_int8_checkpoint_stays_int8_and_un_rotates(monkeypatch, tmp_path) -> None:
    """The end state, not the decision.

    `Int8ConvrotLinear` holds the stored codes, which is the point -- the file's size is the
    resident size. The weight it reconstructs must also be the *un-rotated* one: a checkpoint sent
    down the fp8 path instead would be scaled but never derotated, which loads cleanly and
    generates noise.
    """
    state_dict, originals = _checkpoint()

    model, _ = _load(monkeypatch, tmp_path, state_dict)

    for path in QUANTIZED:
        layer = model.get_submodule(path)
        assert isinstance(layer, Int8ConvrotLinear), path
        assert layer.weight.dtype is torch.int8, path
        assert layer.convrot is True, path
        # int8 keeps ~2 decimal digits, so this is "the rotation was undone", not a bit compare:
        # a weight left rotated is a different tensor entirely, not a rounder one.
        reconstructed = layer._dequantized_weight(torch.device("cpu"), torch.float32)
        original = originals[path]
        assert torch.allclose(reconstructed, original, rtol=0.05, atol=0.05 * original.abs().max()), path


def test_the_biases_of_the_quantized_layers_survive(monkeypatch, tmp_path) -> None:
    """`qkv` and the modulation projections carry biases, and the swap has to keep them.

    `Int8ConvrotLinear` takes its bias from the checkpoint, so a swap that dropped it would leave
    every block's attention and modulation without its offset -- and still load.
    """
    state_dict, _ = _checkpoint()
    expected = {path: state_dict[f"{path}.bias"].clone() for path in QUANTIZED if f"{path}.bias" in state_dict}
    assert expected, "fixture carries no biases on the quantized layers"

    model, _ = _load(monkeypatch, tmp_path, state_dict)

    for path, bias in expected.items():
        assert torch.equal(model.get_submodule(path).bias.to(bias.dtype), bias), path


def test_the_reservation_charges_each_payload_its_stored_width(monkeypatch, tmp_path) -> None:
    """The cache frees what it is asked for, so this number is the load's whole memory contract.

    Charging an int8 payload bf16's two bytes evicts roughly twice what an 11.5 GiB file needs.
    """
    state_dict, _ = _checkpoint()
    before_load = {key: value.clone() for key, value in state_dict.items()}

    _, run = _load(monkeypatch, tmp_path, state_dict)

    # One byte per code; its per-output-channel scale stays at the fp32 it is pinned at; the marker
    # and the activation scale are dropped before the load and charged nothing; everything else
    # arrives at the compute width -- including the layer the checkpoint stored as fp32, charged
    # what it becomes rather than what it was.
    pinned = {f"{path}.{suffix}" for path in QUANTIZED for suffix in ("weight", "weight_scale")}
    expected = 0
    for key, tensor in before_load.items():
        if key.endswith(".comfy_quant") or key == ACTIVATION_SCALE:
            continue
        expected += tensor.nelement() * (tensor.element_size() if key in pinned else torch.bfloat16.itemsize)

    assert run.reserved[-1] == expected


def test_an_int8_weight_without_a_marker_is_refused(monkeypatch, tmp_path) -> None:
    """An unclaimed int8 weight would be cast to bf16 as raw codes: a model of small integers."""
    state_dict, _ = _checkpoint(marker=None)

    with pytest.raises(ValueError, match="int8"):
        _load(monkeypatch, tmp_path, state_dict)


def test_a_checkpoint_that_declares_int8_only_in_its_header_still_loads(monkeypatch, tmp_path) -> None:
    """Per-layer markers are one of the two forms ComfyUI writes, and not the only one in the wild.

    A repack that declares its scheme in `_quantization_metadata` alone would otherwise be refused
    for a marker it never had to write.
    """
    state_dict, originals = _checkpoint(marker=None)
    header = {"_quantization_metadata": json.dumps({"layers": dict.fromkeys(QUANTIZED, MARKER)})}

    model, _ = _load(monkeypatch, tmp_path, state_dict, header=header)

    for path in QUANTIZED:
        layer = model.get_submodule(path)
        assert isinstance(layer, Int8ConvrotLinear), path
        # The marker *body* has to survive that route too, not just the layer name: a header that
        # arrived without `convrot` would build every layer unrotated, and an unrotated decode of a
        # rotated weight loads cleanly and generates noise.
        assert layer.convrot is True, path
        reconstructed = layer._dequantized_weight(torch.device("cpu"), torch.float32)
        original = originals[path]
        assert torch.allclose(reconstructed, original, rtol=0.05, atol=0.05 * original.abs().max()), path


def test_a_per_layer_marker_wins_over_the_header(monkeypatch, tmp_path) -> None:
    # Only the per-layer marker can carry `convrot` and the group size, so where both are present it
    # decides how the weight is decoded. Deliberately the opposite of the fp8 hint merge beside it.
    state_dict, originals = _checkpoint(marker={**MARKER, "convrot": False})
    for path in QUANTIZED:
        weight = originals[path]
        scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
        state_dict[f"{path}.weight"] = torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8)
        state_dict[f"{path}.weight_scale"] = scale.to(torch.float32)
    header = {"_quantization_metadata": json.dumps({"layers": dict.fromkeys(QUANTIZED, MARKER)})}

    model, _ = _load(monkeypatch, tmp_path, state_dict, header=header)

    assert model.get_submodule(QUANTIZED[0]).convrot is False


def test_a_header_that_names_a_layer_this_file_does_not_quantize_is_ignored(monkeypatch, tmp_path) -> None:
    """Stale metadata must not route a plain checkpoint into the int8 branch.

    A repack tool that left `_quantization_metadata` behind on a bf16 export would otherwise take
    the whole file down the int8 path and die at the swap, complaining about a missing
    `weight_scale` for a layer that was never quantized -- pointing at a header the user cannot see.
    """
    state_dict, originals = _checkpoint(marker=None)
    for path in QUANTIZED:
        del state_dict[f"{path}.weight_scale"]
        state_dict[f"{path}.weight"] = originals[path].to(torch.bfloat16)
    header = {"_quantization_metadata": json.dumps({"layers": dict.fromkeys(QUANTIZED, MARKER)})}

    model, _ = _load(monkeypatch, tmp_path, state_dict, header=header)

    assert not any(isinstance(module, Int8ConvrotLinear) for module in model.modules())


def test_an_int8_weight_the_transformer_does_not_consume_is_tolerated(monkeypatch, tmp_path) -> None:
    """A merged export bundles an encoder beside the transformer.

    Those keys are discarded by the non-strict load rather than cast, so they cannot become noise --
    the same reasoning `reject_foreign_quantization_scales` already applies to a foreign scale.
    Refusing the whole file for them would reject a checkpoint that loads correctly today.
    """
    state_dict, _ = _checkpoint()
    state_dict["text_encoders.t5xxl.block.0.attn.q.weight"] = torch.zeros(8, 8, dtype=torch.int8)

    model, _ = _load(monkeypatch, tmp_path, state_dict)

    assert isinstance(model.get_submodule(QUANTIZED[0]), Int8ConvrotLinear)


def test_a_file_mixing_int8_with_scaled_fp8_is_refused(monkeypatch, tmp_path) -> None:
    """Inside the int8 branch the fp8 pipeline is skipped entirely, so an fp8 layer that came along
    would be cast to bf16 without its scale -- off by `1/weight_scale`, with nothing logged."""
    state_dict, _ = _checkpoint()
    fp8 = quantize_scaled_fp8(state_dict["double_blocks.0.img_attn.proj.weight"].float())
    state_dict["double_blocks.0.img_attn.proj.weight"] = fp8.codes
    state_dict["double_blocks.0.img_attn.proj.weight_scale"] = fp8.scale

    with pytest.raises(ValueError, match="mixing int8_tensorwise with scaled fp8"):
        _load(monkeypatch, tmp_path, state_dict)


def test_a_plain_checkpoint_still_takes_the_fp8_path(monkeypatch, tmp_path) -> None:
    # The other half: a file with no int8 in it must reach the scaled-fp8 pipeline unchanged.
    state_dict, originals = _checkpoint()
    for path in QUANTIZED:
        del state_dict[f"{path}.comfy_quant"]
        del state_dict[f"{path}.weight_scale"]
        state_dict[f"{path}.weight"] = originals[path].to(torch.bfloat16)

    model, _ = _load(monkeypatch, tmp_path, state_dict)

    assert not any(isinstance(module, Int8ConvrotLinear) for module in model.modules())
    for path in QUANTIZED:
        assert model.get_submodule(path).weight.dtype is torch.bfloat16, path


def test_the_dense_remainder_is_cast_to_the_compute_dtype(monkeypatch, tmp_path) -> None:
    """`assign=True` hands the checkpoint's storage dtype straight to the module.

    The fixture stores one layer as fp32, which is what makes this able to fail: without the cast an
    fp32 `img_in` lands inside a bf16 model.
    """
    state_dict, _ = _checkpoint()
    assert state_dict[STORED_FP32].dtype is torch.float32, "fixture no longer exercises the cast"

    model, _ = _load(monkeypatch, tmp_path, state_dict)

    assert model.img_in.weight.dtype is torch.bfloat16


def test_the_fp8_storage_pass_is_not_offered_an_int8_model(monkeypatch, tmp_path) -> None:
    """FP8 Storage is a per-model setting offered for every FLUX main model.

    On an int8 build the cast cannot reach the quantized weights at all, and what it *can* reach is
    the dense remainder the repack deliberately left bf16 -- the embedders and the final layer, 3.6%
    of the file. So it is skipped rather than allowed to spend e4m3 on the model's entry and exit.
    """
    from invokeai.backend.model_manager.taxonomy import SubModelType

    state_dict, _ = _checkpoint()
    calls: list[str] = []
    run, config = _driver(monkeypatch, tmp_path, state_dict)
    run.loader._apply_fp8_layerwise_casting = lambda model, *_args: calls.append("cast") or model

    # `_load_model`, not the seam's `_load_from_singlefile`: the FP8 Storage pass sits one level up.
    model = run.loader._load_model(config, SubModelType.Transformer)

    assert calls == []
    assert any(isinstance(module, Int8ConvrotLinear) for module in model.modules())


def test_a_dense_checkpoint_still_reaches_the_fp8_storage_pass(monkeypatch, tmp_path) -> None:
    # The other half: skipping for int8 must not skip for everything else, or the setting quietly
    # stops working for ordinary FLUX.1 checkpoints.
    from invokeai.backend.model_manager.taxonomy import SubModelType

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


def test_a_prefixed_checkpoint_is_refused_by_name_rather_than_loaded_without_its_hints(monkeypatch, tmp_path) -> None:
    """Why the header hints and the state dict cannot fall out of step on a checkpoint that loads.

    The hints are re-keyed into the stripped namespace unconditionally; the state dict is converted
    only when the bundle probe fires, and that probe looks for one key under `model.diffusion_model.`
    alone. So a `diffusion_model.`- or `net.`-prefixed file would keep its prefix while its hints
    lost theirs -- the header below names the prefixed layers, which is exactly the shape that would
    desync -- and every `full_precision_matrix_mult` would name nothing.

    It cannot bite, and the reason is upstream of this loader: `_validate_is_flux` admits a file only
    if `double_blocks.0.img_attn.norm.key_norm.scale` is present bare or under
    `model.diffusion_model.`, so a file in any other namespace never becomes a FLUX config at all.
    The loader's probe therefore cannot be narrower than what reaches it. Pinned here because the two
    refusals below are what a reader of this loader can see; the identification gate is not.
    """
    state_dict, _ = _checkpoint()
    prefixed = {f"diffusion_model.{key}": value for key, value in state_dict.items()}
    header = {"_quantization_metadata": json.dumps({"layers": {f"diffusion_model.{p}": MARKER for p in QUANTIZED}})}

    with pytest.raises(ValueError, match="keys need a conversion this loader did not apply"):
        _load(monkeypatch, tmp_path, prefixed, header=header)


def test_a_prefixed_dense_checkpoint_is_refused_for_want_of_every_parameter(monkeypatch, tmp_path) -> None:
    """The same file with no int8 install to catch it first. Either refusal alone is enough -- they
    are redundant, not each other's only line of defence -- and both are pinned because either is
    the kind of check a later change relaxes.

    What neither sees, and what would make the desync live, is a *new* state-dict prefix stripper on
    this path for a prefix `TRANSFORMER_KEY_PREFIXES` does not know.
    """
    state_dict, originals = _checkpoint(marker=None)
    for path in QUANTIZED:
        del state_dict[f"{path}.weight_scale"]
        state_dict[f"{path}.weight"] = originals[path].to(torch.bfloat16)
    prefixed = {f"diffusion_model.{key}": value for key, value in state_dict.items()}
    header = {
        "_quantization_metadata": json.dumps(
            {"layers": {f"diffusion_model.{p}": {"full_precision_matrix_mult": True} for p in QUANTIZED}}
        )
    }

    with pytest.raises(RuntimeError, match=r"missing \d+ parameter\(s\) that the model requires"):
        _load(monkeypatch, tmp_path, prefixed, header=header)
