"""Loader-level tests for the single-file Ideogram 4 path.

These drive `Ideogram4CheckpointModel._load_model` and assert the *end state* of the module, not
the decisions that produced it: what a scaled-fp8 checkpoint leaves resident, which layers keep
their codes, and which two must not -- `Ideogram4Transformer.forward` and
`Ideogram4EmbedScalar.forward` derive the dtype they cast their inputs to from `input_proj` and
`t_embedding`, so an fp8 weight on either turns every activation into float8.

A tiny `Ideogram4Config` stands in for the released geometry: same module tree, 29 tensors instead
of 458.
"""

import pytest
import torch

from invokeai.backend.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Ideogram4_Config
from invokeai.backend.model_manager.load import load_default
from invokeai.backend.model_manager.load.model_loaders import ideogram4
from invokeai.backend.model_manager.load.model_loaders.ideogram4 import Ideogram4CheckpointModel
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.quantization.int8_convrot import (
    CONVROT_GROUP_SIZE,
    Int8ConvrotLinear,
)
from tests.fixtures.loader_seams import Seam, SeamRun, prepare
from tests.fixtures.quantized_payloads import comfy_quant_marker, quantize_convrot, quantize_scaled_fp8

FP8 = torch.float8_e4m3fn

TINY_CONFIG = Ideogram4Config(
    emb_dim=64,
    num_layers=1,
    num_heads=2,
    intermediate_size=128,
    adanln_dim=16,
    in_channels=8,
    llm_features_dim=32,
    mrope_section=(4, 2, 2),
)

# convrot rotates in groups along the input dim, so the int8 cases need a geometry whose linears are
# a whole number of groups wide. Everything else stays as narrow as the released layout allows.
INT8_CONFIG = Ideogram4Config(
    emb_dim=CONVROT_GROUP_SIZE,
    num_layers=1,
    num_heads=2,
    intermediate_size=CONVROT_GROUP_SIZE,
    adanln_dim=CONVROT_GROUP_SIZE,
    in_channels=8,
    llm_features_dim=32,
    mrope_section=(16, 8, 8),
)

# One Linear the cast must keep quantized, and the two it must not.
KEPT = "layers.0.feed_forward.w1"
SKIPPED = ("input_proj", "t_embedding.mlp_in")


def _checkpoint(paths: tuple[str, ...]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A full state dict for the tiny geometry, with `paths` stored as scaled fp8.

    Returns the checkpoint and the float originals of the quantized layers, so a test can compare
    what the loader produced against what the file meant.
    """
    torch.manual_seed(0)
    reference = Ideogram4Transformer(TINY_CONFIG)
    state_dict = {key: value.clone().to(torch.float32) for key, value in reference.state_dict().items()}

    originals: dict[str, torch.Tensor] = {}
    for path in paths:
        key = f"{path}.weight"
        originals[key] = state_dict[key].clone()
        payload = quantize_scaled_fp8(state_dict[key])
        state_dict[key], state_dict[f"{path}.weight_scale"] = payload.codes, payload.scale

    return state_dict, originals


SEAM = Seam(
    loader=Ideogram4CheckpointModel,
    module=ideogram4,
    entry="_load_model",
    patches_device=True,
)


def _driver(
    monkeypatch,
    tmp_path,
    state_dict: dict,
    *,
    keep_fp8: bool,
    fp8_storage: bool = False,
    geometry: Ideogram4Config = TINY_CONFIG,
):
    checkpoint = tmp_path / "ideogram4_fp8_scaled.safetensors"
    checkpoint.touch()
    config = Main_Checkpoint_Ideogram4_Config.model_construct(
        path=str(checkpoint),
        name="ideogram4",
        branch="conditional",
        default_settings=MainModelDefaultSettings(fp8_storage=fp8_storage),
    )

    def install(patch):
        # The two inputs of the loader's decision, not the decision itself: `_keep_fp8_weights` asks
        # the fp8 matmul and this model's FP8 Storage setting, and either alone is enough. Stubbing
        # the answer would let a loader that consults only one of them stay green.
        patch.setattr(load_default, "should_keep_fp8_weights", lambda _device: keep_fp8)
        patch.setattr(load_default, "_device_supports_fp8_storage", lambda _device, _logger=None: True)
        patch.setattr("invokeai.backend.ideogram4.modeling_ideogram4.Ideogram4Config", lambda: geometry)

    return prepare(SEAM, monkeypatch, state_dict=state_dict, metadata=None, geometry=install), config


def _load(
    monkeypatch,
    tmp_path,
    state_dict,
    *,
    keep_fp8: bool,
    fp8_storage: bool = False,
    geometry: Ideogram4Config = TINY_CONFIG,
) -> tuple[torch.nn.Module, SeamRun]:
    """The model, and the record of how the loader got there -- the second is why several of these
    tests exist at all, so it is returned rather than collected through an out-parameter."""
    run, model_config = _driver(
        monkeypatch, tmp_path, state_dict, keep_fp8=keep_fp8, fp8_storage=fp8_storage, geometry=geometry
    )
    return run.load(model_config, SubModelType.Transformer), run


def test_with_neither_consumer_the_scales_are_folded_into_full_precision_weights(monkeypatch, tmp_path) -> None:
    """Neither the fp8 matmul nor FP8 Storage wants these packed, so folding is right: keeping them
    would dequantize on every forward to save memory nobody asked to save."""
    state_dict, originals = _checkpoint((KEPT, *SKIPPED))

    model, _ = _load(monkeypatch, tmp_path, state_dict, keep_fp8=False)

    for key, original in originals.items():
        weight = model.get_parameter(key)
        assert weight.dtype is torch.float32
        # fp8 has ~2 decimal digits, so this is a "the scale was applied" check, not a bit compare:
        # a dropped scale would be off by 1/scale, which is orders of magnitude.
        assert torch.allclose(weight, original, rtol=0.1, atol=0.02)
    assert not any(hasattr(module, "weight_scale") for module in model.modules())


def test_fp8_storage_alone_keeps_the_checkpoint_in_its_own_scaled_form(monkeypatch, tmp_path) -> None:
    """FP8 Storage is the second consumer, and asking only the matmul made this path *lossy*.

    Folded, the weights reach the layerwise cast as full precision and it re-encodes them as
    *unscaled* fp8 -- for the byte count the file already had, and losing the layers whose scale was
    what kept them representable (~3% flushed to zero, measured on FLUX.2 Klein 4B). Kept, the
    checkpoint's own per-tensor scale stays exact and the cast is skipped as having nothing to do.
    """
    state_dict, _ = _checkpoint((KEPT,))
    before = {key: value.clone() for key, value in state_dict.items()}

    model, run = _load(monkeypatch, tmp_path, state_dict, keep_fp8=False, fp8_storage=True)

    kept = model.get_submodule(KEPT)
    assert kept.weight.dtype is FP8
    assert kept.weight_scale.shape == ()
    assert run.casting_calls == 0
    # Byte for byte what the file holds -- that is the whole argument for keeping them. A fold
    # followed by the storage cast re-encodes without the scale, which on a checkpoint whose scales
    # are not powers of two silently flushes the smallest weights to zero.
    assert torch.equal(kept.weight.view(torch.uint8), before[f"{KEPT}.weight"].view(torch.uint8))
    assert kept.weight_scale == before[f"{KEPT}.weight_scale"]

    # And the reservation is sized for what is actually held. This is the number the fix moves --
    # measured on the released checkpoint, 17.28 GiB down to 8.68 -- and it is what a loader that
    # decided the prediction's flag separately from the fold's would get wrong while every
    # end-state assertion above still passed.
    #
    # The scale is popped out of the dict before this point, so no prediction over the dict can see
    # it; the seam reserves through `reserve_for_load`, which is handed the recovered scales as well and
    # charges them. Four bytes here for a scalar, and 1.455 GiB on an MXFP8 build whose scales are
    # decoded exponent grids -- the reason the charge exists at all.
    kept_codes = before[f"{KEPT}.weight"].numel()
    dense = sum(
        tensor.numel() * torch.float32.itemsize
        for key, tensor in before.items()
        if key != f"{KEPT}.weight" and not key.endswith(".weight_scale")
    )
    recovered_scales = sum(
        tensor.numel() * tensor.element_size() for key, tensor in before.items() if key.endswith(".weight_scale")
    )
    assert recovered_scales > 0, "otherwise this cell would not notice the plan dropping the term"
    assert run.reserved[-1] == kept_codes + dense + recovered_scales


def test_with_the_fp8_matmul_the_codes_stay_and_the_scales_are_attached(monkeypatch, tmp_path) -> None:
    state_dict, _ = _checkpoint((KEPT,))

    model, run = _load(monkeypatch, tmp_path, state_dict, keep_fp8=True)

    kept = model.get_submodule(KEPT)
    assert kept.weight.dtype is FP8
    assert kept.weight_scale.shape == ()
    # Non-persistent: a re-save that carried it would scale the weight twice.
    assert "weight_scale" not in kept.state_dict()
    # And the layerwise storage pass must not have run: its hooks restore the compute dtype without
    # applying `weight_scale` (a silently wrong weight) and disable the matmul this path exists for.
    assert run.casting_calls == 0


@pytest.mark.parametrize("keep_fp8", [True, False], ids=["fp8_compute", "folded"])
def test_room_is_reserved_before_the_scales_are_folded(monkeypatch, tmp_path, keep_fp8: bool) -> None:
    # Two steps widen weights and both must land on reserved room: `split_fp8_scaled_layers`
    # dequantizes what it cannot keep through float32, and -- when neither consumer wants the codes
    # -- the fold widens every scaled layer outright. Reserving first is invisible in the loaded
    # model, so it is asserted where it happens.
    #
    # `keep_fp8=False` is the case that exercises the fold at all; with it true the fold never runs,
    # so that parameter alone could not tell "reserved first" from "folded first".
    state_dict, _ = _checkpoint((KEPT, *SKIPPED))

    _, run = _load(monkeypatch, tmp_path, state_dict, keep_fp8=keep_fp8)

    fp8_at_reservation = sum(1 for dtype in run.dtypes_at_make_room.values() if dtype is FP8)
    assert fp8_at_reservation == 3, "a weight was widened before the cache was asked for room"


@pytest.mark.parametrize("path", SKIPPED)
@pytest.mark.parametrize(
    ("keep_fp8", "fp8_storage"), [(True, False), (False, True)], ids=["fp8_compute", "fp8_storage"]
)
def test_the_two_dtype_deriving_layers_never_stay_quantized(
    monkeypatch, tmp_path, path: str, keep_fp8: bool, fp8_storage: bool
) -> None:
    # The regression this guards: both forwards read `<layer>.weight.dtype` (or its `compute_dtype`)
    # and cast x, t and the conditioning to it. With an fp8 weight there, the first matmul dies with
    # "addmm_cpu" not implemented for 'Float8_e4m3fn' -- and on a device where it does not die, it
    # computes in a dtype the model never intended.
    state_dict, originals = _checkpoint((KEPT, path))

    model, run = _load(monkeypatch, tmp_path, state_dict, keep_fp8=keep_fp8, fp8_storage=fp8_storage)

    skipped = model.get_submodule(path)
    assert skipped.weight.dtype is torch.float32
    assert getattr(skipped, "weight_scale", None) is None
    assert torch.allclose(skipped.weight, originals[f"{path}.weight"], rtol=0.1, atol=0.02)
    # The layer that is allowed to stay fp8 still did.
    assert model.get_submodule(KEPT).weight.dtype is FP8


def test_a_plain_checkpoint_loads_verbatim_and_is_offered_to_the_storage_pass(monkeypatch, tmp_path) -> None:
    state_dict, _ = _checkpoint(())
    expected = {key: value.clone() for key, value in state_dict.items()}

    model, run = _load(monkeypatch, tmp_path, state_dict, keep_fp8=False, fp8_storage=True)

    for key, value in expected.items():
        assert torch.equal(model.get_parameter(key), value), key
    # Nothing here is fp8, so the model's own FP8 Storage setting is the only thing that could make
    # it so -- this is the one path that must reach `_apply_fp8_layerwise_casting`. The setting is on
    # for realism rather than because the assertion needs it: the casting pass is stubbed out by a
    # counter here, so what is pinned is that the loader offers the model to it at all.
    assert run.casting_calls == 1


def test_only_the_transformer_submodel_is_served(monkeypatch, tmp_path) -> None:
    # A single file has no encoder, tokenizer or VAE in it; those are selected on the loader node.
    state_dict, _ = _checkpoint(())
    run, config = _driver(monkeypatch, tmp_path, state_dict, keep_fp8=False)

    with pytest.raises(ValueError, match="holds only a transformer"):
        run.load(config, SubModelType.VAE)


MARKER = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": CONVROT_GROUP_SIZE}

# The released int8 build quantizes the six linears of every block and nothing else. `adaln_modulation`
# is the one with a bias, so it is the one worth driving.
INT8_LAYERS = ("layers.0.attention.o", "layers.0.adaln_modulation")


def _quantize_tensorwise(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel int8 with no rotation."""
    scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
    return torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8), scale.to(torch.float32)


def _int8_checkpoint(
    paths: tuple[str, ...] = INT8_LAYERS, *, convrot: bool = True
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """A full state dict for the tiny geometry with `paths` stored int8_tensorwise (+convrot).

    `convrot=False` is for the layers a repack quantizes that the released build does not: the
    rotation works in groups of 256 along the input dim, and `input_proj` is 8 wide. It is still a
    valid `int8_tensorwise` marker -- what makes the layer interesting is the int8 storage.
    """
    torch.manual_seed(0)
    reference = Ideogram4Transformer(INT8_CONFIG)
    state_dict = {key: value.clone().to(torch.float32) for key, value in reference.state_dict().items()}

    originals: dict[str, torch.Tensor] = {}
    for path in paths:
        key = f"{path}.weight"
        originals[key] = state_dict[key].clone()
        if convrot:
            payload = quantize_convrot(state_dict[key])
            state_dict[key], state_dict[f"{path}.weight_scale"] = payload.codes, payload.scale
        else:
            state_dict[key], state_dict[f"{path}.weight_scale"] = _quantize_tensorwise(state_dict[key])
        state_dict[f"{path}.comfy_quant"] = comfy_quant_marker({**MARKER, "convrot": convrot})

    return state_dict, originals


def _reservation_for(
    before_load: dict[str, torch.Tensor], staying_int8: tuple[str, ...], dtype: torch.dtype = torch.float32
) -> int:
    """What the load occupies, with the four widths spelled out rather than recomputed.

    Takes the state dict as it was *before* the load, which pops the markers and rebinds the dict.

    A payload that stays int8 is one byte and its scale stays at the fp32 it is pinned at. A marker
    is dropped, and so is the scale of a marked layer the split widens -- that one is folded into
    its weight, which arrives at the compute width like every dense tensor. Charging a kept int8
    payload two bytes asks the cache to evict roughly twice what the load needs; charging a widened
    one a single byte leaves the reservation short by exactly the layers the split grew.
    """
    marked = {key[: -len(".comfy_quant")] for key in before_load if key.endswith(".comfy_quant")}
    widened = marked - set(staying_int8)
    pinned = {f"{path}.{suffix}" for path in staying_int8 for suffix in ("weight", "weight_scale")}
    dropped = {f"{path}.comfy_quant" for path in marked} | {f"{path}.weight_scale" for path in widened}
    total = 0
    for key, tensor in before_load.items():
        if key in dropped:
            continue
        total += tensor.nelement() * (tensor.element_size() if key in pinned else dtype.itemsize)
    return total


def test_an_int8_checkpoint_stays_int8_resident_and_un_rotates(monkeypatch, tmp_path) -> None:
    """The end state, not the decision.

    `Int8ConvrotLinear` holds the stored codes and dequantizes per forward, which is the entire
    point: the file's size is the resident size. The weight it reconstructs must also be the
    *un-rotated* one — a checkpoint taken down the fp8 path instead would be scaled but never
    derotated, which loads cleanly and generates noise.
    """
    state_dict, originals = _int8_checkpoint()
    biases = {key: value.clone() for key, value in state_dict.items() if key.endswith(".bias")}

    model, run = _load(monkeypatch, tmp_path, state_dict, keep_fp8=False, geometry=INT8_CONFIG)

    for path in INT8_LAYERS:
        layer = model.get_submodule(path)
        assert isinstance(layer, Int8ConvrotLinear), path
        assert layer.weight.dtype is torch.int8
        assert layer.convrot is True
        # int8 keeps ~2 decimal digits per weight, so this is "the rotation was undone", not a bit
        # compare: a weight left rotated is a different tensor entirely, not a rounder one.
        reconstructed = layer._dequantized_weight(torch.device("cpu"), torch.float32)
        original = originals[f"{path}.weight"]
        assert torch.allclose(reconstructed, original, rtol=0.05, atol=0.05 * original.abs().max())

    # `adaln_modulation` is the one quantized layer with a bias, and the swapped module carries it
    # as a buffer rather than a parameter -- so the strict load below cannot see it go missing.
    assert torch.equal(model.get_submodule("layers.0.adaln_modulation").bias, biases["layers.0.adaln_modulation.bias"])
    # The fp8 storage pass must not run over int8 codes either.
    assert run.casting_calls == 0


def test_an_unusable_activation_scale_does_not_break_the_strict_load(monkeypatch, tmp_path) -> None:
    """W8A8 sidecars are metadata for a mode this path does not implement.

    The load is strict, so a repack that ships `.input_scale` next to its int8 weights would fail
    on an unexpected key -- rejecting a file that is otherwise perfectly loadable.
    """
    state_dict, _ = _int8_checkpoint()
    state_dict[f"{INT8_LAYERS[0]}.input_scale"] = torch.ones((), dtype=torch.float32)

    model, _ = _load(monkeypatch, tmp_path, state_dict, keep_fp8=False, geometry=INT8_CONFIG)

    assert isinstance(model.get_submodule(INT8_LAYERS[0]), Int8ConvrotLinear)


@pytest.mark.parametrize("path", SKIPPED)
def test_a_repack_that_quantizes_a_dtype_deriving_layer_is_dequantized(monkeypatch, tmp_path, path: str) -> None:
    """The int8 half of the skip patterns, which the released build never exercises.

    `Ideogram4Transformer.forward` and `Ideogram4EmbedScalar.forward` read `input_proj` and
    `t_embedding`'s weight dtype and cast x, t and the conditioning to it. An `Int8ConvrotLinear`
    there reports `torch.int8`, which is not a dtype torch computes in -- so the layer has to come
    back dense even though its marker says otherwise.
    """
    state_dict, originals = _int8_checkpoint((path,), convrot=False)
    for key, value in _int8_checkpoint((INT8_LAYERS[0],))[0].items():
        if key.startswith(f"{INT8_LAYERS[0]}."):
            state_dict[key] = value
    before_load = {key: value.clone() for key, value in state_dict.items()}

    model, run = _load(monkeypatch, tmp_path, state_dict, keep_fp8=False, geometry=INT8_CONFIG)

    skipped = model.get_submodule(path)
    assert not isinstance(skipped, Int8ConvrotLinear)
    assert skipped.weight.dtype is torch.float32
    original = originals[f"{path}.weight"]
    assert torch.allclose(skipped.weight, original, rtol=0.05, atol=0.05 * original.abs().max())
    # The block linear it was mixed with still keeps its codes.
    assert isinstance(model.get_submodule(INT8_LAYERS[0]), Int8ConvrotLinear)
    # And the widened layer was reserved for at its widened width: charging it one byte because a
    # marker claims it leaves the reservation short by exactly the layers the split grew.
    assert run.reserved[-1] == _reservation_for(before_load, (INT8_LAYERS[0],))


def test_the_int8_reservation_charges_each_payload_its_stored_width(monkeypatch, tmp_path) -> None:
    # The cache frees what it is asked for, so this number is the load's whole VRAM/RAM contract:
    # two bytes per int8 payload evicts roughly twice what an 8.9 GiB file needs.
    state_dict, _ = _int8_checkpoint()
    before_load = {key: value.clone() for key, value in state_dict.items()}

    _, run = _load(monkeypatch, tmp_path, state_dict, keep_fp8=False, geometry=INT8_CONFIG)

    assert run.reserved[-1] == _reservation_for(before_load, INT8_LAYERS)


def test_a_missing_bias_on_a_quantized_layer_is_refused(monkeypatch, tmp_path) -> None:
    """The one hole the strict load cannot cover.

    `Int8ConvrotLinear` registers `bias` only when the checkpoint has one, so after the swap the
    module's key set mirrors the file instead of the architecture. A repack that drops all-zero
    biases would leave every block's adaLN modulation without its offset -- wrong `scale_msa`,
    `gate_msa`, `scale_mlp`, `gate_mlp` -- and load, cache and render with nothing in the log.
    """
    state_dict, _ = _int8_checkpoint()
    del state_dict["layers.0.adaln_modulation.bias"]

    with pytest.raises(ValueError, match="layers.0.adaln_modulation"):
        _load(monkeypatch, tmp_path, state_dict, keep_fp8=False, geometry=INT8_CONFIG)


def test_a_file_mixing_int8_with_scaled_fp8_is_refused(monkeypatch, tmp_path) -> None:
    """Inside the int8 branch the fp8 pipeline is skipped entirely.

    An fp8 layer that came along for the ride would be cast to the compute dtype without its scale
    -- off by `1/weight_scale` -- while `strict=False` swallows the orphaned scale. There is no
    setting under which this file is loadable, so it is refused rather than half-loaded.
    """
    state_dict, _ = _int8_checkpoint()
    fp8 = quantize_scaled_fp8(state_dict["final_layer.linear.weight"])
    state_dict["final_layer.linear.weight"] = fp8.codes
    state_dict["final_layer.linear.weight_scale"] = fp8.scale

    with pytest.raises(ValueError, match="mixing int8_tensorwise with scaled fp8"):
        _load(monkeypatch, tmp_path, state_dict, keep_fp8=False, geometry=INT8_CONFIG)
