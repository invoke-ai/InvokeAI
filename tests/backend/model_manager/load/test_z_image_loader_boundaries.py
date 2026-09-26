"""Loader-level tests for the Z-Image single-file path.

The state-dict helpers are covered elsewhere. What is pinned here is that the loader *calls*
them: deleting the swap would leave every unit test green while the loader produced a model that
loads cleanly and generates noise. These drive `_load_from_singlefile` itself and check what
reaches the module.
"""

import json

import diffusers
import pytest
import torch
from safetensors import torch as safetensors_torch

from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_ZImage_Config
from invokeai.backend.model_manager.load import load_default
from invokeai.backend.model_manager.load.model_loaders import z_image
from invokeai.backend.model_manager.load.model_loaders.z_image import ZImageCheckpointModel
from invokeai.backend.quantization.int8_convrot import CONVROT_GROUP_SIZE, Int8ConvrotLinear
from invokeai.backend.quantization.nvfp4 import NVFP4Linear
from tests.fixtures.loader_seams import Seam, prepare
from tests.fixtures.quantized_payloads import (
    comfy_quant_marker,
    nvfp4_codes,
    nvfp4_tensors,
    quantize_convrot,
    quantize_scaled_fp8,
)

MARKER = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": CONVROT_GROUP_SIZE}


class _TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(CONVROT_GROUP_SIZE, 4, bias=False)


class _TinyZImage(torch.nn.Module):
    """Stands in for ZImageTransformer2DModel. `layers.` is one of the loader's valid prefixes,
    so the state dict survives its filter."""

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_TinyBlock()])


SEAM = Seam(
    loader=ZImageCheckpointModel,
    module=z_image,
    # The loader calls `safetensors.torch.load_file` through the package, so patching the name on
    # its own module would patch nothing and let it read the empty file the config points at.
    load_file_host=safetensors_torch,
    patches_device=True,
)


def _driver(monkeypatch, tmp_path, state_dict: dict, model_class=None):
    checkpoint = tmp_path / "z_image_int8_convrot.safetensors"
    checkpoint.touch()
    config = Main_Checkpoint_ZImage_Config.model_construct(path=str(checkpoint), name="z-image")

    def geometry(patch):
        patch.setattr(diffusers, "ZImageTransformer2DModel", model_class or _TinyZImage, raising=False)

    return prepare(SEAM, monkeypatch, state_dict=state_dict, geometry=geometry), config


class _TinyMixedZImage(torch.nn.Module):
    """Two blocks, so a checkpoint can be int8 in one and scaled fp8 in the other."""

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_TinyBlock(), _TinyBlock()])


def test_a_mixed_int8_and_scaled_fp8_checkpoint_is_refused(monkeypatch, tmp_path) -> None:
    """This branch skips the fp8 pipeline entirely, so an fp8 weight that came along would be cast
    without its scale -- off by `1/weight_scale` -- while the orphaned scale disappears into the
    load. Z-Image made exactly that mistake: it ran every other step of the int8 install and not
    this check, and the result was a model that loaded cleanly and generated noise.
    """

    torch.manual_seed(0)
    quantized, scale, _restored = quantize_convrot(torch.randn(4, CONVROT_GROUP_SIZE))
    state_dict = {
        "layers.0.proj.weight": quantized,
        "layers.0.proj.weight_scale": scale,
        "layers.0.proj.comfy_quant": comfy_quant_marker(MARKER),
        "layers.1.proj.weight": torch.zeros(4, CONVROT_GROUP_SIZE, dtype=torch.float8_e4m3fn),
        "layers.1.proj.weight_scale": torch.ones(()),
    }
    run, config = _driver(monkeypatch, tmp_path, state_dict, model_class=_TinyMixedZImage)

    with pytest.raises(ValueError, match=r"layers\.1\.proj\.weight_scale"):
        run.load(config)

    # Refused before the cache was asked to evict anything for a load that cannot finish.
    assert not run.reserved


def test_an_int8_checkpoint_loads_int8_resident_and_un_rotated(monkeypatch, tmp_path) -> None:
    torch.manual_seed(0)
    original = torch.randn(4, CONVROT_GROUP_SIZE)
    quantized, scale, _restored = quantize_convrot(original)
    state_dict = {
        "layers.0.proj.weight": quantized,
        "layers.0.proj.weight_scale": scale,
        "layers.0.proj.comfy_quant": comfy_quant_marker(MARKER),
    }
    run, config = _driver(monkeypatch, tmp_path, state_dict)

    model = run.load(config)

    # Resident, not decoded: that is what keeps a 5.8 GB checkpoint at 5.8 GB.
    assert isinstance(model.layers[0].proj, Int8ConvrotLinear)
    assert model.layers[0].proj.weight.dtype is torch.int8

    dequantized = model.layers[0].proj._dequantized_weight(torch.device("cpu"), torch.float32).flatten()
    assert torch.corrcoef(torch.stack([dequantized, original.flatten()]))[0, 1] > 0.999
    # And specifically not the scaled-but-still-rotated weight, which is what a loader that only
    # applied the scale would produce -- silently.
    rotated = (quantized.float() * scale).flatten()
    assert torch.corrcoef(torch.stack([rotated, original.flatten()]))[0, 1].abs() < 0.2


@pytest.mark.parametrize("declared_in_header", [False, True], ids=["unmarked", "declared_in_header"])
def test_an_int8_weight_without_a_per_tensor_marker_is_refused(monkeypatch, tmp_path, declared_in_header) -> None:
    """A quantized weight the loader does not recognise would be handed to a float Linear and only
    fail at forward time, if at all. Refuse at load, and say which layers.

    The header case records a deliberate asymmetry. ComfyUI writes the per-layer flags in either of
    two places, and this loader reads only the per-tensor markers for int8 -- the header is consulted
    for fp8 and nvfp4 hints alone. FLUX.1 (`flux.py:806-822`) and FLUX.2 (`:1204-1221`) merge both;
    Z-Image, its Qwen3 encoder, Krea-2, Krea-2's Qwen3-VL encoder, Ideogram 4 and the PiD decoder
    refuse; MiniMax H3's transformer and its Qwen3-VL encoder have no such check at all, so a
    header-only build dies later in `load_state_dict` on the int8 dtype without naming the scheme.

    Refusing is the *conservative* side, and that is the reason to keep it -- not the survey that
    first motivated this cell. No header entry observed says whether the weight was rotated: across
    every checkpoint on this machine, each `int8_tensorwise` header entry reads
    `{"format": "int8_tensorwise"}` and nothing more, while Ideogram 4's per-tensor markers read
    `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`. Merging the header
    here would therefore build `Int8ConvrotLinear(convrot=False)` over a rotated weight, which loads
    and generates noise. Nothing stops a producer writing `convrot` into the header -- the parser
    passes each entry through verbatim, and nvfp4 entries do carry extra keys -- so this is what has
    been observed, not a property of the format. FLUX gets away with merging because the per-tensor
    marker wins where both are present, and the one build that writes int8 header entries
    (`flux-2-klein-9b-int8-convrot`) is unrotated in both channels.

    What is not a reason: "the transport is chosen per format". That was the first version of this
    docstring and it is false. `flux-2-klein-9b-fp8` is header-only fp8 while `ideogram4_fp8_scaled`
    is marker-only fp8 -- same format, different transport -- and `flux-2-klein-9b-int8-convrot`
    writes int8 entries in *both*. The transport follows the producing tool.
    """
    torch.manual_seed(1)
    quantized, scale, _restored = quantize_convrot(torch.randn(4, CONVROT_GROUP_SIZE))
    state_dict = {"layers.0.proj.weight": quantized, "layers.0.proj.weight_scale": scale}
    run, config = _driver(monkeypatch, tmp_path, state_dict)
    if declared_in_header:
        monkeypatch.setattr(
            z_image,
            "read_safetensors_metadata",
            # Bare on purpose: the damaging shape is a header entry that says the format and not
            # the rotation, which is every int8 header entry observed.
            lambda _path, _logger: {
                "_quantization_metadata": json.dumps({"layers": {"layers.0.proj": {"format": "int8_tensorwise"}}})
            },
        )

    with pytest.raises(ValueError, match=r"int8 weight\(s\) with no `comfy_quant` marker"):
        run.load(config)


def test_an_unquantized_checkpoint_is_unaffected(monkeypatch, tmp_path) -> None:
    torch.manual_seed(2)
    weight = torch.randn(4, CONVROT_GROUP_SIZE)
    run, config = _driver(monkeypatch, tmp_path, {"layers.0.proj.weight": weight})

    model = run.load(config)

    assert isinstance(model.layers[0].proj, torch.nn.Linear)
    assert not isinstance(model.layers[0].proj, Int8ConvrotLinear)
    assert torch.equal(model.layers[0].proj.weight, weight)


class _TinyTimestepEmbedder(torch.nn.Module):
    """Z-Image's `t_embedder` in miniature. Its real forward reads `self.mlp[0].weight.dtype` to
    pick the dtype it casts its activations to, which is why the model declares it
    precision-sensitive -- and why an `Int8ConvrotLinear` there (weight dtype `torch.int8`,
    `is_floating_point()` False, no `compute_dtype` attribute) sends that branch somewhere the
    model was never meant to run."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.ModuleList([torch.nn.Linear(CONVROT_GROUP_SIZE, 4, bias=False)])


class _TinyZImageWithTimestepEmbedder(_TinyZImage):
    _skip_layerwise_casting_patterns = ["t_embedder", "cap_embedder"]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.t_embedder = _TinyTimestepEmbedder()


def test_a_precision_sensitive_layer_is_not_left_int8(monkeypatch, tmp_path) -> None:
    """The model's own `_skip_layerwise_casting_patterns` has to be honored on the int8 branch too.
    The fp8 branch always read it; the int8 branch did not, so Z-Image's timestep embedder stayed
    quantized and its forward picked its activation dtype off a `torch.int8` weight."""
    torch.manual_seed(3)
    sensitive = torch.randn(4, CONVROT_GROUP_SIZE)
    ordinary = torch.randn(4, CONVROT_GROUP_SIZE)
    sensitive_q, sensitive_scale, _ = quantize_convrot(sensitive)
    ordinary_q, ordinary_scale, _ = quantize_convrot(ordinary)
    state_dict = {
        "t_embedder.mlp.0.weight": sensitive_q,
        "t_embedder.mlp.0.weight_scale": sensitive_scale,
        "t_embedder.mlp.0.comfy_quant": comfy_quant_marker(MARKER),
        "layers.0.proj.weight": ordinary_q,
        "layers.0.proj.weight_scale": ordinary_scale,
        "layers.0.proj.comfy_quant": comfy_quant_marker(MARKER),
    }
    run, config = _driver(monkeypatch, tmp_path, state_dict, model_class=_TinyZImageWithTimestepEmbedder)

    model = run.load(config)

    embedder_linear = model.t_embedder.mlp[0]
    assert not isinstance(embedder_linear, Int8ConvrotLinear)
    assert embedder_linear.weight.dtype is torch.float32
    # Dequantized *with* its scale and derotation, not merely cast: the whole point of widening it
    # here rather than dropping the marker.
    assert torch.corrcoef(torch.stack([embedder_linear.weight.flatten(), sensitive.flatten()]))[0, 1] > 0.999

    # Everything else still pays the one byte per weight this scheme exists for.
    assert isinstance(model.layers[0].proj, Int8ConvrotLinear)
    assert model.layers[0].proj.weight.dtype is torch.int8

    # And the reservation covers the widened layer at its post-split width, not at one byte.
    reserved = run.reserved[-1]
    assert reserved >= sensitive.nelement() * 4 + ordinary_q.nelement()


class _TinyAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(64, 128, bias=False)
        self.to_k = torch.nn.Linear(64, 128, bias=False)
        self.to_v = torch.nn.Linear(64, 128, bias=False)


class _TinyNativeBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _TinyAttention()
        self.adaLN_modulation = torch.nn.Sequential(torch.nn.Linear(64, 128))


class _TinyNvfp4TimestepEmbedder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.ModuleList([torch.nn.Linear(64, 128, bias=False)])


class _TinyNativeZImage(torch.nn.Module):
    """What the native-to-diffusers conversion makes of a checkpoint with a fused `attention.qkv`, beside the
    timestep embedder Z-Image declares precision-sensitive."""

    _skip_layerwise_casting_patterns = ["t_embedder", "cap_embedder"]

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.all_x_embedder = torch.nn.ModuleDict({"2-1": torch.nn.Linear(4, 4, bias=False)})
        self.t_embedder = _TinyNvfp4TimestepEmbedder()
        self.layers = torch.nn.ModuleList([_TinyNativeBlock()])


def _nvfp4_layer(
    path: str, positive: torch.Tensor, tile_row_scales: list[float], global_scale: float
) -> dict[str, torch.Tensor]:
    """A block scale constant over each 128-row tile row, which reads the same tiled as row by row --
    so the split under test can be checked without the de-swizzle being part of the expectation."""
    grid = torch.tensor(tile_row_scales).repeat_interleave(128).unsqueeze(1).repeat(1, positive.shape[1] // 16)
    return nvfp4_tensors(path, nvfp4_codes(positive), block_scale=grid, global_scale=global_scale)


def test_an_nvfp4_checkpoint_loads_packed_with_its_qkv_split_on_tile_rows(monkeypatch, tmp_path) -> None:
    """Comfy's nvfp4 Z-Image build quantizes the fused `attention.qkv` and names its layers only in the
    safetensors header. The packed tensors have to leave the state dict before the key conversion and the
    scaled-fp8 extraction -- which drops block scales whose weight is not float8 -- follow the QKV split on
    whole tile rows, and come back as `NVFP4Linear` modules for the strict load. The timestep embedder is
    decoded to the compute dtype instead, since its forward picks its activation dtype off its weight, and a
    bundled encoder's layers are dropped with everything else the transformer does not hold."""
    torch.manual_seed(4)
    qkv, modulation, timestep, bundled = (
        torch.randint(0, 2, (rows, 64), dtype=torch.bool) for rows in (384, 128, 128, 128)
    )
    modulation_bias = torch.randn(128)
    embedder = torch.randn(4, 4)
    state_dict = {
        "x_embedder.weight": embedder,
        **_nvfp4_layer("layers.0.attention.qkv", qkv, [1.0, 2.0, 4.0], global_scale=0.5),
        "layers.0.attention.qkv.input_scale": torch.tensor(1.0),
        **_nvfp4_layer("layers.0.adaLN_modulation.0", modulation, [2.0], global_scale=0.25),
        "layers.0.adaLN_modulation.0.bias": modulation_bias,
        **_nvfp4_layer("t_embedder.mlp.0", timestep, [2.0], global_scale=0.25),
        **_nvfp4_layer("text_encoders.qwen3.layers.0.mlp.up_proj", bundled, [2.0], global_scale=0.25),
    }
    header = {
        path: {"format": "nvfp4"}
        for path in (
            "layers.0.attention.qkv",
            "layers.0.adaLN_modulation.0",
            "t_embedder.mlp.0",
            "text_encoders.qwen3.layers.0.mlp.up_proj",
        )
    }
    run, config = _driver(monkeypatch, tmp_path, state_dict, model_class=_TinyNativeZImage)
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.z_image.read_safetensors_metadata",
        lambda _path, _logger: {"_quantization_metadata": json.dumps({"layers": header})},
    )
    # bf16 rather than the driver's float32, so a layer decoded or charged at any other width shows.
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.z_image.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.bfloat16,
    )

    model = run.load(config)

    bf16 = torch.bfloat16
    x = torch.randn(3, 64, dtype=bf16)
    attention = model.layers[0].attention
    for projection, signs, magnitude in (
        (attention.to_q, qkv[:128], 0.5),
        (attention.to_k, qkv[128:256], 1.0),
        (attention.to_v, qkv[256:], 2.0),
    ):
        assert isinstance(projection, NVFP4Linear)
        assert projection.weight.dtype is torch.uint8
        expected = torch.where(signs, magnitude, -magnitude).to(bf16)
        assert torch.equal(projection(x), torch.nn.functional.linear(x, expected))
    adaln = model.layers[0].adaLN_modulation[0]
    assert isinstance(adaln, NVFP4Linear)
    expected = torch.nn.functional.linear(x, torch.where(modulation, 0.5, -0.5).to(bf16), modulation_bias.to(bf16))
    assert torch.equal(adaln(x), expected)
    assert type(model.t_embedder.mlp[0]) is torch.nn.Linear
    assert torch.equal(model.t_embedder.mlp[0].weight, torch.where(timestep, 0.5, -0.5).to(bf16))
    assert torch.equal(model.all_x_embedder["2-1"].weight, embedder.to(bf16))

    # One reservation for what the model ends up holding: the packed tensors as stored, and the decoded embedder
    # and the dense rest at two bytes. Not the bundled layer, and not the packed layers at their decoded size,
    # which would ask for about 45 KB more.
    packed = (384 + 128) * 32 + (384 + 128) * 4
    dense = 128 * 64 * 2 + 128 * 2 + 4 * 4 * 2
    assert len(run.reserved) == 1
    reserved = run.reserved[0]
    assert packed + dense <= reserved < packed + dense + 1024


def test_a_scaled_fp8_checkpoint_stays_packed_when_storage_is_on(monkeypatch, tmp_path) -> None:
    """The keep decision has to reach the loader, not only the helper it was extracted into.

    With FP8 Storage on, the checkpoint's own scaled fp8 weights are kept rather than folded into
    bf16 and re-quantized by the layerwise cast, which has no scale to apply. A loader that goes
    back to asking only about the fp8 matmul folds them again -- silently, and precisely on the
    hardware most users have, since `fp8_compute` is off by default.
    """
    torch.manual_seed(3)
    original = torch.randn(4, CONVROT_GROUP_SIZE)
    fp8 = quantize_scaled_fp8(original)
    state_dict = {"layers.0.proj.weight": fp8.codes, "layers.0.proj.weight_scale": fp8.scale}

    run, driver_config = _driver(monkeypatch, tmp_path, state_dict)
    config = Main_Checkpoint_ZImage_Config.model_construct(
        path=driver_config.path,
        name="z-image",
        default_settings=MainModelDefaultSettings(fp8_storage=True),
    )
    # The configuration this path exists for: no fp8 matmul, but a device that can hold fp8.
    monkeypatch.setattr(load_default, "should_keep_fp8_weights", lambda _device: False)
    monkeypatch.setattr(load_default, "_device_supports_fp8_storage", lambda _device, _logger=None: True)

    model = run.load(config)

    proj = model.layers[0].proj
    assert proj.weight.dtype is torch.float8_e4m3fn, "folded back to a float dtype"
    assert getattr(proj, "weight_scale", None) is not None, "kept packed but without its scale"
    dequantized = (proj.weight.float() * proj.weight_scale).flatten()
    assert torch.corrcoef(torch.stack([dequantized, original.flatten()]))[0, 1] > 0.999


def test_a_comfyui_prefixed_checkpoint_loads_the_same_as_a_bare_one(monkeypatch, tmp_path) -> None:
    """The prefix strip used to be eight lines inline in this method, and nothing here noticed if it
    went away: a redistribution wrapping every key in `model.diffusion_model.` is accepted by the
    config probes, so it reaches this loader, and unstripped it hands the model a state dict in which
    every key is unexpected. Asserted against the bare load rather than a key list, so the two cannot
    drift apart."""
    torch.manual_seed(0)
    bare = {key: value.clone() for key, value in _TinyZImage().state_dict().items()}
    prefixed = {f"model.diffusion_model.{key}": value for key, value in bare.items()}

    run, config = _driver(monkeypatch, tmp_path, prefixed)
    from_prefixed = run.load(config).state_dict()
    run, config = _driver(monkeypatch, tmp_path, bare)
    from_bare = run.load(config).state_dict()

    assert set(from_prefixed) == set(from_bare)
    for key, value in from_bare.items():
        assert torch.equal(from_prefixed[key], value), key


def test_an_nvfp4_layer_missing_its_global_scale_is_refused_before_the_cache_is_evicted(monkeypatch, tmp_path) -> None:
    """The degraded half-state, at this seam rather than at the detector.

    A packed uint8 weight with a block-scale grid and no `weight_scale_2` is the shape a guard keyed
    on `weight_scale_2` -- the key the decode keys on -- lets straight through. `_find_nvfp4_layers`
    refuses it and `test_nvfp4.py` pins that; what only a seam can answer is whether this loader
    reaches the detector before it asks the cache for room, which nothing but the order of those two
    statements secures.
    """
    state_dict = {
        "x_embedder.weight": torch.randn(4, 4),
        "layers.0.attention.qkv.weight": torch.zeros(384, 32, dtype=torch.uint8),
        "layers.0.attention.qkv.weight_scale": torch.zeros(384, 4).to(torch.float8_e4m3fn),
    }
    run, config = _driver(monkeypatch, tmp_path, state_dict)

    with pytest.raises(ValueError, match="with a weight_scale but no weight_scale_2"):
        run.load(config)

    assert run.reserved == []
