"""The packed int4 convrot layouts: ``convrot_w4a4``, ``int4_tensorwise`` and, via GGUF, ``Q4_CR``.

They share the int8 scheme's scale and rotation and differ in one thing a decoder can get wrong without any
shape or dtype check noticing: two signed codes per byte, the LOW nibble first. Read in nvfp4's order, or
without the rotation, a layer still has the right shape and roughly the right magnitude.
"""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.full_layer import FullLayer
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes, requires_sidecar_patching
from invokeai.backend.quantization.int8_convrot import (
    CONVROT_GROUP_SIZE,
    ConvrotLayout,
    Int4ConvrotLinear,
    extract_int8_convrot_markers,
    marker_layout,
    unpack_int4_codes,
)
from tests.fixtures.device_allocations import PeakDeviceBytes
from tests.fixtures.quantized_payloads import (
    comfy_quant_marker,
    convrot_w4a4_marker,
    int4_tensorwise_marker,
    quantize_convrot_int4,
)

FIXTURE = Path(__file__).parent / "data" / "z_image_turbo_convrot_w4a4_slices.safetensors"


def _swap_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """The same bytes read in nvfp4's order, high nibble first."""
    u = packed.view(torch.uint8)
    return ((u >> 4) | (u << 4)).view(torch.int8)


def test_every_byte_unpacks_to_its_two_signed_nibbles_low_first() -> None:
    packed = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.int8).view(1, 256)

    codes = unpack_int4_codes(packed)

    def signed(nibble: int) -> int:
        return nibble - 16 if nibble >= 8 else nibble

    expected = [value for byte in range(256) for value in (signed(byte & 0x0F), signed(byte >> 4))]
    assert codes.dtype is torch.int8
    assert codes.tolist() == [expected]


def test_the_module_decodes_a_comfy_kitchen_payload_to_its_round_trip() -> None:
    torch.manual_seed(0)
    payload = quantize_convrot_int4(torch.randn(48, 2 * CONVROT_GROUP_SIZE))
    module = Int4ConvrotLinear(payload.packed, payload.scale, convrot=True)

    decoded = module._dequantized_weight(torch.device("cpu"), torch.float32)

    assert (module.out_features, module.in_features) == (48, 2 * CONVROT_GROUP_SIZE)
    assert torch.allclose(decoded, payload.dequantized, atol=1e-5)
    x = torch.randn(5, 2 * CONVROT_GROUP_SIZE)
    assert torch.allclose(module(x), torch.nn.functional.linear(x, payload.dequantized), atol=1e-4)


@pytest.mark.parametrize("layer", ["layers.0.attention.out", "layers.0.feed_forward.w2"])
def test_a_real_checkpoint_slice_decodes_to_its_bf16_source(layer: str) -> None:
    """JoaoZaokk/Z-Image-Turbo-W4A4-ConvRot at revision 71a4af3c75208746513ed4bca0e067b3fbf7d080 (Apache-2.0):
    the first rows of two `convrot_w4a4` layers of `zimage_turbo_w4a4.safetensors`, beside the same rows of
    Tongyi-MAI/Z-Image-Turbo's transformer (Apache-2.0) in bf16. A row slice decodes exactly as it does in the
    full matrix, because the rotation acts within each row.

    Measured when cut: cosine 0.984 for both, the 4-bit floor -- and 0.98-0.99 on Krea-2 Turbo builds of all
    three spellings, which are not redistributable. The two misreadings this pins land near zero: the nibbles
    in nvfp4's order, and the scale applied without undoing the rotation.
    """
    fixture = load_file(FIXTURE)
    packed, scale = fixture[f"{layer}.weight"], fixture[f"{layer}.weight_scale"]
    reference = fixture[f"{layer}.bf16_reference"].float().flatten()

    def cosine(weight: torch.Tensor) -> float:
        return torch.cosine_similarity(weight.flatten(), reference, dim=0).item()

    decoded = Int4ConvrotLinear(packed, scale, convrot=True)._dequantized_weight(torch.device("cpu"), torch.float32)
    assert cosine(decoded) > 0.98
    assert (decoded.norm() / reference.norm()).item() == pytest.approx(1.0, abs=0.05)

    swapped = Int4ConvrotLinear(_swap_nibbles(packed), scale, convrot=True)
    assert abs(cosine(swapped._dequantized_weight(torch.device("cpu"), torch.float32))) < 0.1
    unrotated = Int4ConvrotLinear(packed, scale, convrot=False)
    assert abs(cosine(unrotated._dequantized_weight(torch.device("cpu"), torch.float32))) < 0.2


def test_the_module_registers_the_checkpoints_own_keys_and_reports_its_unpacked_width() -> None:
    """`load_state_dict` assigns the file's tensors straight into these buffers, and LoRA sidecars and the model
    cache's patch shapes are built from `in_features`, which has to be the width the layer computes with."""
    torch.manual_seed(1)
    payload = quantize_convrot_int4(torch.randn(16, CONVROT_GROUP_SIZE))

    module = Int4ConvrotLinear(payload.packed, payload.scale, convrot=True, bias=torch.randn(16))

    assert set(module.state_dict()) == {"weight", "weight_scale", "bias"}
    assert module.state_dict()["weight"].shape == (16, CONVROT_GROUP_SIZE // 2)
    assert module.in_features == CONVROT_GROUP_SIZE


def test_a_lora_and_a_full_layer_patch_ride_as_sidecars_over_the_packed_weight() -> None:
    """Patches cannot be written into packed buffers, so the model has to be recognised as needing sidecars once the
    model cache has wrapped the module, and patches that are not LoRAs have to be handed the unpacked weight shape
    to be reshaped to -- half the stored width would not reshape at all."""
    torch.manual_seed(2)
    payload = quantize_convrot_int4(torch.randn(12, CONVROT_GROUP_SIZE))
    bias = torch.randn(12)
    model = torch.nn.Module()
    model.proj = Int4ConvrotLinear(payload.packed, payload.scale, convrot=True, bias=bias)
    apply_custom_layers_to_model(model)
    up, down, full = torch.randn(12, 3), torch.randn(3, CONVROT_GROUP_SIZE), torch.randn(12, CONVROT_GROUP_SIZE)
    lora = ModelPatchRaw(layers={"lora-proj": LoRALayer(up=up, mid=None, down=down, alpha=None, bias=None)})
    full_diff = ModelPatchRaw(layers={"lora-proj": FullLayer(weight=full, bias=None)})
    x = torch.randn(3, CONVROT_GROUP_SIZE)
    dense = torch.nn.functional.linear(x, payload.dequantized, bias)

    with LayerPatcher.apply_smart_model_patches(
        model=model,
        patches=[(lora, 0.75), (full_diff, 0.5)],
        prefix="lora-",
        dtype=torch.float32,
        force_sidecar_patching=requires_sidecar_patching(model, ModelFormat.Checkpoint),
    ):
        patched = model.proj(x)

    expected = torch.nn.functional.linear(x, payload.dequantized + 0.75 * (up @ down) + 0.5 * full, bias)
    assert torch.allclose(patched, expected, atol=1e-4)
    assert model.proj.get_num_patches() == 0
    assert torch.allclose(model.proj(x), dense, atol=1e-5)


@pytest.mark.parametrize(
    ("marker", "layout"),
    [
        pytest.param(convrot_w4a4_marker(), ConvrotLayout(True, True, 256), id="convrot_w4a4 is rotated by definition"),
        pytest.param(int4_tensorwise_marker(), ConvrotLayout(True, True, 256), id="int4_tensorwise"),
        pytest.param(
            int4_tensorwise_marker(convrot=False, group_size=64), ConvrotLayout(True, False, 64), id="int4 unrotated"
        ),
        pytest.param({"format": "int8_tensorwise"}, ConvrotLayout(False, False, 256), id="int8 without convrot"),
    ],
)
def test_a_marker_declares_its_layout(marker: dict, layout: ConvrotLayout) -> None:
    assert marker_layout(marker) == layout


class TestMarkerExtraction:
    @staticmethod
    def _sd(weight_dtype: torch.dtype = torch.int8, marker: dict | None = None) -> dict:
        sd = {"proj.weight": torch.zeros(4, 128, dtype=weight_dtype), "proj.weight_scale": torch.ones(4)}
        if marker is not None:
            sd["proj.comfy_quant"] = comfy_quant_marker(marker)
        return sd

    def test_int4_markers_are_left_alone_unless_the_loader_opts_in(self) -> None:
        sd = self._sd(marker=convrot_w4a4_marker())

        assert extract_int8_convrot_markers(sd) == {}
        assert "proj.comfy_quant" in sd
        assert extract_int8_convrot_markers(sd, accept_int4=True) == {"proj": convrot_w4a4_marker()}
        assert "proj.comfy_quant" not in sd

    def test_a_layer_the_header_alone_names_is_claimed(self) -> None:
        """The Starnodes converter deletes every per-tensor marker and writes the header only."""
        sd = self._sd()

        markers = extract_int8_convrot_markers(sd, accept_int4=True, header_layers={"proj": convrot_w4a4_marker()})

        assert markers == {"proj": convrot_w4a4_marker()}

    def test_a_per_tensor_marker_wins_over_the_header_even_one_the_loader_does_not_claim(self) -> None:
        sd = self._sd(marker={"format": "asym_w4a8_int8"})

        markers = extract_int8_convrot_markers(sd, accept_int4=True, header_layers={"proj": convrot_w4a4_marker()})

        assert markers == {}

    def test_a_header_entry_for_a_weight_that_is_not_int8_is_stale_and_ignored(self) -> None:
        sd = self._sd(weight_dtype=torch.bfloat16)

        assert extract_int8_convrot_markers(sd, accept_int4=True, header_layers={"proj": convrot_w4a4_marker()}) == {}

    @pytest.mark.parametrize("fmt", ["int8_tensorwise", "int4_tensorwise"])
    def test_a_header_entry_that_does_not_say_whether_the_layer_was_rotated_is_refused(self, fmt: str) -> None:
        """Comfy-Org's own exports write header entries without `convrot` beside per-tensor markers that set it,
        so a header-only file of that origin would decode as unrotated noise."""
        with pytest.raises(ValueError, match="does not say whether the weight was rotated"):
            extract_int8_convrot_markers(self._sd(), accept_int4=True, header_layers={"proj": {"format": fmt}})


def test_an_int8_marker_on_a_packed_payload_is_refused_at_the_swap() -> None:
    """A strict load assigns the quantized buffers whatever shape the file has, so nothing after the swap would
    notice an int4 payload declared as int8 until its forward multiplies half a weight."""
    from invokeai.backend.quantization.int8_convrot import swap_in_int8_linears

    model = torch.nn.Module()
    model.proj = torch.nn.Linear(CONVROT_GROUP_SIZE, 8, bias=False)
    payload = quantize_convrot_int4(torch.randn(8, CONVROT_GROUP_SIZE))
    sd = {"proj.weight": payload.packed, "proj.weight_scale": payload.scale}

    with pytest.raises(ValueError, match=r"decodes to \(8, 128\).*\(8, 256\)"):
        swap_in_int8_linears(model, sd, {"proj": {"format": "int8_tensorwise", "convrot": True}})

    swap_in_int8_linears(model, sd, {"proj": int4_tensorwise_marker()})
    assert isinstance(model.proj, Int4ConvrotLinear)


@pytest.mark.parametrize("streamed", [True, False], ids=["streamed", "resident"])
@pytest.mark.parametrize("convrot", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_transient_estimate_is_the_decode_peak_with_the_payload_streamed(
    streamed: bool, convrot: bool, dtype: torch.dtype
) -> None:
    """The node reserves working memory before partial loading decides which layers stay in RAM, so the estimate
    has to hold for a layer whose packed payload is copied to the compute device per call: unrotated, that copy is
    still alive while the unpack peaks. A resident layer peaks at or below it. The meta device stands in for the
    accelerator."""
    device = torch.device("meta")
    torch.manual_seed(3)
    payload = quantize_convrot_int4(torch.randn(64, 2 * CONVROT_GROUP_SIZE))
    model = torch.nn.Module()
    model.proj = Int4ConvrotLinear(
        payload.packed if streamed else payload.packed.to(device),
        payload.scale if streamed else payload.scale.to(device),
        convrot=convrot,
    )
    model.proj._dequantized_weight(device, dtype)  # Mints the shared Hadamard matrix.

    with PeakDeviceBytes(device) as allocations:
        model.proj._dequantized_weight(device, dtype)

    estimate = peak_dequant_transient_bytes(model, dtype)
    if streamed:
        assert allocations.peak == estimate
    else:
        assert allocations.peak <= estimate


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="measures GPU allocations")
@pytest.mark.parametrize("streamed", [True, False], ids=["streamed", "resident"])
@pytest.mark.parametrize("convrot", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_transient_estimate_covers_what_the_per_forward_decode_allocates(
    streamed: bool, convrot: bool, dtype: torch.dtype
) -> None:
    """The node reserves this much working memory for the decode, and the model's resident size does not include
    it, so an estimate that falls short lets the first forward compete with weights the cache just placed. Runs on
    ROCm too, where `torch.cuda` is the HIP device.

    Measured in requested bytes: peak *allocated* bytes count whole cached blocks, which blocks left over by
    earlier tests in the same process can inflate. The estimate counts the payload as streamed, so a resident
    unrotated layer comes in at (itemsize + 1) / (itemsize + 1.5) of it."""
    device = torch.device("cuda")
    torch.manual_seed(3)
    payload = quantize_convrot_int4(torch.randn(1024, 4096))
    model = torch.nn.Module()
    model.proj = Int4ConvrotLinear(
        payload.packed if streamed else payload.packed.to(device),
        payload.scale if streamed else payload.scale.to(device),
        convrot=convrot,
    )
    model.proj._dequantized_weight(device, dtype)  # Mints the shared Hadamard matrix.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_stats()["requested_bytes.all.current"]

    weight = model.proj._dequantized_weight(device, dtype)
    torch.cuda.synchronize()

    measured = torch.cuda.memory_stats()["requested_bytes.all.peak"] - baseline
    estimate = peak_dequant_transient_bytes(model, dtype)
    assert weight.dtype is dtype
    assert (0.9 * estimate if streamed or convrot else 0.8 * estimate) <= measured <= estimate


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU device")
def test_buffers_left_in_ram_stream_per_call_and_decode_like_resident_ones() -> None:
    """Partial loading leaves some packed layers in RAM, and their forward has to move them to the input's device
    -- packed, before unpacking -- and compute exactly what the resident layer computes."""
    device = torch.device("cuda")
    torch.manual_seed(4)
    payload = quantize_convrot_int4(torch.randn(64, CONVROT_GROUP_SIZE))
    model = torch.nn.Module()
    model.proj = Int4ConvrotLinear(payload.packed, payload.scale, convrot=True, bias=torch.randn(64))
    apply_custom_layers_to_model(model, device_autocasting_enabled=True)
    x = torch.randn(3, CONVROT_GROUP_SIZE, device=device, dtype=torch.bfloat16)

    streamed = model.proj(x)
    model.to(device)
    resident = model.proj(x)

    assert streamed.device == x.device
    assert torch.equal(streamed, resident)
