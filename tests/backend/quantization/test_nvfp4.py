import pytest
import torch

from invokeai.backend.quantization.dequantize_common import (
    FP8_STORAGE_SKIP_PATTERNS,
    read_declared_quantization_formats,
    resolve_target_dtype,
)
from invokeai.backend.quantization.nvfp4 import (
    FP4_E2M1_MAX,
    NVFP4_BLOCK_SIZE,
    dequantize_nvfp4,
    expand_block_scale,
    fp4_e2m1_lut,
    is_nvfp4_state_dict,
    unpack_fp4_e2m1,
)
from invokeai.backend.quantization.scaled_fp8 import dequantize_scaled_fp8

#: The 16 FP4 E2M1 code values, in code order. Bit 3 is the sign.
EXPECTED_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _pack(codes: torch.Tensor) -> torch.Tensor:
    """Pack FP4 codes (0-15) along the last dim, high nibble first — the layout NVFP4 uses."""
    even, odd = codes[..., 0::2], codes[..., 1::2]
    return ((even << 4) | odd).to(torch.uint8)


def _nvfp4_sd(weight: torch.Tensor, *, name: str = "layer") -> dict[str, torch.Tensor]:
    """Encode ``weight`` as an NVFP4 state dict the way a real exporter does.

    Per block of 16 along the last dim, the block scale is ``amax / 6`` so the largest magnitude maps
    to the largest FP4 code; the global scale carries whatever the E4M3 block scale cannot represent.
    """
    rows, cols = weight.shape
    blocks = weight.reshape(rows, cols // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    amax = blocks.abs().amax(-1, keepdim=True)
    block_scale = (amax / FP4_E2M1_MAX).clamp_min(torch.finfo(torch.float32).tiny)

    lut = fp4_e2m1_lut(torch.float32, weight.device)
    normalized = blocks / block_scale
    codes = (normalized[..., None] - lut).abs().argmin(-1).to(torch.uint8)

    return {
        f"{name}.weight": _pack(codes.reshape(rows, cols)),
        f"{name}.weight_scale": block_scale.squeeze(-1).to(torch.float8_e4m3fn),
        f"{name}.weight_scale_2": torch.tensor(1.0),
    }


class TestFp4E2m1Lut:
    def test_lut_matches_the_ocp_e2m1_encoding(self) -> None:
        assert fp4_e2m1_lut(torch.float32, torch.device("cpu")).tolist() == EXPECTED_LUT

    def test_max_magnitude_constant_matches_the_lut(self) -> None:
        assert FP4_E2M1_MAX == max(EXPECTED_LUT)

    def test_codes_are_not_their_own_values(self) -> None:
        # The E2M1 magnitudes are 0, .5, 1, 1.5, 2, 3, 4, 6 — not 0-7. Treating a code as its own
        # value silently distorts the top half of every block, so pin the divergence.
        assert EXPECTED_LUT[5] == 3.0
        assert EXPECTED_LUT[7] == 6.0


class TestUnpackFp4E2m1:
    def test_high_nibble_is_the_even_element(self) -> None:
        """The whole-model failure mode: reversing this swaps every adjacent pair without crashing.

        Verified against Krea-2 Turbo's bit-identical MXFP8 export — high-nibble-first gives 100%
        sign agreement and +0.996 per-block cosine, low-nibble-first gives 49.95% and -0.004.
        """
        lut = fp4_e2m1_lut(torch.float32, torch.device("cpu"))
        # 0x2A -> high nibble 0x2 (= 1.0), low nibble 0xA (= -1.0)
        unpacked = unpack_fp4_e2m1(torch.tensor([[0x2A]], dtype=torch.uint8), lut)
        assert unpacked.tolist() == [[1.0, -1.0]]

    def test_doubles_the_last_dim(self) -> None:
        lut = fp4_e2m1_lut(torch.float32, torch.device("cpu"))
        packed = torch.zeros((3, 8), dtype=torch.uint8)
        assert unpack_fp4_e2m1(packed, lut).shape == (3, 16)

    def test_consecutive_elements_are_adjacent_not_split_into_halves(self) -> None:
        # A split-halves layout (all low nibbles, then all high) would put these values in a
        # different order; independently confirmed on a real checkpoint, where the interleaved
        # layout puts the max code in 100% of scale blocks vs ~85% for split halves.
        lut = fp4_e2m1_lut(torch.float32, torch.device("cpu"))
        packed = torch.tensor([[0x12, 0x34]], dtype=torch.uint8)
        assert unpack_fp4_e2m1(packed, lut).tolist() == [[0.5, 1.0, 1.5, 2.0]]

    def test_rejects_non_uint8_weights(self) -> None:
        lut = fp4_e2m1_lut(torch.float32, torch.device("cpu"))
        with pytest.raises(ValueError, match="must be uint8"):
            unpack_fp4_e2m1(torch.zeros((2, 2), dtype=torch.int8), lut)


class TestExpandBlockScale:
    def test_expands_one_scale_per_block(self) -> None:
        scale = torch.tensor([[1.0, 2.0]])
        expanded = expand_block_scale(scale, 4, name="w")
        assert expanded.tolist() == [[1.0, 1.0, 2.0, 2.0]]

    def test_passes_through_a_fully_expanded_scale(self) -> None:
        scale = torch.tensor([[1.0, 2.0, 3.0]])
        assert expand_block_scale(scale, 3, name="w") is scale

    def test_rejects_a_scale_that_does_not_divide_the_weight(self) -> None:
        with pytest.raises(ValueError, match="does not evenly divide"):
            expand_block_scale(torch.ones((1, 3)), 8, name="some.weight")


class TestIsNvfp4StateDict:
    def test_detects_the_per_tensor_global_scale(self) -> None:
        assert is_nvfp4_state_dict({"a.weight": None, "a.weight_scale": None, "a.weight_scale_2": None})

    def test_does_not_match_scaled_fp8(self) -> None:
        # Both formats use `.weight_scale`; only NVFP4 adds `.weight_scale_2`. Misdetecting here is
        # what would send a packed 4-bit weight through the scaled-fp8 path.
        assert not is_nvfp4_state_dict({"a.weight": None, "a.weight_scale": None})
        assert not is_nvfp4_state_dict({"a.weight": None, "a.scale_weight": None})

    def test_tolerates_non_string_keys(self) -> None:
        assert not is_nvfp4_state_dict({0: None})


class TestDequantizeNvfp4:
    def test_round_trips_values_that_lie_on_the_fp4_grid(self) -> None:
        # Every value is exactly representable, so dequantization must reproduce it bit-for-bit.
        row = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, -0.5]])
        sd = _nvfp4_sd(row)
        assert dequantize_nvfp4(sd, torch.float32) == 1
        torch.testing.assert_close(sd["layer.weight"], row)

    def test_restores_the_logical_shape(self) -> None:
        weight = torch.randn(8, 32)
        sd = _nvfp4_sd(weight)
        assert sd["layer.weight"].shape == (8, 16)  # packed: half width
        dequantize_nvfp4(sd, torch.float32)
        assert sd["layer.weight"].shape == (8, 32)

    def test_approximates_arbitrary_weights_and_stays_finite(self) -> None:
        torch.manual_seed(0)
        weight = torch.randn(16, 64) * 0.05
        sd = _nvfp4_sd(weight)
        dequantize_nvfp4(sd, torch.float32)
        out = sd["layer.weight"]
        assert torch.isfinite(out).all()
        # 1 mantissa bit -> coarse, but the per-block direction must be preserved.
        cosine = torch.nn.functional.cosine_similarity(out.flatten(), weight.flatten(), dim=0)
        assert cosine > 0.99

    def test_applies_the_global_scale(self) -> None:
        sd = _nvfp4_sd(torch.tensor([[6.0] * 16]))
        sd["layer.weight_scale_2"] = torch.tensor(3.0)
        dequantize_nvfp4(sd, torch.float32)
        torch.testing.assert_close(sd["layer.weight"], torch.tensor([[18.0] * 16]))

    def test_removes_all_scale_keys(self) -> None:
        sd = _nvfp4_sd(torch.randn(4, 16))
        dequantize_nvfp4(sd, torch.float32)
        assert set(sd) == {"layer.weight"}

    def test_removes_orphan_scales_with_no_matching_weight(self) -> None:
        # Nothing can be multiplied by these, and leaving them behind would inflate the caller's
        # size accounting and reach load_state_dict as unexpected keys.
        sd = {"ghost.weight_scale": torch.ones(1), "ghost.weight_scale_2": torch.tensor(1.0)}
        assert dequantize_nvfp4(sd, torch.float32) == 0
        assert sd == {}

    def test_is_a_no_op_without_nvfp4_keys(self) -> None:
        sd = {"a.weight": torch.ones(2, 2)}
        assert dequantize_nvfp4(sd, torch.float32) == 0
        assert set(sd) == {"a.weight"}

    def test_honors_the_compute_dtype(self) -> None:
        sd = _nvfp4_sd(torch.randn(4, 16))
        dequantize_nvfp4(sd, torch.bfloat16)
        assert sd["layer.weight"].dtype == torch.bfloat16


class TestFp8StorageStreaming:
    """Dequantizing straight to fp8 is what keeps peak host RAM at the fp8 model size.

    Dequantizing everything to bf16 and re-quantizing afterwards is numerically identical but
    transiently needs the full bf16 model — 23.9 GiB for a 12.8 B-parameter Krea-2 transformer.
    """

    def test_nvfp4_stores_at_the_storage_dtype(self) -> None:
        sd = _nvfp4_sd(torch.randn(4, 16))
        dequantize_nvfp4(sd, torch.bfloat16, storage_dtype=torch.float8_e4m3fn)
        assert sd["layer.weight"].dtype == torch.float8_e4m3fn

    def test_scaled_fp8_stores_at_the_storage_dtype(self) -> None:
        sd = {
            "layer.weight": torch.randn(4, 8).to(torch.float8_e4m3fn),
            "layer.weight_scale": torch.tensor(2.0),
        }
        dequantize_scaled_fp8(sd, torch.bfloat16, storage_dtype=torch.float8_e4m3fn)
        assert sd["layer.weight"].dtype == torch.float8_e4m3fn

    def test_skip_patterns_preserve_precision_sensitive_weights(self) -> None:
        sd = {
            **_nvfp4_sd(torch.randn(4, 16), name="attn.to_q"),
            **_nvfp4_sd(torch.randn(4, 16), name="norm_out.linear"),
        }
        dequantize_nvfp4(sd, torch.bfloat16, storage_dtype=torch.float8_e4m3fn, skip_patterns=FP8_STORAGE_SKIP_PATTERNS)
        assert sd["attn.to_q.weight"].dtype == torch.float8_e4m3fn
        assert sd["norm_out.linear.weight"].dtype == torch.bfloat16

    def test_defaults_to_the_compute_dtype(self) -> None:
        sd = _nvfp4_sd(torch.randn(4, 16))
        dequantize_nvfp4(sd, torch.bfloat16)
        assert sd["layer.weight"].dtype == torch.bfloat16


class TestResolveTargetDtype:
    def test_returns_compute_dtype_when_no_storage_dtype_is_requested(self) -> None:
        assert resolve_target_dtype("a.b", torch.bfloat16, None, FP8_STORAGE_SKIP_PATTERNS) == torch.bfloat16

    def test_returns_storage_dtype_for_an_unskipped_name(self) -> None:
        target = resolve_target_dtype(
            "blocks.0.attn.wq", torch.bfloat16, torch.float8_e4m3fn, FP8_STORAGE_SKIP_PATTERNS
        )
        assert target == torch.float8_e4m3fn

    @pytest.mark.parametrize("name", ["blocks.0.prenorm", "pos_embed.proj", "patch_embed.proj", "proj_in"])
    def test_skips_precision_sensitive_names(self, name: str) -> None:
        target = resolve_target_dtype(name, torch.bfloat16, torch.float8_e4m3fn, FP8_STORAGE_SKIP_PATTERNS)
        assert target == torch.bfloat16


class TestReadDeclaredQuantizationFormats:
    def test_reads_the_declared_formats(self, tmp_path) -> None:
        import json

        from safetensors.torch import save_file

        path = tmp_path / "m.safetensors"
        metadata = {
            "_quantization_metadata": json.dumps(
                {"layers": {"blocks.0.attn.wq": {"format": "nvfp4"}, "blocks.1.mlp.up": {"format": "MXFP8"}}}
            )
        }
        save_file({"a": torch.ones(2)}, path, metadata=metadata)
        assert read_declared_quantization_formats(path) == {"nvfp4", "mxfp8"}

    def test_returns_empty_for_a_file_without_metadata(self, tmp_path) -> None:
        from safetensors.torch import save_file

        path = tmp_path / "m.safetensors"
        save_file({"a": torch.ones(2)}, path)
        assert read_declared_quantization_formats(path) == set()

    @pytest.mark.parametrize("name", ["empty.safetensors", "model.gguf", "missing.safetensors"])
    def test_never_raises_on_an_unreadable_file(self, tmp_path, name: str) -> None:
        # This only ever rejects a load, so it must not become a failure mode of its own.
        path = tmp_path / name
        if "missing" not in name:
            path.touch()
        assert read_declared_quantization_formats(path) == set()

    def test_returns_empty_for_malformed_metadata(self, tmp_path) -> None:
        from safetensors.torch import save_file

        path = tmp_path / "m.safetensors"
        save_file({"a": torch.ones(2)}, path, metadata={"_quantization_metadata": "{not json"})
        assert read_declared_quantization_formats(path) == set()
