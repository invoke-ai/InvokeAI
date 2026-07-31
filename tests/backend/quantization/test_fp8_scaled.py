import pytest
import torch

from invokeai.backend.quantization.fp8_scaled import (
    FP8_DTYPE,
    Fp8ScaledLayer,
    attach_fp8_scales,
    dequantize_fp8_scaled,
    dequantize_weight,
    device_supports_fp8_matmul,
    extract_fp8_scaled_layers,
    parse_quantization_metadata,
    scaled_mm_linear,
    set_fp8_matmul_enabled,
)

cuda_fp8 = pytest.mark.skipif(
    not (torch.cuda.is_available() and device_supports_fp8_matmul(torch.device("cuda"))),
    reason="requires a CUDA device with fp8 tensor cores (SM 8.9+)",
)


def _fp8_weight(out_f: int, in_f: int, per_channel: bool = False):
    w = torch.randn(out_f, in_f, dtype=torch.bfloat16) * 0.02
    if per_channel:
        scale = (w.abs().amax(dim=1) / torch.finfo(FP8_DTYPE).max).float().clamp(min=1e-12)
        q = (w / scale.reshape(-1, 1)).to(FP8_DTYPE)
    else:
        scale = (w.abs().amax() / torch.finfo(FP8_DTYPE).max).float().clamp(min=1e-12)
        q = (w / scale).to(FP8_DTYPE)
    return q, scale


class TestExtract:
    def test_pops_scales_and_keys_layer_by_path(self):
        q, scale = _fp8_weight(32, 16)
        sd = {"blk.0.lin.weight": q, "blk.0.lin.weight_scale": scale, "blk.0.lin.bias": torch.zeros(32)}
        layers = extract_fp8_scaled_layers(sd)
        assert set(layers) == {"blk.0.lin"}
        assert "blk.0.lin.weight_scale" not in sd, "scale keys must be removed so the sd loads cleanly"
        assert sd["blk.0.lin.weight"].dtype == FP8_DTYPE, "weights must stay quantized"

    def test_accepts_scale_weight_suffix(self):
        q, scale = _fp8_weight(32, 16)
        sd = {"lin.weight": q, "lin.scale_weight": scale}
        assert set(extract_fp8_scaled_layers(sd)) == {"lin"}

    def test_input_scale_is_captured(self):
        q, scale = _fp8_weight(32, 16)
        sd = {"lin.weight": q, "lin.weight_scale": scale, "lin.input_scale": torch.tensor(0.5)}
        assert extract_fp8_scaled_layers(sd)["lin"].input_scale == pytest.approx(0.5)

    def test_scale_without_fp8_weight_is_dropped(self):
        """A scale applied to an already-dequantized weight would corrupt it."""
        sd = {"lin.weight": torch.randn(32, 16, dtype=torch.bfloat16), "lin.weight_scale": torch.tensor(2.0)}
        assert extract_fp8_scaled_layers(sd) == {}
        assert "lin.weight_scale" not in sd

    def test_strips_stray_marker_keys(self):
        q, scale = _fp8_weight(32, 16)
        sd = {
            "lin.weight": q,
            "lin.weight_scale": scale,
            "scaled_fp8": torch.tensor(0.0),
            "comfy_quant_x": torch.tensor(1),
        }
        extract_fp8_scaled_layers(sd)
        assert set(sd) == {"lin.weight"}

    def test_full_precision_hint_from_metadata(self):
        q, scale = _fp8_weight(32, 16)
        sd = {"a.weight": q, "a.weight_scale": scale, "b.weight": q.clone(), "b.weight_scale": scale.clone()}
        meta = {"_quantization_metadata": '{"layers": {"a": {"full_precision_matrix_mult": true}, "b": {}}}'}
        layers = extract_fp8_scaled_layers(sd, meta)
        assert layers["a"].full_precision_matmul is True
        assert layers["b"].full_precision_matmul is False

    def test_malformed_metadata_is_ignored(self):
        assert parse_quantization_metadata({"_quantization_metadata": "not json"}) == {}
        assert parse_quantization_metadata(None) == {}

    def test_layer_hints_override_metadata(self):
        q, scale = _fp8_weight(32, 16)
        sd = {"renamed.lin.weight": q, "renamed.lin.weight_scale": scale}
        # Metadata uses the pre-rename path, so only the explicitly remapped hints can match.
        meta = {"_quantization_metadata": '{"layers": {"native.lin": {"full_precision_matrix_mult": true}}}'}
        layers = extract_fp8_scaled_layers(
            dict(sd), metadata=meta, layer_hints={"renamed.lin": {"full_precision_matrix_mult": True}}
        )
        assert layers["renamed.lin"].full_precision_matmul is True

        # Without the remap the flag silently matches nothing - the regression this guards against.
        layers = extract_fp8_scaled_layers(dict(sd), metadata=meta)
        assert layers["renamed.lin"].full_precision_matmul is False


class TestKrea2MetadataRemap:
    def test_native_layer_paths_are_remapped_like_the_state_dict(self):
        """The quantization metadata names layers natively; the scales are keyed after renaming."""
        from invokeai.backend.model_manager.load.model_loaders.krea2 import _remap_native_layer_paths

        mapping = _remap_native_layer_paths(
            ["blocks.0.attn.wq", "blocks.0.attn.wo", "blocks.3.mlp.down", "txtfusion.refiner_blocks.1.attn.wk"]
        )
        assert mapping["blocks.0.attn.wq"] == "transformer_blocks.0.attn.to_q"
        assert mapping["blocks.0.attn.wo"] == "transformer_blocks.0.attn.to_out.0"
        assert mapping["blocks.3.mlp.down"] == "transformer_blocks.3.ff.down"
        assert mapping["txtfusion.refiner_blocks.1.attn.wk"] == "text_fusion.refiner_blocks.1.attn.to_k"


class TestDequantize:
    @pytest.mark.parametrize("per_channel", [False, True])
    def test_roundtrip_close_to_original(self, per_channel: bool):
        w = torch.randn(64, 32, dtype=torch.bfloat16) * 0.02
        scale_src = w.abs().amax(dim=1) if per_channel else w.abs().amax()
        scale = (scale_src / torch.finfo(FP8_DTYPE).max).float().clamp(min=1e-12)
        q = (w / (scale.reshape(-1, 1) if per_channel else scale)).to(FP8_DTYPE)

        out = dequantize_weight(q, scale, torch.bfloat16)
        assert out.dtype == torch.bfloat16
        assert ((out.float() - w.float()).norm() / w.float().norm()).item() < 0.05

    def test_missing_scale_is_a_plain_cast(self):
        """fp8_storage layerwise casting produces scale-free fp8 weights."""
        q = torch.randn(8, 8, dtype=torch.bfloat16).to(FP8_DTYPE)
        assert torch.equal(dequantize_weight(q, None, torch.bfloat16), q.to(torch.bfloat16))

    def test_state_dict_dequantization_matches_helper(self):
        q, scale = _fp8_weight(64, 32)
        sd = {"lin.weight": q}
        layers = {"lin": Fp8ScaledLayer(weight_scale=scale)}
        dequantize_fp8_scaled(sd, layers)
        assert torch.equal(sd["lin.weight"], dequantize_weight(q, scale, torch.bfloat16))


class TestAttach:
    def test_registers_non_persistent_buffers(self):
        lin = torch.nn.Linear(16, 32, bias=False)
        q, scale = _fp8_weight(32, 16)
        lin.weight = torch.nn.Parameter(q, requires_grad=False)
        model = torch.nn.Sequential(lin)

        assert attach_fp8_scales(model, {"0": Fp8ScaledLayer(weight_scale=scale, full_precision_matmul=True)}) == 1
        assert torch.equal(model[0].weight_scale, scale)
        assert model[0]._fp8_full_precision_matmul is True
        # Re-saving a model whose scales landed in state_dict() would double-scale on reload.
        assert "0.weight_scale" not in model.state_dict()

    def test_skips_non_fp8_modules(self):
        model = torch.nn.Sequential(torch.nn.Linear(16, 32))
        assert attach_fp8_scales(model, {"0": Fp8ScaledLayer(weight_scale=torch.tensor(1.0))}) == 0


@cuda_fp8
class TestScaledMm:
    @pytest.mark.parametrize("per_channel", [False, True])
    @pytest.mark.parametrize("tokens", [64, 100])  # 100 is deliberately not a multiple of 16
    def test_matches_reference_linear(self, per_channel: bool, tokens: int):
        dev = torch.device("cuda")
        q, scale = _fp8_weight(256, 128, per_channel)
        q, scale = q.to(dev), scale.to(dev)
        w_ref = dequantize_weight(q, scale, torch.bfloat16)
        x = torch.randn(tokens, 128, device=dev, dtype=torch.bfloat16)

        got = scaled_mm_linear(x, q, scale)
        expected = torch.nn.functional.linear(x, w_ref)

        assert got.shape == expected.shape
        rel = ((got.float() - expected.float()).norm() / expected.float().norm()).item()
        assert rel < 0.08, f"relative error {rel:.4f} too high"

    def test_preserves_leading_dims_and_bias(self):
        dev = torch.device("cuda")
        q, scale = _fp8_weight(64, 32)
        q, scale = q.to(dev), scale.to(dev)
        bias = torch.randn(64, device=dev, dtype=torch.bfloat16)
        x = torch.randn(2, 48, 32, device=dev, dtype=torch.bfloat16)

        got = scaled_mm_linear(x, q, scale, bias)
        expected = torch.nn.functional.linear(x, dequantize_weight(q, scale, torch.bfloat16), bias)
        assert got.shape == (2, 48, 64)
        assert ((got.float() - expected.float()).norm() / expected.float().norm()).item() < 0.08

    def test_static_input_scale_path(self):
        dev = torch.device("cuda")
        q, scale = _fp8_weight(64, 32)
        q, scale = q.to(dev), scale.to(dev)
        x = torch.randn(32, 32, device=dev, dtype=torch.bfloat16)
        static = (x.abs().amax() / torch.finfo(FP8_DTYPE).max).float()

        got = scaled_mm_linear(x, q, scale, input_scale=static)
        expected = torch.nn.functional.linear(x, dequantize_weight(q, scale, torch.bfloat16))
        assert ((got.float() - expected.float()).norm() / expected.float().norm()).item() < 0.08


class TestCustomLinearIntegration:
    """The fp8 matmul must be opt-in and must degrade to the dequantized path, never raise."""

    def _module(self, device: torch.device, in_f=64, out_f=128, device_autocasting: bool = False):
        from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
            apply_custom_layers_to_model,
        )

        lin = torch.nn.Linear(in_f, out_f, bias=False)
        q, scale = _fp8_weight(out_f, in_f)
        lin.weight = torch.nn.Parameter(q.to(device), requires_grad=False)
        lin.register_buffer("weight_scale", scale.to(device), persistent=False)
        model = torch.nn.Sequential(lin).to(device)
        apply_custom_layers_to_model(model, device_autocasting_enabled=device_autocasting)
        return model

    @cuda_fp8
    @pytest.mark.parametrize("device_autocasting", [False, True])
    def test_fp8_path_runs_regardless_of_device_autocasting(self, device_autocasting: bool):
        """`apply_custom_layers_to_model` leaves autocasting off for fully-resident models.

        A fp8 check that only lives in `_autocast_forward` is therefore skipped in the common case:
        `forward` falls through to the dtype-mismatch branch and silently dequantizes instead. This
        regression cost the entire speedup while every other test stayed green.
        """
        dev = torch.device("cuda")
        model = self._module(dev, device_autocasting=device_autocasting)
        x = torch.randn(32, 64, device=dev, dtype=torch.bfloat16)
        custom = model[0]

        calls = []
        original = custom._maybe_fp8_forward
        custom._maybe_fp8_forward = lambda inp: (calls.append(1), original(inp))[1]

        set_fp8_matmul_enabled(True)
        try:
            out = model(x)
        finally:
            set_fp8_matmul_enabled(False)
            custom._maybe_fp8_forward = original

        assert calls, "the fp8 branch was never consulted"
        assert out.shape == (32, 128)
        # And it must actually have taken the fp8 path, not just been asked.
        w_ref = dequantize_weight(custom.weight, custom.weight_scale, torch.bfloat16)
        assert not torch.equal(out, torch.nn.functional.linear(x, w_ref)), (
            "output is bit-identical to the dequantized path, so _scaled_mm did not run"
        )

    def test_disabled_by_default_uses_scaled_dequant(self):
        """With the matmul off, the weight must still be dequantized *with* its scale."""
        set_fp8_matmul_enabled(False)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._module(dev)
        x = torch.randn(32, 64, device=dev, dtype=torch.bfloat16)

        got = model(x)
        w_ref = dequantize_weight(model[0].weight, model[0].weight_scale, torch.bfloat16)
        expected = torch.nn.functional.linear(x, w_ref)
        assert torch.equal(got, expected)

    @cuda_fp8
    def test_full_precision_flag_forces_fallback(self):
        set_fp8_matmul_enabled(True)
        try:
            dev = torch.device("cuda")
            model = self._module(dev)
            model[0]._fp8_full_precision_matmul = True
            x = torch.randn(32, 64, device=dev, dtype=torch.bfloat16)

            got = model(x)
            w_ref = dequantize_weight(model[0].weight, model[0].weight_scale, torch.bfloat16)
            assert torch.equal(got, torch.nn.functional.linear(x, w_ref))
        finally:
            set_fp8_matmul_enabled(False)

    @cuda_fp8
    def test_sidecar_patched_fp8_layer_uses_the_fp8_matmul(self):
        """LoRAs on fp8 weights are force-routed to sidecar patching (layer_patcher), and the
        sidecar wrapper calls `_autocast_forward`. The fp8 branch must survive that route, and the
        LoRA residual must still be added on top."""
        from invokeai.backend.patches.layers.lora_layer import LoRALayer

        dev = torch.device("cuda")
        model = self._module(dev)
        custom = model[0]
        rank = 8
        lora = LoRALayer(
            up=torch.randn(custom.out_features, rank, device=dev, dtype=torch.bfloat16) * 0.05,
            mid=None,
            down=torch.randn(rank, custom.in_features, device=dev, dtype=torch.bfloat16) * 0.05,
            alpha=float(rank),
            bias=None,
        )
        custom.add_patch(lora, 1.0)
        x = torch.randn(32, 64, device=dev, dtype=torch.bfloat16)

        set_fp8_matmul_enabled(True)
        try:
            patched = model(x)
        finally:
            set_fp8_matmul_enabled(False)
        unpatched_ref = torch.nn.functional.linear(
            x, dequantize_weight(custom.weight, custom.weight_scale, torch.bfloat16)
        )

        assert patched.shape == (32, 128)
        # The patch must actually change the output...
        assert not torch.allclose(patched, unpatched_ref, atol=1e-2)
        # ...and the base contribution must still be roughly the fp8 linear, not garbage.
        residual = (lora.get_weight(1.0) * lora.scale()).to(torch.bfloat16)
        expected = unpatched_ref + torch.nn.functional.linear(x, residual)
        rel = ((patched.float() - expected.float()).norm() / expected.float().norm()).item()
        assert rel < 0.1, f"sidecar-patched fp8 output diverges by {rel:.4f}"

    @cuda_fp8
    def test_unaligned_features_fall_back(self):
        set_fp8_matmul_enabled(True)
        try:
            dev = torch.device("cuda")
            model = self._module(dev, in_f=60, out_f=120)  # not multiples of 16
            x = torch.randn(16, 60, device=dev, dtype=torch.bfloat16)
            assert model(x).shape == (16, 120)  # must not raise
        finally:
            set_fp8_matmul_enabled(False)

    @cuda_fp8
    def test_enabled_path_agrees_with_fallback(self):
        dev = torch.device("cuda")
        model = self._module(dev)
        x = torch.randn(48, 64, device=dev, dtype=torch.bfloat16)

        set_fp8_matmul_enabled(False)
        baseline = model(x)
        set_fp8_matmul_enabled(True)
        try:
            fp8 = model(x)
        finally:
            set_fp8_matmul_enabled(False)

        rel = ((fp8.float() - baseline.float()).norm() / baseline.float().norm()).item()
        assert rel < 0.1, f"fp8 matmul diverges from the dequantized path by {rel:.4f}"
