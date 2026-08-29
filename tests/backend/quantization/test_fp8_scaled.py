import contextlib
import logging
from unittest import mock

import pytest
import torch

from invokeai.backend.quantization.fp8_scaled import (
    FP8_DTYPE,
    Fp8ScaledLayer,
    attach_fp8_scales,
    cast_state_dict,
    count_fp8_weights,
    dequantize_fp8_scaled,
    dequantize_weight,
    detach_layer_sidechannel,
    device_supports_fp8_matmul,
    expand_weight_scale,
    extract_comfy_quant_hints,
    extract_fp8_scaled_layers,
    is_matmul_usable_scale,
    is_scale_metadata_key,
    iter_weight_scale_pairs,
    parse_quantization_metadata,
    predict_cast_state_dict_size,
    reattach_layer_sidechannel,
    reset_fp8_matmul_support_cache,
    scaled_mm_linear,
    set_fp8_matmul_enabled,
    set_full_precision_hints_respected,
    split_fp8_scaled_layers,
    strip_layer_path_prefix,
    warn_on_unattached_scales,
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


def _comfy_quant_blob(full_precision: bool):
    """The per-layer marker some ComfyUI exports write instead of the header entry."""
    payload = f'{{"format": "float8_e4m3fn", "full_precision_matrix_mult": {"true" if full_precision else "false"}}}'
    return torch.tensor(list(payload.encode("utf-8")), dtype=torch.uint8)


class TestRawFp8:
    """Checkpoints that ship fp8 weights with no weight_scale at all."""

    def test_cast_state_dict_preserves_fp8_when_asked(self):
        q, _ = _fp8_weight(32, 16)
        sd = {"lin.weight": q, "lin.bias": torch.zeros(32), "norm.weight": torch.ones(16)}
        kept = cast_state_dict(sd, torch.bfloat16, keep_fp8=True)
        assert kept == 1
        assert sd["lin.weight"].dtype is FP8_DTYPE, "raw fp8 must survive the load"
        assert sd["lin.bias"].dtype is torch.bfloat16
        assert sd["norm.weight"].dtype is torch.bfloat16

    def test_cast_state_dict_dequantizes_when_matmul_unavailable(self):
        """Without the matmul, staying quantized costs a dequantize per forward for no gain."""
        q, _ = _fp8_weight(32, 16)
        sd = {"lin.weight": q}
        assert cast_state_dict(sd, torch.bfloat16, keep_fp8=False) == 0
        assert sd["lin.weight"].dtype is torch.bfloat16

    def test_e5m2_is_never_preserved(self):
        """scaled_mm cannot take e5m2 as the weight operand on Ada, so keeping it buys nothing."""
        sd = {"lin.weight": torch.zeros(32, 16).to(torch.float8_e5m2)}
        assert cast_state_dict(sd, torch.bfloat16, keep_fp8=True) == 0
        assert sd["lin.weight"].dtype is torch.bfloat16

    def test_only_linear_weights_stay_quantized(self):
        """The regression an end-to-end run caught: checkpoints exist that quantize *everything*.

        A Z-Image checkpoint had 243 of its 453 fp8 tensors 1-D — biases, norm weights, a learned
        pad token. Keeping those quantized saves nothing usable and breaks inference: the fp8 value
        flows into the activations and the next Linear receives an fp8 *input*, which dies in
        `x.abs()` with `"abs_cuda" not implemented for 'Float8_e4m3fn'`.
        """
        model = torch.nn.Sequential()
        model.add_module("lin", torch.nn.Linear(16, 32))
        model.add_module("norm", torch.nn.LayerNorm(32))
        q, _ = _fp8_weight(32, 16)
        sd = {
            "lin.weight": q,  # keep: a Linear weight the matmul can use
            "lin.bias": torch.zeros(32).to(FP8_DTYPE),  # dequantize: bias
            "norm.weight": torch.ones(32).to(FP8_DTYPE),  # dequantize: 1-D norm
            "pad_token": torch.zeros(1, 32).to(FP8_DTYPE),  # dequantize: not a module weight
        }
        assert cast_state_dict(sd, torch.bfloat16, keep_fp8=True, model=model) == 1
        assert sd["lin.weight"].dtype is FP8_DTYPE
        for key in ("lin.bias", "norm.weight", "pad_token"):
            assert sd[key].dtype is torch.bfloat16, f"{key} must not stay fp8"

    def test_skip_patterns_dequantize_named_modules(self):
        """Modules a model marks precision-sensitive must be dequantized even if they are Linears.

        Z-Image declares `_skip_layerwise_casting_patterns = ["t_embedder", "cap_embedder"]`, and
        `TimestepEmbedder.forward` casts its activations to `self.mlp[0].weight.dtype` — an fp8
        weight there makes the activations fp8.
        """
        model = torch.nn.Sequential()
        model.add_module("t_embedder", torch.nn.Linear(16, 32))
        model.add_module("blocks", torch.nn.Linear(16, 32))
        q, _ = _fp8_weight(32, 16)
        sd = {"t_embedder.weight": q, "blocks.weight": q.clone()}
        assert cast_state_dict(sd, torch.bfloat16, keep_fp8=True, model=model, skip_patterns=["t_embedder"]) == 1
        assert sd["t_embedder.weight"].dtype is torch.bfloat16
        assert sd["blocks.weight"].dtype is FP8_DTYPE

    def test_count_fp8_weights(self):
        model = torch.nn.Sequential(torch.nn.Linear(16, 32, bias=False), torch.nn.Linear(32, 16, bias=False))
        assert count_fp8_weights(model) == 0
        model[0].weight = torch.nn.Parameter(_fp8_weight(32, 16)[0], requires_grad=False)
        assert count_fp8_weights(model) == 1

    @cuda_fp8
    def test_fp8_weight_without_scale_uses_the_tensor_cores(self):
        """The runtime already supports scale-less fp8 — `weight_scale` is optional in
        `scaled_mm_linear`, and `_can_use_fp8_matmul` only requires the fp8 dtype. This pins that
        down so a future change cannot quietly make a scale mandatory."""
        from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
            apply_custom_layers_to_model,
        )

        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(64, 64, bias=False)).cuda()
        net[0].weight.data = (net[0].weight.data * 0.02).to(FP8_DTYPE)
        apply_custom_layers_to_model(net)
        lin = net[0]
        x = torch.randn(1, 32, 64, device="cuda", dtype=torch.bfloat16)

        set_fp8_matmul_enabled(True)
        try:
            assert getattr(lin, "weight_scale", None) is None
            assert lin._can_use_fp8_matmul(x) is True
            out = net(x)
        finally:
            set_fp8_matmul_enabled(None)

        reference = torch.nn.functional.linear(x, dequantize_weight(lin.weight, None, x.dtype))
        rel = ((out.float() - reference.float()).norm() / reference.float().norm()).item()
        assert torch.isfinite(out).all()
        assert rel < 0.1, f"unit-scaled fp8 matmul drifted too far from bf16: {rel:.4f}"


class TestInputScale:
    def test_calibrated_scale_is_kept(self):
        q, scale = _fp8_weight(32, 16)
        sd = {"lin.weight": q, "lin.weight_scale": scale, "lin.input_scale": torch.tensor(0.017)}
        layer = extract_fp8_scaled_layers(sd)["lin"]
        assert layer.input_scale is not None
        assert pytest.approx(layer.input_scale.item()) == 0.017
        assert "lin.input_scale" not in sd

    def test_scale_input_spelling_is_accepted(self):
        """`.scale_input` is the other spelling in the wild; ignoring it discards the calibration."""
        q, scale = _fp8_weight(32, 16)
        sd = {"lin.weight": q, "lin.scale_weight": scale, "lin.scale_input": torch.tensor(0.017)}
        layer = extract_fp8_scaled_layers(sd)["lin"]
        assert pytest.approx(layer.input_scale.item()) == 0.017
        assert not [k for k in sd if "scale_input" in k], "must be popped, not left for load_state_dict"

    @pytest.mark.parametrize(
        "value", [1.0, 0.0, -0.5, float("nan"), float("inf")], ids=["placeholder", "zero", "negative", "nan", "inf"]
    )
    def test_unusable_scales_fall_back_to_dynamic(self, value: float):
        """A 1.0 input_scale is an uncalibrated placeholder. Using it means *no* activation scaling,
        so everything above the fp8 max saturates -- strictly worse than the per-forward amax it
        would replace. Zero/negative/non-finite cannot be a divisor at all."""
        q, scale = _fp8_weight(32, 16)
        sd = {"lin.weight": q, "lin.weight_scale": scale, "lin.input_scale": torch.tensor(value)}
        assert extract_fp8_scaled_layers(sd)["lin"].input_scale is None

    @cuda_fp8
    def test_placeholder_scale_saturates_activations_above_the_fp8_range(self):
        """End-to-end cost of trusting a 1.0 placeholder.

        fp8_e4m3 is a floating-point format, so a scale factor does not buy relative precision the
        way it would for int8 — for activations inside +/-448 both paths are equivalent. The damage
        appears only above the representable range, where an unscaled cast clamps hard while the
        dynamic amax scale maps the whole tensor into range. Transformer activations do reach that
        regime, which is why a calibrated input_scale exists at all.
        """
        torch.manual_seed(0)
        q, scale = _fp8_weight(64, 64)
        q, scale = q.cuda(), scale.cuda()
        x = (torch.randn(32, 64, dtype=torch.bfloat16, device="cuda") * 2000.0).unsqueeze(0)
        assert x.abs().max() > 448, "test must exercise the saturating regime"
        reference = torch.nn.functional.linear(x, dequantize_weight(q, scale, x.dtype))

        def err(got: torch.Tensor) -> float:
            return ((got.float() - reference.float()).norm() / reference.float().norm()).item()

        dynamic = scaled_mm_linear(x, q, scale, None, input_scale=None)
        unscaled = scaled_mm_linear(x, q, scale, None, input_scale=torch.tensor(1.0, device="cuda"))
        assert err(dynamic) < err(unscaled) / 2, f"dynamic={err(dynamic):.4f} unscaled={err(unscaled):.4f}"


class TestComfyQuantHints:
    def test_reads_per_layer_markers_and_pops_them(self):
        sd = {
            "blk.0.lin.weight": _fp8_weight(32, 16)[0],
            "blk.0.lin.comfy_quant": _comfy_quant_blob(True),
            "blk.1.lin.comfy_quant": _comfy_quant_blob(False),
        }
        hints = extract_comfy_quant_hints(sd)
        assert hints["blk.0.lin"]["full_precision_matrix_mult"] is True
        assert hints["blk.1.lin"]["full_precision_matrix_mult"] is False
        assert not [k for k in sd if "comfy_quant" in k], "markers must not reach load_state_dict"

    def test_full_precision_flag_reaches_the_layer(self):
        """The regression this guards: a checkpoint carrying the flags *only* in per-layer markers
        had them silently ignored, so layers the producer marked unsafe were multiplied in fp8."""
        q, scale = _fp8_weight(32, 16)
        sd = {"blk.0.lin.weight": q, "blk.0.lin.weight_scale": scale, "blk.0.lin.comfy_quant": _comfy_quant_blob(True)}
        hints = extract_comfy_quant_hints(sd)
        layers = extract_fp8_scaled_layers(sd, layer_hints=hints)
        assert layers["blk.0.lin"].full_precision_matmul is True

        # Reading only the header (no hints) is what used to happen - the flag matches nothing.
        sd2 = {"blk.0.lin.weight": q, "blk.0.lin.weight_scale": scale, "blk.0.lin.comfy_quant": _comfy_quant_blob(True)}
        assert extract_fp8_scaled_layers(sd2).get("blk.0.lin").full_precision_matmul is False

    def test_nul_padded_and_malformed_blobs(self):
        padded = torch.cat([_comfy_quant_blob(True), torch.zeros(8, dtype=torch.uint8)])
        sd = {
            "a.comfy_quant": padded,
            "b.comfy_quant": torch.tensor(list(b"not json"), dtype=torch.uint8),
        }
        hints = extract_comfy_quant_hints(sd)
        assert hints["a"]["full_precision_matrix_mult"] is True
        # A malformed marker is a lost hint, never a failed load.
        assert "b" not in hints
        assert not sd


class TestQwen3VLKeyRemap:
    def test_scale_keys_and_hint_paths_land_on_the_same_module(self):
        """attach_fp8_scales resolves hint paths against the *model*, so the state-dict remap and the
        hint remap must agree - otherwise every recovered scale silently matches nothing."""
        from invokeai.backend.model_manager.load.model_loaders.krea2 import (
            _qwen3vl_target_key,
            _remap_qwen3vl_singlefile_keys,
        )

        q, scale = _fp8_weight(32, 16)
        sd = _remap_qwen3vl_singlefile_keys(
            {
                "model.layers.0.mlp.down_proj.weight": q,
                "model.layers.0.mlp.down_proj.weight_scale": scale,
                "model.visual.blocks.0.attn.qkv.weight": torch.zeros(16, 16),
            }
        )
        assert "language_model.layers.0.mlp.down_proj.weight_scale" in sd
        assert "visual.blocks.0.attn.qkv.weight" in sd

        hints = {_qwen3vl_target_key("model.layers.0.mlp.down_proj"): {"full_precision_matrix_mult": True}}
        layers = extract_fp8_scaled_layers(sd, layer_hints=hints)
        assert set(layers) == {"language_model.layers.0.mlp.down_proj"}
        assert layers["language_model.layers.0.mlp.down_proj"].full_precision_matmul is True


class TestFullPrecisionHintToggle:
    def test_marker_is_applied_by_default(self):
        q, scale = _fp8_weight(32, 16)
        module = torch.nn.Linear(16, 32, bias=False)
        module.weight = torch.nn.Parameter(q, requires_grad=False)
        model = torch.nn.Sequential()
        model.add_module("lin", module)
        attach_fp8_scales(model, {"lin": Fp8ScaledLayer(weight_scale=scale, full_precision_matmul=True)})
        assert module._fp8_full_precision_matmul is True

    def test_marker_suppressed_when_hints_are_off(self):
        """Turning the hints off must reach the module flag CustomLinear reads, not just the parse."""
        q, scale = _fp8_weight(32, 16)
        module = torch.nn.Linear(16, 32, bias=False)
        module.weight = torch.nn.Parameter(q, requires_grad=False)
        model = torch.nn.Sequential()
        model.add_module("lin", module)
        set_full_precision_hints_respected(False)
        try:
            attach_fp8_scales(model, {"lin": Fp8ScaledLayer(weight_scale=scale, full_precision_matmul=True)})
        finally:
            set_full_precision_hints_respected(None)
        assert module._fp8_full_precision_matmul is False


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


class TestSplitScaledLayers:
    """A scaled layer must never reach `cast_state_dict` still holding an unapplied scale."""

    def _model(self):
        model = torch.nn.Sequential()
        model.add_module("time_embed", torch.nn.Sequential())
        model.time_embed.add_module("linear_2", torch.nn.Linear(16, 32))
        model.add_module("attn", torch.nn.Linear(16, 32))
        return model

    def test_skip_pattern_layer_is_dequantized_with_its_scale(self):
        """The Krea-2 regression: `time_embed` matches the model's skip patterns, so the cast would
        turn its fp8 weight into bf16 *codes* — the scale silently dropped, the weight off by
        1/weight_scale, and `attach_fp8_scales` unable to repair it afterwards."""
        model = self._model()
        w = torch.randn(32, 16, dtype=torch.bfloat16) * 0.02
        scale = (w.abs().amax() / torch.finfo(FP8_DTYPE).max).float().clamp(min=1e-12)
        q = (w / scale).to(FP8_DTYPE)
        sd = {"time_embed.linear_2.weight": q, "attn.weight": q.clone()}
        layers = {
            "time_embed.linear_2": Fp8ScaledLayer(weight_scale=scale),
            "attn": Fp8ScaledLayer(weight_scale=scale),
        }

        remaining = split_fp8_scaled_layers(
            sd, layers, torch.bfloat16, model=model, skip_patterns=["time_embed", "norm"]
        )

        assert set(remaining) == {"attn"}, "only the layer that can stay quantized is left to attach"
        assert sd["time_embed.linear_2.weight"].dtype is torch.bfloat16
        rel = ((sd["time_embed.linear_2.weight"].float() - w.float()).norm() / w.float().norm()).item()
        assert rel < 0.05, f"the scale was not applied on the way down: rel-err {rel:.4f}"

        # And the survivor is untouched by the subsequent cast, so its scale still attaches.
        assert cast_state_dict(sd, torch.bfloat16, keep_fp8=True, model=model, skip_patterns=["time_embed"]) == 1
        model.load_state_dict(sd, assign=True, strict=False)
        assert attach_fp8_scales(model, remaining) == 1
        assert torch.equal(model.attn.weight_scale, scale)

    def test_non_linear_scaled_weight_is_dequantized_with_its_scale(self):
        """Same failure without any skip pattern: a scaled tensor that is not an nn.Linear weight
        fails `_is_fp8_matmul_weight`, so the cast would strip its scale."""
        model = self._model()
        q, scale = _fp8_weight(32, 16)
        sd = {"pad_token.weight": q}
        remaining = split_fp8_scaled_layers(
            sd, {"pad_token": Fp8ScaledLayer(weight_scale=scale)}, torch.bfloat16, model=model
        )
        assert remaining == {}
        assert torch.equal(sd["pad_token.weight"], dequantize_weight(q, scale, torch.bfloat16))

    def test_warns_when_a_scale_finds_no_module(self):
        logger = logging.getLogger("fp8-test")
        with mock.patch.object(logger, "warning") as warn:
            warn_on_unattached_scales(logger, "Krea-2", 1, {"a": object(), "b": object()})
        assert warn.call_count == 1
        assert "1 of 2" in warn.call_args[0][0]

    def test_silent_when_every_scale_landed(self):
        logger = logging.getLogger("fp8-test")
        with mock.patch.object(logger, "warning") as warn:
            warn_on_unattached_scales(logger, "Krea-2", 2, {"a": object(), "b": object()})
        assert warn.call_count == 0


class TestPredictedSize:
    def test_matches_what_cast_state_dict_actually_leaves(self):
        """`make_room` reserves against this, so a mismatch is a silent under-reservation."""
        model = torch.nn.Sequential()
        model.add_module("lin", torch.nn.Linear(16, 32))
        model.add_module("t_embedder", torch.nn.Linear(16, 32))
        model.add_module("norm", torch.nn.LayerNorm(32))
        q, _ = _fp8_weight(32, 16)
        sd = {
            "lin.weight": q,
            "lin.bias": torch.zeros(32).to(FP8_DTYPE),
            "t_embedder.weight": q.clone(),
            "norm.weight": torch.ones(32).to(FP8_DTYPE),
            "pad_token": torch.zeros(1, 32).to(FP8_DTYPE),
        }
        kwargs = {"model": model, "skip_patterns": ["t_embedder"]}
        # What the loaders used to reserve: 1 byte/element for *every* fp8 tensor.
        naive = sum(t.nelement() * (t.element_size() if t.dtype is FP8_DTYPE else 2) for t in sd.values())

        predicted = predict_cast_state_dict_size(sd, torch.bfloat16, keep_fp8=True, **kwargs)
        cast_state_dict(sd, torch.bfloat16, keep_fp8=True, **kwargs)
        actual = sum(t.nelement() * t.element_size() for t in sd.values())
        assert predicted == actual

        # Only `lin.weight` stays 1 byte/element; the rest lands at 2, so the old sum under-counted.
        assert naive < actual

    def test_all_dequantized_when_keep_fp8_is_off(self):
        q, _ = _fp8_weight(32, 16)
        sd = {"lin.weight": q}
        assert predict_cast_state_dict_size(sd, torch.bfloat16, keep_fp8=False) == q.nelement() * 2


class TestDeviceSupport:
    def test_probe_decides_rather_than_the_capability_number(self):
        """ROCm reports the gfx arch from `get_device_capability`, so RDNA3 (gfx1100 -> (11, 0))
        passes a `>= (8, 9)` test and then raises on every forward. The probe is what decides."""
        reset_fp8_matmul_support_cache()
        with mock.patch("torch.cuda.is_available", return_value=True):
            with mock.patch("torch.cuda.get_device_capability", return_value=(11, 0)):
                with mock.patch(
                    "invokeai.backend.quantization.fp8_scaled._probe_fp8_matmul", return_value=False
                ) as probe:
                    assert device_supports_fp8_matmul(torch.device("cuda", 0)) is False
                    assert probe.called
        reset_fp8_matmul_support_cache()

    def test_probe_result_is_cached_per_device(self):
        reset_fp8_matmul_support_cache()
        with mock.patch("torch.cuda.is_available", return_value=True):
            with mock.patch("torch.cuda.get_device_capability", return_value=(8, 9)):
                with mock.patch(
                    "invokeai.backend.quantization.fp8_scaled._probe_fp8_matmul", return_value=True
                ) as probe:
                    assert device_supports_fp8_matmul(torch.device("cuda", 0)) is True
                    assert device_supports_fp8_matmul(torch.device("cuda", 0)) is True
                    assert probe.call_count == 1
        reset_fp8_matmul_support_cache()

    def test_non_cuda_never_probes(self):
        reset_fp8_matmul_support_cache()
        with mock.patch("invokeai.backend.quantization.fp8_scaled._probe_fp8_matmul") as probe:
            assert device_supports_fp8_matmul(torch.device("cpu")) is False
            assert not probe.called


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

    @pytest.fixture(autouse=True)
    def _restore_matmul_override(self):
        """Clear the override after each test, rather than leaving it pinned to ``False``.

        These tests reset with ``set_fp8_matmul_enabled(False)``, which is not the neutral state --
        it is an explicit "off" that outlives the class and silently overrides the config for
        anything that runs later in the same process.
        """
        yield
        set_fp8_matmul_enabled(None)

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


class TestScaleSpellingHelpers:
    """Both spellings of the weight scale must be handled everywhere.

    Reading only `.weight_scale` is the mistake that keeps recurring in the per-loader dequant
    helpers. Depending on what the loader strips afterwards it either deletes a `.scale_weight`
    without applying it — leaving the weight off by `1/weight_scale`, silently — or leaves the key
    behind for `load_state_dict(..., strict=True)` to reject.
    """

    @pytest.mark.parametrize("spelling", [".weight_scale", ".scale_weight"])
    def test_pairs_are_found_in_either_spelling(self, spelling: str) -> None:
        sd = {"blk.weight": torch.ones(2, 2), f"blk{spelling}": torch.tensor(0.5)}

        assert list(iter_weight_scale_pairs(sd)) == [("blk.weight", f"blk{spelling}")]

    def test_a_scale_without_its_weight_is_not_paired(self) -> None:
        # Pairing it would invent a weight key the checkpoint never had.
        sd = {"other.weight": torch.ones(1), "blk.weight_scale": torch.tensor(0.5)}

        assert list(iter_weight_scale_pairs(sd)) == []

    def test_non_string_keys_are_ignored(self) -> None:
        # `.pt`/`.ckpt` sources can carry int keys; `endswith` would raise on them.
        assert list(iter_weight_scale_pairs({0: torch.ones(1)})) == []

    @pytest.mark.parametrize(
        "key",
        [".weight_scale", ".scale_weight", ".input_scale", ".scale_input"],
    )
    def test_metadata_keys_are_recognized_in_either_spelling(self, key: str) -> None:
        assert is_scale_metadata_key(f"blk{key}")

    @pytest.mark.parametrize("key", ["blk.comfy_quant", "scaled_fp8"])
    def test_marker_keys_are_recognized(self, key: str) -> None:
        assert is_scale_metadata_key(key)

    @pytest.mark.parametrize("key", ["blk.weight", "blk.bias", "norm.scale", 0])
    def test_model_tensors_are_not_mistaken_for_metadata(self, key: object) -> None:
        # `norm.scale` is a real learned parameter in several architectures - stripping it would
        # delete weights, and it is why this cannot just match "scale" anywhere in the key.
        assert not is_scale_metadata_key(key)


def _fp8(rows: int = 4, cols: int = 4, value: float = 1.0) -> torch.Tensor:
    return (torch.ones(rows, cols) * value).to(FP8_DTYPE)


class _Linear(torch.nn.Module):
    def __init__(self, out_features: int = 4, in_features: int = 4) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(in_features, out_features, bias=False)


class TestE5m2ScaleRecovery:
    """`float8_e5m2` may not stay quantized, but it must not lose its scale on the way to bf16."""

    def test_the_scale_is_recovered_and_folded(self) -> None:
        # Gating extraction on e4m3fn alone popped the scale key and then dropped it, so
        # `cast_state_dict` did a plain `.to(bf16)` and the weight came out off by 1/weight_scale.
        sd = {"lin.weight": (torch.ones(4, 4) * 2).to(torch.float8_e5m2), "lin.weight_scale": torch.tensor(0.25)}

        layers = extract_fp8_scaled_layers(sd)
        assert "lin" in layers

        dequantize_fp8_scaled(sd, layers, torch.bfloat16)
        assert torch.allclose(sd["lin.weight"].float(), torch.full((4, 4), 0.5))

    def test_it_is_never_left_quantized(self) -> None:
        """`scaled_mm_linear` cannot take e5m2 as the weight operand, so it must be widened."""
        sd = {"lin.weight": (torch.ones(4, 4) * 2).to(torch.float8_e5m2), "lin.weight_scale": torch.tensor(0.25)}

        kept = split_fp8_scaled_layers(sd, extract_fp8_scaled_layers(sd), torch.bfloat16, model=_Linear())

        assert kept == {}
        assert sd["lin.weight"].dtype is torch.bfloat16
        assert torch.allclose(sd["lin.weight"].float(), torch.full((4, 4), 0.5))


class TestBlockWiseScale:
    """A 2-D ``weight_scale`` has one entry per *block* of weight elements, not per row."""

    def test_the_layout_survives_extraction(self) -> None:
        # Flattening it destroys the block geometry, and the multiply then fails on shape.
        sd = {"b.weight": _fp8(64, 128), "b.weight_scale": torch.full((64, 2), 0.5)}

        layers = extract_fp8_scaled_layers(sd)

        assert tuple(layers["b"].weight_scale.shape) == (64, 2)

    def test_it_is_expanded_rather_than_raising(self) -> None:
        # Before: `RuntimeError: The size of tensor a (64) must match the size of tensor b (128)`,
        # i.e. with fp8_compute off - the default - the model failed to load at all.
        sd = {"b.weight": _fp8(64, 128), "b.weight_scale": torch.full((64, 2), 0.5)}

        dequantize_fp8_scaled(sd, extract_fp8_scaled_layers(sd), torch.bfloat16)

        assert sd["b.weight"].shape == (64, 128)
        assert torch.allclose(sd["b.weight"].float(), torch.full((64, 128), 0.5))

    def test_it_is_never_left_quantized(self) -> None:
        """`scaled_mm_linear` can apply a per-tensor or per-row scale and nothing else.

        Left quantized, the mismatch surfaces mid-generation inside the kernel instead.
        Dequantizing here - before `predict_cast_state_dict_size` runs - also keeps the RAM
        reservation honest.
        """
        sd = {"lin.weight": _fp8(4, 4), "lin.weight_scale": torch.full((4, 2), 0.5)}

        kept = split_fp8_scaled_layers(sd, extract_fp8_scaled_layers(sd), torch.bfloat16, model=_Linear())

        assert kept == {}
        assert sd["lin.weight"].dtype is torch.bfloat16

    def test_a_per_row_scale_still_stays_quantized(self) -> None:
        sd = {"lin.weight": _fp8(4, 4), "lin.weight_scale": torch.full((4,), 0.5)}

        kept = split_fp8_scaled_layers(sd, extract_fp8_scaled_layers(sd), torch.bfloat16, model=_Linear())

        assert set(kept) == {"lin"}
        assert sd["lin.weight"].dtype is FP8_DTYPE


class TestSidechannelDetachReattach:
    """Key converters rename ``.weight``; the side channel has to be carried across separately."""

    def test_entries_follow_their_module_to_the_new_path(self) -> None:
        sd = {
            "old.linear.weight": _fp8(),
            "old.linear.weight_scale": torch.tensor(0.5),
            "old.linear.input_scale": torch.tensor(0.25),
        }

        detached = detach_layer_sidechannel(sd)
        assert list(sd) == ["old.linear.weight"], "scales must be out of the converter's way"

        converted = {"new.linear.weight": sd["old.linear.weight"]}
        orphaned = reattach_layer_sidechannel(converted, detached, {"old.linear": "new.linear"})

        assert orphaned == []
        layers = extract_fp8_scaled_layers(converted)
        assert set(layers) == {"new.linear"}
        assert layers["new.linear"].input_scale is not None

    def test_a_module_the_converter_drops_is_reported_not_swallowed(self) -> None:
        """A silently dropped scale is exactly the failure this pair exists to prevent."""
        sd = {"gone.weight": _fp8(), "gone.weight_scale": torch.tensor(0.5)}

        detached = detach_layer_sidechannel(sd)
        orphaned = reattach_layer_sidechannel({}, detached, {})

        assert orphaned == ["gone"]

    def test_both_scale_spellings_are_detached(self) -> None:
        sd = {
            "a.weight": _fp8(),
            "a.scale_weight": torch.tensor(0.5),
            "b.weight": _fp8(),
            "b.weight_scale": torch.tensor(0.5),
        }

        detached = detach_layer_sidechannel(sd)

        assert set(detached) == {"a", "b"}


class TestStripLayerPathPrefix:
    """`_quantization_metadata` is read from the file header, so its names keep the prefix."""

    def test_the_checkpoint_prefix_is_removed(self) -> None:
        hints = {"model.diffusion_model.blocks.0.attn.wq": {"full_precision_matrix_mult": True}}

        assert strip_layer_path_prefix(hints) == {"blocks.0.attn.wq": {"full_precision_matrix_mult": True}}

    def test_unprefixed_names_are_passed_through_not_dropped(self) -> None:
        """Running the names through a prefix *filter* truncated a partially-prefixed header, and
        the strict-zip that read the result back aborted the whole load with a ValueError."""
        hints = {"net.blocks.0.attn.q_proj": {}, "final_layer.linear": {}}

        assert set(strip_layer_path_prefix(hints)) == {"blocks.0.attn.q_proj", "final_layer.linear"}


class TestNonFloatTensorsAreNotCast:
    """Integer payloads are not weights; casting them to the compute dtype corrupts them."""

    def test_cast_state_dict_leaves_them_alone(self) -> None:
        sd = {"w.weight": torch.ones(2, 2), "ids": torch.arange(4, dtype=torch.int64)}

        cast_state_dict(sd, torch.bfloat16, keep_fp8=False)

        assert sd["w.weight"].dtype is torch.bfloat16
        assert sd["ids"].dtype is torch.int64

    def test_the_size_prediction_agrees(self) -> None:
        sd = {"ids": torch.arange(4, dtype=torch.int64)}

        assert predict_cast_state_dict_size(sd, torch.bfloat16, keep_fp8=False) == 4 * 8


@contextlib.contextmanager
def _probe_on_cpu():
    """Let `_probe_fp8_matmul` run its allocations on CPU so the caching policy can be tested.

    The probe allocates real tensors on the device *before* it reaches `torch._scaled_mm`. On a
    CPU-only runner that allocation raises first, the mocked matmul never runs, and the test
    measures the wrong thing — it passed on a CUDA box and failed on CI for exactly this reason.
    float8 tensors are allocatable on CPU, so dropping the device argument is enough.
    """
    real_zeros, real_ones = torch.zeros, torch.ones

    def on_cpu(real):
        def alloc(*args, **kwargs):
            kwargs.pop("device", None)
            return real(*args, **kwargs)

        return alloc

    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.get_device_capability", return_value=(8, 9)),
        mock.patch("torch.cuda.current_device", return_value=0),
        mock.patch("torch.zeros", on_cpu(real_zeros)),
        mock.patch("torch.ones", on_cpu(real_ones)),
    ):
        yield


class TestProbeFailureCaching:
    """The probe runs during a model load, i.e. under real VRAM pressure."""

    def test_an_allocation_failure_is_not_cached(self) -> None:
        reset_fp8_matmul_support_cache()
        device = torch.device("cuda", 0)
        with _probe_on_cpu():
            with mock.patch("torch._scaled_mm", side_effect=torch.OutOfMemoryError("transient")):
                assert device_supports_fp8_matmul(device) is False
            # A momentary OOM must not disable fp8 for the rest of the process.
            with mock.patch("torch._scaled_mm", return_value=torch.zeros(1)):
                assert device_supports_fp8_matmul(device) is True
        reset_fp8_matmul_support_cache()

    def test_a_genuine_unsupported_op_is_cached(self) -> None:
        reset_fp8_matmul_support_cache()
        device = torch.device("cuda", 0)
        with _probe_on_cpu():
            with mock.patch("torch._scaled_mm", side_effect=RuntimeError("not supported on this device")):
                assert device_supports_fp8_matmul(device) is False
            # No second probe: the answer cannot change at runtime.
            with mock.patch("torch._scaled_mm", side_effect=AssertionError("must not be probed again")):
                assert device_supports_fp8_matmul(device) is False
        reset_fp8_matmul_support_cache()


class TestMxfp8IsRefused:
    """MXFP8 block scales are refused rather than guessed at.

    Established against a real pair of checkpoints: the MXFP8 and the scaled-fp8 build of
    `krea2TurboOfficialComfy` share all 174 bf16 tensors bit-for-bit, so the scaled build is an
    exact reference. Decoding the uint8 exponents as `2**(v-127)` and expanding them 32-wide
    reaches a correlation of only 0.60 against it and generates a pure-noise image; the measured
    per-block scale has no monotonic relation to the byte (112 and 116 give the same true scale),
    which points at a swizzled scale layout.

    Refusing matters because the block-wise expansion is what makes such a file *loadable*: without
    a guard it produces garbage silently, which is strictly worse than the shape error it used to
    raise.
    """

    def test_a_uint8_block_scale_is_rejected_with_an_actionable_message(self) -> None:
        sd = {
            "lin.weight": torch.full((4, 64), 2.0).to(FP8_DTYPE),
            "lin.weight_scale": torch.full((4, 2), 125, dtype=torch.uint8),
        }

        with pytest.raises(NotImplementedError) as excinfo:
            extract_fp8_scaled_layers(sd)

        message = str(excinfo.value)
        assert "lin" in message, "the failing layer must be named"
        assert "MXFP8" in message
        assert "noise" in message, "say what happens if it were loaded anyway"

    def test_float_scales_are_unaffected(self) -> None:
        """The guard keys off the dtype, so ordinary scaled-fp8 checkpoints must still load."""
        sd = {
            "lin.weight": torch.full((4, 64), 2.0).to(FP8_DTYPE),
            "lin.weight_scale": torch.full((4, 2), 0.25),
        }

        layers = extract_fp8_scaled_layers(sd)

        assert tuple(layers["lin"].weight_scale.shape) == (4, 2)


class TestNonScalarInputScale:
    """A multi-element ``input_scale`` is dropped, not raised on.

    `scaled_mm_linear` scales activations by one value, so there is nothing to do with a
    per-channel or per-block activation scale -- but reshaping it to a scalar unconditionally
    turned such a checkpoint into a shape error at load time. Every other malformed side-channel in
    this module is either skipped or refused with an actionable message; this one was neither.
    """

    def _sd(self, input_scale: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "blocks.0.attn.weight": torch.zeros(16, 16, dtype=torch.float32).to(FP8_DTYPE),
            "blocks.0.attn.weight_scale": torch.tensor(2.0),
            "blocks.0.attn.input_scale": input_scale,
        }

    def test_a_per_channel_input_scale_does_not_abort_the_load(self) -> None:
        layers = extract_fp8_scaled_layers(self._sd(torch.full((16,), 0.5)))

        assert set(layers) == {"blocks.0.attn"}
        # Dropped, so the forward falls back to the dynamic amax path -- which is correct.
        assert layers["blocks.0.attn"].input_scale is None
        # The weight scale is unaffected by the unusable activation scale.
        assert layers["blocks.0.attn"].weight_scale is not None

    def test_a_scalar_input_scale_is_still_kept(self) -> None:
        layers = extract_fp8_scaled_layers(self._sd(torch.tensor(0.5)))

        assert layers["blocks.0.attn"].input_scale is not None


class TestMatmulUsableScale:
    """`is_matmul_usable_scale` decides what may reach `_scaled_mm` un-dequantized."""

    def test_a_per_tensor_scale_is_usable(self) -> None:
        assert is_matmul_usable_scale(torch.zeros(32, 16), torch.tensor(2.0)) is True

    def test_a_per_output_channel_scale_is_usable(self) -> None:
        assert is_matmul_usable_scale(torch.zeros(32, 16), torch.full((32,), 2.0)) is True

    def test_a_one_dimensional_scale_of_the_wrong_length_is_not(self) -> None:
        """The branch that keeps a mislabelled scale out of the kernel.

        A 1-D scale whose length is not the row count is neither per-tensor nor
        per-output-channel. Letting it through raises inside `_scaled_mm` mid-generation instead of
        dequantizing the layer up front, which is the whole point of asking.
        """
        assert is_matmul_usable_scale(torch.zeros(32, 16), torch.full((16,), 2.0)) is False

    def test_a_block_wise_scale_is_not(self) -> None:
        assert is_matmul_usable_scale(torch.zeros(32, 16), torch.full((4, 2), 2.0)) is False


class TestReattachToNonWeightDestination:
    """The reattach guard must not assume every module stores its parameter as ``weight``.

    A producer that quantizes norms -- the "quantizes everything" class this module documents --
    writes ``<path>.scale``. Testing for ``<destination>.weight`` rejects such a destination for the
    wrong reason and drops a scale it could have placed, with only a log line to show for it.
    """

    def test_a_destination_whose_tensor_is_not_named_weight_still_receives_its_scale(self) -> None:
        sd = {"blocks.0.norm_q.scale": torch.zeros(16, dtype=torch.float32).to(FP8_DTYPE)}
        detached = {"blocks.0.qnorm": [(".weight_scale", torch.tensor(2.0))]}

        orphaned = reattach_layer_sidechannel(sd, detached, {"blocks.0.qnorm": "blocks.0.norm_q"})

        assert orphaned == []
        assert "blocks.0.norm_q.weight_scale" in sd

    def test_a_destination_absent_from_the_state_dict_is_still_reported(self) -> None:
        sd = {"blocks.0.attn.weight": torch.zeros(4, 4)}
        detached = {"blocks.0.dropped": [(".weight_scale", torch.tensor(2.0))]}

        orphaned = reattach_layer_sidechannel(sd, detached, {})

        assert orphaned == ["blocks.0.dropped"]
        assert not [k for k in sd if k.endswith(".weight_scale")]


class TestProbeTransientFailures:
    """Only a definitively unsupported device may be cached; everything else re-probes.

    Listing the transient wordings is the wrong way round: an OOM and a cuBLAS workspace failure
    are two of the ways a loaded machine can fail this call, and any wording not on the list would
    be cached as permanent -- reproducing the failure the OOM branch was written to avoid.
    """

    def test_an_unrecognized_runtime_error_is_not_cached(self) -> None:
        reset_fp8_matmul_support_cache()
        device = torch.device("cuda", 0)
        with _probe_on_cpu():
            with mock.patch("torch._scaled_mm", side_effect=RuntimeError("CUDA driver reset")):
                assert device_supports_fp8_matmul(device) is False
            # Inconclusive, so the next load re-probes rather than the process losing fp8.
            with mock.patch("torch._scaled_mm", return_value=torch.zeros(1)):
                assert device_supports_fp8_matmul(device) is True
        reset_fp8_matmul_support_cache()

    def test_a_capability_error_is_cached(self) -> None:
        reset_fp8_matmul_support_cache()
        device = torch.device("cuda", 0)
        message = "torch._scaled_mm is only supported on CUDA devices with compute capability >= 8.9"
        with _probe_on_cpu():
            with mock.patch("torch._scaled_mm", side_effect=RuntimeError(message)):
                assert device_supports_fp8_matmul(device) is False
            with mock.patch("torch._scaled_mm", side_effect=AssertionError("must not be probed again")):
                assert device_supports_fp8_matmul(device) is False
        reset_fp8_matmul_support_cache()


class TestExpandWeightScaleAxis:
    """A per-output-channel scale must scale rows, not columns.

    `(out, in) * (out,)` broadcasts on the *last* axis, so a bare multiply scales input channels --
    wrong on a square weight, a shape error on any other. Both legacy folds used to do exactly that.
    """

    def test_a_per_channel_scale_lines_up_with_the_rows(self) -> None:
        weight = torch.ones(3, 2)
        scale = torch.tensor([1.0, 2.0, 3.0])

        result = weight * expand_weight_scale(weight, scale)

        assert torch.equal(result, torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]))

    def test_a_non_square_weight_no_longer_raises(self) -> None:
        weight = torch.ones(4, 2)

        # Without the expansion this is a broadcast error, not merely a wrong number.
        assert (weight * expand_weight_scale(weight, torch.arange(4, dtype=torch.float32))).shape == (4, 2)
