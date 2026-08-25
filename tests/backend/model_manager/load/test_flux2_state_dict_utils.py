"""Unit tests for the FLUX.2 BFL->diffusers state-dict converters.

Fixtures are captured from real single-file checkpoints (see the fixture module docstrings).
The meta-device tests instantiate the actual diffusers architectures with `init_empty_weights`
(no real weights, no GPU) and assert that every converted key is a real parameter -- the same
kind of check that would have caught the Qwen VL remap regression.
"""

import accelerate
import torch

from invokeai.backend.model_manager.load.model_loaders.flux2_state_dict_utils import (
    _flux2_swap_scale_shift,
    convert_flux2_bfl_to_diffusers,
    convert_flux2_vae_bfl_to_diffusers,
)
from tests.backend.model_manager.load.state_dicts.flux2_transformer_bfl_keys import (
    state_dict_keys as flux2_transformer_keys,
)
from tests.backend.model_manager.load.state_dicts.flux2_vae_bfl_keys import (
    state_dict_keys as flux2_vae_keys,
)
from tests.backend.model_manager.load.state_dicts.utils import keys_to_mock_state_dict


class TestConvertFlux2Transformer:
    def test_fused_qkv_is_split_and_blocks_renamed(self):
        sd = keys_to_mock_state_dict(flux2_transformer_keys)

        converted = convert_flux2_bfl_to_diffusers(sd)

        # Fused img/txt QKV are split into separate projections.
        assert "transformer_blocks.0.attn.to_q.weight" in converted
        assert "transformer_blocks.0.attn.to_k.weight" in converted
        assert "transformer_blocks.0.attn.to_v.weight" in converted
        assert "transformer_blocks.0.attn.add_q_proj.weight" in converted
        # No fused/BFL-named keys remain.
        assert not any("img_attn.qkv" in k or "double_blocks." in k or "single_blocks." in k for k in converted)
        # Top-level renames.
        assert "x_embedder.weight" in converted
        assert "context_embedder.weight" in converted
        assert "proj_out.weight" in converted

    def test_converted_keys_are_all_real_transformer_params(self):
        """Meta-device coverage: every converted key must exist in Flux2Transformer2DModel."""
        from diffusers import Flux2Transformer2DModel

        converted = convert_flux2_bfl_to_diffusers(keys_to_mock_state_dict(flux2_transformer_keys))

        # The fixture keeps block 0 of each stack -> a single-layer model covers it.
        with accelerate.init_empty_weights():
            model = Flux2Transformer2DModel(num_layers=1, num_single_layers=1)
        params = set(model.state_dict().keys())

        unmatched = sorted(k for k in converted if k not in params)
        assert not unmatched, f"converted keys with no matching model parameter: {unmatched}"


class TestConvertFlux2Vae:
    def test_full_bijective_coverage_against_arch(self):
        """The full VAE fixture must convert to exactly the AutoencoderKLFlux2 parameter set."""
        from diffusers import AutoencoderKLFlux2

        converted = convert_flux2_vae_bfl_to_diffusers(keys_to_mock_state_dict(flux2_vae_keys))

        with accelerate.init_empty_weights():
            vae = AutoencoderKLFlux2(block_out_channels=(128, 256, 512, 512))
        params = set(vae.state_dict().keys())

        unmatched = sorted(k for k in converted if k not in params)
        missing = sorted(k for k in params if k not in converted)
        assert not unmatched, f"converted keys with no matching VAE parameter: {unmatched}"
        assert not missing, f"VAE parameters not covered by the converted checkpoint: {missing}"

    def test_up_block_order_is_reversed(self):
        # BFL decoder.up.X maps to diffusers up_blocks.(3 - X).
        sd = {
            "decoder.up.0.block.0.norm1.weight": torch.empty(1),
            "decoder.up.3.block.0.norm1.weight": torch.empty(1),
        }
        converted = convert_flux2_vae_bfl_to_diffusers(sd)
        assert "decoder.up_blocks.3.resnets.0.norm1.weight" in converted
        assert "decoder.up_blocks.0.resnets.0.norm1.weight" in converted

    def test_mid_attention_conv_weights_are_squeezed_to_linear(self):
        # BFL stores mid attention as Conv2d [out, in, 1, 1]; diffusers uses Linear [out, in].
        sd = {"encoder.mid.attn_1.q.weight": torch.empty(8, 8, 1, 1)}
        converted = convert_flux2_vae_bfl_to_diffusers(sd)
        assert converted["encoder.mid_block.attentions.0.to_q.weight"].shape == (8, 8)


class TestSwapScaleShift:
    def test_swaps_the_two_halves(self):
        # First half = shift, second half = scale; diffusers wants them swapped.
        weight = torch.cat([torch.zeros(2), torch.ones(2)])  # [shift=0, scale=1]
        swapped = _flux2_swap_scale_shift(weight)
        assert torch.allclose(swapped, torch.cat([torch.ones(2), torch.zeros(2)]))

    def test_leaves_malformed_tensor_untouched(self):
        weight = torch.ones(3)  # odd length -> cannot be split
        assert torch.allclose(_flux2_swap_scale_shift(weight), weight)


class TestFlux2RawFp8Gate:
    """`_dequantize_fp8_weights` runs before `cast_state_dict`, so anything it converts is gone.

    Its trailing loop used to cast *every* float8 tensor unconditionally, which meant nothing fp8
    ever reached `cast_state_dict` and the FLUX.2 half of the raw-fp8 path could not execute.
    """

    def _dequantize(self, sd, keep_fp8):
        from invokeai.backend.model_manager.load.model_loaders.flux import Flux2CheckpointModel

        # The method does not touch `self`; calling it unbound avoids building a whole loader.
        return Flux2CheckpointModel._dequantize_fp8_weights(None, sd, keep_fp8=keep_fp8)

    def test_raw_fp8_linear_weights_survive_when_kept(self):
        sd = {
            "double_blocks.0.img_attn.qkv.weight": torch.zeros(48, 16).to(torch.float8_e4m3fn),
            "double_blocks.0.img_attn.qkv.bias": torch.zeros(48).to(torch.float8_e4m3fn),
            "double_blocks.0.img_norm.scale": torch.ones(16).to(torch.float8_e4m3fn),
        }
        out = self._dequantize(sd, keep_fp8=True)
        assert out["double_blocks.0.img_attn.qkv.weight"].dtype is torch.float8_e4m3fn
        # 1-D tensors are never usable on the tensor cores and must not stay quantized.
        assert out["double_blocks.0.img_attn.qkv.bias"].dtype is torch.bfloat16
        assert out["double_blocks.0.img_norm.scale"].dtype is torch.bfloat16

    def test_everything_is_dequantized_when_not_kept(self):
        sd = {"double_blocks.0.img_attn.qkv.weight": torch.zeros(48, 16).to(torch.float8_e4m3fn)}
        out = self._dequantize(sd, keep_fp8=False)
        assert out["double_blocks.0.img_attn.qkv.weight"].dtype is torch.bfloat16

    def test_scaled_fp8_is_still_folded_even_when_keeping(self):
        """A weight with a `weight_scale` is dequantized *with* its scale, as before — only
        scale-less fp8 is kept, because only that is safe to hand to `_scaled_mm` unscaled."""
        sd = {
            "double_blocks.0.img_attn.qkv.weight": torch.ones(48, 16).to(torch.float8_e4m3fn),
            "double_blocks.0.img_attn.qkv.weight_scale": torch.tensor(4.0),
        }
        out = self._dequantize(sd, keep_fp8=True)
        assert out["double_blocks.0.img_attn.qkv.weight"].dtype is torch.bfloat16
        assert torch.allclose(out["double_blocks.0.img_attn.qkv.weight"], torch.full((48, 16), 4.0).bfloat16())
        assert "double_blocks.0.img_attn.qkv.weight_scale" not in out


class TestAdaLnSwapIsMirroredOntoTheScale:
    """`final_layer.adaLN_modulation.1.weight` has its two halves swapped (BFL vs diffusers order).

    A per-output-channel `weight_scale` has one entry per row, so it has to be swapped identically.
    Copying it verbatim leaves rows 0..n/2 holding the original second half while still carrying the
    first half's scale factors - every row scaled by another row's factor. It is the one converter
    transform that reorders rows and is not the fused-qkv split.
    """

    def test_a_per_channel_scale_is_swapped_with_its_weight(self) -> None:
        weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        sd = {
            "final_layer.adaLN_modulation.1.weight": weight,
            "final_layer.adaLN_modulation.1.weight_scale": torch.arange(1, 7, dtype=torch.float32),
        }

        converted = convert_flux2_bfl_to_diffusers(sd)

        assert converted["norm_out.linear.weight"][:, 0].tolist() == [12, 16, 20, 0, 4, 8]
        assert converted["norm_out.linear.weight_scale"].tolist() == [4, 5, 6, 1, 2, 3]

    def test_the_scale_weight_spelling_is_swapped_too(self) -> None:
        sd = {
            "final_layer.adaLN_modulation.1.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
            "final_layer.adaLN_modulation.1.scale_weight": torch.arange(1, 7, dtype=torch.float32),
        }

        converted = convert_flux2_bfl_to_diffusers(sd)

        assert converted["norm_out.linear.scale_weight"].tolist() == [4, 5, 6, 1, 2, 3]

    def test_per_tensor_scales_and_markers_are_left_alone(self) -> None:
        """They describe the whole layer, so reordering them would be wrong.

        The `input_scale` is per-tensor by construction (Ada rejects per-row activation scaling) and
        `comfy_quant` is a JSON byte blob, not a vector.
        """
        sd = {
            "final_layer.adaLN_modulation.1.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
            "final_layer.adaLN_modulation.1.weight_scale": torch.tensor(0.5),
            "final_layer.adaLN_modulation.1.input_scale": torch.tensor(0.25),
            "final_layer.adaLN_modulation.1.comfy_quant": torch.tensor(list(b'{"a":1}'), dtype=torch.uint8),
        }

        converted = convert_flux2_bfl_to_diffusers(sd)

        assert converted["norm_out.linear.weight_scale"].item() == 0.5
        assert converted["norm_out.linear.input_scale"].item() == 0.25
        assert bytes(converted["norm_out.linear.comfy_quant"].tolist()) == b'{"a":1}'

    def test_other_layers_are_not_reordered(self) -> None:
        """Only this one key is row-permuted; a scale elsewhere must be copied verbatim."""
        sd = {
            "final_layer.linear.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
            "final_layer.linear.weight_scale": torch.arange(1, 7, dtype=torch.float32),
        }

        converted = convert_flux2_bfl_to_diffusers(sd)

        scale = next(v for k, v in converted.items() if k.endswith(".weight_scale"))
        assert scale.tolist() == [1, 2, 3, 4, 5, 6]
