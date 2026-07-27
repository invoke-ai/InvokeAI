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
