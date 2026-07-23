"""Unit tests for the Anima single-file prefix-stripping helper."""

import torch

from invokeai.backend.model_manager.load.model_loaders.anima import _strip_anima_bundle_prefix
from tests.backend.model_manager.load.state_dicts.anima_comfyui_keys import state_dict_keys as anima_keys
from tests.backend.model_manager.load.state_dicts.utils import keys_to_mock_state_dict


class TestStripAnimaBundlePrefix:
    def test_official_net_prefix_is_stripped(self):
        sd = keys_to_mock_state_dict(anima_keys)
        assert all(k.startswith("net.") for k in sd)

        out = _strip_anima_bundle_prefix(sd)

        assert len(out) == len(sd)
        assert not any(k.startswith("net.") for k in out)
        # Every key had exactly its `net.` prefix removed.
        assert {"net." + k for k in out} == set(sd.keys())

    def test_comfyui_bundle_keeps_only_transformer_keys(self):
        # ComfyUI bundles the transformer under `model.diffusion_model.` alongside the VAE and
        # text encoder, which must be dropped.
        sd = {
            "model.diffusion_model.blocks.0.attn.qkv.weight": torch.empty(1),
            "model.diffusion_model.final_layer.weight": torch.empty(1),
            "first_stage_model.encoder.conv_in.weight": torch.empty(1),
            "cond_stage_model.transformer.embeddings.weight": torch.empty(1),
        }

        out = _strip_anima_bundle_prefix(sd)

        assert set(out.keys()) == {"blocks.0.attn.qkv.weight", "final_layer.weight"}

    def test_no_known_prefix_is_a_noop(self):
        sd = {"blocks.0.attn.qkv.weight": torch.empty(1)}
        assert _strip_anima_bundle_prefix(sd) is sd
