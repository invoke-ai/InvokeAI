"""Tests for SDNQ FLUX.2 single-file model identification.

A single-file SDNQ FLUX.2 transformer must identify as ``Main_SDNQ_Flux2_Config`` (base=flux2),
not as ``Main_SDNQ_FLUX_Config`` (base=flux) and not as unknown.

The only supported single-file layout is *bare diffusers* (``transformer_blocks.*`` /
``context_embedder.*``): SDNQ tooling quantizes diffusers state dicts, and the single-file SDNQ
FLUX.2 loader (``Flux2SDNQCheckpointModel._load_from_singlefile``) reads exactly those bare keys with
no BFL→diffusers conversion and no ``model.diffusion_model.`` prefix stripping. ``Main_SDNQ_Flux2_Config``
therefore accepts only the bare diffusers layout and must REJECT a BFL / ComfyUI
(``model.diffusion_model.double_blocks.*``) checkpoint — otherwise identification would classify a
checkpoint the loader then fails to load with missing/unexpected keys.

``Main_SDNQ_FLUX_Config`` (base=flux) must also reject FLUX.2 state dicts in either layout, so an
ambiguous checkpoint is never handed to the FLUX.1 loader.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.taxonomy import BaseModelType

_REQUIRED_FIELDS = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/test.safetensors",
    "file_size": 1000,
    "name": "flux-2-klein-4b-sdnq",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}

# 3 × Qwen3-4B hidden size (2560) — the FLUX.2 Klein 4B context dimension.
_KLEIN_4B_CONTEXT_DIM = 7680


def _make_mock_mod(filename: str, state_dict: dict) -> MagicMock:
    mod = MagicMock()
    mod.path = Path(f"/fake/models/{filename}")
    mod.name = filename
    mod.load_state_dict.return_value = state_dict
    return mod


def _bare_flux2_sdnq_state_dict() -> dict:
    """FLUX.2 SDNQ transformer in the bare diffusers layout (the real shipped single-file layout),
    with an SDNQ weight+scale pair."""
    return {
        "transformer_blocks.0.attn.to_q.weight": torch.zeros(3072, 3072, dtype=torch.uint8),
        "transformer_blocks.0.attn.to_q.scale": torch.zeros(3072, 24, 1),
        "context_embedder.weight": torch.zeros(3072, _KLEIN_4B_CONTEXT_DIM),
        "x_embedder.weight": torch.zeros(3072, 128),
        "img_in.weight": torch.zeros(3072, 128),
    }


def _prefixed_flux2_sdnq_state_dict() -> dict:
    """FLUX.2 SDNQ transformer in BFL / ComfyUI prefixed layout — not a real single-file SDNQ
    artifact, kept only to exercise the FLUX.1-config rejection guard."""
    return {
        "model.diffusion_model.double_blocks.0.img_attn.qkv.weight": torch.zeros(9216, 3072, dtype=torch.uint8),
        "model.diffusion_model.double_blocks.0.img_attn.qkv.scale": torch.zeros(9216, 24, 1),
        "model.diffusion_model.context_embedder.weight": torch.zeros(3072, _KLEIN_4B_CONTEXT_DIM),
        "model.diffusion_model.img_in.weight": torch.zeros(3072, 128),
    }


@patch("invokeai.backend.model_manager.configs.main.raise_if_not_file")
@patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields")
class TestSDNQFlux2Identification:
    def test_bare_diffusers_identifies_as_flux2(self, _rfo, _rif):
        """A bare diffusers-layout SDNQ FLUX.2 checkpoint (the layout the loader supports) identifies
        as base=flux2."""
        from invokeai.backend.model_manager.configs.main import Main_SDNQ_Flux2_Config

        mod = _make_mock_mod("flux-2-klein-4b-sdnq.safetensors", _bare_flux2_sdnq_state_dict())
        config = Main_SDNQ_Flux2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
        assert config.base is BaseModelType.Flux2

    def test_prefixed_bfl_layout_is_rejected(self, _rfo, _rif):
        """A BFL / ComfyUI (model.diffusion_model.double_blocks.*) SDNQ FLUX.2 checkpoint must NOT be
        accepted: the single-file SDNQ FLUX.2 loader only consumes the bare diffusers layout (no BFL
        prefix stripping / conversion), so identification must reject it rather than classify a
        checkpoint the loader then fails to load."""
        from invokeai.backend.model_manager.configs.main import Main_SDNQ_Flux2_Config

        mod = _make_mock_mod("flux-2-klein-4b-sdnq.safetensors", _prefixed_flux2_sdnq_state_dict())
        with pytest.raises(NotAMatchError):
            Main_SDNQ_Flux2_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})

    @pytest.mark.parametrize(
        "state_dict_factory",
        [_bare_flux2_sdnq_state_dict, _prefixed_flux2_sdnq_state_dict],
        ids=["bare-diffusers", "prefixed"],
    )
    def test_flux1_config_rejects_flux2(self, _rfo, _rif, state_dict_factory):
        """The FLUX.1 SDNQ config must NOT accept a FLUX.2 state dict (would load with wrong loader)."""
        from invokeai.backend.model_manager.configs.main import Main_SDNQ_FLUX_Config

        mod = _make_mock_mod("flux-2-klein-4b-sdnq.safetensors", state_dict_factory())
        with pytest.raises(NotAMatchError):
            Main_SDNQ_FLUX_Config.from_model_on_disk(mod, {**_REQUIRED_FIELDS})
