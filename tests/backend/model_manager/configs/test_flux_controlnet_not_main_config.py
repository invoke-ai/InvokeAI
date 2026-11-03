"""Regression test for XLabs FLUX ControlNet checkpoints being identified as FLUX main models.

An XLabs FLUX ControlNet checkpoint (e.g. flux-canny-controlnet-v3.safetensors) contains two FLUX
double blocks, so it carries the `double_blocks.0.img_attn.norm.key_norm.scale` key that
`Main_Checkpoint_FLUX_Config` keys off. It also matches `is_state_dict_xlabs_controlnet`. Both config
classes used to match, and the factory tie-break ranks main models first, so the file was installed as
a main model and could not be used as a ControlNet (invoke-ai/InvokeAI#9832).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.backend.model_manager.configs.controlnet import ControlNet_Checkpoint_FLUX_Config
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_FLUX_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType

_OVERRIDE_FIELDS: dict[str, object] = {
    "hash": "blake3:fakehash",
    "path": "/fake/models/flux-canny-controlnet-v3.safetensors",
    "file_size": 1000,
    "name": "flux-canny-controlnet-v3",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}

# Keys mirroring an XLabs FLUX ControlNet checkpoint: the controlnet-specific keys that
# is_state_dict_xlabs_controlnet looks for, plus the two FLUX double blocks (and img_in / guidance_in)
# the file actually contains. There are no single_blocks or final_layer keys.
XLABS_CONTROLNET_KEYS = {
    "controlnet_blocks.0.weight": SimpleNamespace(shape=(3072, 3072)),
    "controlnet_blocks.0.bias": SimpleNamespace(shape=(3072,)),
    "input_hint_block.0.weight": SimpleNamespace(shape=(16, 3, 3, 3)),
    "input_hint_block.0.bias": SimpleNamespace(shape=(16,)),
    "pos_embed_input.weight": SimpleNamespace(shape=(3072, 64)),
    "pos_embed_input.bias": SimpleNamespace(shape=(3072,)),
    "double_blocks.0.img_attn.norm.key_norm.scale": SimpleNamespace(shape=(128,)),
    "double_blocks.0.img_attn.qkv.weight": SimpleNamespace(shape=(9216, 3072)),
    "img_in.weight": SimpleNamespace(shape=(3072, 64)),
    "guidance_in.out_layer.weight": SimpleNamespace(shape=(3072, 3072)),
    "time_in.in_layer.weight": SimpleNamespace(shape=(3072, 256)),
    "txt_in.weight": SimpleNamespace(shape=(3072, 4096)),
    "vector_in.in_layer.weight": SimpleNamespace(shape=(3072, 768)),
}


def _make_mod(state_dict: dict[str, object]) -> MagicMock:
    mod = MagicMock()
    mod.load_state_dict.return_value = state_dict
    mod.metadata.return_value = {}
    return mod


def test_xlabs_controlnet_does_not_match_flux_main():
    mod = _make_mod(dict(XLABS_CONTROLNET_KEYS))

    with pytest.raises(NotAMatchError, match="FLUX ControlNet"):
        Main_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))


def test_xlabs_controlnet_matches_flux_controlnet():
    mod = _make_mod(dict(XLABS_CONTROLNET_KEYS))

    config = ControlNet_Checkpoint_FLUX_Config.from_model_on_disk(mod, dict(_OVERRIDE_FIELDS))

    assert config.base is BaseModelType.Flux
    assert config.type is ModelType.ControlNet
    assert config.format is ModelFormat.Checkpoint
