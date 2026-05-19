import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from invokeai.backend.hidiffusion.hidiffusion import (
    remove_hidiffusion as real_remove_hidiffusion,
)
from invokeai.backend.hidiffusion.hidiffusion import (
    switching_threshold_ratio_dict,
)
from invokeai.backend.hidiffusion.hidiffusion import (
    text_to_img_controlnet_switching_threshold_ratio_dict,
)
from invokeai.backend.stable_diffusion.hidiffusion_utils import hidiffusion_patch


class DummySubmodule:
    pass


class PatchedSubmodule(DummySubmodule):
    _parent = DummySubmodule


class DummyUNet:
    def __init__(self):
        self.num_upsamplers = 3
        self.layer = DummySubmodule()

    def named_modules(self):
        return [("", self), ("layer", self.layer)]


def test_hidiffusion_patch_restores_state_when_apply_hidiffusion_raises():
    original_switching = copy.deepcopy(switching_threshold_ratio_dict)
    original_controlnet = copy.deepcopy(text_to_img_controlnet_switching_threshold_ratio_dict)

    model = SimpleNamespace(
        unet=DummyUNet(),
        _name_or_path="original-model-name",
        config=SimpleNamespace(_name_or_path="original-config-name"),
    )
    hook = MagicMock()

    def fake_apply_hidiffusion(patched_model, **_kwargs):
        assert patched_model._name_or_path == "patched-model-name"
        assert patched_model.config._name_or_path == "patched-model-name"

        first_switching_entry = next(iter(switching_threshold_ratio_dict.values()))
        first_controlnet_entry = next(iter(text_to_img_controlnet_switching_threshold_ratio_dict.values()))
        assert first_switching_entry["T1_ratio"] == 0.25
        assert first_switching_entry["T2_ratio"] == 0.1
        assert first_controlnet_entry["T1_ratio"] == 0.25
        assert first_controlnet_entry["T2_ratio"] == 0.1

        patched_model.unet.num_upsamplers = 99
        patched_model.unet.layer.info = {"hooks": [hook]}
        patched_model.unet.layer.__class__ = PatchedSubmodule
        raise RuntimeError("hidiffusion boom")

    try:
        with (
            patch("invokeai.backend.hidiffusion.hidiffusion.apply_hidiffusion", side_effect=fake_apply_hidiffusion),
            patch(
                "invokeai.backend.hidiffusion.hidiffusion.remove_hidiffusion",
                wraps=real_remove_hidiffusion,
            ) as mock_remove_hidiffusion,
        ):
            with pytest.raises(RuntimeError, match="hidiffusion boom"):
                with hidiffusion_patch(
                    model,
                    name_or_path="patched-model-name",
                    t1_ratio=0.25,
                    t2_ratio=0.1,
                ):
                    pass

        assert mock_remove_hidiffusion.call_count == 1
        assert switching_threshold_ratio_dict == original_switching
        assert text_to_img_controlnet_switching_threshold_ratio_dict == original_controlnet
        assert model.unet.num_upsamplers == 3
        assert model.unet.layer.__class__ is DummySubmodule
        assert model.unet.layer.info["hooks"] == []
        hook.remove.assert_called_once()
        assert model._name_or_path == "original-model-name"
        assert model.config._name_or_path == "original-config-name"
    finally:
        switching_threshold_ratio_dict.clear()
        switching_threshold_ratio_dict.update(original_switching)
        text_to_img_controlnet_switching_threshold_ratio_dict.clear()
        text_to_img_controlnet_switching_threshold_ratio_dict.update(original_controlnet)
