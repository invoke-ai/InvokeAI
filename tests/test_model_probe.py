from pathlib import Path

import pytest
from torch import tensor

from invokeai.backend.model_manager import BaseModelType, ModelRepoVariant
from invokeai.backend.model_manager.config import InvalidModelConfigException
from invokeai.backend.model_manager.probe import (
    CkptType,
    ModelProbe,
    VaeFolderProbe,
    get_default_settings_controlnet_t2i_adapter,
    get_default_settings_main,
)


@pytest.mark.parametrize(
    "vae_path,expected_type",
    [
        ("sd-vae-ft-mse", BaseModelType.StableDiffusion1),
        ("sdxl-vae", BaseModelType.StableDiffusionXL),
        ("taesd", BaseModelType.StableDiffusion1),
        ("taesdxl", BaseModelType.StableDiffusionXL),
    ],
)
def test_get_base_type(vae_path: str, expected_type: BaseModelType, datadir: Path):
    sd1_vae_path = datadir / "vae" / vae_path
    probe = VaeFolderProbe(sd1_vae_path)
    base_type = probe.get_base_type()
    assert base_type == expected_type
    repo_variant = probe.get_repo_variant()
    assert repo_variant == ModelRepoVariant.Default


def test_repo_variant(datadir: Path):
    probe = VaeFolderProbe(datadir / "vae" / "taesdxl-fp16")
    repo_variant = probe.get_repo_variant()
    assert repo_variant == ModelRepoVariant.FP16


def test_controlnet_t2i_default_settings():
    assert get_default_settings_controlnet_t2i_adapter("some_canny_model").preprocessor == "canny_image_processor"
    assert (
        get_default_settings_controlnet_t2i_adapter("some_depth_model").preprocessor == "depth_anything_image_processor"
    )
    assert get_default_settings_controlnet_t2i_adapter("some_pose_model").preprocessor == "dw_openpose_image_processor"
    assert get_default_settings_controlnet_t2i_adapter("i like turtles") is None


def test_default_settings_main():
    assert get_default_settings_main(BaseModelType.StableDiffusion1).width == 512
    assert get_default_settings_main(BaseModelType.StableDiffusion1).height == 512
    assert get_default_settings_main(BaseModelType.StableDiffusion2).width == 512
    assert get_default_settings_main(BaseModelType.StableDiffusion2).height == 512
    assert get_default_settings_main(BaseModelType.StableDiffusionXL).width == 1024
    assert get_default_settings_main(BaseModelType.StableDiffusionXL).height == 1024
    assert get_default_settings_main(BaseModelType.StableDiffusionXLRefiner) is None
    assert get_default_settings_main(BaseModelType.Any) is None


def test_probe_handles_state_dict_with_integer_keys():
    # This structure isn't supported by invoke, but we still need to handle it gracefully. See #6044
    state_dict_with_integer_keys: CkptType = {
        320: (
            {
                "linear1.weight": tensor([1.0]),
                "linear1.bias": tensor([1.0]),
                "linear2.weight": tensor([1.0]),
                "linear2.bias": tensor([1.0]),
            },
            {
                "linear1.weight": tensor([1.0]),
                "linear1.bias": tensor([1.0]),
                "linear2.weight": tensor([1.0]),
                "linear2.bias": tensor([1.0]),
            },
        ),
    }
    with pytest.raises(InvalidModelConfigException):
        ModelProbe.get_model_type_from_checkpoint(Path("embedding.pt"), state_dict_with_integer_keys)
