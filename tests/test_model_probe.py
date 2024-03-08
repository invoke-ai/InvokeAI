from pathlib import Path

import pytest

from invokeai.backend.model_manager import BaseModelType, ModelRepoVariant
from invokeai.backend.model_manager.probe import ControlAdapterProbe, VaeFolderProbe


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
    should_be_canny = ControlAdapterProbe.get_default_settings("some_canny_model")
    assert should_be_canny and should_be_canny.preprocessor == "canny_image_processor"

    should_be_depth_anything = ControlAdapterProbe.get_default_settings("some_depth_model")
    assert should_be_depth_anything and should_be_depth_anything.preprocessor == "depth_anything_image_processor"

    should_be_dw_openpose = ControlAdapterProbe.get_default_settings("some_pose_model")
    assert should_be_dw_openpose and should_be_dw_openpose.preprocessor == "dw_openpose_image_processor"

    should_be_none = ControlAdapterProbe.get_default_settings("i like turtles")
    assert should_be_none is None
