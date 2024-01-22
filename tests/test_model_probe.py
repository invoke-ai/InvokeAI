from pathlib import Path

import pytest

from invokeai.backend import BaseModelType
from invokeai.backend.model_manager.probe import VaeFolderProbe


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
    assert repo_variant == 'default'

def test_repo_variant(datadir: Path):
    probe = VaeFolderProbe(datadir / "vae" / "taesdxl-fp16")
    repo_variant = probe.get_repo_variant()
    assert repo_variant == 'fp16'
