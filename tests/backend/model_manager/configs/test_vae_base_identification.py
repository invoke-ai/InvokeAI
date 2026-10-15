"""Which base a standalone `AutoencoderKL` VAE is filed under.

FLUX.1's and SD3's autoencoders are one network: same tensors, same shapes, 16 latent channels. Only
the constants that normalise the latents differ, and a single file does not carry them. So a folder
is decided by its `config.json`, a single file by an explicit `base`, then by the backbone its name
states, then by the FLUX.1 default. Each case also checks that no two VAE configs claim the file,
because identification takes whichever match it meets first.

The two published `vae/` folders are covered as fixtures in `tests/model_identification`; these
cells cover the rules around them.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelSourceType, ModelType

FLUX = BaseModelType.Flux
SD3 = BaseModelType.StableDiffusion3


def _checkpoint(path: Path, latent_channels: int = 16) -> Path:
    """A single-file VAE as identification sees it: a decoder whose first convolution reads the latents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "encoder.conv_in.weight": torch.zeros(8, 3, 3, 3),
            "decoder.conv_in.weight": torch.zeros(8, latent_channels, 3, 3),
        },
        path,
    )
    return path


def _folder(path: Path, **config: float) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKL", **config}))
    latent_channels = int(config.get("latent_channels", 4))
    save_file(
        {"decoder.conv_in.weight": torch.zeros(8, latent_channels, 3, 3)}, path / "diffusion_pytorch_model.safetensors"
    )
    return path


def _identified_base(path: Path, **override_fields: object) -> BaseModelType | None:
    result = ModelConfigFactory.from_model_on_disk(path, override_fields or None, allow_unknown=False)
    vae_matches = [m for m in result.details.values() if isinstance(m, Config_Base) and m.type is ModelType.VAE]
    assert len(vae_matches) <= 1, f"more than one VAE config claims {path.name}: {vae_matches}"
    return result.config.base if result.config is not None else None


class TestASingleFile:
    @pytest.mark.parametrize(
        ("relative_path", "expected"),
        [
            # BFL's own file names no backbone, and every unnamed 16-channel VAE was FLUX.1 before SD3 had a class.
            ("ae.safetensors", FLUX),
            ("sd3_vae.safetensors", SD3),
            ("sd3.5_large_vae.safetensors", SD3),
            ("sd35_vae.safetensors", SD3),
            ("sd35l_vae.safetensors", SD3),
            ("sd3m_vae.safetensors", SD3),
            # A Hugging Face download keeps the repository's name in its folder.
            ("stable-diffusion-3.5-large/diffusion_pytorch_model.safetensors", SD3),
            # The file's own name is more specific than the folder it sits in.
            ("sd3/flux1_vae.safetensors", FLUX),
            # Two backbones in one name decide nothing.
            ("flux_vs_sd3_vae.safetensors", FLUX),
            # SDXL is not a 16-channel base, so the name contradicts the weights and is ignored.
            ("sdxl_vae.safetensors", FLUX),
        ],
    )
    def test_a_16_channel_vae_is_filed_by_the_backbone_its_name_states(
        self, tmp_path: Path, relative_path: str, expected: BaseModelType
    ) -> None:
        assert _identified_base(_checkpoint(tmp_path / relative_path)) is expected

    def test_an_install_source_names_the_backbone_a_generic_file_name_does_not(self, tmp_path: Path) -> None:
        path = _checkpoint(tmp_path / "0f5c3b1e" / "diffusion_pytorch_model.safetensors")
        base = _identified_base(
            path,
            source="stabilityai/stable-diffusion-3.5-large::vae/diffusion_pytorch_model.safetensors",
            source_type=ModelSourceType.HFRepoID,
        )
        assert base is SD3

    @pytest.mark.parametrize(
        ("file_name", "override", "expected"),
        [
            ("ae.safetensors", SD3, SD3),
            ("sd3_vae.safetensors", FLUX, FLUX),
        ],
    )
    def test_an_explicit_base_outranks_the_name(
        self, tmp_path: Path, file_name: str, override: BaseModelType, expected: BaseModelType
    ) -> None:
        assert _identified_base(_checkpoint(tmp_path / file_name), base=override) is expected

    @pytest.mark.parametrize(
        ("latent_channels", "override"),
        [
            # The weights say 16 channels; no override turns that into an SD1 VAE.
            (16, BaseModelType.StableDiffusion1),
            # No family has 8 channels, so no base can load it.
            (8, FLUX),
        ],
    )
    def test_an_explicit_base_outside_the_latent_family_is_refused(
        self, tmp_path: Path, latent_channels: int, override: BaseModelType
    ) -> None:
        path = _checkpoint(tmp_path / "vae.safetensors", latent_channels=latent_channels)
        assert _identified_base(path, base=override) is None

    def test_an_explicit_base_is_no_longer_overruled_within_the_4_channel_family(self, tmp_path: Path) -> None:
        """It used to be accepted and then re-derived from the name, which filed an SD2 VAE as Unknown."""
        path = _checkpoint(tmp_path / "vae.safetensors", latent_channels=4)
        assert _identified_base(path) is BaseModelType.StableDiffusion1
        assert _identified_base(path, base=BaseModelType.StableDiffusion2) is BaseModelType.StableDiffusion2


class TestAFolder:
    @pytest.mark.parametrize("override", [None, FLUX, SD3])
    def test_a_16_channel_config_normalised_as_neither_flux_nor_sd3_is_not_filed_under_either(
        self, tmp_path: Path, override: BaseModelType | None
    ) -> None:
        """CogView 4's constants; taef1 and taesd3 carry a `scaling_factor` of 1.0 too. Such a folder used to be
        filed as SD1, and filing it under FLUX.1 or SD3 -- guessed or asked for -- would decode their latents
        with the wrong constants."""
        path = _folder(tmp_path / "vae", latent_channels=16, scaling_factor=1.0, shift_factor=0.0)
        assert _identified_base(path, **({"base": override} if override else {})) is None

    def test_an_explicit_base_does_not_outrank_the_constants_in_the_config(self, tmp_path: Path) -> None:
        """The config is not a guess: it states the normalisation the weights were trained with."""
        path = _folder(tmp_path / "vae", latent_channels=16, scaling_factor=0.3611, shift_factor=0.1159)
        assert _identified_base(path, base=SD3) is None

    def test_an_explicit_base_outranks_the_4_channel_heuristic(self, tmp_path: Path) -> None:
        path = _folder(tmp_path / "vae", latent_channels=4, scaling_factor=0.18215)
        assert _identified_base(path) is BaseModelType.StableDiffusion1
        assert _identified_base(path, base=BaseModelType.StableDiffusionXL) is BaseModelType.StableDiffusionXL
