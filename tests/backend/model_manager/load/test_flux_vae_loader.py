"""The FLUX.1 VAE loader: the config it builds, and the two packagings it accepts.

The FLUX.1 autoencoder is loaded as a diffusers `AutoencoderKL` rather than through InvokeAI's own
port of the BFL reference. That is the same network -- measured in fp32 at `maxdiff 1.6e-06` on a
latent and `1.5e-05` on an image -- and it is what gives the FLUX.1 nodes a tiled *encode*, which
the port never had.

Two things can break silently and are pinned here:

* **the config.** It is derived from `AutoEncoderParams` and leaves six fields to diffusers'
  defaults. A changed default in a future diffusers would build a different autoencoder out of the
  same checkpoint, so the constructed config is compared against the values `black-forest-labs`
  publishes rather than against the code that produced it.
* **the refusal.** `VAE_Checkpoint_FLUX_Config` recognises a FLUX VAE by `encoder.conv_in` plus 16
  latent channels, and files every 16-channel checkpoint whose name says nothing under `flux`. The SD 3.5
  and CogView 4 autoencoders are identical in shape to this one -- 244 keys, same shapes, differing
  only in `scaling_factor`/`shift_factor` -- so a standalone diffusers-layout file, which carries no
  config, cannot be attributed to a latent space. The loader refuses it rather than stamping FLUX's
  constants on weights that may not be FLUX's.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from invokeai.backend.flux.util import get_flux_vae_diffusers_config
from invokeai.backend.model_manager.load.model_loaders.flux import (
    _FLUX_VAE_BFL_MARKER,
    _FLUX_VAE_DIFFUSERS_MARKER,
    FluxVAELoader,
)
from invokeai.backend.util.vae_tiling_scope import scoped_vae_tiling

# `black-forest-labs/FLUX.1-dev::vae/config.json` and `…-schnell::vae/config.json`, which are
# byte-identical to each other (774 bytes). Written out rather than fetched: the point is an
# expectation that does not move when the code does.
PUBLISHED_CONFIG = {
    "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "force_upcast": True,
    "in_channels": 3,
    "latent_channels": 16,
    "latents_mean": None,
    "latents_std": None,
    "layers_per_block": 2,
    "mid_block_add_attention": True,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 1024,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "use_post_quant_conv": False,
    "use_quant_conv": False,
}


def _loader(dtype: torch.dtype = torch.bfloat16) -> FluxVAELoader:
    loader = FluxVAELoader.__new__(FluxVAELoader)
    loader._torch_dtype = dtype
    loader._torch_device = torch.device("cpu")
    return loader


def _config_for(path: Path) -> MagicMock:
    from invokeai.backend.model_manager.configs.vae import VAE_Checkpoint_Config_Base

    config = MagicMock(spec=VAE_Checkpoint_Config_Base)
    config.path = str(path)
    return config


class TestTheConstructedConfig:
    def test_it_matches_what_black_forest_labs_publishes(self):
        built = dict(AutoencoderKL(**get_flux_vae_diffusers_config()).config)

        def normalise(value):
            return list(value) if isinstance(value, (list, tuple)) else value

        differs = {
            field: (normalise(built.get(field, "<absent>")), normalise(expected))
            for field, expected in PUBLISHED_CONFIG.items()
            if normalise(built.get(field, "<absent>")) != normalise(expected)
        }
        assert not differs

        # Also the other direction: a *new* constructor argument in a future diffusers, with a
        # default that changes the network, would pass the loop above by not being in it.
        # Underscore-prefixed keys are diffusers' own bookkeeping (`_class_name`,
        # `_diffusers_version`, `_use_default_values`), not network geometry.
        unexpected = sorted(field for field in set(built) - set(PUBLISHED_CONFIG) if not field.startswith("_"))
        assert not unexpected, f"diffusers grew config fields this test does not know about: {unexpected}"

    def test_it_is_derived_from_the_bfl_parameters_rather_than_transcribed(self):
        """`AutoEncoderParams` stays the single source of truth: the PiD nodes read
        `scale_factor`/`shift_factor` off it, and a value edited there has to reach the loader."""
        from invokeai.backend.flux.util import get_flux_ae_params

        params = get_flux_ae_params()
        config = get_flux_vae_diffusers_config()
        assert config["latent_channels"] == params.z_channels
        assert config["scaling_factor"] == params.scale_factor
        assert config["shift_factor"] == params.shift_factor
        assert config["block_out_channels"] == tuple(params.ch * mult for mult in params.ch_mult)
        assert config["layers_per_block"] == params.num_res_blocks


class TestWhatIsRefused:
    def test_a_diffusers_layout_checkpoint_is_refused_because_it_cannot_be_attributed(self, tmp_path):
        """The hazard this refusal exists for.

        Identification sends any 16-channel autoencoder here as `flux`. SD 3.5's VAE has the same
        244 keys with the same shapes and differs only in `scaling_factor`/`shift_factor`, so
        loading a standalone diffusers-layout file means stamping FLUX's constants on weights that
        may be SD 3.5's -- which loads cleanly, generates without error, and produces an image
        normalised by the wrong constant that nothing downstream can distinguish from an intended
        one. `is_flux_family_vae` cannot catch it either: it reads the config the loader synthesised.
        """
        from safetensors.torch import save_file

        reference = AutoencoderKL(**get_flux_vae_diffusers_config())
        assert _FLUX_VAE_DIFFUSERS_MARKER in reference.state_dict(), "the marker must exist in this layout"

        path = tmp_path / "diffusion_pytorch_model.safetensors"
        save_file({k: v.contiguous() for k, v in list(reference.state_dict().items())[:8]}, path)

        with pytest.raises(ValueError, match="carries no config"):
            _loader()._load_model(_config_for(path))

    def test_an_sd35_shaped_checkpoint_is_not_silently_loaded_as_flux(self, tmp_path):
        """The same hazard stated as the thing a user would actually install.

        SD 3.5's VAE is published in the diffusers layout and is 16-channel; saved under its published
        name, `diffusion_pytorch_model.safetensors`, it names no backbone, so identification labels it
        `flux`. If this ever starts loading, the cell above is the one that explains why it must not.
        """
        from safetensors.torch import save_file

        sd35 = AutoencoderKL(**dict(get_flux_vae_diffusers_config(), scaling_factor=1.5305, shift_factor=0.0609))
        path = tmp_path / "diffusion_pytorch_model.safetensors"
        save_file({k: v.contiguous() for k, v in list(sd35.state_dict().items())[:8]}, path)

        with pytest.raises(ValueError):
            _loader()._load_model(_config_for(path))

    def test_a_checkpoint_in_no_known_layout_is_refused_by_name(self, tmp_path):
        from safetensors.torch import save_file

        path = tmp_path / "not_a_flux_vae.safetensors"
        save_file({"encoder.conv_in.weight": torch.zeros(4, 3, 3, 3)}, path)

        with pytest.raises(ValueError) as excinfo:
            _loader()._load_model(_config_for(path))

        # The marker is named: "not a FLUX VAE" about a file identification already accepted is not
        # actionable on its own.
        assert _FLUX_VAE_BFL_MARKER in str(excinfo.value)


_REAL_VAE_ENV = "INVOKEAI_FLUX_VAE_CORPUS"


def _real_bfl_vae() -> Path | None:
    root = os.environ.get(_REAL_VAE_ENV)
    if not root:
        return None
    path = Path(root) / "ae.safetensors"
    return path if path.exists() else None


@pytest.mark.slow
class TestTheRealCheckpoint:
    """The BFL `ae.safetensors`, when it is on disk -- the packaging every FLUX install uses.

    Set `INVOKEAI_FLUX_VAE_CORPUS` to a directory holding it. Skipped otherwise: the file is 320 MiB
    and CI has no business downloading it.
    """

    def test_it_loads_completely_through_the_conversion(self):
        path = _real_bfl_vae()
        if path is None:
            pytest.skip(f"set {_REAL_VAE_ENV} to a directory holding ae.safetensors")

        model = _loader(torch.float32)._load_model(_config_for(path))
        unfilled = [k for k, v in model.state_dict().items() if v.is_meta]
        assert not unfilled, f"{len(unfilled)} tensors were never filled by the conversion"

    def test_a_tiled_decode_stays_inside_the_measured_accuracy_band(self):
        """The property the deleted `test_autoencoder_tiling.py` asserted, on the class that now
        does the work.

        A tiled pass is not the untiled one: each tile is convolved against zero padding at its own
        borders, and the GroupNorms and mid-block attention are global. The node warns the user about
        exactly that, and this is the bound behind the warning.

        **The bound is on the mean, not the max.** Measured here across three seeds, fp32 on CPU,
        the max swings between 0.125 and 0.231 at 768px -- unstructured Gaussian latents put the VAE
        far outside the range a denoiser produces, and one outlier pixel moves it. The mean is stable
        to three digits over the same seeds (0.00305-0.00318 at 768px, 0.00383-0.00402 at 1536px),
        so it is the quantity that answers "did the geometry change". The max is kept as a loose
        sanity ceiling.
        """
        path = _real_bfl_vae()
        if path is None:
            pytest.skip(f"set {_REAL_VAE_ENV} to a directory holding ae.safetensors")

        vae = _loader(torch.float32)._load_model(_config_for(path)).eval()
        torch.manual_seed(0)
        for edge, mean_band in ((768, 0.006), (1536, 0.008)):
            latents = torch.randn(1, 16, edge // 8, edge // 8)
            with torch.no_grad():
                with scoped_vae_tiling(vae, None):
                    untiled = vae.decode(latents, return_dict=False)[0]
                with scoped_vae_tiling(vae, 0):
                    tiled = vae.decode(latents, return_dict=False)[0]
            assert tiled.shape == untiled.shape
            difference = (tiled - untiled).abs()
            assert difference.mean().item() <= mean_band, (
                f"{edge}px: mean drift {difference.mean().item():.5f} exceeds {mean_band}"
            )
            assert difference.max().item() <= 0.5, f"{edge}px: max drift {difference.max().item():.4f}"


class TestTheFamilyGuard:
    """`isinstance(vae, AutoencoderKL)` used to mean "the FLUX autoencoder" and no longer does.

    The FLUX.1 VAE is an `AutoencoderKL` now -- and so are SD 1.5, SDXL, SD 3.5, CogView 4 and
    Z-Image's. Two of those even share the 16-channel latent width, so the nodes that used to be
    protected by the class check need something that still separates the families.
    """

    @staticmethod
    def _vae(**overrides) -> AutoencoderKL:
        config = {
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ("DownEncoderBlock2D",) * 4,
            "up_block_types": ("UpDecoderBlock2D",) * 4,
            "block_out_channels": (4, 4, 4, 4),
            "layers_per_block": 1,
            "norm_num_groups": 2,
            "sample_size": 64,
            "latent_channels": 16,
            "scaling_factor": 0.3611,
            "shift_factor": 0.1159,
        }
        config.update(overrides)
        return AutoencoderKL(**config)

    def test_the_flux_autoencoder_passes(self):
        from invokeai.backend.flux.util import is_flux_family_vae

        assert is_flux_family_vae(self._vae()) is True

    def test_an_sd_vae_is_rejected_on_its_latent_width(self):
        from invokeai.backend.flux.util import is_flux_family_vae

        assert is_flux_family_vae(self._vae(latent_channels=4, scaling_factor=0.18215, shift_factor=None)) is False

    def test_an_sd35_vae_is_rejected_although_it_has_the_same_latent_width(self):
        """The case the channel count alone does not catch, and the reason the test is on the
        normalisation: SD 3.5 is 16-channel too, and its latents mean something else."""
        from invokeai.backend.flux.util import is_flux_family_vae

        assert is_flux_family_vae(self._vae(scaling_factor=1.5305, shift_factor=0.0609)) is False

    def test_something_without_a_config_is_rejected_rather_than_raising(self):
        from invokeai.backend.flux.util import is_flux_family_vae

        assert is_flux_family_vae(object()) is False
