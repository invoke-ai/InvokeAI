from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from invokeai.backend.stable_diffusion.vae_tiling import patch_vae_tiling_params


def test_patch_vae_tiling_params():
    """Smoke test the patch_vae_tiling_params(...) context manager. The main purpose of this unit test is to detect if
    diffusers ever changes the attributes of the AutoencoderKL class that we expect to exist.
    """
    vae = AutoencoderKL()

    with patch_vae_tiling_params(vae, 1, 2, 3):
        pass
