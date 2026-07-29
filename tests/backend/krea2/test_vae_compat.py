import accelerate
import pytest
from diffusers.models.autoencoders import AutoencoderKLWan

from invokeai.backend.krea2.vae_compat import as_qwen_image_vae


def test_as_qwen_image_vae_preserves_the_cached_model_and_its_hooks() -> None:
    with accelerate.init_empty_weights():
        model = AutoencoderKLWan()

    hook = model.register_forward_pre_hook(lambda *_args: None)

    converted = as_qwen_image_vae(model)

    assert converted is model
    assert hook.id in converted._forward_pre_hooks


def test_as_qwen_image_vae_rejects_incompatible_model() -> None:
    with pytest.raises(TypeError, match="Expected AutoencoderKLQwenImage or AutoencoderKLWan"):
        as_qwen_image_vae(object())


def test_as_qwen_image_vae_accepts_default_16_channel_wan() -> None:
    # The default AutoencoderKLWan config is the Qwen-Image geometry (16 channels, 8x spatial, no patching).
    with accelerate.init_empty_weights():
        model = AutoencoderKLWan()
    assert as_qwen_image_vae(model) is model


def test_as_qwen_image_vae_rejects_48_channel_wan_2_2_vae() -> None:
    # Wan 2.2's VAE is 48-channel and would fail the 16-vs-48 per-channel normalization.
    with accelerate.init_empty_weights():
        model = AutoencoderKLWan(z_dim=48)
    with pytest.raises(ValueError, match="not Qwen-Image-compatible"):
        as_qwen_image_vae(model)


def test_as_qwen_image_vae_rejects_patchified_wan_vae() -> None:
    with accelerate.init_empty_weights():
        model = AutoencoderKLWan(patch_size=2, is_residual=True)
    with pytest.raises(ValueError, match="not Qwen-Image-compatible"):
        as_qwen_image_vae(model)
