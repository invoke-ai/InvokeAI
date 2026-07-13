import pytest

from invokeai.backend.krea2.vae_compat import as_qwen_image_vae


def test_as_qwen_image_vae_rejects_incompatible_model() -> None:
    with pytest.raises(TypeError, match="Expected AutoencoderKLQwenImage or AutoencoderKLWan"):
        as_qwen_image_vae(object())
