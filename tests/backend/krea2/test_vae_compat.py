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
