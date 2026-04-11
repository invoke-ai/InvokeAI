from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.app.invocations.anima_denoise import AnimaDenoiseInvocation
from invokeai.app.invocations.cogview4_denoise import CogView4DenoiseInvocation
from invokeai.app.invocations.flux2_denoise import Flux2DenoiseInvocation
from invokeai.app.invocations.flux_denoise import FluxDenoiseInvocation
from invokeai.app.invocations.sd3_denoise import SD3DenoiseInvocation
from invokeai.app.invocations.z_image_denoise import ZImageDenoiseInvocation


def test_flux_prepare_noise_uses_external_noise():
    invocation = FluxDenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    expected = torch.zeros(1, 16, 8, 8)
    mock_context.tensors.load.return_value = expected

    with patch("invokeai.app.invocations.flux_denoise.get_noise") as mock_get_noise:
        noise = invocation._prepare_noise_tensor(mock_context, torch.bfloat16, torch.device("cpu"))

    assert torch.equal(noise, expected.to(dtype=torch.bfloat16))
    mock_get_noise.assert_not_called()


def test_flux_prepare_noise_rejects_invalid_shape():
    invocation = FluxDenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = torch.zeros(1, 15, 8, 8)

    with pytest.raises(ValueError, match="Expected noise with shape"):
        invocation._prepare_noise_tensor(mock_context, torch.bfloat16, torch.device("cpu"))


def test_flux2_prepare_noise_uses_external_noise():
    invocation = Flux2DenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    expected = torch.zeros(1, 32, 8, 8)
    mock_context.tensors.load.return_value = expected

    with patch("invokeai.app.invocations.flux2_denoise.get_noise_flux2") as mock_get_noise:
        noise = invocation._prepare_noise_tensor(mock_context, torch.bfloat16, torch.device("cpu"))

    assert torch.equal(noise, expected.to(dtype=torch.bfloat16))
    mock_get_noise.assert_not_called()


def test_flux2_prepare_noise_rejects_invalid_shape():
    invocation = Flux2DenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = torch.zeros(1, 16, 8, 8)

    with pytest.raises(ValueError, match="Expected noise with shape"):
        invocation._prepare_noise_tensor(mock_context, torch.bfloat16, torch.device("cpu"))


def test_sd3_prepare_noise_uses_external_noise():
    invocation = SD3DenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    expected = torch.zeros(1, 16, 8, 8)
    mock_context.tensors.load.return_value = expected

    with patch.object(invocation, "_get_noise") as mock_get_noise:
        noise = invocation._prepare_noise_tensor(mock_context, 16, torch.bfloat16, torch.device("cpu"))

    assert torch.equal(noise, expected.to(dtype=torch.bfloat16))
    mock_get_noise.assert_not_called()


def test_sd3_prepare_noise_rejects_invalid_shape():
    invocation = SD3DenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = torch.zeros(1, 8, 8, 8)

    with pytest.raises(ValueError, match="Expected noise with shape"):
        invocation._prepare_noise_tensor(mock_context, 16, torch.bfloat16, torch.device("cpu"))


def test_cogview4_prepare_noise_uses_external_noise():
    invocation = CogView4DenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    expected = torch.zeros(1, 16, 8, 8)
    mock_context.tensors.load.return_value = expected

    with patch.object(invocation, "_get_noise") as mock_get_noise:
        noise = invocation._prepare_noise_tensor(mock_context, 16, torch.bfloat16, torch.device("cpu"))

    assert torch.equal(noise, expected.to(dtype=torch.bfloat16))
    mock_get_noise.assert_not_called()


def test_cogview4_prepare_noise_rejects_invalid_shape():
    invocation = CogView4DenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = torch.zeros(1, 4, 8, 8)

    with pytest.raises(ValueError, match="Expected noise with shape"):
        invocation._prepare_noise_tensor(mock_context, 16, torch.bfloat16, torch.device("cpu"))


def test_z_image_prepare_noise_uses_external_noise():
    invocation = ZImageDenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    expected = torch.zeros(1, 16, 8, 8)
    mock_context.tensors.load.return_value = expected

    with patch.object(invocation, "_get_noise") as mock_get_noise:
        noise = invocation._prepare_noise_tensor(mock_context, torch.bfloat16, torch.device("cpu"))

    assert torch.equal(noise, expected.to(dtype=torch.bfloat16))
    mock_get_noise.assert_not_called()


def test_z_image_prepare_noise_rejects_invalid_shape():
    invocation = ZImageDenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = torch.zeros(1, 8, 8, 8)

    with pytest.raises(ValueError, match="Expected noise with shape"):
        invocation._prepare_noise_tensor(mock_context, torch.bfloat16, torch.device("cpu"))


def test_anima_prepare_noise_uses_external_noise():
    invocation = AnimaDenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    expected = torch.zeros(1, 16, 1, 8, 8)
    mock_context.tensors.load.return_value = expected

    with patch.object(invocation, "_get_noise") as mock_get_noise:
        noise = invocation._prepare_noise_tensor(mock_context, torch.bfloat16, torch.device("cpu"))

    assert torch.equal(noise, expected.to(dtype=torch.bfloat16))
    mock_get_noise.assert_not_called()


def test_anima_prepare_noise_rejects_invalid_rank():
    invocation = AnimaDenoiseInvocation.model_construct(
        width=64, height=64, seed=0, noise=MagicMock(latents_name="noise")
    )
    mock_context = MagicMock()
    mock_context.tensors.load.return_value = torch.zeros(1, 16, 8, 8)

    with pytest.raises(ValueError, match="Expected noise with shape"):
        invocation._prepare_noise_tensor(mock_context, torch.bfloat16, torch.device("cpu"))
