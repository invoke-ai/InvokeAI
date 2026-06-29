from unittest.mock import MagicMock

import pytest
import torch


@pytest.mark.parametrize(
    ("noise_type", "width", "height", "expected_shape"),
    [
        ("SD", 64, 64, (1, 4, 8, 8)),
        ("FLUX", 64, 64, (1, 16, 8, 8)),
        ("FLUX.2", 64, 64, (1, 32, 8, 8)),
        ("SD3", 64, 64, (1, 16, 8, 8)),
        ("CogView4", 64, 64, (1, 16, 8, 8)),
        ("Z-Image", 64, 64, (1, 16, 8, 8)),
        ("Anima", 64, 64, (1, 16, 1, 8, 8)),
    ],
)
def test_noise_invocation_generates_expected_shapes(noise_type: str, width: int, height: int, expected_shape):
    from invokeai.app.invocations.noise import NoiseInvocation

    mock_context = MagicMock()
    mock_context.tensors.save.return_value = "noise-name"

    invocation = NoiseInvocation(noise_type=noise_type, width=width, height=height, seed=123)

    output = invocation.invoke(mock_context)

    saved_tensor = mock_context.tensors.save.call_args.kwargs["tensor"]
    assert saved_tensor.shape == expected_shape
    assert output.noise.seed == 123
    assert output.width == width
    assert output.height == height


def test_noise_invocation_defaults_to_sd_shape():
    from invokeai.app.invocations.noise import NoiseInvocation

    mock_context = MagicMock()
    mock_context.tensors.save.return_value = "noise-name"

    invocation = NoiseInvocation(width=64, height=64, seed=1)

    invocation.invoke(mock_context)

    saved_tensor = mock_context.tensors.save.call_args.kwargs["tensor"]
    assert saved_tensor.shape == (1, 4, 8, 8)


@pytest.mark.parametrize(
    ("noise_type", "width", "height", "message"),
    [
        ("SD", 66, 64, "multiple of 8"),
        ("FLUX", 72, 64, "multiple of 16"),
        ("FLUX.2", 64, 72, "multiple of 16"),
        ("SD3", 72, 64, "multiple of 16"),
        ("Z-Image", 64, 72, "multiple of 16"),
        ("CogView4", 64, 80, "multiple of 32"),
        ("Anima", 66, 64, "multiple of 8"),
    ],
)
def test_noise_invocation_rejects_invalid_dimensions(noise_type: str, width: int, height: int, message: str):
    from invokeai.app.invocations.noise import NoiseInvocation

    mock_context = MagicMock()

    with pytest.raises(ValueError, match=message):
        invocation = NoiseInvocation(noise_type=noise_type, width=width, height=height, seed=0)
        invocation.invoke(mock_context)


def test_noise_invocation_is_deterministic_for_identical_inputs():
    from invokeai.app.invocations.noise import NoiseInvocation

    mock_context = MagicMock()
    mock_context.tensors.save.side_effect = ["noise-1", "noise-2"]

    invocation = NoiseInvocation(noise_type="FLUX", width=64, height=64, seed=7)

    invocation.invoke(mock_context)
    first = mock_context.tensors.save.call_args_list[0].kwargs["tensor"]
    invocation.invoke(mock_context)
    second = mock_context.tensors.save.call_args_list[1].kwargs["tensor"]
    assert torch.equal(first, second)


@pytest.mark.parametrize(("noise_type", "expected_shape"), [("FLUX", (1, 16, 8, 8)), ("FLUX.2", (1, 32, 8, 8))])
def test_generate_noise_tensor_honors_use_cpu_false_for_flux_variants(noise_type: str, expected_shape):
    from invokeai.app.invocations.latent_noise import generate_noise_tensor

    noise = generate_noise_tensor(
        noise_type=noise_type,
        width=64,
        height=64,
        seed=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
        use_cpu=False,
    )

    assert noise.shape == expected_shape
