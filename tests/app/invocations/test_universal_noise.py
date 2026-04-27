from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.model import ModelIdentifierField, TransformerField


def build_transformer_field(base):
    return TransformerField(
        transformer=ModelIdentifierField(
            key="model-key",
            hash="model-hash",
            name="model-name",
            base=base,
            type="main",
            submodel_type=None,
        ),
        loras=[],
    )


def build_transformer_info(in_channels: int):
    return SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(in_channels=in_channels)))


@pytest.mark.parametrize(
    ("noise_type", "width", "height", "expected_shape"),
    [
        ("SD", 64, 64, (1, 4, 8, 8)),
        ("FLUX", 64, 64, (1, 16, 8, 8)),
        ("FLUX.2", 64, 64, (1, 32, 8, 8)),
        ("Z-Image", 64, 64, (1, 16, 8, 8)),
        ("Anima", 64, 64, (1, 16, 1, 8, 8)),
    ],
)
def test_universal_noise_positive_fixed_architectures(noise_type: str, width: int, height: int, expected_shape):
    from invokeai.app.invocations.universal_noise import UniversalNoiseInvocation

    mock_context = MagicMock()
    mock_context.tensors.save.return_value = "noise-name"

    invocation = UniversalNoiseInvocation(noise_type=noise_type, width=width, height=height, seed=123)

    output = invocation.invoke(mock_context)

    saved_tensor = mock_context.tensors.save.call_args.kwargs["tensor"]
    assert saved_tensor.shape == expected_shape
    assert output.noise.seed == 123
    assert output.width == width
    assert output.height == height


@pytest.mark.parametrize(
    ("noise_type", "base", "in_channels"),
    [
        ("SD3", "sd-3", 16),
        ("CogView4", "cogview4", 32),
    ],
)
def test_universal_noise_positive_transformer_architectures(noise_type: str, base: str, in_channels: int):
    from invokeai.app.invocations.universal_noise import UniversalNoiseInvocation

    mock_context = MagicMock()
    mock_context.tensors.save.return_value = "noise-name"
    mock_context.models.load.return_value = build_transformer_info(in_channels)

    invocation = UniversalNoiseInvocation(
        noise_type=noise_type,
        width=64,
        height=64,
        seed=321,
        transformer=build_transformer_field(base),
    )

    output = invocation.invoke(mock_context)

    saved_tensor = mock_context.tensors.save.call_args.kwargs["tensor"]
    assert saved_tensor.shape == (1, in_channels, 8, 8)
    assert output.noise.seed == 321


@pytest.mark.parametrize("noise_type", ["SD3", "CogView4"])
def test_universal_noise_requires_transformer_for_model_driven_types(noise_type: str):
    from invokeai.app.invocations.universal_noise import UniversalNoiseInvocation

    mock_context = MagicMock()
    invocation = UniversalNoiseInvocation(noise_type=noise_type, width=64, height=64, seed=0)

    with pytest.raises(ValueError, match="requires a transformer input"):
        invocation.invoke(mock_context)


@pytest.mark.parametrize(
    ("noise_type", "base"),
    [
        ("SD3", "flux"),
        ("CogView4", "sd-3"),
    ],
)
def test_universal_noise_rejects_incompatible_transformer(noise_type: str, base: str):
    from invokeai.app.invocations.universal_noise import UniversalNoiseInvocation

    mock_context = MagicMock()
    invocation = UniversalNoiseInvocation(
        noise_type=noise_type,
        width=64,
        height=64,
        seed=0,
        transformer=build_transformer_field(base),
    )

    with pytest.raises(ValueError, match="Incompatible transformer base"):
        invocation.invoke(mock_context)


@pytest.mark.parametrize("noise_type", ["SD", "FLUX", "FLUX.2", "Z-Image", "Anima"])
def test_universal_noise_rejects_transformer_for_fixed_architectures(noise_type: str):
    from invokeai.app.invocations.universal_noise import UniversalNoiseInvocation

    mock_context = MagicMock()
    invocation = UniversalNoiseInvocation(
        noise_type=noise_type,
        width=64,
        height=64,
        seed=0,
        transformer=build_transformer_field("sd-3"),
    )

    with pytest.raises(ValueError, match="does not accept a transformer input"):
        invocation.invoke(mock_context)


@pytest.mark.parametrize(
    ("noise_type", "width", "height", "message"),
    [
        ("SD", 66, 64, "must be a multiple of 8"),
        ("FLUX", 64, 66, "must be a multiple of 16"),
        ("FLUX.2", 66, 64, "must be a multiple of 16"),
        ("Z-Image", 64, 66, "must be a multiple of 16"),
        ("CogView4", 64, 80, "must be a multiple of 32"),
        ("Anima", 66, 64, "must be a multiple of 8"),
    ],
)
def test_universal_noise_rejects_invalid_dimensions(noise_type: str, width: int, height: int, message: str):
    from invokeai.app.invocations.universal_noise import UniversalNoiseInvocation

    mock_context = MagicMock()
    invocation = UniversalNoiseInvocation(noise_type=noise_type, width=width, height=height, seed=0)

    with pytest.raises(ValueError, match=message):
        invocation.invoke(mock_context)


def test_universal_noise_is_deterministic_for_identical_inputs():
    from invokeai.app.invocations.universal_noise import UniversalNoiseInvocation

    mock_context = MagicMock()
    mock_context.tensors.save.side_effect = ["noise-1", "noise-2"]

    invocation = UniversalNoiseInvocation(noise_type="FLUX", width=64, height=64, seed=7)

    invocation.invoke(mock_context)
    first = mock_context.tensors.save.call_args_list[0].kwargs["tensor"]
    invocation.invoke(mock_context)
    second = mock_context.tensors.save.call_args_list[1].kwargs["tensor"]
    assert torch.equal(first, second)
