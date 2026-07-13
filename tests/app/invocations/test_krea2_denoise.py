import pytest

from invokeai.app.invocations.fields import DenoiseMaskField
from invokeai.app.invocations.krea2_denoise import Krea2DenoiseInvocation


@pytest.mark.parametrize(("denoising_start", "denoising_end"), [(0.75, 0.25), (0.5, 0.5)])
def test_validate_inputs_rejects_empty_or_inverted_denoising_range(
    denoising_start: float, denoising_end: float
) -> None:
    invocation = Krea2DenoiseInvocation.model_construct(
        denoising_start=denoising_start, denoising_end=denoising_end, denoise_mask=None
    )

    with pytest.raises(ValueError, match="denoising_start must be less than denoising_end"):
        invocation._validate_inputs()


def test_validate_inputs_rejects_denoise_mask_without_latents() -> None:
    invocation = Krea2DenoiseInvocation.model_construct(
        denoising_start=0.0,
        denoising_end=1.0,
        denoise_mask=DenoiseMaskField(mask_name="mask"),
        latents=None,
    )

    with pytest.raises(ValueError, match="Initial latents are required when a denoise mask is provided"):
        invocation._validate_inputs()
