import math
from typing import Literal, Optional

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, LatentsField, OutputField
from invokeai.app.invocations.model import TransformerField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.misc import SEED_MAX
from invokeai.backend.flux.sampling_utils import get_noise as get_flux_noise
from invokeai.backend.flux2.sampling_utils import get_noise_flux2
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.util.devices import TorchDevice

UniversalNoiseType = Literal["SD", "FLUX", "FLUX.2", "SD3", "CogView4", "Z-Image", "Anima"]


def validate_noise_dimensions(noise_type: UniversalNoiseType, width: int, height: int) -> None:
    multiple_of = 8
    if noise_type in ("FLUX", "FLUX.2", "Z-Image", "SD3"):
        multiple_of = 16
    elif noise_type == "CogView4":
        multiple_of = 32

    if width % multiple_of != 0 or height % multiple_of != 0:
        raise ValueError(f"{noise_type} noise width and height must be a multiple of {multiple_of}")


def get_transformer_channels(
    context: InvocationContext, noise_type: UniversalNoiseType, transformer: Optional[TransformerField]
) -> int:
    if noise_type == "SD3":
        expected_base = BaseModelType.StableDiffusion3
    elif noise_type == "CogView4":
        expected_base = BaseModelType.CogView4
    else:
        if transformer is not None:
            raise ValueError(f"{noise_type} does not accept a transformer input")
        return 0

    if transformer is None:
        raise ValueError(f"{noise_type} noise requires a transformer input")
    if transformer.transformer.base != expected_base:
        raise ValueError(
            f"Incompatible transformer base for {noise_type}: expected {expected_base.value}, got "
            f"{transformer.transformer.base.value}"
        )

    loaded_model = context.models.load(transformer.transformer)
    in_channels = loaded_model.model.config.in_channels
    assert isinstance(in_channels, int)
    return in_channels


def get_expected_noise_shape(
    noise_type: UniversalNoiseType, width: int, height: int, num_channels: int | None = None
) -> tuple[int, ...]:
    validate_noise_dimensions(noise_type, width, height)

    if noise_type == "SD":
        return (1, 4, height // LATENT_SCALE_FACTOR, width // LATENT_SCALE_FACTOR)
    if noise_type == "FLUX":
        return (1, 16, 2 * math.ceil(height / 16), 2 * math.ceil(width / 16))
    if noise_type == "FLUX.2":
        return (1, 32, 2 * math.ceil(height / 16), 2 * math.ceil(width / 16))
    if noise_type in ("SD3", "CogView4"):
        if num_channels is None:
            raise ValueError(f"{noise_type} noise requires num_channels")
        return (1, num_channels, height // LATENT_SCALE_FACTOR, width // LATENT_SCALE_FACTOR)
    if noise_type == "Z-Image":
        return (1, 16, height // LATENT_SCALE_FACTOR, width // LATENT_SCALE_FACTOR)
    if noise_type == "Anima":
        return (1, 16, 1, height // LATENT_SCALE_FACTOR, width // LATENT_SCALE_FACTOR)
    raise ValueError(f"Unsupported noise type: {noise_type}")


def validate_noise_tensor_shape(
    noise: torch.Tensor, noise_type: UniversalNoiseType, width: int, height: int, num_channels: int | None = None
) -> None:
    expected_shape = get_expected_noise_shape(noise_type, width, height, num_channels)
    if tuple(noise.shape) != expected_shape:
        raise ValueError(f"Expected noise with shape {expected_shape}, got {tuple(noise.shape)}")


def generate_noise_tensor(
    noise_type: UniversalNoiseType,
    width: int,
    height: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    num_channels: int | None = None,
) -> torch.Tensor:
    validate_noise_dimensions(noise_type, width, height)

    if noise_type == "SD":
        return torch.randn(
            1,
            4,
            height // LATENT_SCALE_FACTOR,
            width // LATENT_SCALE_FACTOR,
            dtype=TorchDevice.choose_torch_dtype(device=device),
            device="cpu",
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).to("cpu")
    if noise_type == "FLUX":
        return get_flux_noise(num_samples=1, height=height, width=width, device=device, dtype=dtype, seed=seed).to(
            "cpu"
        )
    if noise_type == "FLUX.2":
        return get_noise_flux2(num_samples=1, height=height, width=width, device=device, dtype=dtype, seed=seed).to(
            "cpu"
        )
    if noise_type in ("SD3", "CogView4"):
        if num_channels is None:
            raise ValueError(f"{noise_type} noise requires num_channels")
        return torch.randn(
            1,
            num_channels,
            height // LATENT_SCALE_FACTOR,
            width // LATENT_SCALE_FACTOR,
            device="cpu",
            dtype=torch.float16,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).to("cpu")
    if noise_type == "Z-Image":
        return torch.randn(
            1,
            16,
            height // LATENT_SCALE_FACTOR,
            width // LATENT_SCALE_FACTOR,
            device="cpu",
            dtype=torch.float32,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).to("cpu")
    if noise_type == "Anima":
        return torch.randn(
            1,
            16,
            1,
            height // LATENT_SCALE_FACTOR,
            width // LATENT_SCALE_FACTOR,
            device="cpu",
            dtype=torch.float32,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).to("cpu")
    raise ValueError(f"Unsupported noise type: {noise_type}")


@invocation_output("universal_noise_output")
class UniversalNoiseOutput(BaseInvocationOutput):
    """Invocation output for universal architecture-specific noise."""

    noise: LatentsField = OutputField(description=FieldDescriptions.noise)
    width: int = OutputField(description=FieldDescriptions.width)
    height: int = OutputField(description=FieldDescriptions.height)

    @classmethod
    def build(cls, latents_name: str, latents: torch.Tensor, seed: int) -> "UniversalNoiseOutput":
        return cls(
            noise=LatentsField(latents_name=latents_name, seed=seed),
            width=latents.shape[-1] * LATENT_SCALE_FACTOR,
            height=latents.shape[-2] * LATENT_SCALE_FACTOR,
        )


@invocation(
    "universal_noise",
    title="Universal Noise",
    tags=["latents", "noise"],
    category="latents",
    version="1.0.0",
)
class UniversalNoiseInvocation(BaseInvocation):
    """Generate architecture-specific latent noise for supported denoisers."""

    noise_type: UniversalNoiseType = InputField(description="Architecture-specific noise type.")
    width: int = InputField(default=512, gt=0, description=FieldDescriptions.width)
    height: int = InputField(default=512, gt=0, description=FieldDescriptions.height)
    seed: int = InputField(default=0, ge=0, le=SEED_MAX, description=FieldDescriptions.seed)
    transformer: TransformerField | None = InputField(default=None, input=Input.Connection, title="Transformer")

    def invoke(self, context: InvocationContext) -> UniversalNoiseOutput:
        validate_noise_dimensions(self.noise_type, self.width, self.height)
        num_channels = get_transformer_channels(context, self.noise_type, self.transformer)
        noise = generate_noise_tensor(
            noise_type=self.noise_type,
            width=self.width,
            height=self.height,
            seed=self.seed,
            device=TorchDevice.choose_torch_device(),
            dtype=TorchDevice.choose_torch_dtype(),
            num_channels=num_channels if num_channels > 0 else None,
        )
        name = context.tensors.save(tensor=noise)
        return UniversalNoiseOutput.build(latents_name=name, latents=noise, seed=self.seed)
