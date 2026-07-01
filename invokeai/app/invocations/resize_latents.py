from typing import Literal

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
)
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.util.devices import TorchDevice

LATENTS_INTERPOLATION_MODE = Literal["nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"]


@invocation(
    "lresize",
    title="Resize Latents",
    tags=["latents", "resize"],
    category="latents",
    version="1.0.2",
)
class ResizeLatentsInvocation(BaseInvocation):
    """Resizes latents to explicit width/height (in pixels). Provided dimensions are floor-divided by 8."""

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    width: int = InputField(
        ge=64,
        multiple_of=LATENT_SCALE_FACTOR,
        description=FieldDescriptions.width,
    )
    height: int = InputField(
        ge=64,
        multiple_of=LATENT_SCALE_FACTOR,
        description=FieldDescriptions.width,
    )
    mode: LATENTS_INTERPOLATION_MODE = InputField(default="bilinear", description=FieldDescriptions.interp_mode)
    antialias: bool = InputField(default=False, description=FieldDescriptions.torch_antialias)

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = context.tensors.load(self.latents.latents_name)
        device = TorchDevice.choose_torch_device()

        resized_latents = torch.nn.functional.interpolate(
            latents.to(device),
            size=(self.height // LATENT_SCALE_FACTOR, self.width // LATENT_SCALE_FACTOR),
            mode=self.mode,
            antialias=self.antialias if self.mode in ["bilinear", "bicubic"] else False,
        )

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        resized_latents = resized_latents.to("cpu")

        TorchDevice.empty_cache()

        name = context.tensors.save(tensor=resized_latents)
        return LatentsOutput.build(latents_name=name, latents=resized_latents, seed=self.latents.seed)


@invocation(
    "lscale",
    title="Scale Latents",
    tags=["latents", "resize"],
    category="latents",
    version="1.0.2",
)
class ScaleLatentsInvocation(BaseInvocation):
    """Scales latents by a given factor."""

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    scale_factor: float = InputField(gt=0, description=FieldDescriptions.scale_factor)
    mode: LATENTS_INTERPOLATION_MODE = InputField(default="bilinear", description=FieldDescriptions.interp_mode)
    antialias: bool = InputField(default=False, description=FieldDescriptions.torch_antialias)

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = context.tensors.load(self.latents.latents_name)

        device = TorchDevice.choose_torch_device()

        # resizing
        resized_latents = torch.nn.functional.interpolate(
            latents.to(device),
            scale_factor=self.scale_factor,
            mode=self.mode,
            antialias=self.antialias if self.mode in ["bilinear", "bicubic"] else False,
        )

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        resized_latents = resized_latents.to("cpu")
        TorchDevice.empty_cache()

        name = context.tensors.save(tensor=resized_latents)
        return LatentsOutput.build(latents_name=name, latents=resized_latents, seed=self.latents.seed)
