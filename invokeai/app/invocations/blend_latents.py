from typing import Any, Union

import numpy as np
import numpy.typing as npt
import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, LatentsField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "lblend",
    title="Blend Latents",
    tags=["latents", "blend"],
    category="latents",
    version="1.0.3",
)
class BlendLatentsInvocation(BaseInvocation):
    """Blend two latents using a given alpha. Latents must have same size."""

    latents_a: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    latents_b: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    alpha: float = InputField(default=0.5, description=FieldDescriptions.blend_alpha)

    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents_a = context.tensors.load(self.latents_a.latents_name)
        latents_b = context.tensors.load(self.latents_b.latents_name)

        if latents_a.shape != latents_b.shape:
            raise Exception("Latents to blend must be the same size.")

        device = TorchDevice.choose_torch_device()

        def slerp(
            t: Union[float, npt.NDArray[Any]],  # FIXME: maybe use np.float32 here?
            v0: Union[torch.Tensor, npt.NDArray[Any]],
            v1: Union[torch.Tensor, npt.NDArray[Any]],
            DOT_THRESHOLD: float = 0.9995,
        ) -> Union[torch.Tensor, npt.NDArray[Any]]:
            """
            Spherical linear interpolation
            Args:
                t (float/np.ndarray): Float value between 0.0 and 1.0
                v0 (np.ndarray): Starting vector
                v1 (np.ndarray): Final vector
                DOT_THRESHOLD (float): Threshold for considering the two vectors as
                                    colineal. Not recommended to alter this.
            Returns:
                v2 (np.ndarray): Interpolation vector between v0 and v1
            """
            inputs_are_torch = False
            if not isinstance(v0, np.ndarray):
                inputs_are_torch = True
                v0 = v0.detach().cpu().numpy()
            if not isinstance(v1, np.ndarray):
                inputs_are_torch = True
                v1 = v1.detach().cpu().numpy()

            dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
            if np.abs(dot) > DOT_THRESHOLD:
                v2 = (1 - t) * v0 + t * v1
            else:
                theta_0 = np.arccos(dot)
                sin_theta_0 = np.sin(theta_0)
                theta_t = theta_0 * t
                sin_theta_t = np.sin(theta_t)
                s0 = np.sin(theta_0 - theta_t) / sin_theta_0
                s1 = sin_theta_t / sin_theta_0
                v2 = s0 * v0 + s1 * v1

            if inputs_are_torch:
                v2_torch: torch.Tensor = torch.from_numpy(v2).to(device)
                return v2_torch
            else:
                assert isinstance(v2, np.ndarray)
                return v2

        # blend
        bl = slerp(self.alpha, latents_a, latents_b)
        assert isinstance(bl, torch.Tensor)
        blended_latents: torch.Tensor = bl  # for type checking convenience

        # https://discuss.huggingface.co/t/memory-usage-by-later-pipeline-stages/23699
        blended_latents = blended_latents.to("cpu")

        TorchDevice.empty_cache()

        name = context.tensors.save(tensor=blended_latents)
        return LatentsOutput.build(latents_name=name, latents=blended_latents, seed=self.latents_a.seed)
