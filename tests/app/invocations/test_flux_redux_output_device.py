"""FluxReduxInvocation is idle_gpu_offloadable: it may run on a borrowed idle GPU while its
consumer (FLUX denoise) runs on the session's GPU. Like the other offloadable encoders, its saved
conditioning must be moved to the CPU — otherwise the embeddings stay resident on the borrowed
device and downstream concatenation operates on mixed devices."""

from unittest.mock import MagicMock

import torch

from invokeai.app.invocations.fields import ImageField
from invokeai.app.invocations.flux_redux import FluxReduxInvocation


def test_flux_redux_conditioning_is_saved_on_cpu(monkeypatch):
    invocation = FluxReduxInvocation.model_construct(
        image=ImageField(image_name="img"), mask=None, downsampling_factor=1, weight=1.0
    )

    # A stand-in for a tensor living on a (borrowed) GPU: .detach().to("cpu") yields the CPU copy.
    gpu_conditioning = MagicMock(spec=torch.Tensor)
    cpu_conditioning = MagicMock(spec=torch.Tensor)
    gpu_conditioning.detach.return_value.to.return_value = cpu_conditioning

    monkeypatch.setattr(invocation, "_siglip_encode", lambda context, image: MagicMock())
    monkeypatch.setattr(invocation, "_flux_redux_encode", lambda context, encoded_x: gpu_conditioning)

    context = MagicMock()
    context.tensors.save.return_value = "tensor-name"

    invocation.invoke(context)

    gpu_conditioning.detach.return_value.to.assert_called_once_with("cpu")
    context.tensors.save.assert_called_once_with(cpu_conditioning)
