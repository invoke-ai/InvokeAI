"""Flux2DevTextEncoderInvocation is idle_gpu_offloadable: it may run on a borrowed idle GPU
whose device-pool lock is released the moment the node returns. Like the other offloadable
encoders (flux_text_encoder, flux2_klein_text_encoder), its saved conditioning must be detached
and moved to the CPU — otherwise the embeddings stay resident on the borrowed device, pinning
VRAM on a GPU another session may immediately start using."""

from unittest.mock import MagicMock

import torch

from invokeai.app.invocations.text_encoder.flux2_dev_text_encoder import Flux2DevTextEncoderInvocation


def test_flux2_dev_conditioning_is_saved_on_cpu(monkeypatch):
    invocation = Flux2DevTextEncoderInvocation.model_construct(
        prompt="a prompt", mistral_encoder=MagicMock(), max_seq_len=512, mask=None
    )

    # Stand-in for a tensor living on a (borrowed) GPU: .detach().to("cpu") yields the CPU copy.
    gpu_mistral = MagicMock(spec=torch.Tensor)
    cpu_mistral = torch.zeros(1, 8, 15360, dtype=torch.bfloat16)
    gpu_mistral.detach.return_value.to.return_value = cpu_mistral

    monkeypatch.setattr(invocation, "_encode_prompt", lambda context, exit_stack: gpu_mistral)

    context = MagicMock()
    context.conditioning.save.return_value = "cond-name"

    invocation.invoke(context)

    gpu_mistral.detach.return_value.to.assert_called_once_with("cpu")
    saved = context.conditioning.save.call_args.args[0]
    info = saved.conditionings[0]
    assert info.t5_embeds is cpu_mistral
    # The placeholder clip_embeds is created from the (now CPU) mistral tensor's device.
    assert info.clip_embeds.device.type == "cpu"
