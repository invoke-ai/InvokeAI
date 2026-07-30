"""Flux2KleinTextEncoderInvocation is idle_gpu_offloadable: it may run on a borrowed idle GPU
whose device-pool lock is released the moment the node returns. Like the other offloadable
encoders (flux_text_encoder, flux_redux), its saved conditioning must be detached and moved to
the CPU — otherwise the embeddings stay resident on the borrowed device, pinning VRAM on a GPU
another session may immediately start using (JPPhoto merge blocker, 2026-07-22)."""

from unittest.mock import MagicMock

import torch

from invokeai.app.invocations.flux2_klein_text_encoder import Flux2KleinTextEncoderInvocation


def test_flux2_klein_conditioning_is_saved_on_cpu(monkeypatch):
    invocation = Flux2KleinTextEncoderInvocation.model_construct(
        prompt="a prompt", qwen3_encoder=MagicMock(), max_seq_len=512, mask=None
    )

    # Stand-ins for tensors living on a (borrowed) GPU: .detach().to("cpu") yields the CPU copy.
    gpu_qwen3 = MagicMock(spec=torch.Tensor)
    gpu_pooled = MagicMock(spec=torch.Tensor)
    cpu_qwen3 = MagicMock(spec=torch.Tensor)
    cpu_pooled = MagicMock(spec=torch.Tensor)
    gpu_qwen3.detach.return_value.to.return_value = cpu_qwen3
    gpu_pooled.detach.return_value.to.return_value = cpu_pooled

    monkeypatch.setattr(invocation, "_encode_prompt", lambda context, exit_stack: (gpu_qwen3, gpu_pooled))

    context = MagicMock()
    context.conditioning.save.return_value = "cond-name"

    invocation.invoke(context)

    gpu_qwen3.detach.return_value.to.assert_called_once_with("cpu")
    gpu_pooled.detach.return_value.to.assert_called_once_with("cpu")
    saved = context.conditioning.save.call_args.args[0]
    info = saved.conditionings[0]
    assert info.t5_embeds is cpu_qwen3
    assert info.clip_embeds is cpu_pooled
