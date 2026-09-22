"""The Wan, Krea-2, Ideogram 4 and ERNIE-Image text encoders are idle_gpu_offloadable: each may run
on a borrowed idle GPU whose device-pool lock is released the moment the node returns. Like the
other offloadable encoders (see test_flux2_klein_output_device.py, test_flux_redux_output_device.py),
their saved conditioning must be detached and moved to the CPU — otherwise the embeddings stay
resident on the borrowed device, pinning VRAM on a GPU another session may immediately start using,
and the denoise loop that picks them up on the session's own GPU sees mixed devices.

The flag is declared on the @invocation decorator, so nothing inside these node bodies hints that
the contract exists; these tests are the guard against a future edit dropping the .to("cpu").
"""

from unittest.mock import MagicMock

import torch


def _gpu_tensor_yielding(cpu_tensor: MagicMock) -> MagicMock:
    """A stand-in for a tensor on a (borrowed) GPU: .detach().to("cpu") yields the CPU copy."""
    gpu_tensor = MagicMock(spec=torch.Tensor)
    gpu_tensor.detach.return_value.to.return_value = cpu_tensor
    return gpu_tensor


def test_wan_conditioning_is_saved_on_cpu(monkeypatch):
    from invokeai.app.invocations.wan_text_encoder import WanTextEncoderInvocation

    invocation = WanTextEncoderInvocation.model_construct(prompt="a prompt", wan_t5_encoder=MagicMock())

    cpu_embeds = MagicMock(spec=torch.Tensor)
    cpu_mask = MagicMock(spec=torch.Tensor)
    gpu_embeds = _gpu_tensor_yielding(cpu_embeds)
    gpu_mask = _gpu_tensor_yielding(cpu_mask)
    monkeypatch.setattr(invocation, "_encode", lambda context: (gpu_embeds, gpu_mask))

    context = MagicMock()
    context.conditioning.save.return_value = "cond-name"

    invocation.invoke(context)

    gpu_embeds.detach.return_value.to.assert_called_once_with("cpu")
    gpu_mask.detach.return_value.to.assert_called_once_with("cpu")
    info = context.conditioning.save.call_args.args[0].conditionings[0]
    assert info.prompt_embeds is cpu_embeds
    assert info.prompt_attention_mask is cpu_mask


def test_wan_conditioning_tolerates_a_full_attention_mask(monkeypatch):
    """_encode returns None for the mask when every token is valid; that must not blow up the
    CPU move."""
    from invokeai.app.invocations.wan_text_encoder import WanTextEncoderInvocation

    invocation = WanTextEncoderInvocation.model_construct(prompt="a prompt", wan_t5_encoder=MagicMock())

    cpu_embeds = MagicMock(spec=torch.Tensor)
    monkeypatch.setattr(invocation, "_encode", lambda context: (_gpu_tensor_yielding(cpu_embeds), None))

    context = MagicMock()
    context.conditioning.save.return_value = "cond-name"

    invocation.invoke(context)

    info = context.conditioning.save.call_args.args[0].conditionings[0]
    assert info.prompt_embeds is cpu_embeds
    assert info.prompt_attention_mask is None


def test_krea2_conditioning_is_saved_on_cpu(monkeypatch):
    from invokeai.app.invocations.krea2_text_encoder import Krea2TextEncoderInvocation

    invocation = Krea2TextEncoderInvocation.model_construct(prompt="a prompt", mask=None, qwen3_vl_encoder=MagicMock())

    cpu_embeds = MagicMock(spec=torch.Tensor)
    cpu_mask = MagicMock(spec=torch.Tensor)
    gpu_embeds = _gpu_tensor_yielding(cpu_embeds)
    gpu_mask = _gpu_tensor_yielding(cpu_mask)
    monkeypatch.setattr(invocation, "_encode", lambda context: (gpu_embeds, gpu_mask))

    context = MagicMock()
    context.conditioning.save.return_value = "cond-name"

    invocation.invoke(context)

    gpu_embeds.detach.return_value.to.assert_called_once_with("cpu")
    gpu_mask.detach.return_value.to.assert_called_once_with("cpu")
    info = context.conditioning.save.call_args.args[0].conditionings[0]
    assert info.prompt_embeds is cpu_embeds
    assert info.prompt_embeds_mask is cpu_mask


def test_krea2_regional_mask_is_passed_through_untouched(monkeypatch):
    """The optional regional `mask` is a TensorField (a name reference), not a tensor: the encoder
    must forward it as-is and leave the load/device placement to krea2_denoise. Touching it here
    would resolve a tensor onto the borrowed GPU."""
    from invokeai.app.invocations.fields import TensorField
    from invokeai.app.invocations.krea2_text_encoder import Krea2TextEncoderInvocation

    mask_field = TensorField(tensor_name="regional-mask")
    invocation = Krea2TextEncoderInvocation.model_construct(
        prompt="a prompt", mask=mask_field, qwen3_vl_encoder=MagicMock()
    )
    monkeypatch.setattr(
        invocation, "_encode", lambda context: (_gpu_tensor_yielding(MagicMock(spec=torch.Tensor)), None)
    )

    context = MagicMock()
    context.conditioning.save.return_value = "cond-name"

    output = invocation.invoke(context)

    assert output.conditioning.mask is mask_field
    context.tensors.load.assert_not_called()


def test_ideogram4_conditioning_is_saved_on_cpu(monkeypatch):
    import invokeai.app.invocations.ideogram4_text_encoder as ideogram4_module
    from invokeai.app.invocations.ideogram4_text_encoder import Ideogram4TextEncoderInvocation

    invocation = Ideogram4TextEncoderInvocation.model_construct(prompt="a prompt", qwen3_encoder=MagicMock())

    cpu_embeds = MagicMock(spec=torch.Tensor)
    gpu_embeds = _gpu_tensor_yielding(cpu_embeds)
    monkeypatch.setattr(ideogram4_module, "encode_qwen3vl_prompt", lambda prompt, tokenizer, text_encoder: gpu_embeds)

    context = MagicMock()
    # `with model_on_device() as (_, model)` — the mock's __enter__ must unpack into two values.
    context.models.load.return_value.model_on_device.return_value.__enter__.return_value = (None, MagicMock())
    context.conditioning.save.return_value = "cond-name"

    invocation.invoke(context)

    gpu_embeds.detach.return_value.to.assert_called_once_with("cpu")
    info = context.conditioning.save.call_args.args[0].conditionings[0]
    assert info.prompt_embeds is cpu_embeds


def test_ernie_image_conditioning_is_saved_on_cpu(monkeypatch):
    from invokeai.app.invocations.ernie_image_text_encoder import ErnieImageTextEncoderInvocation

    invocation = ErnieImageTextEncoderInvocation.model_construct(prompt="a prompt", text_encoder=MagicMock())

    cpu_embeds = MagicMock(spec=torch.Tensor)
    gpu_embeds = _gpu_tensor_yielding(cpu_embeds)
    monkeypatch.setattr(invocation, "_encode_prompt", lambda context, prompt: gpu_embeds)

    context = MagicMock()
    context.conditioning.save.return_value = "cond-name"

    invocation.invoke(context)

    gpu_embeds.detach.return_value.to.assert_called_once_with("cpu")
    info = context.conditioning.save.call_args.args[0].conditionings[0]
    assert info.prompt_embeds is cpu_embeds
