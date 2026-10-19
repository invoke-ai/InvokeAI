"""`ernie_image_prompt_enhancer` exists to keep long, per-execution work *off* a borrowed idle GPU.

It was split out of `ernie_image_text_encoder`, which is `idle_gpu_offloadable`: the offload holds
the lent GPU's exclusive-use lock for the whole node, so a session dequeued onto that GPU blocks
until it returns. An encoder forward is short and its model load amortizes into the borrowed
device's cache; the enhancer's autoregressive `generate()` runs afresh on every generation and never
amortizes. These tests pin the split — the enhancer must stay un-offloadable, and the encoder must
stay free of enhancer work.
"""

from unittest.mock import MagicMock

import pytest


def test_prompt_enhancer_is_not_idle_gpu_offloadable():
    """The whole point of the split. Marking this node would reintroduce the stall the split fixes."""
    from invokeai.app.invocations.ernie_image.ernie_image_prompt_enhancer import ErnieImagePromptEnhancerInvocation

    assert ErnieImagePromptEnhancerInvocation.idle_gpu_offloadable is False


def test_text_encoder_no_longer_carries_enhancer_fields():
    """If the enhancer is ever merged back into the encoder, the encoder's `idle_gpu_offloadable=True`
    silently becomes wrong again — nothing else would fail."""
    from invokeai.app.invocations.text_encoder.ernie_image_text_encoder import ErnieImageTextEncoderInvocation

    fields = set(ErnieImageTextEncoderInvocation.model_fields)
    assert ErnieImageTextEncoderInvocation.idle_gpu_offloadable is True
    assert not {f for f in fields if f.startswith("pe_")}
    assert "prompt_enhancer" not in fields
    assert "use_prompt_enhancer" not in fields


def test_prompt_passes_through_when_no_enhancer_is_connected():
    """The loader emits `prompt_enhancer=None` for a pipeline that ships no PE submodel. That must
    not load a model or fail — the graph builder still wires this node when the toggle is on."""
    from invokeai.app.invocations.ernie_image.ernie_image_prompt_enhancer import ErnieImagePromptEnhancerInvocation

    invocation = ErnieImagePromptEnhancerInvocation.model_construct(prompt="a prompt", prompt_enhancer=None)
    context = MagicMock()

    output = invocation.invoke(context)

    assert output.value == "a prompt"
    context.models.load.assert_not_called()


def test_generation_is_capped_and_cancelable(monkeypatch):
    """Two properties of the `generate()` call that nothing else covers:

    - `max_new_tokens` is clamped to PE_MAX_NEW_TOKENS. A tokenizer config without
      `model_max_length` makes transformers substitute int(1e30), so the tokenizer's value alone is
      not a bound.
    - a `StoppingCriteria` is passed, so a cancel takes effect mid-rewrite rather than after it.
    """
    import invokeai.app.invocations.ernie_image.ernie_image_prompt_enhancer as pe_module
    from invokeai.app.invocations.ernie_image.ernie_image_prompt_enhancer import (
        PE_MAX_NEW_TOKENS,
        ErnieImagePromptEnhancerInvocation,
    )

    invocation = ErnieImagePromptEnhancerInvocation.model_construct(
        prompt="a prompt", prompt_enhancer=MagicMock(), width=832, height=1216, temperature=0.6, top_p=0.95
    )

    tokenizer = MagicMock()
    tokenizer.model_max_length = int(1e30)  # the transformers sentinel for "unset"
    tokenizer.decode.return_value = "  an enhanced prompt  "
    tokenizer.return_value = MagicMock(**{"to.return_value": {"input_ids": MagicMock(shape=[1, 3])}})
    lm = MagicMock()
    lm.generate.return_value = [[0, 1, 2, 3, 4]]

    monkeypatch.setattr(pe_module, "PreTrainedTokenizerBase", MagicMock)
    monkeypatch.setattr(pe_module, "PreTrainedModel", MagicMock)

    context = MagicMock()
    context.util.is_canceled.return_value = False
    loaded = {id(invocation.prompt_enhancer.tokenizer): tokenizer, id(invocation.prompt_enhancer.text_encoder): lm}

    def _load(identifier):
        info = MagicMock()
        info.model_on_device.return_value.__enter__.return_value = (None, loaded[id(identifier)])
        return info

    context.models.load.side_effect = _load

    output = invocation.invoke(context)

    assert output.value == "an enhanced prompt"
    kwargs = lm.generate.call_args.kwargs
    assert kwargs["max_new_tokens"] == PE_MAX_NEW_TOKENS
    assert len(kwargs["stopping_criteria"]) == 1
    # The criterion reports the session's cancel state, so `generate()` stops when the queue item is.
    criterion = kwargs["stopping_criteria"][0]
    assert criterion(MagicMock(), MagicMock()) is False
    context.util.is_canceled.return_value = True
    assert criterion(MagicMock(), MagicMock()) is True


def test_cancel_discards_the_partial_rewrite(monkeypatch):
    """The stopping criterion makes `generate()` return a truncated sequence. Encoding that would
    silently generate an image from half a prompt, so the node must raise instead."""
    import invokeai.app.invocations.ernie_image.ernie_image_prompt_enhancer as pe_module
    from invokeai.app.invocations.ernie_image.ernie_image_prompt_enhancer import ErnieImagePromptEnhancerInvocation
    from invokeai.app.services.session_processor.session_processor_common import CanceledException

    invocation = ErnieImagePromptEnhancerInvocation.model_construct(
        prompt="a prompt", prompt_enhancer=MagicMock(), width=1024, height=1024, temperature=0.6, top_p=0.95
    )

    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    tokenizer.return_value = MagicMock(**{"to.return_value": {"input_ids": MagicMock(shape=[1, 3])}})
    lm = MagicMock()
    lm.generate.return_value = [[0, 1, 2]]

    monkeypatch.setattr(pe_module, "PreTrainedTokenizerBase", MagicMock)
    monkeypatch.setattr(pe_module, "PreTrainedModel", MagicMock)

    context = MagicMock()
    context.util.is_canceled.return_value = True
    context.models.load.return_value.model_on_device.return_value.__enter__.side_effect = [
        (None, tokenizer),
        (None, lm),
    ]

    with pytest.raises(CanceledException):
        invocation.invoke(context)

    tokenizer.decode.assert_not_called()
