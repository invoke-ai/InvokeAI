"""The Krea-2 nodes expose two extension seams so node packs can add per-token behaviour without
copying `_run_diffusion` (281 lines) or restating the encoder's token layout.

These tests pin the contract: the defaults must behave exactly as the inlined code did, and a subclass
must be able to replace each seam. Without them a refactor could silently narrow the seam and only
break out-of-tree code.
"""

from contextlib import ExitStack
from types import SimpleNamespace

import torch

from invokeai.app.invocations.krea2_denoise import Krea2DenoiseInvocation
from invokeai.backend.krea2.attention import Krea2RegionalPromptingState
from invokeai.backend.krea2.regional_prompting import Krea2RegionalPromptingExtension, Krea2TextConditioning
from invokeai.backend.krea2.text_encoding import KREA2_BODY_MAX_LENGTH, KREA2_START_IDX, encode_krea2_prompt


def _extension(mask: torch.Tensor | None) -> Krea2RegionalPromptingExtension:
    conditioning = Krea2TextConditioning(prompt_embeds=torch.ones(1, 2, 12, 8), mask=mask)
    return Krea2RegionalPromptingExtension.from_text_conditionings([conditioning], image_seq_len=4)


def test_default_attention_payload_is_the_regional_mask() -> None:
    invocation = Krea2DenoiseInvocation.model_construct()
    with_regions = _extension(torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]))
    without_regions = _extension(None)

    payload = invocation._build_attention_payload(with_regions, torch.float32)

    assert payload is not None
    assert torch.equal(payload, with_regions.get_attention_mask())
    # No regional masks means no mask is allocated at all -- unchanged from before the refactor.
    assert invocation._build_attention_payload(without_regions, torch.float32) is None


def test_default_install_and_clear_drive_the_shared_state() -> None:
    state = Krea2RegionalPromptingState()
    mask = torch.eye(4, dtype=torch.bool)

    Krea2DenoiseInvocation._install_attention_payload(state, mask)
    assert state.attention_mask is mask

    # Cleanup must not leave a potentially multi-GB mask on the cached transformer's processors.
    Krea2DenoiseInvocation._clear_attention_state(state)
    assert state.attention_mask is None


def test_install_attention_processors_wires_state_and_registers_cleanup() -> None:
    invocation = Krea2DenoiseInvocation.model_construct()
    installed: dict[str, object] = {}

    transformer = SimpleNamespace(
        attn_processors={"transformer_blocks.0.attn.processor": object()},
        set_attn_processor=lambda processors: installed.update(processors=processors),
    )

    with ExitStack() as exit_stack:
        state = invocation._install_attention_processors(transformer, exit_stack)
        state.set_attention_mask(torch.eye(4, dtype=torch.bool))
        assert isinstance(state, Krea2RegionalPromptingState)
        assert "transformer_blocks.0.attn.processor" in installed["processors"]

    # Leaving the exit stack must have run the cleanup callback.
    assert state.attention_mask is None


def test_a_subclass_can_replace_every_attention_seam() -> None:
    """The contract a node pack depends on: swap the state, the payload and the per-pass install."""

    class _PackState(Krea2RegionalPromptingState):
        extra: object = None

    class _PackDenoise(Krea2DenoiseInvocation):
        def _install_attention_processors(self, transformer, exit_stack):
            state = _PackState()
            exit_stack.callback(self._clear_attention_state, state)
            return state

        @staticmethod
        def _clear_attention_state(state) -> None:
            state.attention_mask = None
            state.extra = "cleared"

        def _build_attention_payload(self, extension, inference_dtype):
            return ("mask", extension.get_attention_mask())

        @staticmethod
        def _install_attention_payload(state, payload) -> None:
            state.extra = payload

    invocation = _PackDenoise.model_construct()
    with ExitStack() as exit_stack:
        state = invocation._install_attention_processors(SimpleNamespace(), exit_stack)
        assert isinstance(state, _PackState)

        payload = invocation._build_attention_payload(_extension(None), torch.float32)
        invocation._install_attention_payload(state, payload)
        assert state.extra == ("mask", None)

    assert state.extra == "cleared"


class _WordTokenizer:
    """Whitespace tokenizer with real character offsets, standing in for Qwen2TokenizerFast."""

    is_fast = True

    def __call__(
        self,
        text,
        max_length=None,
        truncation=False,
        padding=None,
        return_tensors=None,
        return_offsets_mapping=False,
    ):
        if max_length is None:
            input_ids = torch.arange(91, 96, dtype=torch.long).unsqueeze(0)
            return SimpleNamespace(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

        offsets = []
        cursor = 0
        for word in text.split(" "):
            if word:
                start = text.index(word, cursor)
                offsets.append((start, start + len(word)))
                cursor = start + len(word)
        offsets = offsets[:max_length]

        input_ids = torch.zeros((1, max_length), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        input_ids[:, : len(offsets)] = torch.arange(1, len(offsets) + 1, dtype=torch.long)
        attention_mask[:, : len(offsets)] = 1
        result = SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)
        if return_offsets_mapping:
            offset_mapping = torch.zeros((1, max_length, 2), dtype=torch.long)
            offset_mapping[0, : len(offsets)] = torch.tensor(offsets, dtype=torch.long)
            result.offset_mapping = offset_mapping
        return result


class _StubEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, *, input_ids, attention_mask, position_ids, **_kwargs):
        seq_len = input_ids.shape[1]
        return SimpleNamespace(hidden_states=tuple(torch.zeros((1, seq_len, 4)) for _ in range(36)))


def test_encode_without_the_callback_asks_for_no_offsets(monkeypatch) -> None:
    # The common path must make the exact same tokenizer call it always did.
    seen: list[bool] = []
    tokenizer = _WordTokenizer()
    original = tokenizer.__call__

    def _spy(text, **kwargs):
        if kwargs.get("max_length") is not None:
            seen.append("return_offsets_mapping" in kwargs)
        return original(text, **kwargs)

    monkeypatch.setattr(
        "invokeai.backend.krea2.text_encoding.TorchDevice.choose_bfloat16_safe_dtype", lambda _d: torch.float32
    )
    embeds, mask, values = encode_krea2_prompt("a prompt", _spy, _StubEncoder())

    assert seen == [False]
    assert values is None
    assert embeds.shape == (1, 512, 12, 4)
    assert mask.shape == (1, 512)


def test_encode_aligns_callback_values_with_the_conditioning(monkeypatch) -> None:
    # Whatever the callback returns is sliced by the same prefix drop as the embeddings, so a caller's
    # per-token vector lines up with the conditioning without knowing the layout.
    monkeypatch.setattr(
        "invokeai.backend.krea2.text_encoding.TorchDevice.choose_bfloat16_safe_dtype", lambda _d: torch.float32
    )
    captured: dict[str, torch.Tensor] = {}

    def build(offset_mapping: torch.Tensor) -> torch.Tensor:
        captured["offsets"] = offset_mapping
        values = torch.ones(offset_mapping.shape[0], dtype=torch.float32)
        # Mark the token right after the prefix drop so we can assert where it lands.
        values[KREA2_START_IDX] = 0.25
        return values

    filler = " ".join(f"w{i}" for i in range(KREA2_START_IDX + 4))
    embeds, mask, values = encode_krea2_prompt(filler, _WordTokenizer(), _StubEncoder(), build)

    assert captured["offsets"].shape == (KREA2_BODY_MAX_LENGTH, 2)
    assert values is not None
    assert values.shape == mask.shape == (1, 512)
    assert values[0, 0].item() == 0.25
    assert values[0, 1].item() == 1.0


def test_encode_returns_none_when_the_callback_yields_nothing(monkeypatch) -> None:
    monkeypatch.setattr(
        "invokeai.backend.krea2.text_encoding.TorchDevice.choose_bfloat16_safe_dtype", lambda _d: torch.float32
    )

    _, _, values = encode_krea2_prompt("a prompt", _WordTokenizer(), _StubEncoder(), lambda _offsets: None)

    assert values is None
