from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from invokeai.app.invocations.anima_text_encoder import AnimaTextEncoderInvocation


class FakeQwen3Encoder(torch.nn.Module):
    """Mimics the Qwen3 0.6B encoder.

    Its `.device` property reports CPU (as HF `PreTrainedModel.device` would when partial loading has offloaded
    every weight to RAM), while the intended compute device is carried separately by the LoadedModel. The forward
    records the device of its inputs so the test can assert where the encode actually ran.
    """

    def __init__(self):
        super().__init__()
        self.register_parameter("cpu_param", torch.nn.Parameter(torch.ones(1)))
        self.forward_input_device: torch.device | None = None

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(self, input_ids: torch.Tensor, attention_mask=None, output_hidden_states: bool = False):
        assert output_hidden_states
        self.forward_input_device = input_ids.device
        hidden = input_ids.unsqueeze(-1).float()
        return SimpleNamespace(hidden_states=[hidden, hidden + 1])


class FakeQwen3Tokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, prompt, **kwargs):
        del prompt, kwargs
        return SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
            attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
        )


class FakeT5Tokenizer:
    def __call__(self, prompt, **kwargs):
        del prompt, kwargs
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))


class FakeLoadedModel:
    def __init__(self, model, compute_device=torch.device("cpu")):
        self._model = model
        self._compute_device = compute_device

    @property
    def compute_device(self) -> torch.device:
        return self._compute_device

    @contextmanager
    def model_on_device(self):
        yield (None, self._model)


def _run_encode(monkeypatch, compute_device: torch.device) -> FakeQwen3Encoder:
    module_path = "invokeai.app.invocations.anima_text_encoder"
    text_encoder = FakeQwen3Encoder()
    tokenizer = FakeQwen3Tokenizer()

    mock_context = MagicMock()
    mock_context.models.load.side_effect = [
        FakeLoadedModel(text_encoder, compute_device=compute_device),
        FakeLoadedModel(tokenizer),
    ]
    mock_context.util.signal_progress = MagicMock()

    # isinstance() guards in the invocation must accept the fakes.
    monkeypatch.setattr(f"{module_path}.PreTrainedModel", FakeQwen3Encoder)
    monkeypatch.setattr(f"{module_path}.PreTrainedTokenizerBase", FakeQwen3Tokenizer)
    monkeypatch.setattr(f"{module_path}.LayerPatcher.apply_smart_model_patches", lambda **kwargs: nullcontext())
    # Step 2 tokenizes with the bundled T5-XXL tokenizer; avoid touching the real bundled files.
    monkeypatch.setattr(f"{module_path}.load_bundled_t5_tokenizer", lambda: FakeT5Tokenizer())

    invocation = AnimaTextEncoderInvocation.model_construct(
        prompt="test prompt",
        qwen3_encoder=SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace(), loras=[]),
        mask=None,
    )

    invocation._encode_prompt(mock_context)
    return text_encoder


def test_anima_qwen3_encode_uses_compute_device(monkeypatch):
    # Regression test for #9373: the encoder's weights are offloaded to CPU (`.device` == CPU), but its intended
    # compute device is the accelerator. The encode must run on the intended compute device, not the current
    # residency, or the whole encode silently runs on the CPU.
    compute_device = torch.device("meta")
    text_encoder = _run_encode(monkeypatch, compute_device)
    assert text_encoder.forward_input_device == compute_device


def test_anima_qwen3_encode_uses_cpu_for_cpu_only_model(monkeypatch):
    # A cpu_only encoder has compute_device == CPU; the encode must run on the CPU.
    text_encoder = _run_encode(monkeypatch, torch.device("cpu"))
    assert text_encoder.forward_input_device == torch.device("cpu")
