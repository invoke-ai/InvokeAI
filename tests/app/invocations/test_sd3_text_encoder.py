from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from invokeai.app.invocations.sd3_text_encoder import Sd3TextEncoderInvocation
from invokeai.backend.model_manager.taxonomy import ModelFormat


class FakeSd3ClipTextEncoder(torch.nn.Module):
    def __init__(self, effective_device: torch.device):
        super().__init__()
        self.register_parameter("cpu_param", torch.nn.Parameter(torch.ones(1)))
        self.register_buffer("active_buffer", torch.ones(1, device=effective_device))
        self.dtype = torch.float32
        self.forward_input_device: torch.device | None = None

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False):
        assert output_hidden_states
        self.forward_input_device = input_ids.device
        hidden = input_ids.unsqueeze(-1).float()
        return SimpleNamespace(hidden_states=[hidden, hidden + 1], __getitem__=lambda self, idx: hidden)


class FakeClipOutput(SimpleNamespace):
    def __getitem__(self, idx):
        del idx
        return self.hidden_states[-1]


class FakeClipTokenizer:
    def __call__(self, prompt, padding, max_length=None, truncation=None, return_tensors=None):
        del prompt, padding, max_length, truncation, return_tensors
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))

    def batch_decode(self, input_ids):
        del input_ids
        return ["decoded"]


class FakeSd3T5Encoder(torch.nn.Module):
    def __init__(self, effective_device: torch.device):
        super().__init__()
        self.register_parameter("cpu_param", torch.nn.Parameter(torch.ones(1)))
        self.register_buffer("active_buffer", torch.ones(1, device=effective_device))
        self.forward_input_device: torch.device | None = None

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(self, input_ids: torch.Tensor):
        self.forward_input_device = input_ids.device
        hidden = input_ids.unsqueeze(-1).float()
        return (hidden,)


class FakeT5Tokenizer:
    def __call__(self, prompt, padding, max_length=None, truncation=None, add_special_tokens=None, return_tensors=None):
        del prompt, padding, max_length, truncation, add_special_tokens, return_tensors
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))

    def batch_decode(self, input_ids):
        del input_ids
        return ["decoded"]


class FakeLoadedModel:
    def __init__(self, model, config=None):
        self._model = model
        self.config = config

    @contextmanager
    def model_on_device(self):
        yield (None, self._model)

    def __enter__(self):
        return self._model

    def __exit__(self, exc_type, exc, tb):
        return False


def test_sd3_clip_encode_uses_effective_device(monkeypatch):
    module_path = "invokeai.app.invocations.sd3_text_encoder"
    effective_device = torch.device("meta")
    text_encoder = FakeSd3ClipTextEncoder(effective_device)
    tokenizer = FakeClipTokenizer()

    def forward(input_ids: torch.Tensor, output_hidden_states: bool = False):
        assert output_hidden_states
        text_encoder.forward_input_device = input_ids.device
        hidden = input_ids.unsqueeze(-1).float()
        return FakeClipOutput(hidden_states=[hidden, hidden + 1])

    text_encoder.forward = forward  # type: ignore[method-assign]

    mock_context = MagicMock()
    mock_context.models.load.side_effect = [
        FakeLoadedModel(text_encoder, config=SimpleNamespace(format=ModelFormat.Diffusers)),
        FakeLoadedModel(tokenizer),
    ]
    mock_context.util.signal_progress = MagicMock()

    monkeypatch.setattr(f"{module_path}.CLIPTextModel", FakeSd3ClipTextEncoder)
    monkeypatch.setattr(f"{module_path}.CLIPTextModelWithProjection", FakeSd3ClipTextEncoder)
    monkeypatch.setattr(f"{module_path}.CLIPTokenizer", FakeClipTokenizer)
    monkeypatch.setattr(f"{module_path}.LayerPatcher.apply_smart_model_patches", lambda **kwargs: nullcontext())

    invocation = Sd3TextEncoderInvocation.model_construct(
        clip_l=SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace(), loras=[]),
        clip_g=SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace(), loras=[]),
        t5_encoder=None,
        prompt="test prompt",
    )

    invocation._clip_encode(
        context=mock_context,
        clip_model=SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace(), loras=[]),
    )

    assert text_encoder.forward_input_device == effective_device


def test_sd3_t5_encode_uses_effective_device(monkeypatch):
    module_path = "invokeai.app.invocations.sd3_text_encoder"
    effective_device = torch.device("meta")
    text_encoder = FakeSd3T5Encoder(effective_device)
    tokenizer = FakeT5Tokenizer()

    mock_context = MagicMock()
    mock_context.models.load.side_effect = [FakeLoadedModel(text_encoder), FakeLoadedModel(tokenizer)]
    mock_context.util.signal_progress = MagicMock()
    mock_context.logger.warning = MagicMock()

    monkeypatch.setattr(f"{module_path}.T5EncoderModel", FakeSd3T5Encoder)
    monkeypatch.setattr(f"{module_path}.T5Tokenizer", FakeT5Tokenizer)
    monkeypatch.setattr(f"{module_path}.T5TokenizerFast", FakeT5Tokenizer)

    invocation = Sd3TextEncoderInvocation.model_construct(
        clip_l=SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace(), loras=[]),
        clip_g=SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace(), loras=[]),
        t5_encoder=SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace()),
        prompt="test prompt",
    )

    invocation._t5_encode(mock_context, max_seq_len=16)

    assert text_encoder.forward_input_device == effective_device
