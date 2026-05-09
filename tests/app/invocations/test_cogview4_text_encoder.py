from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from invokeai.app.invocations.cogview4_text_encoder import CogView4TextEncoderInvocation


class FakeGlmModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
        self.repaired = False
        self.forward_input_device: torch.device | None = None

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False):
        assert output_hidden_states
        if not self.repaired:
            raise RuntimeError("model must be repaired before forward")

        self.forward_input_device = input_ids.device
        hidden = input_ids.unsqueeze(-1).float()
        return SimpleNamespace(hidden_states=[hidden, hidden + 1])


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, prompt, padding, max_length=None, truncation=None, add_special_tokens=None, return_tensors=None):
        del prompt, padding, max_length, truncation, add_special_tokens, return_tensors
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))

    def batch_decode(self, input_ids):
        del input_ids
        return ["decoded"]


class FakeLoadedModel:
    def __init__(self, model):
        self._model = model
        self.repair_calls = 0

    @contextmanager
    def model_on_device(self):
        yield (None, self._model)

    def repair_required_tensors_on_device(self) -> int:
        self.repair_calls += 1
        self._model.repaired = True
        return 1


def test_cogview4_text_encoder_repairs_model_before_forward(monkeypatch):
    fake_model = FakeGlmModel()
    fake_tokenizer = FakeTokenizer()
    fake_model_info = FakeLoadedModel(fake_model)
    fake_tokenizer_info = FakeLoadedModel(fake_tokenizer)

    mock_context = MagicMock()
    mock_context.models.load.side_effect = [fake_model_info, fake_tokenizer_info]
    mock_context.util.signal_progress = MagicMock()
    mock_context.logger.warning = MagicMock()

    invocation = CogView4TextEncoderInvocation.model_construct(
        prompt="test prompt",
        glm_encoder=SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace()),
    )

    module_path = "invokeai.app.invocations.cogview4_text_encoder"
    monkeypatch.setattr(f"{module_path}.GlmModel", FakeGlmModel)
    monkeypatch.setattr(f"{module_path}.PreTrainedTokenizerFast", FakeTokenizer)

    embeds = invocation._glm_encode(mock_context, max_seq_len=16)

    assert fake_model_info.repair_calls == 1
    mock_context.logger.warning.assert_called_once()
    mock_context.util.signal_progress.assert_called_once_with("Running GLM text encoder")
    assert fake_model.forward_input_device == torch.device("cpu")
    assert embeds.shape == (1, 16, 1)
