from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from invokeai.app.invocations.compel import SDXLPromptInvocationBase


class FakeClipTextEncoder(torch.nn.Module):
    def __init__(self, effective_device: torch.device):
        super().__init__()
        self.register_parameter("cpu_param", torch.nn.Parameter(torch.ones(1)))
        self.register_buffer("active_buffer", torch.ones(1, device=effective_device))
        self.dtype = torch.float32

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")


class FakeTokenizer:
    pass


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


class FakeCompel:
    last_init_device: torch.device | None = None

    def __init__(self, *args, device: torch.device, **kwargs):
        del args, kwargs
        FakeCompel.last_init_device = device
        self.conditioning_provider = SimpleNamespace(
            get_pooled_embeddings=lambda prompts: torch.ones((len(prompts), 4), dtype=torch.float32)
        )

    @staticmethod
    def parse_prompt_string(prompt: str) -> str:
        return prompt

    def build_conditioning_tensor_for_conjunction(self, conjunction: str):
        del conjunction
        return torch.ones((1, 4, 4), dtype=torch.float32), {}


@contextmanager
def fake_apply_ti(tokenizer, text_encoder, ti_list):
    del text_encoder, ti_list
    yield tokenizer, object()


def test_sdxl_run_clip_compel_uses_effective_device_for_partially_loaded_model(monkeypatch):
    module_path = "invokeai.app.invocations.compel"
    effective_device = torch.device("meta")
    text_encoder = FakeClipTextEncoder(effective_device=effective_device)
    tokenizer = FakeTokenizer()
    text_encoder_info = FakeLoadedModel(text_encoder, config=SimpleNamespace(base="sdxl"))
    tokenizer_info = FakeLoadedModel(tokenizer)

    mock_context = MagicMock()
    mock_context.models.load.side_effect = [text_encoder_info, tokenizer_info]
    mock_context.config.get.return_value.log_tokenization = False
    mock_context.util.signal_progress = MagicMock()

    monkeypatch.setattr(f"{module_path}.CLIPTextModel", FakeClipTextEncoder)
    monkeypatch.setattr(f"{module_path}.CLIPTextModelWithProjection", FakeClipTextEncoder)
    monkeypatch.setattr(f"{module_path}.CLIPTokenizer", FakeTokenizer)
    monkeypatch.setattr(f"{module_path}.Compel", FakeCompel)
    monkeypatch.setattr(f"{module_path}.generate_ti_list", lambda prompt, base, context: [])
    monkeypatch.setattr(f"{module_path}.LayerPatcher.apply_smart_model_patches", lambda **kwargs: nullcontext())
    monkeypatch.setattr(f"{module_path}.ModelPatcher.apply_clip_skip", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(f"{module_path}.ModelPatcher.apply_ti", fake_apply_ti)

    base = SDXLPromptInvocationBase()
    cond, pooled = base.run_clip_compel(
        context=mock_context,
        clip_field=SimpleNamespace(
            text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace(), loras=[], skipped_layers=0
        ),
        prompt="test prompt",
        get_pooled=False,
        lora_prefix="lora_te1_",
        zero_on_empty=False,
    )

    assert FakeCompel.last_init_device == effective_device
    assert cond.shape == (1, 4, 4)
    assert pooled is None


def test_sdxl_run_clip_compel_uses_cpu_for_fully_cpu_model(monkeypatch):
    module_path = "invokeai.app.invocations.compel"
    text_encoder = FakeClipTextEncoder(effective_device=torch.device("cpu"))
    tokenizer = FakeTokenizer()
    text_encoder_info = FakeLoadedModel(text_encoder, config=SimpleNamespace(base="sdxl"))
    tokenizer_info = FakeLoadedModel(tokenizer)

    mock_context = MagicMock()
    mock_context.models.load.side_effect = [text_encoder_info, tokenizer_info]
    mock_context.config.get.return_value.log_tokenization = False
    mock_context.util.signal_progress = MagicMock()

    monkeypatch.setattr(f"{module_path}.CLIPTextModel", FakeClipTextEncoder)
    monkeypatch.setattr(f"{module_path}.CLIPTextModelWithProjection", FakeClipTextEncoder)
    monkeypatch.setattr(f"{module_path}.CLIPTokenizer", FakeTokenizer)
    monkeypatch.setattr(f"{module_path}.Compel", FakeCompel)
    monkeypatch.setattr(f"{module_path}.generate_ti_list", lambda prompt, base, context: [])
    monkeypatch.setattr(f"{module_path}.LayerPatcher.apply_smart_model_patches", lambda **kwargs: nullcontext())
    monkeypatch.setattr(f"{module_path}.ModelPatcher.apply_clip_skip", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(f"{module_path}.ModelPatcher.apply_ti", fake_apply_ti)

    base = SDXLPromptInvocationBase()
    base.run_clip_compel(
        context=mock_context,
        clip_field=SimpleNamespace(
            text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace(), loras=[], skipped_layers=0
        ),
        prompt="test prompt",
        get_pooled=False,
        lora_prefix="lora_te1_",
        zero_on_empty=False,
    )

    assert FakeCompel.last_init_device == torch.device("cpu")
