import threading
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from accelerate import init_empty_weights
from compel import Compel, ReturnedEmbeddingsType
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from invokeai.app.invocations.text_encoder.compel import SDXLPromptInvocationBase
from invokeai.backend.model_manager.load.model_cache.model_cache import MODEL_LOAD_LOCK
from invokeai.backend.model_patcher import ModelPatcher
from invokeai.backend.textual_inversion import TextualInversionManager, TextualInversionModelRaw
from invokeai.backend.util.devices import TorchDevice


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
    def __init__(self, model, config=None, compute_device=torch.device("cpu")):
        self._model = model
        self.config = config
        self._compute_device = compute_device

    @property
    def compute_device(self) -> torch.device:
        return self._compute_device

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


def test_sdxl_run_clip_compel_uses_compute_device_for_partially_loaded_model(monkeypatch):
    # Regression test for #9373: the encoder's weights are all offloaded to CPU (effective device == CPU), but its
    # intended compute device is the accelerator. compel must build on the intended compute device, not the current
    # residency, or the whole encode silently runs on the CPU.
    module_path = "invokeai.app.invocations.text_encoder.compel"
    compute_device = torch.device("meta")
    text_encoder = FakeClipTextEncoder(effective_device=torch.device("cpu"))
    tokenizer = FakeTokenizer()
    text_encoder_info = FakeLoadedModel(
        text_encoder, config=SimpleNamespace(base="sdxl"), compute_device=compute_device
    )
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

    assert FakeCompel.last_init_device == compute_device
    assert cond.shape == (1, 4, 4)
    assert pooled is None


def test_sdxl_run_clip_compel_uses_cpu_for_cpu_only_model(monkeypatch):
    # A cpu_only text encoder has compute_device == CPU; compel must build on the CPU.
    module_path = "invokeai.app.invocations.text_encoder.compel"
    text_encoder = FakeClipTextEncoder(effective_device=torch.device("cpu"))
    tokenizer = FakeTokenizer()
    text_encoder_info = FakeLoadedModel(
        text_encoder, config=SimpleNamespace(base="sdxl"), compute_device=torch.device("cpu")
    )
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


@pytest.mark.parametrize("model_class", [CLIPTextModel, CLIPTextModelWithProjection])
@torch.no_grad()
def test_sdxl_ti_conditioning_survives_concurrent_model_construction(monkeypatch, model_class):
    tokenizer = CLIPTokenizer(
        vocab={
            token: i
            for i, token in enumerate(["<|startoftext|>", "<|endoftext|>", "a</w>", "b</w>", "c</w>", "d</w>", "e</w>"])
        },
        merges=[],
        model_max_length=16,
    )
    config = CLIPTextConfig(
        vocab_size=7,
        hidden_size=4 if model_class is CLIPTextModelWithProjection else 8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        projection_dim=4,
        max_position_embeddings=16,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        encoder, reference = model_class(config).eval(), model_class(config).eval()
    reference.load_state_dict(encoder.state_dict())
    weights = encoder.get_input_embeddings().weight.detach().clone()
    ti = TextualInversionModelRaw()
    ti.embedding = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    ti.embedding_2 = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100
    vectors = ti.embedding_2 if model_class is CLIPTextModelWithProjection else ti.embedding
    prompt = "a <concept> b"

    # Independent, sequential reference: install the known two vectors and expansion directly.
    reference_tokenizer = CLIPTokenizer(vocab=tokenizer.get_vocab(), merges=[], model_max_length=16)
    reference_tokenizer.add_tokens(["<concept>", "<concept-!pad-1>"])
    reference.resize_token_embeddings(10, pad_to_multiple_of=8)
    ids = reference_tokenizer.convert_tokens_to_ids(["<concept>", "<concept-!pad-1>"])
    reference.get_input_embeddings().weight[ids] = vectors
    manager = TextualInversionManager(reference_tokenizer)
    manager.pad_tokens[ids[0]] = [ids[1]]
    reference_compel = Compel(
        tokenizer=reference_tokenizer,
        text_encoder=reference,
        textual_inversion_manager=manager,
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        truncate_long_prompts=False,
        device=torch.device("cpu"),
    )
    expected = reference_compel(prompt)

    # Only the expensive model-service boundary is doubled. TI lookup, LoRA scope, the shared
    # non-reentrant lock, real Accelerate global patch, resize and Compel inference remain real.
    context = MagicMock()
    context.config.get.return_value.log_tokenization = False
    loaded = {
        "encoder": FakeLoadedModel(encoder, config=SimpleNamespace(base="sdxl")),
        "tokenizer": FakeLoadedModel(tokenizer),
        "concept": SimpleNamespace(model=ti, config=SimpleNamespace(base="sdxl")),
    }
    context.models.load.side_effect = loaded.__getitem__
    clip = SimpleNamespace(text_encoder="encoder", tokenizer="tokenizer", loras=[], skipped_layers=0)
    monkeypatch.setattr(TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(TorchDevice, "choose_torch_dtype", staticmethod(lambda *args, **kwargs: torch.float32))
    ready, release, attempted = threading.Event(), threading.Event(), threading.Event()
    results, errors = [], []
    unprotected = []
    meta_active = False
    apply_ti = ModelPatcher.apply_ti
    resize = encoder.resize_token_embeddings
    wait = MODEL_LOAD_LOCK._cond.wait

    @contextmanager
    def observed_ti(*args, **kwargs):
        ready.set()
        assert release.wait(20), "coordinator did not release the caller"
        with apply_ti(*args, **kwargs) as result:
            yield result

    def observed_resize(*args, **kwargs):
        if meta_active:
            unprotected.append(True)
            try:
                return resize(*args, **kwargs)
            finally:
                # Keep the real global patch installed until construction returns or raises.
                attempted.set()
        return resize(*args, **kwargs)

    def observed_wait(*args, **kwargs):
        if threading.current_thread() is worker:
            attempted.set()
        return wait(*args, **kwargs)

    def invoke():
        try:
            with torch.no_grad():
                results.append(
                    SDXLPromptInvocationBase().run_clip_compel(
                        context,
                        clip,
                        prompt,
                        model_class is CLIPTextModelWithProjection,
                        "lora_te1_",
                        zero_on_empty=False,
                    )
                )
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(ModelPatcher, "apply_ti", observed_ti)
    monkeypatch.setattr(encoder, "resize_token_embeddings", observed_resize)
    monkeypatch.setattr(MODEL_LOAD_LOCK._cond, "wait", observed_wait)
    worker = threading.Thread(target=invoke)
    worker.start()
    try:
        assert ready.wait(20), f"actual caller did not reach TI: {errors!r}"
        with MODEL_LOAD_LOCK.write_lock(), init_empty_weights():
            meta_active = True
            release.set()
            assert attempted.wait(20), "caller neither waited nor attempted an unprotected resize"
            meta_active = False
    finally:
        release.set()
        worker.join(20)

    assert not worker.is_alive(), "actual caller did not resume after construction"
    assert not unprotected, f"actual caller resized under a process-global meta patch: {errors!r}"
    assert not errors
    actual, pooled = results[0]
    torch.testing.assert_close(actual, expected)
    if model_class is CLIPTextModelWithProjection:
        torch.testing.assert_close(pooled, reference_compel.conditioning_provider.get_pooled_embeddings([prompt]))
    else:
        assert pooled is None
    assert encoder.get_input_embeddings().weight.device.type == "cpu"
    assert encoder.get_input_embeddings().num_embeddings == 8
    torch.testing.assert_close(encoder.get_input_embeddings().weight[:7], weights)
    assert len(tokenizer) == 7
