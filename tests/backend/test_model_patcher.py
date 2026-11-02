import threading
from time import monotonic

import pytest
import torch
from accelerate import init_empty_weights
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from transformers.modeling_utils import PreTrainedModel

from invokeai.backend.model_manager.load.model_cache.model_cache import MODEL_LOAD_LOCK
from invokeai.backend.model_manager.load.optimizations import skip_torch_weight_init
from invokeai.backend.model_patcher import ModelPatcher
from invokeai.backend.textual_inversion import TextualInversionModelRaw
from invokeai.backend.util.devices import TorchDevice

WAIT_TIMEOUT = 20


@pytest.fixture
def tokenizer() -> CLIPTokenizer:
    vocab = {
        token: i
        for i, token in enumerate(["<|startoftext|>", "<|endoftext|>", "a</w>", "b</w>", "c</w>", "d</w>", "e</w>"])
    }
    return CLIPTokenizer(vocab=vocab, merges=[], model_max_length=16)


@pytest.fixture
def text_encoder(tokenizer: CLIPTokenizer) -> CLIPTextModel:
    return CLIPTextModel(
        CLIPTextConfig(
            vocab_size=len(tokenizer),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=16,
            projection_dim=4,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=1,
        )
    )


@pytest.fixture
def ti() -> TextualInversionModelRaw:
    result = TextualInversionModelRaw()
    result.embedding = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    return result


@pytest.fixture(autouse=True)
def isolate_cpu_initializers_and_check_lock(monkeypatch):
    def assert_idle():
        assert MODEL_LOAD_LOCK._readers == 0
        assert MODEL_LOAD_LOCK._writers_waiting == 0
        assert not MODEL_LOAD_LOCK._writer_active

    assert_idle()
    monkeypatch.setattr(TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    modules = (torch.nn.Linear, torch.nn.modules.conv._ConvNd, torch.nn.Embedding)
    originals = [module.reset_parameters for module in modules]
    original_wait = MODEL_LOAD_LOCK._cond.wait
    deadline = monotonic() + WAIT_TIMEOUT

    def bounded_wait(timeout=None):
        # Unwind real lock contexts on deadlock, including a future non-reentrant outer lock.
        remaining = deadline - monotonic()
        assert remaining > 0, "model load lock did not make progress"
        notified = original_wait(remaining if timeout is None else min(timeout, remaining))
        assert notified, "model load lock did not make progress"
        return notified

    monkeypatch.setattr(MODEL_LOAD_LOCK._cond, "wait", bounded_wait)
    try:
        yield
    finally:
        # A native-base red exposes leaked global initializers; preserve later tests, but never reset the lock.
        for module, original in zip(modules, originals, strict=True):
            module.reset_parameters = original
        assert_idle()


@pytest.mark.parametrize("phase", ["padding", "growth", "teardown", "error_teardown"])
def test_ti_resizing_excludes_global_meta_construction(tokenizer, text_encoder, ti, monkeypatch, phase):
    """Drive each real Transformers resize against accelerate's process-global meta patch.

    The handshake observes either an actual blocked writer or completion of an unprotected
    resize. There is no sleep/latency assertion and no fake embedding or fake load lock.
    """
    ready = threading.Event()
    proceed = threading.Event()
    attempted = threading.Event()
    construction_done = threading.Event()
    modules = (torch.nn.Linear, torch.nn.modules.conv._ConvNd, torch.nn.Embedding)
    initializers = [module.reset_parameters for module in modules]
    errors: list[BaseException] = []
    original_weights = text_encoder.get_input_embeddings().weight.detach().clone()
    original_resize = text_encoder.resize_token_embeddings
    original_wait = MODEL_LOAD_LOCK._cond.wait
    original_add = CLIPTokenizer.add_tokens
    target_call = {"padding": 1, "growth": 2, "teardown": 3, "error_teardown": 3}[phase]
    calls = 0
    unprotected_resize = False
    patch_installed = False

    def pause():
        ready.set()
        assert proceed.wait(WAIT_TIMEOUT), "test coordinator did not release TI"

    def add_tokens(self, *args, **kwargs):
        result = original_add(self, *args, **kwargs)
        if phase == "growth":
            pause()
        return result

    def resize(*args, **kwargs):
        nonlocal calls, unprotected_resize
        calls += 1
        if calls == target_call and patch_installed:
            unprotected_resize = True
            try:
                return original_resize(*args, **kwargs)
            finally:
                attempted.set()
                if phase == "growth":
                    # Force the unsafe nested save/restore to finish after the loader restores its originals.
                    assert construction_done.wait(WAIT_TIMEOUT), "construction did not unwind"
        return original_resize(*args, **kwargs)

    def wait(*args, **kwargs):
        if threading.current_thread() is worker:
            attempted.set()
        return original_wait(*args, **kwargs)

    def apply():
        try:
            if phase == "padding":
                pause()
            with ModelPatcher.apply_ti(tokenizer, text_encoder, [("concept", ti)]) as (patched, _):
                token_id = patched.convert_tokens_to_ids("<concept>")
                torch.testing.assert_close(text_encoder.get_input_embeddings().weight[token_id], ti.embedding[0])
                if phase in ("teardown", "error_teardown"):
                    pause()
                if phase == "error_teardown":
                    raise ValueError("conditioning failed")
        except ValueError as error:
            if phase != "error_teardown" or str(error) != "conditioning failed":
                errors.append(error)
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(CLIPTokenizer, "add_tokens", add_tokens)
    monkeypatch.setattr(text_encoder, "resize_token_embeddings", resize)
    monkeypatch.setattr(MODEL_LOAD_LOCK._cond, "wait", wait)
    worker = threading.Thread(target=apply, daemon=True)
    worker.start()
    try:
        assert ready.wait(WAIT_TIMEOUT), f"TI did not reach {phase}: {errors!r}"
        with MODEL_LOAD_LOCK.write_lock(), skip_torch_weight_init(), init_empty_weights():
            patch_installed = True
            proceed.set()
            assert attempted.wait(WAIT_TIMEOUT), "TI neither waited for the writer nor finished its unprotected resize"
            patch_installed = False
    finally:
        construction_done.set()
        proceed.set()
        worker.join(WAIT_TIMEOUT)

    assert not worker.is_alive(), "TI failed to resume after construction released the lock"
    if phase == "growth":
        assert [module.reset_parameters for module in modules] == initializers
    assert not unprotected_resize, f"{phase} resized while accelerate's meta patch was installed: {errors!r}"
    assert not errors
    assert text_encoder.get_input_embeddings().weight.device.type == "cpu"
    assert text_encoder.get_input_embeddings().num_embeddings == 8
    torch.testing.assert_close(text_encoder.get_input_embeddings().weight[: len(tokenizer)], original_weights)
    assert len(tokenizer) == 7


@torch.no_grad()
def test_ti_skips_unused_initialization_and_releases_lock_across_yield(tokenizer, text_encoder, ti, monkeypatch):
    projected = CLIPTextModelWithProjection(text_encoder.config)
    # Observe real work, not just resize kwargs. The ordinary resize below proves both counters are live.
    calls = {"random": 0, "mean_covariance": 0}

    def count(name, original):
        def observed(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)

        return observed

    monkeypatch.setattr(torch.Tensor, "normal_", count("random", torch.Tensor.normal_))
    monkeypatch.setattr(
        PreTrainedModel,
        "_init_added_embeddings_weights_with_mean",
        count("mean_covariance", PreTrainedModel._init_added_embeddings_weights_with_mean),
    )
    # Exercise the other supported CLIP class too; its bare Embedding initializer must also remain a no-op.
    projected.resize_token_embeddings(8)
    assert calls["random"] > 0 and calls["mean_covariance"] > 0

    completed = threading.Event()
    errors = []

    def load():
        try:
            with MODEL_LOAD_LOCK.write_lock():
                pass
        except BaseException as error:
            errors.append(error)
        finally:
            completed.set()

    for model in (text_encoder, projected):
        calls.update(random=0, mean_covariance=0)
        original_weights = model.get_input_embeddings().weight.detach().clone()
        loader = None
        try:
            with ModelPatcher.apply_ti(tokenizer, model, [("concept", ti)]):
                if model is text_encoder:
                    loader = threading.Thread(target=load, daemon=True)
                    loader.start()
                    assert completed.wait(WAIT_TIMEOUT), "TI yield blocked model construction"
                    assert not errors, f"TI yield retained the construction lock: {errors!r}"
        finally:
            if loader is not None:
                loader.join(WAIT_TIMEOUT)
                assert not loader.is_alive()

        assert calls == {"random": 0, "mean_covariance": 0}
        torch.testing.assert_close(
            model.get_input_embeddings().weight[: len(tokenizer)], original_weights[: len(tokenizer)], rtol=0, atol=0
        )
