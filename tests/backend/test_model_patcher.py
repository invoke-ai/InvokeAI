import threading

import pytest
import torch
from accelerate import init_empty_weights
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from invokeai.backend.model_manager.load.model_cache.model_cache import MODEL_LOAD_LOCK
from invokeai.backend.model_manager.load.optimizations import skip_torch_weight_init
from invokeai.backend.model_patcher import ModelPatcher
from invokeai.backend.textual_inversion import TextualInversionModelRaw
from invokeai.backend.util.devices import TorchDevice


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
def isolate_cpu_and_weight_initializers(monkeypatch):
    # These tests exercise real CPU tensors irrespective of the application's configured device.
    monkeypatch.setattr(TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    # The red regression deliberately exposes a leaked process-global patch. Never let that
    # contaminate another test, even when an assertion fails on the unfixed implementation.
    modules = (torch.nn.Linear, torch.nn.modules.conv._ConvNd, torch.nn.Embedding)
    originals = [module.reset_parameters for module in modules]
    yield
    for module, original in zip(modules, originals, strict=True):
        module.reset_parameters = original


@pytest.mark.parametrize("phase", ["padding", "growth", "teardown", "error_teardown"])
def test_ti_resizing_excludes_global_meta_construction(tokenizer, text_encoder, ti, monkeypatch, phase):
    """Drive each real Transformers resize against accelerate's process-global meta patch.

    The handshake observes either an actual blocked writer or completion of an unprotected
    resize. There is no sleep/latency assertion and no fake embedding or fake load lock.
    """
    ready = threading.Event()
    proceed = threading.Event()
    attempted = threading.Event()
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
        assert proceed.wait(20), "test coordinator did not release TI"

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
    worker = threading.Thread(target=apply)
    worker.start()
    try:
        assert ready.wait(20), f"TI did not reach {phase}: {errors!r}"
        with MODEL_LOAD_LOCK.write_lock(), init_empty_weights():
            patch_installed = True
            proceed.set()
            assert attempted.wait(20), "TI neither waited for the writer nor finished its unprotected resize"
            patch_installed = False
    finally:
        proceed.set()
        worker.join(20)

    assert not worker.is_alive(), "TI failed to resume after construction released the lock"
    assert not unprotected_resize, f"{phase} resized while accelerate's meta patch was installed: {errors!r}"
    assert not errors
    assert text_encoder.get_input_embeddings().weight.device.type == "cpu"
    assert text_encoder.get_input_embeddings().num_embeddings == 8
    torch.testing.assert_close(text_encoder.get_input_embeddings().weight[: len(tokenizer)], original_weights)
    assert len(tokenizer) == 7


def test_ti_and_model_loading_do_not_leak_global_weight_initializers(tokenizer, text_encoder, ti, monkeypatch):
    ti_inside_patch = threading.Event()
    loader_attempted = threading.Event()
    ti_left_patch = threading.Event()
    errors: list[BaseException] = []
    modules = (torch.nn.Linear, torch.nn.modules.conv._ConvNd, torch.nn.Embedding)
    originals = [module.reset_parameters for module in modules]
    original_resize = text_encoder.resize_token_embeddings
    original_wait = MODEL_LOAD_LOCK._cond.wait
    calls = 0

    def resize(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original_resize(*args, **kwargs)
        if calls == 2:
            ti_inside_patch.set()
            assert loader_attempted.wait(20), "loader never attempted concurrent construction"
        return result

    def wait(*args, **kwargs):
        if threading.current_thread() is loader:
            loader_attempted.set()
        return original_wait(*args, **kwargs)

    def apply():
        try:
            with ModelPatcher.apply_ti(tokenizer, text_encoder, [("concept", ti)]):
                ti_left_patch.set()
        except BaseException as error:
            errors.append(error)
        finally:
            ti_left_patch.set()

    def load():
        try:
            with MODEL_LOAD_LOCK.write_lock(), skip_torch_weight_init():
                loader_attempted.set()
                assert ti_left_patch.wait(20), "TI did not leave its initialization patch"
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(text_encoder, "resize_token_embeddings", resize)
    monkeypatch.setattr(MODEL_LOAD_LOCK._cond, "wait", wait)
    worker = threading.Thread(target=apply)
    loader = threading.Thread(target=load)
    worker.start()
    try:
        assert ti_inside_patch.wait(20), f"TI did not enter its initialization patch: {errors!r}"
        loader.start()
    finally:
        if not loader.ident:
            loader_attempted.set()
        worker.join(20)
        if loader.ident:
            loader.join(20)

    assert not worker.is_alive() and not loader.is_alive()
    assert not errors
    assert [module.reset_parameters for module in modules] == originals


@pytest.mark.parametrize("projected", [False, True])
def test_ti_restores_original_tokens_and_selects_matching_sdxl_vectors(tokenizer, ti, projected):
    cls = CLIPTextModelWithProjection if projected else CLIPTextModel
    hidden_size = 4 if projected else 8
    model = cls(
        CLIPTextConfig(
            vocab_size=7,
            hidden_size=hidden_size,
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
    original_weights = model.get_input_embeddings().weight.detach().clone()
    model.eval()
    inputs = torch.tensor([[0, 2, 1]])
    original_output = model(input_ids=inputs).last_hidden_state.detach().clone()
    ti.embedding = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    ti.embedding_2 = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100
    expected = ti.embedding_2 if projected else ti.embedding

    loader = None
    try:
        with ModelPatcher.apply_ti(tokenizer, model, [("concept", ti)]) as (patched, manager):
            ids = [patched.convert_tokens_to_ids(token) for token in ("<concept>", "<concept-!pad-1>")]
            torch.testing.assert_close(model.get_input_embeddings().weight[ids], expected)
            torch.testing.assert_close(model.get_input_embeddings().weight[:7], original_weights)
            torch.testing.assert_close(model(input_ids=inputs).last_hidden_state, original_output)
            assert manager.expand_textual_inversion_token_ids_if_necessary([ids[0]]) == ids
            # The process-global writer must not be retained across caller inference/model work.
            acquired = threading.Event()

            def acquire():
                with MODEL_LOAD_LOCK.write_lock():
                    acquired.set()

            loader = threading.Thread(target=acquire)
            loader.start()
            assert acquired.wait(20), "TI context body must permit concurrent model loading"
    finally:
        if loader is not None:
            loader.join(20)
            assert not loader.is_alive()

    assert model.get_input_embeddings().num_embeddings == 8
    torch.testing.assert_close(model.get_input_embeddings().weight[:7], original_weights)
    torch.testing.assert_close(model(input_ids=inputs).last_hidden_state, original_output)
    assert len(tokenizer) == 7


def test_empty_ti_does_not_resize_or_take_construction_lock(tokenizer, text_encoder):
    # This is deliberately non-reentrant. The empty fast path must not acquire it at all.
    original = text_encoder.get_input_embeddings()
    done = threading.Event()
    errors: list[BaseException] = []

    def apply():
        try:
            with ModelPatcher.apply_ti(tokenizer, text_encoder, []) as (patched, manager):
                assert patched is tokenizer
                assert manager.pad_tokens == {}
                assert text_encoder.get_input_embeddings() is original
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    worker = threading.Thread(target=apply)
    try:
        with MODEL_LOAD_LOCK.write_lock():
            worker.start()
            assert done.wait(20), "empty TI must not block behind model construction"
    finally:
        worker.join(20)
    assert not worker.is_alive()
    assert not errors


def test_ti_failed_resize_restores_embeddings_and_initializers(tokenizer, text_encoder, ti, monkeypatch):
    original_weights = text_encoder.get_input_embeddings().weight.detach().clone()
    original_resize = text_encoder.resize_token_embeddings
    original_reset = torch.nn.Embedding.reset_parameters
    failure = RuntimeError("resize failed after constructing an embedding")
    calls = 0

    def resize(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original_resize(*args, **kwargs)
        if calls == 2:
            raise failure
        return result

    monkeypatch.setattr(text_encoder, "resize_token_embeddings", resize)
    with pytest.raises(RuntimeError) as raised:
        with ModelPatcher.apply_ti(tokenizer, text_encoder, [("concept", ti)]):
            pytest.fail("failed resize must not yield a patched encoder")
    assert raised.value is failure
    assert calls == 3
    assert torch.nn.Embedding.reset_parameters is original_reset
    assert text_encoder.get_input_embeddings().num_embeddings == 8
    torch.testing.assert_close(text_encoder.get_input_embeddings().weight[:7], original_weights)
