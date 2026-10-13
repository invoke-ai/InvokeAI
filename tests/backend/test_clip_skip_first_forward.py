"""CLIP skip must not change future conditioning on a cached text encoder."""

import pytest
import torch
from compel import Compel, ReturnedEmbeddingsType
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from invokeai.backend.model_patcher import ModelPatcher

NUM_LAYERS = 4
MODEL_CLASSES = [CLIPTextModel, CLIPTextModelWithProjection]


def _encoders(model_class):
    config = CLIPTextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=2,
        vocab_size=64,
        projection_dim=8,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        fresh = model_class(config).eval()
        reference = model_class(config).eval()
    reference.load_state_dict(fresh.state_dict())
    return fresh, reference, torch.tensor([[0, 5, 6, 2]])


def _assert_restored(actual, expected):
    assert len(actual.hidden_states) == NUM_LAYERS + 1
    for actual_state, expected_state in zip(actual.hidden_states, expected.hidden_states, strict=True):
        torch.testing.assert_close(actual_state, expected_state)
    torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state)
    if hasattr(expected, "text_embeds"):
        torch.testing.assert_close(actual.text_embeds, expected.text_embeds)
    else:
        torch.testing.assert_close(actual.pooler_output, expected.pooler_output)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
@pytest.mark.parametrize("skip", [1, 2])
@pytest.mark.parametrize("capture_first", [False, True])
@torch.no_grad()
def test_first_skipped_forward_preserves_later_hidden_states(model_class, skip, capture_first):
    fresh, reference, ids = _encoders(model_class)
    expected = reference(ids, output_hidden_states=True)
    layers = getattr(fresh, "text_model", fresh).encoder.layers
    original_layers = tuple(layers)

    with ModelPatcher.apply_clip_skip(fresh, skip):
        skipped = fresh(ids, output_hidden_states=capture_first)
        if capture_first:
            assert len(skipped.hidden_states) == NUM_LAYERS - skip + 1
            for actual, target in zip(skipped.hidden_states, expected.hidden_states[:-skip], strict=True):
                torch.testing.assert_close(actual, target)
        torch.testing.assert_close(
            skipped.last_hidden_state,
            getattr(fresh, "text_model", fresh).final_layer_norm(expected.hidden_states[-(skip + 1)]),
        )

    assert tuple(layers) == original_layers
    _assert_restored(fresh(ids, output_hidden_states=True), expected)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
@pytest.mark.parametrize("skip", [1, 2])
@pytest.mark.parametrize("fail_inside_forward", [False, True])
@torch.no_grad()
def test_failed_first_skipped_context_preserves_later_conditioning(model_class, skip, fail_inside_forward):
    fresh, reference, ids = _encoders(model_class)
    expected = reference(ids, output_hidden_states=True)
    layers = getattr(fresh, "text_model", fresh).encoder.layers
    original_layers = tuple(layers)

    def fail_forward(_module, _inputs, _output):
        raise RuntimeError("conditioning failed")

    failure_hook = layers[0].register_forward_hook(fail_forward) if fail_inside_forward else None
    try:
        with pytest.raises(RuntimeError, match="conditioning failed"):
            with ModelPatcher.apply_clip_skip(fresh, skip):
                fresh(ids, output_hidden_states=True)
                raise RuntimeError("conditioning failed")
    finally:
        if failure_hook is not None:
            failure_hook.remove()

    assert tuple(layers) == original_layers
    _assert_restored(fresh(ids, output_hidden_states=True), expected)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
@pytest.mark.parametrize("first_skip", [1, 2])
@torch.no_grad()
def test_repeated_skip_contexts_do_not_lose_or_duplicate_hidden_states(model_class, first_skip):
    fresh, reference, ids = _encoders(model_class)
    expected = reference(ids, output_hidden_states=True)

    for skip in [first_skip, 0, 2, 1, 0]:
        with ModelPatcher.apply_clip_skip(fresh, skip):
            actual = fresh(ids, output_hidden_states=True)
            target_states = expected.hidden_states[:-skip] if skip else expected.hidden_states
            assert len(actual.hidden_states) == len(target_states)
            for actual_state, expected_state in zip(actual.hidden_states, target_states, strict=True):
                torch.testing.assert_close(actual_state, expected_state)
        _assert_restored(fresh(ids, output_hidden_states=True), expected)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
@pytest.mark.parametrize("skip", [0, 1, 2])
@torch.no_grad()
def test_warmed_encoder_retains_existing_skip_semantics(model_class, skip):
    fresh, reference, ids = _encoders(model_class)
    expected = reference(ids, output_hidden_states=True)
    _assert_restored(fresh(ids, output_hidden_states=True), expected)

    with ModelPatcher.apply_clip_skip(fresh, skip):
        actual = fresh(ids, output_hidden_states=True)
        target_states = expected.hidden_states[:-skip] if skip else expected.hidden_states
        assert len(actual.hidden_states) == len(target_states)
        for actual_state, expected_state in zip(actual.hidden_states, target_states, strict=True):
            torch.testing.assert_close(actual_state, expected_state)
    _assert_restored(fresh(ids, output_hidden_states=True), expected)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
@torch.no_grad()
def test_first_skipped_compel_prompt_preserves_later_sdxl_conditioning(model_class):
    fresh, reference, _ids = _encoders(model_class)
    tokenizer = CLIPTokenizer(
        vocab={"<|startoftext|>": 0, "<|pad|>": 1, "<|endoftext|>": 2, "a</w>": 3, "b</w>": 4},
        merges=[],
        pad_token="<|pad|>",
        model_max_length=8,
    )
    assert tokenizer("a b", add_special_tokens=False).input_ids == [3, 4]

    def conditioning_provider(encoder):
        return Compel(
            tokenizer=tokenizer,
            text_encoder=encoder,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            truncate_long_prompts=False,
            device=torch.device("cpu"),
        )

    with ModelPatcher.apply_clip_skip(fresh, 1):
        conditioning_provider(fresh)("a b")

    for prompt in ["a b", "a++ b", "a b " * 5]:
        # SDXL creates a new Compel per prompt while reusing the cached encoder.
        actual_compel = conditioning_provider(fresh)
        reference_compel = conditioning_provider(reference)
        actual = actual_compel(prompt)
        expected = reference_compel(prompt)
        torch.testing.assert_close(actual, expected)
        if prompt == "a b":
            # An independent tensor oracle also guards the consumer's penultimate, non-normalized selection.
            ids = tokenizer(prompt, padding="max_length", return_tensors="pt").input_ids
            reference_output = reference(ids, output_hidden_states=True)
            torch.testing.assert_close(actual, reference_output.hidden_states[-2])
        elif prompt == "a++ b":
            assert not torch.allclose(actual, conditioning_provider(fresh)("a b"))
        else:
            assert actual.shape[1] > tokenizer.model_max_length

        if model_class is CLIPTextModelWithProjection:
            torch.testing.assert_close(
                actual_compel.conditioning_provider.get_pooled_embeddings([prompt]),
                reference_compel.conditioning_provider.get_pooled_embeddings([prompt]),
            )
