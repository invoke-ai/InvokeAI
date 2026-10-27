"""Krea-2 edit mode: reference images read by the Qwen3-VL vision tower into the prompt conditioning.

The reference is ai-toolkit's ``encode_krea_prompt`` / ``_prep_vlm_images``, which trains Krea-2 edit LoRAs.
Everything runs on a tiny but real ``Qwen3VLModel`` with the bundled tokenizer and processor, so the token
ids are the release's and the vision tower is the architecture's own. The vocabulary is kept at full size
for that reason; only the widths shrink.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from invokeai.backend.krea2.sampling_utils import KREA2_SELECT_LAYERS, KREA2_START_IDX
from invokeai.backend.krea2.text_encoding import (
    KREA2_BODY_MAX_LENGTH,
    KREA2_PREFIX,
    KREA2_SUFFIX,
    encode_krea2_prompt,
    encode_krea2_prompt_with_images,
    prepare_vlm_reference,
    tokenize_krea2_prompt_with_images,
    working_memory_bytes,
)
from invokeai.backend.model_manager.util.qwen3_vl import attach_qwen3vl_visual_tower, drop_qwen3vl_visual_tower
from invokeai.backend.qwen3_vl.qwen3_vl_assets import build_qwen3_vl_processor, load_bundled_qwen3_vl_tokenizer

HIDDEN = 64


def _tiny_config(*, vision_out: int = HIDDEN):
    from transformers import Qwen3VLConfig

    return Qwen3VLConfig(
        text_config={
            "hidden_size": HIDDEN,
            "intermediate_size": 128,
            # Krea-2 taps layers up to 35, so the depth is the real one.
            "num_hidden_layers": 36,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 151936,
            "rope_scaling": {
                "mrope_interleaved": True,
                "mrope_section": [6, 5, 5],
                "rope_type": "default",
                "rope_theta": 5000000,
            },
        },
        vision_config={
            "hidden_size": HIDDEN,
            "intermediate_size": 128,
            "depth": 2,
            "num_heads": 2,
            "out_hidden_size": vision_out,
            "deepstack_visual_indexes": [0],
        },
    )


def _tiny_encoder():
    from transformers import Qwen3VLModel

    torch.manual_seed(0)
    return Qwen3VLModel(_tiny_config()).eval()


@pytest.fixture(scope="module")
def tokenizer():
    return load_bundled_qwen3_vl_tokenizer()


@pytest.fixture(scope="module")
def processor(tokenizer):
    return build_qwen3_vl_processor(tokenizer)


@pytest.fixture(autouse=True)
def _float32_conditioning(monkeypatch):
    """The conditioning is cast to the device's bf16-safe dtype; keep it float32 so equality is exact."""
    monkeypatch.setattr(
        "invokeai.backend.krea2.text_encoding.TorchDevice.choose_bfloat16_safe_dtype", lambda _device: torch.float32
    )


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _image(height: int, width: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(3, height, width, generator=generator)


# --- token layout ----------------------------------------------------------------------------------


def test_prompt_layout_names_each_reference_after_the_system_prefix(tokenizer, processor) -> None:
    """ai-toolkit's layout, token for token: prefix, then ``Picture N: <vision_start><pad>*n<vision_end>`` per
    reference with no separator, the prompt, and the assistant suffix appended after the processor pass.
    Each pad run is one token per 2x2 merged 16px patches."""
    # Both sizes are multiples of the 32px merged patch and inside the processor's pixel budget, so the
    # processor keeps them as they are: 8 x 12 and 10 x 10 merged patches.
    images = [_image(256, 384), _image(320, 320, seed=1)]

    inputs = tokenize_krea2_prompt_with_images("a red fox", images, processor, tokenizer)

    pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    expected = (
        _ids(tokenizer, KREA2_PREFIX)
        + _ids(tokenizer, "Picture 1: ")
        + [start, *[pad] * (8 * 12), end]
        + _ids(tokenizer, "Picture 2: ")
        + [start, *[pad] * (10 * 10), end]
        + _ids(tokenizer, "a red fox")
        + _ids(tokenizer, KREA2_SUFFIX)
    )
    assert inputs["input_ids"][0].tolist() == expected
    assert inputs["image_grid_thw"].tolist() == [[1, 16, 24], [1, 20, 20]]
    # The M-RoPE token types mark exactly the pads, the appended suffix included as text.
    assert torch.equal(inputs["mm_token_type_ids"][0].bool(), inputs["input_ids"][0] == pad)
    assert inputs["attention_mask"].dtype is torch.bool
    assert bool(inputs["attention_mask"].all())


def test_the_prefix_drop_removes_the_system_turn_and_keeps_every_image_token(tokenizer, processor) -> None:
    inputs = tokenize_krea2_prompt_with_images("x", [_image(256, 256)], processor, tokenizer)
    pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    assert len(_ids(tokenizer, KREA2_PREFIX)) == KREA2_START_IDX
    kept = inputs["input_ids"][0, KREA2_START_IDX:]
    assert int((kept == pad).sum()) == int((inputs["input_ids"] == pad).sum()) == 8 * 8


def test_a_long_prompt_is_not_truncated_with_images(tokenizer, processor) -> None:
    """Truncating would cut a pad run apart, which the model refuses; ai-toolkit encodes at natural length."""
    prompt = " ".join(["word"] * 700)

    inputs = tokenize_krea2_prompt_with_images(prompt, [_image(256, 256)], processor, tokenizer)

    assert inputs["input_ids"].shape[1] > KREA2_BODY_MAX_LENGTH
    assert inputs["input_ids"][0, -5:].tolist() == _ids(tokenizer, KREA2_SUFFIX)


# --- the 384^2 vision budget -----------------------------------------------------------------------


def test_a_large_reference_is_downscaled_to_the_vision_budget_keeping_its_aspect() -> None:
    # scale = sqrt(384^2 / (1024 * 768)) = 0.4330; 768 -> 332.55 -> 333, 1024 -> 443.4 -> 443.
    pixels = prepare_vlm_reference(Image.new("RGB", (1024, 768), (255, 0, 0)))

    assert pixels.shape == (3, 333, 443)
    assert pixels.dtype is torch.float32
    assert torch.allclose(pixels[0], torch.ones(333, 443))
    assert torch.allclose(pixels[1:], torch.zeros(2, 333, 443))


def test_a_small_reference_is_never_upscaled() -> None:
    image = Image.new("RGB", (200, 100), (51, 102, 204))

    pixels = prepare_vlm_reference(image)

    assert pixels.shape == (3, 100, 200)
    assert pixels[:, 0, 0].tolist() == pytest.approx([0.2, 0.4, 0.8])


def test_a_thin_reference_is_held_at_the_minimum_side() -> None:
    pixels = prepare_vlm_reference(Image.new("RGB", (1000, 10)))

    assert pixels.shape == (3, 28, 1000)


def test_alpha_is_dropped_rather_than_composited() -> None:
    """ai-toolkit and ComfyUI's LoadImage both discard alpha; compositing over white would brighten a
    transparent reference into something the LoRA never saw."""
    pixels = prepare_vlm_reference(Image.new("RGBA", (64, 64), (0, 0, 0, 0)))

    assert torch.equal(pixels, torch.zeros(3, 64, 64))


def test_the_downscale_is_an_antialiased_bicubic_on_the_float_image() -> None:
    """Pinned by its pixels, not its size. A one-pixel checkerboard is all detail the smaller grid cannot hold:
    an antialiased resize averages it to mid-grey, a plain bicubic samples it into near black and white, and a
    uint8 round trip rounds what the trained path keeps in float."""
    checker = ((np.indices((480, 640)).sum(axis=0) % 2) * 255).astype(np.uint8)
    float_image = torch.from_numpy(checker.astype(np.float32) / 255.0).expand(1, 3, -1, -1)
    # sqrt(384^2 / (640 * 480)) = 0.6928: 480 -> 332.55 -> 333, 640 -> 443.4 -> 443.
    expected = torch.nn.functional.interpolate(float_image, size=(333, 443), mode="bicubic", antialias=True)
    expected = expected.squeeze(0).clamp(0, 1)
    # The pattern only tells the resizes apart if antialiasing really flattens it.
    assert float((expected - 0.5).abs().max()) < 0.05

    pixels = prepare_vlm_reference(Image.fromarray(checker))

    torch.testing.assert_close(pixels, expected, rtol=0, atol=1e-6)


def test_the_downscaled_reference_stays_in_range() -> None:
    """Bicubic overshoots at hard edges; the pass expects pixels in [0, 1], as ai-toolkit clamps them."""
    checker = Image.new("L", (1200, 1200))
    checker.putdata([255 * ((x // 3 + y // 3) % 2) for y in range(1200) for x in range(1200)])

    pixels = prepare_vlm_reference(checker)

    assert pixels.shape == (3, 384, 384)
    assert float(pixels.min()) >= 0.0
    assert float(pixels.max()) <= 1.0


# --- the encode ------------------------------------------------------------------------------------


def test_the_image_encode_matches_a_model_that_never_lost_its_tower(tokenizer, processor) -> None:
    """The lent tower must make the encoder behave exactly as if it had kept it: the same call ai-toolkit
    makes (no position ids, so Qwen3-VL builds its 3-D M-RoPE) on the intact model is the reference."""
    model = _tiny_encoder()
    inputs = tokenize_krea2_prompt_with_images(
        "a red fox", [_image(256, 384), _image(320, 320, seed=1)], processor, tokenizer
    )
    with torch.no_grad():
        hidden = model(**inputs, output_hidden_states=True, use_cache=False).hidden_states
    expected = torch.stack([hidden[i] for i in KREA2_SELECT_LAYERS], dim=2)[:, KREA2_START_IDX:]

    tower = model.visual
    drop_qwen3vl_visual_tower(model)
    with torch.no_grad():
        embeds, mask = encode_krea2_prompt_with_images(inputs, model, tower)

    length = inputs["input_ids"].shape[1] - KREA2_START_IDX
    assert embeds.shape == (1, length, 12, HIDDEN)
    assert torch.equal(embeds, expected)
    assert mask.shape == (1, length)
    assert bool(mask.all())
    assert isinstance(model.visual, torch.nn.Identity)


def test_the_references_change_the_conditioning(tokenizer, processor) -> None:
    """Guards the test above against passing vacuously: with another image the prompt tokens read differently."""
    model = _tiny_encoder()
    tower = model.visual
    drop_qwen3vl_visual_tower(model)

    with torch.no_grad():
        first, _ = encode_krea2_prompt_with_images(
            tokenize_krea2_prompt_with_images("a fox", [_image(256, 256)], processor, tokenizer), model, tower
        )
        second, _ = encode_krea2_prompt_with_images(
            tokenize_krea2_prompt_with_images("a fox", [_image(256, 256, seed=7)], processor, tokenizer), model, tower
        )

    assert first.shape == second.shape
    assert not torch.equal(first[:, -8:], second[:, -8:])


def _expected_text_only(model, tokenizer, prompt: str) -> torch.Tensor:
    """diffusers' text-only layout restated: prefix + prompt padded to 541, the suffix after, 1-D positions."""
    body = _ids(tokenizer, KREA2_PREFIX + prompt)
    pad_id = tokenizer.pad_token_id
    suffix = _ids(tokenizer, KREA2_SUFFIX)
    input_ids = torch.tensor([body + [pad_id] * (KREA2_BODY_MAX_LENGTH - len(body)) + suffix])
    mask = torch.tensor([[True] * len(body) + [False] * (KREA2_BODY_MAX_LENGTH - len(body)) + [True] * len(suffix)])
    positions = (mask.long().cumsum(-1) - 1).clamp(min=0).unsqueeze(0).expand(3, -1, -1)
    with torch.no_grad():
        hidden = model(
            input_ids=input_ids, attention_mask=mask, position_ids=positions, output_hidden_states=True, use_cache=False
        ).hidden_states
    return torch.stack([hidden[i] for i in KREA2_SELECT_LAYERS], dim=2)[:, KREA2_START_IDX:]


def test_the_text_only_conditioning_is_unchanged_and_untouched_by_an_image_encode(tokenizer, processor) -> None:
    """Text-only prompts keep the diffusers-parity path, and an image encode on the same encoder leaves no
    trace: the 3-D rope deltas the multimodal forward caches on the model must not leak into it."""
    model = _tiny_encoder()
    tower = model.visual
    drop_qwen3vl_visual_tower(model)
    expected = _expected_text_only(model, tokenizer, "a red fox")

    with torch.no_grad():
        before, before_mask, _ = encode_krea2_prompt("a red fox", tokenizer, model)
        encode_krea2_prompt_with_images(
            tokenize_krea2_prompt_with_images("a red fox", [_image(256, 256)], processor, tokenizer), model, tower
        )
        after, _, _ = encode_krea2_prompt("a red fox", tokenizer, model)

    assert before.shape == (1, 512, 12, HIDDEN)
    assert torch.equal(before, expected)
    assert torch.equal(after, before)
    assert int(before_mask.sum()) == len(_ids(tokenizer, KREA2_PREFIX + "a red fox")) - KREA2_START_IDX + 5


# --- the swap --------------------------------------------------------------------------------------


def test_the_stub_is_restored_when_the_forward_raises() -> None:
    model = _tiny_encoder()
    tower = model.visual
    drop_qwen3vl_visual_tower(model)
    stub = model.visual

    with pytest.raises(RuntimeError, match="boom"):
        with attach_qwen3vl_visual_tower(model, tower):
            assert model.visual is tower
            raise RuntimeError("boom")

    assert model.visual is stub
    assert list(model.visual.parameters()) == []


def test_a_model_that_still_has_its_tower_is_refused() -> None:
    model = _tiny_encoder()
    own = model.visual

    with pytest.raises(RuntimeError, match="no dropped visual tower"):
        with attach_qwen3vl_visual_tower(model, own):
            pass

    assert model.visual is own


def test_a_tower_of_another_width_is_refused() -> None:
    from transformers import Qwen3VLModel

    model = _tiny_encoder()
    drop_qwen3vl_visual_tower(model)
    foreign = Qwen3VLModel(_tiny_config(vision_out=2 * HIDDEN)).visual

    with pytest.raises(ValueError, match="different Qwen3-VL"):
        with attach_qwen3vl_visual_tower(model, foreign):
            pass

    assert isinstance(model.visual, torch.nn.Identity)


def test_working_memory_grows_with_the_square_of_the_sequence() -> None:
    """The attention term dominates long sequences: doubling the tokens must more than double the reserve."""
    assert working_memory_bytes(2000) > 2 * working_memory_bytes(1000) > 0


def test_conditioning_dtype_follows_the_device(monkeypatch, tokenizer, processor) -> None:
    """The denoise loop reads the conditioning in the device's bf16-safe dtype, as on the text-only path."""
    monkeypatch.setattr(
        "invokeai.backend.krea2.text_encoding.TorchDevice.choose_bfloat16_safe_dtype", lambda _device: torch.float16
    )
    model = _tiny_encoder()
    tower = model.visual
    drop_qwen3vl_visual_tower(model)

    with torch.no_grad():
        embeds, _ = encode_krea2_prompt_with_images(
            tokenize_krea2_prompt_with_images("a fox", [_image(256, 256)], processor, tokenizer), model, tower
        )

    assert embeds.dtype is torch.float16


def test_the_rope_deltas_are_restored_after_an_image_forward(tokenizer, processor) -> None:
    """The multimodal forward caches 3-D rope deltas on the model; the encoder must leave with its own."""
    model = _tiny_encoder()
    tower = model.visual
    drop_qwen3vl_visual_tower(model)
    sentinel = torch.tensor([[7]])
    model.rope_deltas = sentinel
    inputs = tokenize_krea2_prompt_with_images("a fox", [_image(256, 256)], processor, tokenizer)

    with torch.no_grad(), attach_qwen3vl_visual_tower(model, tower):
        model(**inputs, use_cache=False)
        assert model.rope_deltas is not sentinel

    assert model.rope_deltas is sentinel


def test_the_rope_deltas_are_restored_when_the_forward_raises() -> None:
    model = _tiny_encoder()
    tower = model.visual
    drop_qwen3vl_visual_tower(model)
    model.rope_deltas = None

    with pytest.raises(RuntimeError, match="boom"):
        with attach_qwen3vl_visual_tower(model, tower):
            model.rope_deltas = torch.tensor([[3]])
            raise RuntimeError("boom")

    assert model.rope_deltas is None


def test_a_reference_the_processor_refuses_is_named(tokenizer, processor) -> None:
    """A 6000x20 strip keeps its width and is held at 28 px tall: an aspect ratio over the processor's 200."""
    strip = prepare_vlm_reference(Image.new("RGB", (6000, 20)))

    with pytest.raises(ValueError, match=r"reference image 2 .*6000x28 px.*aspect ratio"):
        tokenize_krea2_prompt_with_images("x", [_image(256, 256), strip], processor, tokenizer)
