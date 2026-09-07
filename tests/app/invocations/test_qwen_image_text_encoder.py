"""Tests for the Qwen Image text encoder prompt building and image resizing."""

import json
import weakref
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration

from invokeai.app.invocations.model import ModelIdentifierField, QwenVLEncoderField
from invokeai.app.invocations.qwen_image_text_encoder import (
    _GENERATE_DROP_IDX,
    QwenImageTextEncoderInvocation,
    _build_prompt,
)
from invokeai.backend.model_manager.load.model_cache.model_cache import MODEL_LOAD_LOCK
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.util.devices import TorchDevice


class TestBuildPrompt:
    """Test the _build_prompt function for edit vs generate modes."""

    def test_no_images_uses_generate_template(self):
        """With 0 images, should use the generate (txt2img) template with no vision placeholder."""
        prompt = _build_prompt("a beautiful sunset", 0)
        assert "a beautiful sunset" in prompt
        assert "<|im_start|>assistant" in prompt
        # Generate mode: no vision placeholders, uses the "describe the image" system prompt
        assert "<|vision_start|>" not in prompt
        assert "Describe the image by detailing" in prompt

    def test_no_images_does_not_use_edit_template(self):
        """With 0 images, should NOT use the edit system prompt."""
        prompt = _build_prompt("a beautiful sunset", 0)
        assert "Describe the key features of the input image" not in prompt

    def test_edit_mode_one_image(self):
        """With 1 image, should use the edit template with one vision placeholder."""
        prompt = _build_prompt("change hair to red", 1)
        assert "Describe the key features of the input image" in prompt
        assert prompt.count("<|vision_start|><|image_pad|><|vision_end|>") == 1
        assert "change hair to red" in prompt
        # Should NOT use the generate system prompt
        assert "Describe the image by detailing" not in prompt

    def test_edit_mode_multiple_images(self):
        """With multiple images, should include one placeholder per image."""
        prompt = _build_prompt("combine these images", 3)
        assert prompt.count("<|vision_start|><|image_pad|><|vision_end|>") == 3
        assert "combine these images" in prompt

    def test_generate_template_has_correct_structure(self):
        """Generate template should have system + user + assistant roles."""
        prompt = _build_prompt("test prompt", 0)
        assert prompt.startswith("<|im_start|>system\n")
        assert "<|im_end|>\n<|im_start|>user\n" in prompt
        assert prompt.endswith("<|im_start|>assistant\n")

    def test_edit_template_has_correct_structure(self):
        """Edit template should have system + user (with image) + assistant roles."""
        prompt = _build_prompt("test prompt", 1)
        assert prompt.startswith("<|im_start|>system\n")
        assert "<|im_end|>\n<|im_start|>user\n" in prompt
        assert "<|vision_start|>" in prompt
        assert prompt.endswith("<|im_start|>assistant\n")

    def test_prompt_special_characters(self):
        """Prompt with special characters should be included verbatim."""
        prompt = _build_prompt("add {curly} braces & <angle> brackets", 0)
        assert "add {curly} braces & <angle> brackets" in prompt


class TestResizeForVLEncoder:
    """Test the image resizing logic for the VL encoder."""

    def test_large_image_is_resized(self):
        """A large image should be resized to ~target_pixels."""
        img = Image.new("RGB", (2048, 2048))
        resized = QwenImageTextEncoderInvocation._resize_for_vl_encoder(img, target_pixels=512 * 512)
        w, h = resized.size
        # Should be much smaller than original
        assert w < 2048
        assert h < 2048
        # Total pixels should be approximately target
        assert abs(w * h - 512 * 512) < 10000  # within ~10k pixels

    def test_small_image_is_resized(self):
        """A small image should also be resized to ~target_pixels."""
        img = Image.new("RGB", (64, 64))
        resized = QwenImageTextEncoderInvocation._resize_for_vl_encoder(img, target_pixels=512 * 512)
        w, h = resized.size
        # Should be larger than original
        assert w > 64
        assert h > 64

    def test_aspect_ratio_preserved(self):
        """Aspect ratio should be approximately preserved."""
        img = Image.new("RGB", (800, 400))  # 2:1 aspect ratio
        resized = QwenImageTextEncoderInvocation._resize_for_vl_encoder(img, target_pixels=512 * 512)
        w, h = resized.size
        original_ratio = 800 / 400  # 2.0
        new_ratio = w / h
        # Allow some deviation due to rounding to multiples of 32
        assert abs(new_ratio - original_ratio) < 0.3

    def test_dimensions_are_multiples_of_32(self):
        """Output dimensions should be multiples of 32."""
        img = Image.new("RGB", (1000, 750))
        resized = QwenImageTextEncoderInvocation._resize_for_vl_encoder(img, target_pixels=512 * 512)
        w, h = resized.size
        assert w % 32 == 0
        assert h % 32 == 0

    def test_square_image(self):
        """A square image should produce approximately square output."""
        img = Image.new("RGB", (1024, 1024))
        resized = QwenImageTextEncoderInvocation._resize_for_vl_encoder(img, target_pixels=512 * 512)
        w, h = resized.size
        assert abs(w - h) <= 32  # within one grid step

    def test_portrait_image(self):
        """A portrait image should produce portrait output."""
        img = Image.new("RGB", (600, 1200))
        resized = QwenImageTextEncoderInvocation._resize_for_vl_encoder(img, target_pixels=512 * 512)
        w, h = resized.size
        assert h > w  # should remain portrait

    def test_landscape_image(self):
        """A landscape image should produce landscape output."""
        img = Image.new("RGB", (1200, 600))
        resized = QwenImageTextEncoderInvocation._resize_for_vl_encoder(img, target_pixels=512 * 512)
        w, h = resized.size
        assert w > h  # should remain landscape


class TestQuantizedEncoderLoad:
    """The BitsAndBytes path bypasses the model cache, so it must ask the cache for VRAM itself and load onto the
    worker's execution device explicitly (issue #9147: `device_map="auto"` spilled to the CPU because the cached
    transformer and VAE still filled the card, and BnB int8 refused to run that way).
    """

    TOTAL_SIZE = 16 * 2**30  # bf16 Qwen2.5-VL-7B on disk

    @staticmethod
    def _make_invocation(quantization: str) -> QwenImageTextEncoderInvocation:
        encoder = ModelIdentifierField(
            key="enc", hash="h", name="qwen-vl", base=BaseModelType.QwenImage, type=ModelType.QwenVLEncoder
        )
        return QwenImageTextEncoderInvocation(
            prompt="a cat",
            qwen_vl_encoder=QwenVLEncoderField(tokenizer=encoder, text_encoder=encoder),
            quantization=quantization,
        )

    def _make_context(self, tmp_path: Path, events: list[str], single_file: bool = False) -> MagicMock:
        if single_file:
            model_root = tmp_path / "encoder.safetensors"
            model_root.write_bytes(b"")
        else:
            model_root = tmp_path / "qwen-vl"
            text_encoder_dir = model_root / "text_encoder"
            text_encoder_dir.mkdir(parents=True)
            index = {"metadata": {"total_size": self.TOTAL_SIZE}, "weight_map": {}}
            (text_encoder_dir / "model.safetensors.index.json").write_text(json.dumps(index))

        context = MagicMock()
        context.models.get_absolute_path.return_value = model_root
        context.models.make_room_in_vram.side_effect = lambda *a, **k: events.append("make_room") or 0
        return context

    @pytest.mark.parametrize(("quantization", "ratio"), [("int8", 0.6), ("nf4", 0.4)])
    def test_makes_room_in_vram_before_loading_onto_the_execution_device(
        self, tmp_path: Path, quantization: str, ratio: float
    ):
        events: list[str] = []
        context = self._make_context(tmp_path, events)
        fake_model = MagicMock()
        device = torch.device("cuda:1")
        seen: dict = {}

        def fake_from_pretrained(path, **kwargs):
            events.append("from_pretrained")
            seen.update(kwargs)
            # The load must run under the model-load lock so a concurrent cache construction on another worker
            # cannot hijack its parameter assignment onto the meta device.
            seen["locked"] = MODEL_LOAD_LOCK._readers > 0 or MODEL_LOAD_LOCK._writer_active
            return fake_model

        with (
            patch.object(Qwen2_5_VLForConditionalGeneration, "from_pretrained", side_effect=fake_from_pretrained),
            patch.object(TorchDevice, "choose_torch_device", return_value=device),
        ):
            text_encoder, returned_device, cleanup = self._make_invocation(quantization)._load_quantized_encoder(
                context
            )

        assert events == ["make_room", "from_pretrained"]
        context.models.make_room_in_vram.assert_called_once_with(int(self.TOTAL_SIZE * ratio))
        assert seen["device_map"] == {"": device}, "must never use device_map='auto'"
        assert seen["locked"]
        assert text_encoder is fake_model
        assert returned_device == device
        cleanup()

    def test_single_file_checkpoint_falls_back_to_the_cache_without_making_room(self, tmp_path: Path):
        """A single-file encoder cannot be BnB-quantized; it goes through the cache, which makes its own room."""
        events: list[str] = []
        context = self._make_context(tmp_path, events, single_file=True)
        invocation = self._make_invocation("int8")
        sentinel = (MagicMock(), torch.device("cuda"), None)

        with (
            patch.object(invocation, "_load_cached_encoder", return_value=sentinel) as cached,
            patch.object(Qwen2_5_VLForConditionalGeneration, "from_pretrained") as from_pretrained,
        ):
            result = invocation._load_quantized_encoder(context)

        assert result is sentinel
        cached.assert_called_once_with(context)
        from_pretrained.assert_not_called()
        context.models.make_room_in_vram.assert_not_called()


class TestQuantizedEncoderRelease:
    """The quantized encoder lives outside the cache, so `_encode` is its only owner. Its cleanup callback empties
    the CUDA cache, which only returns the ~9 GB of encoder weights to the driver if nothing still references the
    model at that point - otherwise they stay reserved by torch and the cache under-budgets the next load.
    """

    HIDDEN = 3584

    class _FakeEncoder(torch.nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.hidden = hidden

        def forward(self, input_ids, attention_mask, **_):
            hidden_states = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.hidden)
            return MagicMock(hidden_states=[hidden_states])

    def test_encoder_is_released_before_cleanup_runs(self, tmp_path: Path):
        model_root = tmp_path / "qwen-vl"
        (model_root / "tokenizer").mkdir(parents=True)
        context = MagicMock()
        context.models.get_absolute_path.return_value = model_root

        encoder = self._FakeEncoder(self.HIDDEN)
        encoder_ref = weakref.ref(encoder)
        alive_at_cleanup: list[bool] = []

        def cleanup():
            alive_at_cleanup.append(encoder_ref() is not None)

        seq_len = _GENERATE_DROP_IDX + 5
        model_inputs = MagicMock()
        model_inputs.input_ids = torch.zeros(1, seq_len, dtype=torch.long)
        model_inputs.attention_mask = torch.ones(1, seq_len, dtype=torch.long)
        model_inputs.to.return_value = model_inputs
        processor = MagicMock(return_value=model_inputs)

        # Hand the encoder over through a one-shot side effect: a `return_value` tuple would keep the mock holding a
        # strong reference of its own and mask the ownership being tested.
        handoff = [encoder]
        invocation = TestQuantizedEncoderLoad._make_invocation("int8")
        with (
            patch.object(
                invocation,
                "_load_quantized_encoder",
                side_effect=lambda _ctx: (handoff.pop(), torch.device("cpu"), cleanup),
            ),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
            patch("transformers.Qwen2_5_VLProcessor", return_value=processor),
        ):
            del encoder  # `_encode` now holds the only strong reference
            prompt_embeds, mask = invocation._encode(context, images=[])

        assert alive_at_cleanup == [False], "cleanup ran while the encoder was still referenced"
        assert prompt_embeds.shape == (1, 5, self.HIDDEN)
        assert mask is None
