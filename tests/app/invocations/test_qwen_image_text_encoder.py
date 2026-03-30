"""Tests for the Qwen Image text encoder prompt building and image resizing."""

from PIL import Image

from invokeai.app.invocations.qwen_image_text_encoder import (
    QwenImageTextEncoderInvocation,
    _build_prompt,
)


class TestBuildPrompt:
    """Test the _build_prompt function for edit vs generate modes."""

    def test_no_images_still_has_placeholder(self):
        """With 0 images, should still include one vision placeholder (for the VL encoder)."""
        prompt = _build_prompt("a beautiful sunset", 0)
        assert "a beautiful sunset" in prompt
        assert "<|im_start|>assistant" in prompt
        # On the edit branch, 0 images still uses the edit template with one placeholder
        assert "<|vision_start|><|image_pad|><|vision_end|>" in prompt

    def test_edit_mode_one_image(self):
        """With 1 image, should use the edit template with one vision placeholder."""
        prompt = _build_prompt("change hair to red", 1)
        assert "Describe the key features of the input image" in prompt
        assert prompt.count("<|vision_start|><|image_pad|><|vision_end|>") == 1
        assert "change hair to red" in prompt

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
