from typing import Literal

import torch
from PIL import Image as PILImage

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    ImageField,
    Input,
    InputField,
    UIComponent,
)
from invokeai.app.invocations.model import QwenVLEncoderField
from invokeai.app.invocations.primitives import QwenImageEditConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.load.model_cache.utils import get_effective_device
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    QwenImageEditConditioningInfo,
)

# The Qwen Image Edit pipeline uses a specific system prompt and drops the first
# N tokens (the system prompt prefix) from the embeddings.  These constants are
# taken directly from the diffusers QwenImageEditPipeline.
_SYSTEM_PROMPT = (
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. "
    "Generate a new image that meets the user's requirements while maintaining consistency "
    "with the original input where appropriate."
)
_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
_DROP_IDX = 64


def _build_prompt(user_prompt: str, num_images: int) -> str:
    """Build the full prompt with one vision placeholder per reference image."""
    image_tokens = _IMAGE_PLACEHOLDER * max(num_images, 1)
    return (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{image_tokens}{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@invocation(
    "qwen_image_edit_text_encoder",
    title="Prompt - Qwen Image Edit",
    tags=["prompt", "conditioning", "qwen_image_edit"],
    category="conditioning",
    version="1.2.0",
    classification=Classification.Prototype,
)
class QwenImageEditTextEncoderInvocation(BaseInvocation):
    """Encodes text and reference images for Qwen Image Edit using Qwen2.5-VL."""

    prompt: str = InputField(description="Text prompt describing the desired edit.", ui_component=UIComponent.Textarea)
    reference_images: list[ImageField] = InputField(
        default=[],
        description="Reference images to guide the edit. The model can use multiple reference images.",
    )
    qwen_vl_encoder: QwenVLEncoderField = InputField(
        title="Qwen VL Encoder",
        description=FieldDescriptions.qwen_vl_encoder,
        input=Input.Connection,
    )
    quantization: Literal["none", "int8", "nf4"] = InputField(
        default="none",
        description="Quantize the Qwen VL encoder to reduce VRAM usage. "
        "'nf4' (4-bit) saves the most memory, 'int8' (8-bit) is a middle ground.",
    )

    @staticmethod
    def _resize_for_vl_encoder(image: PILImage.Image, target_pixels: int = 512 * 512) -> PILImage.Image:
        """Resize image to fit within target_pixels while preserving aspect ratio.

        Matches the diffusers pipeline's calculate_dimensions logic: the image is resized
        so its total pixel count is approximately target_pixels, with dimensions rounded to
        multiples of 32. This prevents large images from producing too many vision tokens
        which can overwhelm the text prompt.
        """
        w, h = image.size
        aspect = w / h
        # Compute dimensions that preserve aspect ratio at ~target_pixels total
        new_w = int((target_pixels * aspect) ** 0.5)
        new_h = int(target_pixels / new_w)
        # Round to multiples of 32
        new_w = max(32, (new_w // 32) * 32)
        new_h = max(32, (new_h // 32) * 32)
        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), resample=PILImage.LANCZOS)
        return image

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> QwenImageEditConditioningOutput:
        # Load and resize reference images to ~1M pixels (matching diffusers pipeline)
        pil_images: list[PILImage.Image] = []
        for img_field in self.reference_images:
            pil_img = context.images.get_pil(img_field.image_name)
            pil_img = self._resize_for_vl_encoder(pil_img.convert("RGB"))
            pil_images.append(pil_img)

        prompt_embeds, prompt_mask = self._encode(context, pil_images)
        prompt_embeds = prompt_embeds.detach().to("cpu")
        prompt_mask = prompt_mask.detach().to("cpu") if prompt_mask is not None else None

        conditioning_data = ConditioningFieldData(
            conditionings=[QwenImageEditConditioningInfo(prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_mask)]
        )
        conditioning_name = context.conditioning.save(conditioning_data)
        return QwenImageEditConditioningOutput.build(conditioning_name)

    def _encode(
        self, context: InvocationContext, images: list[PILImage.Image]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode text prompt and reference images using Qwen2.5-VL.

        Matches the diffusers QwenImageEditPipeline._get_qwen_prompt_embeds logic:
        1. Format prompt with the edit-specific system template
        2. Run through Qwen2.5-VL to get hidden states
        3. Extract valid (non-padding) tokens and drop the system prefix
        4. Return padded embeddings + attention mask
        """
        from transformers import AutoTokenizer, Qwen2_5_VLProcessor

        try:
            from transformers import Qwen2_5_VLImageProcessor as _ImageProcessorCls
        except ImportError:
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import (  # type: ignore[no-redef]
                Qwen2VLImageProcessor as _ImageProcessorCls,
            )

        try:
            from transformers import Qwen2_5_VLVideoProcessor as _VideoProcessorCls
        except ImportError:
            from transformers.models.qwen2_vl.video_processing_qwen2_vl import (  # type: ignore[no-redef]
                Qwen2VLVideoProcessor as _VideoProcessorCls,
            )

        # Format the prompt with one vision placeholder per reference image
        text = _build_prompt(self.prompt, len(images))

        # Build the processor
        tokenizer_config = context.models.get_config(self.qwen_vl_encoder.tokenizer)
        model_root = context.models.get_absolute_path(tokenizer_config)
        tokenizer_dir = model_root / "tokenizer"

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)

        image_processor = None
        for search_dir in [model_root / "processor", tokenizer_dir, model_root, model_root / "image_processor"]:
            if (search_dir / "preprocessor_config.json").exists():
                image_processor = _ImageProcessorCls.from_pretrained(str(search_dir), local_files_only=True)
                break
        if image_processor is None:
            image_processor = _ImageProcessorCls()

        processor = Qwen2_5_VLProcessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            video_processor=_VideoProcessorCls(),
        )

        context.util.signal_progress("Running Qwen2.5-VL text/vision encoder")

        if self.quantization != "none":
            text_encoder, device, cleanup = self._load_quantized_encoder(context)
        else:
            text_encoder, device, cleanup = self._load_cached_encoder(context)

        try:
            model_inputs = processor(
                text=[text],
                images=images if images else None,
                padding=True,
                return_tensors="pt",
            ).to(device=device)

            outputs = text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=getattr(model_inputs, "pixel_values", None),
                image_grid_thw=getattr(model_inputs, "image_grid_thw", None),
                output_hidden_states=True,
            )

            # Use last hidden state (matching diffusers pipeline)
            hidden_states = outputs.hidden_states[-1]

            # Extract valid (non-padding) tokens using the attention mask,
            # then drop the first _DROP_IDX tokens (system prompt prefix).
            attn_mask = model_inputs.attention_mask
            bool_mask = attn_mask.bool()
            valid_lengths = bool_mask.sum(dim=1)
            selected = hidden_states[bool_mask]
            split_hidden = torch.split(selected, valid_lengths.tolist(), dim=0)

            # Drop system prefix tokens and build padded output
            trimmed = [h[_DROP_IDX:] for h in split_hidden]
            attn_mask_list = [torch.ones(h.size(0), dtype=torch.long, device=device) for h in trimmed]
            max_seq_len = max(h.size(0) for h in trimmed)

            prompt_embeds = torch.stack(
                [torch.cat([h, h.new_zeros(max_seq_len - h.size(0), h.size(1))]) for h in trimmed]
            )
            encoder_attention_mask = torch.stack(
                [torch.cat([m, m.new_zeros(max_seq_len - m.size(0))]) for m in attn_mask_list]
            )

            prompt_embeds = prompt_embeds.to(dtype=torch.bfloat16)
        finally:
            if cleanup is not None:
                cleanup()

        # If all tokens are valid (no padding), mask is not needed
        if encoder_attention_mask.all():
            encoder_attention_mask = None

        return prompt_embeds, encoder_attention_mask

    def _load_cached_encoder(self, context: InvocationContext):
        """Load the text encoder through the model cache (no quantization)."""
        from transformers import Qwen2_5_VLForConditionalGeneration

        text_encoder_info = context.models.load(self.qwen_vl_encoder.text_encoder)
        ctx = text_encoder_info.model_on_device()
        _, text_encoder = ctx.__enter__()
        device = get_effective_device(text_encoder)
        assert isinstance(text_encoder, Qwen2_5_VLForConditionalGeneration)
        return text_encoder, device, lambda: ctx.__exit__(None, None, None)

    def _load_quantized_encoder(self, context: InvocationContext):
        """Load the text encoder with BitsAndBytes quantization, bypassing the model cache.

        BnB-quantized models are pinned to GPU and can't be moved between devices,
        so they can't go through the standard model cache. We keep a module-level
        cache keyed on (model_key, quantization) to avoid reloading on every call.
        """
        from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        encoder_config = context.models.get_config(self.qwen_vl_encoder.text_encoder)
        model_root = context.models.get_absolute_path(encoder_config)
        encoder_path = model_root / "text_encoder"

        cache_key = (encoder_config.key, self.quantization)
        if cache_key in _quantized_encoder_cache:
            text_encoder = _quantized_encoder_cache[cache_key]
            device = next(text_encoder.parameters()).device
            return text_encoder, device, None

        if self.quantization == "nf4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:  # int8
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        context.util.signal_progress("Loading Qwen2.5-VL encoder (quantized)")
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(encoder_path),
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        _quantized_encoder_cache[cache_key] = text_encoder
        device = next(text_encoder.parameters()).device
        return text_encoder, device, None


# Module-level cache for BnB-quantized encoders (they can't go through the model cache
# because they're pinned to GPU). Keyed by (model_key, quantization_type).
_quantized_encoder_cache: dict[tuple[str, str], object] = {}
