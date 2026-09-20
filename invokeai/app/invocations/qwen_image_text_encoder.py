import gc
import json
import mmap
import traceback
from pathlib import Path
from typing import Any, Callable, Literal

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
from invokeai.app.invocations.primitives import QwenImageConditioningOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.load.model_cache.model_cache import MB, MODEL_LOAD_LOCK
from invokeai.backend.model_manager.load.model_util import calc_model_size_by_fs
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    QwenImageConditioningInfo,
)
from invokeai.backend.util.devices import TorchDevice

# Prompt templates and drop indices for the two Qwen Image model modes.
# These are taken directly from the diffusers pipelines.

# Image editing mode (QwenImagePipeline)
_EDIT_SYSTEM_PROMPT = (
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. "
    "Generate a new image that meets the user's requirements while maintaining consistency "
    "with the original input where appropriate."
)
_EDIT_DROP_IDX = 64

# Text-to-image mode (QwenImagePipeline)
_GENERATE_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:"
)
_GENERATE_DROP_IDX = 34

# Fraction of the on-disk (bf16) encoder that stays resident after BitsAndBytes quantization. Linear weights
# shrink to 8 or 4 bits, but embeddings, norms, biases and the (excluded) lm_head stay in bf16, and nf4 carries
# per-block absmax scales, so the ratios sit above the pure 1/2 and 1/4. Over-estimating only offloads a little
# more of the cached models to RAM, whereas under-estimating leaves the encoder without enough VRAM, so both
# values are deliberately conservative.
_QUANTIZED_SIZE_RATIO: dict[str, float] = {"int8": 0.6, "nf4": 0.4}

_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

# The encoder as handed from a loader to `_encode`: the model, the device to run it on, and a callback that releases
# it when `_encode` is done.
_LoadedEncoder = tuple[torch.nn.Module, torch.device, Callable[[], None]]


def _read_checkpoint(encoder_path: Path) -> dict[str, torch.Tensor] | None:
    """Map the encoder's safetensors shards and pull their pages into memory.

    Returns `None` when the checkpoint has no safetensors shards (e.g. a `.bin` one); the caller then lets transformers
    read it. The tensors are zero-copy views onto the mapped files, so holding all of them costs page cache rather
    than RAM. Touching one byte per page is what makes the disk read happen *here*, outside the model-load lock,
    instead of lazily inside `from_pretrained`, which has to run under it: with the pages already cached the touch is
    nearly free (~60 ms for the 16 GB Qwen2.5-VL-7B checkpoint), and cold it runs at the disk's sequential speed.
    Call it *after* offloading cached models to RAM, not before: the offload's anonymous memory is what would evict
    freshly read pages on a RAM-constrained host, forcing a second read under the lock.
    """
    from safetensors.torch import load_file

    index_path = encoder_path / "model.safetensors.index.json"
    if index_path.is_file():
        with open(index_path) as index_file:
            shard_names = sorted(set(json.load(index_file)["weight_map"].values()))
    elif (encoder_path / "model.safetensors").is_file():
        shard_names = ["model.safetensors"]
    else:
        return None

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        state_dict.update(load_file(encoder_path / shard_name))
    for tensor in state_dict.values():
        # A strided read of one byte per page faults the whole tensor in through the very mapping transformers will
        # read from; the sum itself is discarded.
        tensor.reshape(-1).view(torch.uint8)[:: mmap.PAGESIZE].sum()
    return state_dict


def _build_prompt(user_prompt: str, num_images: int) -> str:
    """Build the full prompt with the appropriate template based on whether reference images are provided."""
    if num_images > 0:
        # Edit mode: include vision placeholders for reference images
        image_tokens = _IMAGE_PLACEHOLDER * num_images
        return (
            f"<|im_start|>system\n{_EDIT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{image_tokens}{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        # Generate mode: text-only prompt
        return (
            f"<|im_start|>system\n{_GENERATE_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


@invocation(
    "qwen_image_text_encoder",
    title="Prompt - Qwen Image",
    tags=["prompt", "conditioning", "qwen_image"],
    category="conditioning",
    version="1.2.0",
    classification=Classification.Prototype,
    idle_gpu_offloadable=True,
)
class QwenImageTextEncoderInvocation(BaseInvocation):
    """Encodes text and reference images for Qwen Image using Qwen2.5-VL."""

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
    def invoke(self, context: InvocationContext) -> QwenImageConditioningOutput:
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
            conditionings=[QwenImageConditioningInfo(prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_mask)]
        )
        conditioning_name = context.conditioning.save(conditioning_data)
        return QwenImageConditioningOutput.build(conditioning_name)

    def _encode(
        self, context: InvocationContext, images: list[PILImage.Image]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode text prompt and reference images using Qwen2.5-VL.

        Matches the diffusers QwenImagePipeline._get_qwen_prompt_embeds logic:
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

        # Single-file checkpoints (e.g. ComfyUI fp8_scaled): model_root is the
        # safetensors file itself, so there's no tokenizer/processor folder
        # alongside it. Fall back to the canonical Qwen2.5-VL repo on HF (small
        # ~10 MB download for tokenizer+processor configs, cached for offline use).
        if model_root.is_file():
            HF_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
            try:
                tokenizer = AutoTokenizer.from_pretrained(HF_REPO, local_files_only=True)
            except OSError:
                tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
            try:
                image_processor = _ImageProcessorCls.from_pretrained(HF_REPO, local_files_only=True)
            except OSError:
                try:
                    image_processor = _ImageProcessorCls.from_pretrained(HF_REPO)
                except Exception:
                    image_processor = _ImageProcessorCls()
        else:
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
            prompt_embeds, encoder_attention_mask = self._run_encoder(text_encoder, model_inputs, bool(images))
        except BaseException as exc:
            # The in-flight traceback references the forward's frames, and through their locals the activations
            # (the full-vocabulary logits and every layer's hidden states, hundreds of MB at prompt length) and,
            # on the quantized path, the model itself - so the release below would otherwise be a no-op exactly
            # when VRAM is scarcest (an OOM inside the forward is the likeliest error here). Clearing the finished
            # frames drops those references while keeping the traceback's line information, which is all the
            # error report uses. This applies to the cache-owned encoder too: the cache keeps the model, but the
            # activations are the forward's alone.
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            # Drop this frame's references before `cleanup` runs: the quantized encoder is only released once
            # nothing holds it, and `cleanup` calls empty_cache() right after. With the model still alive here,
            # that empty_cache() ran too early and ~9 GB of encoder weights stayed *reserved* by torch after the
            # node finished. The cache budgets from allocated + driver-free VRAM, so reserved-but-unused memory
            # looked like it was in use and the next model (the transformer) was needlessly partial-loaded. The
            # activations live in `_run_encoder`'s frame and are gone by now for the same reason.
            del text_encoder
            cleanup()

        # If all tokens are valid (no padding), mask is not needed
        if encoder_attention_mask.all():
            encoder_attention_mask = None

        return prompt_embeds, encoder_attention_mask

    @staticmethod
    def _run_encoder(
        text_encoder: torch.nn.Module, model_inputs: Any, edit_mode: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the encoder and build the padded embeddings + mask.

        Kept out of `_encode` on purpose: the full-vocabulary logits and per-layer hidden states the forward returns
        are only referenced by this frame, so they are released as soon as it returns (or is cleared on error),
        before `_encode` empties the CUDA cache.
        """
        device = model_inputs.input_ids.device
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
        # then drop the system prompt prefix tokens.
        # The drop index differs between edit mode (64) and generate mode (34).
        drop_idx = _EDIT_DROP_IDX if edit_mode else _GENERATE_DROP_IDX

        attn_mask = model_inputs.attention_mask
        bool_mask = attn_mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_hidden = torch.split(selected, valid_lengths.tolist(), dim=0)

        # Drop system prefix tokens and build padded output
        trimmed = [h[drop_idx:] for h in split_hidden]
        attn_mask_list = [torch.ones(h.size(0), dtype=torch.long, device=device) for h in trimmed]
        max_seq_len = max(h.size(0) for h in trimmed)

        prompt_embeds = torch.stack([torch.cat([h, h.new_zeros(max_seq_len - h.size(0), h.size(1))]) for h in trimmed])
        encoder_attention_mask = torch.stack(
            [torch.cat([m, m.new_zeros(max_seq_len - m.size(0))]) for m in attn_mask_list]
        )

        return prompt_embeds.to(dtype=torch.bfloat16), encoder_attention_mask

    def _load_cached_encoder(self, context: InvocationContext) -> _LoadedEncoder:
        """Load the text encoder through the model cache (no quantization). The cache stays the model's owner."""
        from transformers import Qwen2_5_VLForConditionalGeneration

        text_encoder_info = context.models.load(self.qwen_vl_encoder.text_encoder)
        ctx = text_encoder_info.model_on_device()
        _, text_encoder = ctx.__enter__()
        # Use the encoder's intended compute device, not its current parameter residency: partial loading may have
        # temporarily offloaded all weights to RAM, which would wrongly run the whole encode on the CPU.
        device = text_encoder_info.compute_device
        assert isinstance(text_encoder, Qwen2_5_VLForConditionalGeneration)

        def release() -> None:
            ctx.__exit__(None, None, None)

        return text_encoder, device, release

    def _load_quantized_encoder(self, context: InvocationContext) -> _LoadedEncoder:
        """Load the text encoder with BitsAndBytes quantization, bypassing the model cache.

        BnB-quantized models are pinned to GPU and can't be moved between devices,
        so they can't go through the standard model cache. The model is loaded fresh
        each time and freed after use via the cleanup callback.

        Because the load bypasses the cache, it also bypasses the cache's usual
        make-room-for-the-model-being-locked step, so this path has to ask the cache
        for VRAM explicitly. Without that, whatever the resident transformer/VAE left
        free is all the encoder gets: `device_map="auto"` then silently plans to spill
        layers to the CPU, which BnB int8 refuses with "Some modules are dispatched on
        the CPU or the disk" (issue #9147).
        """
        import warnings

        from transformers import BitsAndBytesConfig, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

        encoder_config = context.models.get_config(self.qwen_vl_encoder.text_encoder)
        model_root = context.models.get_absolute_path(encoder_config)
        if model_root.is_file():
            # Single-file checkpoint (e.g. ComfyUI fp8_scaled): BnB can't load from
            # a single file, and the checkpoint is already FP8-compressed anyway.
            # Fall back to the cached path; the user effectively gets fp8 instead of
            # int8/nf4, which is comparable in size.
            return self._load_cached_encoder(context)
        encoder_path = model_root / "text_encoder"

        if self.quantization == "nf4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:  # int8
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load onto this worker's execution device, never `device_map="auto"`: "auto" sizes its plan from whatever
        # VRAM is free *right now* and quietly spills to the CPU when the cached models fill the card, and it shards
        # across every visible GPU - in multi-GPU mode onto cards that belong to other workers, whose caches never
        # learn of the intrusion. The encoder therefore lives on this session's device only, and a genuine shortfall
        # surfaces as an OOM carrying the numbers (below) instead of a misleading offload error.
        device = TorchDevice.choose_torch_device()
        quantized_bytes = int(calc_model_size_by_fs(encoder_path) * _QUANTIZED_SIZE_RATIO[self.quantization])
        if quantized_bytes == 0:
            context.logger.warning(
                f"Could not determine the size of the Qwen2.5-VL encoder weights in {encoder_path}, so no room will "
                "be made for the quantized encoder in VRAM; the load may fail if the cached models fill the device."
            )

        context.util.signal_progress("Loading Qwen2.5-VL encoder (quantized)")

        # Three steps, each under the lock its kind of work requires (see MODEL_LOAD_LOCK), so that the process-global
        # lock is held no longer than necessary - while a writer holds it, or waits for it, every VRAM move on every
        # worker stalls, and every reader delays every cold model construction:
        # 1. the offload is a VRAM move like any other (`load_state_dict(assign=True)` -> `register_parameter`), so it
        #    takes the read lock, acquired *before* the cache lock `make_room_in_vram` takes, per the lock-ordering
        #    contract;
        # 2. the checkpoint read takes no lock: it creates no parameters and touches no global state, and it is the
        #    slow part on anything but NVMe. It runs after the offload so the offload's RAM cannot evict its pages;
        # 3. the construction takes the WRITE lock, like every construction. It cannot run alongside another
        #    construction: `from_pretrained` builds under process-global save/restore patches (`torch.set_default_dtype`,
        #    `PreTrainedModel.tie_weights`, `torch.linspace`), which two overlapping builds would restore in the wrong
        #    order and leave installed. It cannot run alongside VRAM moves either, in either direction: a cache
        #    construction's `accelerate.init_empty_weights` patch would strand the weights it assigns (through
        #    `setattr` -> `register_parameter`) on the meta device, and its own `torch.set_default_dtype` would change
        #    what a concurrent move allocates. TestTransformersLoadPathAssumptions pins both facts against the
        #    installed transformers.
        vram_available: int | None = None
        state_dict: dict[str, torch.Tensor] | None = None
        try:
            # A CPU execution device has no VRAM to make room in (recent bitsandbytes can quantize on the CPU).
            if quantized_bytes > 0 and device.type != "cpu":
                with MODEL_LOAD_LOCK.read_lock():
                    vram_available = context.models.make_room_in_vram(quantized_bytes)
                if vram_available < quantized_bytes:
                    context.logger.warning(
                        f"Only {max(vram_available, 0) / MB:.0f} MB of VRAM could be made available on {device} for "
                        f"the {self.quantization}-quantized Qwen2.5-VL encoder (~{quantized_bytes / MB:.0f} MB); "
                        "locked (in-use) models cannot be offloaded. The load may run out of memory."
                    )

            model_config = Qwen2_5_VLConfig.from_pretrained(str(encoder_path), local_files_only=True)
            state_dict = _read_checkpoint(encoder_path)

            with MODEL_LOAD_LOCK.write_lock(), warnings.catch_warnings():
                # BnB int8 internally casts bfloat16→float16; the warning is harmless
                warnings.filterwarnings("ignore", message="MatMul8bitLt.*cast.*float16")
                text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    None if state_dict is not None else str(encoder_path),
                    config=model_config,
                    state_dict=state_dict,
                    quantization_config=bnb_config,
                    device_map={"": device},
                    dtype=torch.bfloat16,
                    local_files_only=True,
                )
        except BaseException as exc:
            # A failed load (an OOM is the expected way) leaves the partially built model referenced only by the
            # in-flight traceback's frames. Without releasing it here its weights stay allocated until the session
            # processor drops the exception, and *reserved* by torch after that, so the next load under-budgets -
            # the very leak `_encode` guards against on the success path. This frame is still executing and is
            # skipped by `clear_frames`, hence the explicit drop of the checkpoint.
            traceback.clear_frames(exc.__traceback__)
            state_dict = None
            gc.collect()
            try:
                TorchDevice.empty_cache()
            except Exception as cache_exc:
                # A sick device context fails here too; the original error is the one worth reporting.
                context.logger.warning(f"Could not empty the device cache after the failed load: {cache_exc}")
            if isinstance(exc, torch.OutOfMemoryError):
                raise torch.OutOfMemoryError(self._oom_message(device, quantized_bytes, vram_available)) from exc
            raise

        # Hand the model out without keeping a reference in this closure, so that once `_encode` drops its own the
        # weights are actually free by the time empty_cache() runs.
        def cleanup() -> None:
            gc.collect()
            TorchDevice.empty_cache()

        return text_encoder, device, cleanup

    def _oom_message(self, device: torch.device, quantized_bytes: int, vram_available: int | None) -> str:
        if quantized_bytes == 0:
            sizing = "its size could not be estimated, so no room was made for it"
        elif vram_available is None:
            sizing = (
                f"it needs about {quantized_bytes / MB:.0f} MB, and offloading the cached models to make room failed"
            )
        else:
            sizing = (
                f"it needs about {quantized_bytes / MB:.0f} MB and {vram_available / MB:.0f} MB was available after "
                "offloading the cached models (locked, in-use models cannot be offloaded)"
            )
        return (
            f"Not enough VRAM on {device} for the {self.quantization}-quantized Qwen2.5-VL encoder: {sizing}. Try "
            "'nf4' quantization, or 'none', which loads the encoder through the model cache and can partially "
            "offload it. In a multi-GPU setup the encoder is loaded onto this session's device only."
        )
