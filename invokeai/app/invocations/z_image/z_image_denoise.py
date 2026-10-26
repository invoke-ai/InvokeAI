import math
from contextlib import ExitStack
from typing import Callable, Iterator, Optional

import torch
import torchvision.transforms as tv_transforms
from diffusers.models.transformers.transformer_z_image import SEQ_MULTI_OF
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from PIL import Image
from torchvision.transforms.functional import resize as tv_resize
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    ZImageConditioningField,
)
from invokeai.app.invocations.latent_noise import validate_noise_tensor_shape
from invokeai.app.invocations.model import TransformerField, VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.invocations.vae.z_image_image_to_latents import ZImageImageToLatentsInvocation
from invokeai.app.invocations.z_image.z_image_control import ZImageControlField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.schedulers import (
    ZIMAGE_SCHEDULER_LABELS,
    ZIMAGE_SCHEDULER_MAP,
    ZIMAGE_SCHEDULER_NAME_VALUES,
    set_heun_timesteps_from_sigmas,
)
from invokeai.backend.fun_controlnet.conditioning import build_fun_control_context
from invokeai.backend.fun_controlnet.hooks import FunControlHooks, install_fun_control, should_apply
from invokeai.backend.fun_controlnet.z_image import ZImageFunControl
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher, PatchSpec
from invokeai.backend.patches.lora_conversions.z_image_lora_constants import Z_IMAGE_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.quantization.dequantizing_linear import peak_dequant_transient_bytes, requires_sidecar_patching
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.rectified_flow.schedule_shift import ShiftFit, time_shift
from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    PipelineIntermediateState,
    image_resized_to_grid_as_tensor,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ZImageConditioningInfo
from invokeai.backend.util.attention import sdpa_score_matrix_bytes
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.fp8 import get_model_compute_dtype
from invokeai.backend.util.sage_attention import sage_attention_scope
from invokeai.backend.z_image.extensions.regional_prompting_extension import ZImageRegionalPromptingExtension
from invokeai.backend.z_image.text_conditioning import ZImageTextConditioning
from invokeai.backend.z_image.z_image_transformer_patch import patch_transformer_for_regional_prompting

# The transformer's width and head count, which its ControlNets share: 30 heads of 128.
Z_IMAGE_HIDDEN_SIZE = 3840
Z_IMAGE_ATTENTION_HEADS = 30
# Peak reserved activation bytes per attended token on a fused SDPA kernel; see `_estimate_working_memory`.
Z_IMAGE_BYTES_PER_TOKEN = int(0.25 * 1024**2)


def _padded(tokens: int) -> int:
    """The transformer pads the image and the caption token runs to a multiple of `SEQ_MULTI_OF` each."""
    return tokens + (-tokens) % SEQ_MULTI_OF


# The shift fit of diffusers' ZImagePipeline. 4096 image tokens is a 1024x1024 frame -- one megapixel, and where the
# reference fit ends. Extrapolated past it the shift runs away fast -- 25 at 2048px, 810 at 3072px and 103_777 at
# 4096px -- and `_get_sigmas` flattens with it. Because `denoising_start` is clipped by *index* here, the step count
# survives but the entry point does not: a canvas img2img pass at strength 0.5 would start from sigma 0.88 at 1536px,
# 0.96 at 2048px and 1.0000 at 4096px, i.e. throw the input image away and generate a new one. Below one megapixel
# the hold changes nothing, and an explicit `shift` on the invocation still overrides it.
Z_IMAGE_SHIFT_FIT = ShiftFit(base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15)


@invocation(
    "z_image_denoise",
    title="Denoise - Z-Image",
    tags=["image", "z-image"],
    category="latents",
    version="1.6.1",
    classification=Classification.Prototype,
)
class ZImageDenoiseInvocation(BaseInvocation):
    """Run the denoising process with a Z-Image model.

    Supports regional prompting by connecting multiple conditioning inputs with masks.
    """

    # If latents is provided, this means we are doing image-to-image.
    latents: Optional[LatentsField] = InputField(
        default=None, description=FieldDescriptions.latents, input=Input.Connection
    )
    noise: Optional[LatentsField] = InputField(
        default=None, description=FieldDescriptions.noise, input=Input.Connection
    )
    # denoise_mask is used for image-to-image inpainting. Only the masked region is modified.
    denoise_mask: Optional[DenoiseMaskField] = InputField(
        default=None, description=FieldDescriptions.denoise_mask, input=Input.Connection
    )
    denoising_start: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.denoising_start)
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    add_noise: bool = InputField(default=True, description="Add noise based on denoising start.")
    transformer: TransformerField = InputField(
        description=FieldDescriptions.z_image_model, input=Input.Connection, title="Transformer"
    )
    positive_conditioning: ZImageConditioningField | list[ZImageConditioningField] = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: ZImageConditioningField | list[ZImageConditioningField] | None = InputField(
        default=None, description=FieldDescriptions.negative_cond, input=Input.Connection
    )
    # Z-Image-Turbo works best without CFG (guidance_scale=1.0)
    guidance_scale: float = InputField(
        default=1.0,
        ge=1.0,
        description="Guidance scale for classifier-free guidance. 1.0 = no CFG (recommended for Z-Image-Turbo). "
        "Values > 1.0 amplify guidance.",
        title="Guidance Scale",
    )
    width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")
    # Z-Image-Turbo uses 8 steps by default
    steps: int = InputField(default=8, gt=0, description="Number of denoising steps. 8 recommended for Z-Image-Turbo.")
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")
    # Z-Image Control support
    control: Optional[ZImageControlField] = InputField(
        default=None,
        description="Z-Image control conditioning for spatial control (Canny, HED, Depth, Pose, MLSD).",
        input=Input.Connection,
    )
    # VAE for encoding control images (required when using control)
    vae: Optional[VAEField] = InputField(
        default=None,
        description=FieldDescriptions.vae + " Required for control conditioning.",
        input=Input.Connection,
    )
    # Shift override for the sigma schedule. If None, shift is auto-calculated from image dimensions.
    shift: Optional[float] = InputField(
        default=None,
        ge=0.0,
        description="Override the timestep shift for the sigma schedule. "
        "Leave blank to auto-calculate based on image dimensions (recommended). "
        "This is the shift itself, not the mu it is derived from: the auto-calculated value runs "
        "from about 1.6 on small frames to 3.16 at one megapixel and above. Lower values keep more "
        "of the input image, higher values less.",
        title="Shift",
    )
    # Scheduler selection for the denoising process
    scheduler: ZIMAGE_SCHEDULER_NAME_VALUES = InputField(
        default="euler",
        description="Scheduler (sampler) for the denoising process. Euler is the default and recommended. "
        "Heun is 2nd-order (better quality, 2x slower). LCM works with Turbo only (not Base).",
        ui_choice_labels=ZIMAGE_SCHEDULER_LABELS,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_diffusion(context)
        latents = latents.detach().to("cpu")

        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)

    def _prep_inpaint_mask(self, context: InvocationContext, latents: torch.Tensor) -> torch.Tensor | None:
        """Prepare the inpaint mask."""
        if self.denoise_mask is None:
            return None
        mask = context.tensors.load(self.denoise_mask.mask_name)

        # Invert mask: 0.0 = regions to denoise, 1.0 = regions to preserve
        mask = 1.0 - mask

        _, _, latent_height, latent_width = latents.shape
        mask = tv_resize(
            img=mask,
            size=[latent_height, latent_width],
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
            antialias=False,
        )

        mask = mask.to(device=latents.device, dtype=latents.dtype)
        return mask

    def _load_text_conditioning(
        self,
        context: InvocationContext,
        cond_field: ZImageConditioningField | list[ZImageConditioningField],
        img_height: int,
        img_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[ZImageTextConditioning]:
        """Load Z-Image text conditioning with optional regional masks.

        Args:
            context: The invocation context.
            cond_field: Single conditioning field or list of fields.
            img_height: Height of the image token grid (H // patch_size).
            img_width: Width of the image token grid (W // patch_size).
            dtype: Target dtype.
            device: Target device.

        Returns:
            List of ZImageTextConditioning objects with embeddings and masks.
        """
        # Normalize to a list
        cond_list = [cond_field] if isinstance(cond_field, ZImageConditioningField) else cond_field

        text_conditionings: list[ZImageTextConditioning] = []
        for cond in cond_list:
            # Load the text embeddings
            cond_data = context.conditioning.load(cond.conditioning_name)
            assert len(cond_data.conditionings) == 1
            z_image_conditioning = cond_data.conditionings[0]
            assert isinstance(z_image_conditioning, ZImageConditioningInfo)
            z_image_conditioning = z_image_conditioning.to(dtype=dtype, device=device)
            prompt_embeds = z_image_conditioning.prompt_embeds

            # Load the mask, if provided
            mask: torch.Tensor | None = None
            if cond.mask is not None:
                mask = context.tensors.load(cond.mask.tensor_name)
                mask = mask.to(device=device)
                mask = ZImageRegionalPromptingExtension.preprocess_regional_prompt_mask(
                    mask, img_height, img_width, dtype, device
                )

            text_conditionings.append(ZImageTextConditioning(prompt_embeds=prompt_embeds, mask=mask))

        return text_conditionings

    def _get_noise(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        seed: int,
    ) -> torch.Tensor:
        """Generate initial noise tensor."""
        # Generate noise as float32 on CPU for maximum compatibility,
        # then cast to target dtype/device
        rand_device = "cpu"
        rand_dtype = torch.float32

        return torch.randn(
            batch_size,
            num_channels_latents,
            int(height) // LATENT_SCALE_FACTOR,
            int(width) // LATENT_SCALE_FACTOR,
            device=rand_device,
            dtype=rand_dtype,
            generator=torch.Generator(device=rand_device).manual_seed(seed),
        ).to(device=device, dtype=dtype)

    def _get_sigmas(self, shift: float, num_steps: int) -> list[float]:
        """`num_steps + 1` sigmas from 1 to 0, linearly time-shifted by `shift` (`time_shift`)."""
        sigmas = []
        for i in range(num_steps + 1):
            t = 1.0 - i / num_steps  # Goes from 1.0 to 0.0
            # The ends are fixed. Evaluated, t = 0 would divide by zero, and so would t = 1 at a shift of 0.
            sigmas.append(1.0 if t >= 1 else 0.0 if t <= 0 else time_shift(shift, t))
        return sigmas

    @staticmethod
    def _estimate_working_memory(
        image_seq_len: int,
        text_seq_len: int,
        num_loras: int,
        num_control_blocks: int = 0,
        regional_attention_bias_bytes: int = 0,
        has_attention_mask: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> int:
        """Estimate peak transformer working memory (bytes) so the model cache keeps that much VRAM free.

        The transformer attends over one unified sequence of image and caption tokens, each run padded to a
        multiple of ``SEQ_MULTI_OF``. On a fused SDPA kernel its activations scale linearly with that sequence.
        Measured as peak reserved memory in bf16 on CUDA with a 512-token caption, by
        `scripts/calibrate_z_image_working_memory.py`: 0.99 GiB at 1024px (4608 tokens), 1.96 GiB at 1440px (8640)
        and 3.85 GiB at 2048px (16896) -- 0.22 to 0.23 MB per token, unchanged between 2 and 4 blocks, because a
        no-grad forward frees each block's intermediates. 0.25 MB per token bounds every point on its own, with
        room for the ~5% more ROCm needed on FLUX.2. The fixed base covers what does not scale with the sequence
        -- fp8 weights cast per forward, and allocator slack across steps; the caller adds the dequantization copy
        of an int8, nvfp4 or GGUF build. Conditional and unconditional
        passes run one after the other, so only the longer caption counts.

        Where SDPA has no fused kernel for these shapes it materializes the score matrices, and those dominate:
        on ROCm under Windows, whose fused kernels are disabled, a 1024px generation spilled 7.7 GB into shared
        system memory while this node reserved only the default working memory. ``sdpa_score_matrix_bytes`` adds
        the term only where it is really built. An attention mask narrows which fused kernels a build can use, so
        the call is told whether one is passed: regional prompting's additive bias. The forward passes none at batch
        size 1 otherwise, and a ControlNet's blocks take exactly the arguments the base blocks take.

        A ControlNet first runs its own blocks over a sequence of the same length and keeps one hint per block,
        alive until the last layer it adds to. Measured on the hooked forward over the plain one at 1024px: 1.8
        hint-sized tensors (sequence x width x element size) for the 6-block 1.0 layout, 0.9 for the 3-block 2.1
        Lite and 11.6 for the 15-block 2.0 and 2.1, so the n + 4 charged bound every layout.
        """
        GB = 1024**3
        seq_len = _padded(image_seq_len) + _padded(text_seq_len)
        estimated = seq_len * Z_IMAGE_BYTES_PER_TOKEN + int(1.0 * GB)
        estimated += regional_attention_bias_bytes
        estimated += sdpa_score_matrix_bytes(
            device=device if device is not None else TorchDevice.choose_torch_device(),
            dtype=dtype,
            num_heads=Z_IMAGE_ATTENTION_HEADS,
            head_dim=Z_IMAGE_HIDDEN_SIZE // Z_IMAGE_ATTENTION_HEADS,
            seq_len=seq_len,
            has_attn_mask=has_attention_mask,
            # The attention goes through diffusers' `dispatch_attention_fn`, which can route around torch's SDPA.
            via_diffusers_dispatch=True,
        )
        if num_control_blocks:
            estimated += (num_control_blocks + 4) * seq_len * Z_IMAGE_HIDDEN_SIZE * dtype.itemsize
        estimated += int(0.5 * num_loras * GB)
        return estimated

    @staticmethod
    def _regional_attention_bias_bytes(
        regional_attn_mask: torch.Tensor | None, image_seq_len: int, dtype: torch.dtype
    ) -> int:
        """What `regional_forward` allocates per positive pass for regional prompting: a boolean copy of the mask,
        the additive bias over the padded unified sequence, and the image-to-image block `torch.where` fills it
        from. The mask itself is already on the device when the estimate is taken. Measured at 1024px on CUDA: 68 MiB
        over the plain forward, against 93 MiB charged."""
        if regional_attn_mask is None:
            return 0
        mask_side = regional_attn_mask.shape[0]
        unified_seq_len = _padded(image_seq_len) + _padded(mask_side - image_seq_len)
        return (
            mask_side * mask_side + (unified_seq_len * unified_seq_len + image_seq_len * image_seq_len) * dtype.itemsize
        )

    def _run_diffusion(self, context: InvocationContext) -> torch.Tensor:
        device = TorchDevice.choose_torch_device()
        inference_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        transformer_info = context.models.load(self.transformer.transformer)
        control_info = self._load_control(context)

        # Calculate image token grid dimensions
        patch_size = 2  # Z-Image uses patch_size=2
        latent_height = self.height // LATENT_SCALE_FACTOR
        latent_width = self.width // LATENT_SCALE_FACTOR
        img_token_height = latent_height // patch_size
        img_token_width = latent_width // patch_size
        img_seq_len = img_token_height * img_token_width

        # Load positive conditioning with regional masks
        pos_text_conditionings = self._load_text_conditioning(
            context=context,
            cond_field=self.positive_conditioning,
            img_height=img_token_height,
            img_width=img_token_width,
            dtype=inference_dtype,
            device=device,
        )

        # Create regional prompting extension
        regional_extension = ZImageRegionalPromptingExtension.from_text_conditionings(
            text_conditionings=pos_text_conditionings,
            img_seq_len=img_seq_len,
        )

        # Get the concatenated prompt embeddings for the transformer
        pos_prompt_embeds = regional_extension.regional_text_conditioning.prompt_embeds

        # Load negative conditioning if provided and guidance_scale != 1.0
        # CFG formula: pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        # At cfg_scale=1.0: pred = pred_cond (no effect, skip uncond computation)
        # This matches FLUX's convention where 1.0 means "no CFG"
        neg_prompt_embeds: torch.Tensor | None = None
        do_classifier_free_guidance = (
            not math.isclose(self.guidance_scale, 1.0) and self.negative_conditioning is not None
        )
        if do_classifier_free_guidance:
            assert self.negative_conditioning is not None
            # Load all negative conditionings and concatenate embeddings
            # Note: We ignore masks for negative conditioning as regional negative prompting is not fully supported
            neg_text_conditionings = self._load_text_conditioning(
                context=context,
                cond_field=self.negative_conditioning,
                img_height=img_token_height,
                img_width=img_token_width,
                dtype=inference_dtype,
                device=device,
            )
            # Concatenate all negative embeddings
            neg_prompt_embeds = torch.cat([tc.prompt_embeds for tc in neg_text_conditionings], dim=0)

        # Calculate shift based on image sequence length, or use override
        if self.shift is not None:
            shift = self.shift
        else:
            shift = Z_IMAGE_SHIFT_FIT.shift(img_seq_len)

        # Generate sigma schedule with time shift
        sigmas = self._get_sigmas(shift, self.steps)

        # Apply denoising_start and denoising_end clipping
        if self.denoising_start > 0 or self.denoising_end < 1:
            # Calculate start and end indices based on denoising range
            total_sigmas = len(sigmas)
            start_idx = int(self.denoising_start * (total_sigmas - 1))
            end_idx = int(self.denoising_end * (total_sigmas - 1)) + 1
            sigmas = sigmas[start_idx:end_idx]

        is_heun = self.scheduler == "heun"
        if is_heun:
            # Heun divides by every sigma it evaluates the model at, so an interval of zero length at sigma 0 is
            # 0/0 and turns the latents into NaN; an explicit shift of 0 schedules [1, 0, ..., 0]. Such an interval
            # has nothing to integrate (Euler steps through it unchanged), so Heun runs only the others and counts
            # its progress over them. With none left, the pass returns the blended latents, as Euler would.
            sigmas = [
                sigma for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:], strict=True) if sigma != sigma_next
            ] + sigmas[-1:]

        total_steps = len(sigmas) - 1

        # Load input latents if provided (image-to-image)
        init_latents = context.tensors.load(self.latents.latents_name) if self.latents else None
        if init_latents is not None:
            init_latents = init_latents.to(device=device, dtype=inference_dtype)

        # Generate initial noise.
        # If noise will never be consumed, avoid validating/loading it.
        should_ignore_noise = init_latents is not None and not self.add_noise and self.denoise_mask is None
        noise: torch.Tensor | None
        if should_ignore_noise:
            noise = None
        else:
            noise = self._prepare_noise_tensor(context, inference_dtype, device)

        # Prepare input latent image
        if init_latents is not None:
            if self.add_noise:
                assert noise is not None
                # Noise the init latents using the first sigma from the clipped
                # InvokeAI schedule. Euler and Heun start from this sigma; LCM
                # creates its own schedule, whose first sigma can differ.
                s_0 = sigmas[0]
                latents = s_0 * noise + (1.0 - s_0) * init_latents
            else:
                latents = init_latents
        else:
            if self.denoising_start > 1e-5:
                raise ValueError("denoising_start should be 0 when initial latents are not provided.")
            assert noise is not None
            latents = noise

        # Short-circuit if no denoising steps
        if total_steps <= 0:
            return latents

        # Prepare inpaint extension
        inpaint_mask = self._prep_inpaint_mask(context, latents)
        inpaint_extension: RectifiedFlowInpaintExtension | None = None
        if inpaint_mask is not None:
            if init_latents is None:
                raise ValueError("Initial latents are required when using an inpaint mask (image-to-image inpainting)")
            assert noise is not None
            inpaint_extension = RectifiedFlowInpaintExtension(
                init_latents=init_latents,
                inpaint_mask=inpaint_mask,
                noise=noise,
            )

        step_callback = self._build_step_callback(context)

        # Initialize the diffusers scheduler if not using built-in Euler
        scheduler: SchedulerMixin | None = None
        use_scheduler = self.scheduler != "euler"

        if use_scheduler:
            scheduler_class = ZIMAGE_SCHEDULER_MAP[self.scheduler]
            scheduler = scheduler_class(
                num_train_timesteps=1000,
                shift=1.0,
            )
            if is_heun:
                # Heun runs the shifted, clipped schedule above, so it starts at the sigma the img2img
                # preblend used and stops at the requested end.
                set_heun_timesteps_from_sigmas(scheduler, sigmas[:-1], sigmas[-1], device)
            else:
                # LCM owns its schedule: clipping only sets its step count. Its first sigma can
                # differ from sigmas[0] used in the img2img preblend above.
                scheduler.set_timesteps(num_inference_steps=total_steps, device=device)

            # For Heun scheduler, the number of actual steps may differ
            num_scheduler_steps = len(scheduler.timesteps)
        else:
            num_scheduler_steps = total_steps

        with ExitStack() as exit_stack:
            # Get transformer config to determine if it's quantized
            transformer_config = context.models.get_config(self.transformer.transformer)

            if transformer_config.format not in (
                ModelFormat.Diffusers,
                ModelFormat.Checkpoint,
                ModelFormat.GGUFQuantized,
                ModelFormat.SDNQQuantized,
            ):
                raise ValueError(f"Unsupported Z-Image model format: {transformer_config.format}")

            # Encoded ahead of the transformer's lock, so the VAE does not have to fit beside it.
            control_context: torch.Tensor | None = None
            num_control_blocks = 0
            if control_info is not None:
                assert isinstance(control_info.model, ZImageFunControl)
                control_context = self._build_control_context(context, control_info.model).to(
                    device=device, dtype=inference_dtype
                )
                num_control_blocks = len(control_info.model.control_layers)

            # Ask the model cache to keep the forward's working memory free; it offloads as much of the
            # transformer as that takes. Without it the cache reserves only `device_working_mem_gb`, which a
            # build that materializes the score matrices (ROCm on Windows, MPS) overruns several times at 1024px.
            regional_attn_mask = regional_extension.regional_attn_mask
            working_mem_bytes = self._estimate_working_memory(
                image_seq_len=img_seq_len,
                text_seq_len=max(
                    pos_prompt_embeds.shape[0], neg_prompt_embeds.shape[0] if neg_prompt_embeds is not None else 0
                ),
                num_loras=len(self.transformer.loras),
                num_control_blocks=num_control_blocks,
                regional_attention_bias_bytes=self._regional_attention_bias_bytes(
                    regional_attn_mask, img_seq_len, inference_dtype
                ),
                has_attention_mask=regional_attn_mask is not None,
                device=device,
                dtype=inference_dtype,
            )
            # An int8_tensorwise, nvfp4 or GGUF build materializes each quantized linear's weight per forward, alive
            # alongside the activations above and invisible to them. Read from the unlocked model; zero for
            # every other build.
            working_mem_bytes += peak_dequant_transient_bytes(transformer_info.model, inference_dtype)

            (cached_weights, transformer) = exit_stack.enter_context(
                transformer_info.model_on_device(working_mem_bytes=working_mem_bytes)
            )

            # Whether LoRA has to go on as a sidecar. Asked of the loaded module tree, not the
            # format: a `checkpoint` Z-Image may be an int8_tensorwise or nvfp4 build, whose quantized
            # Linears carry their weights as buffers rather than parameters, so the fallbacks inside
            # the patcher -- which iterate `module.parameters()` -- find nothing and choose direct
            # patching on a module that has no patchable weights.
            model_is_quantized = requires_sidecar_patching(transformer, transformer_config.format)

            control_hooks: FunControlHooks | None = None
            if control_info is not None:
                assert self.control is not None and control_context is not None
                # Locked with the same working memory: the transformer can no longer be offloaded, so a control
                # placed with only the default reservation would fill the room kept for the forward. The loader cast
                # it to the device's bfloat16-safe dtype, `inference_dtype` here; it casts what crosses into its chain
                # and the hints back (`ZImageFunControl._block_runner` / `add_hint`), so nothing cache-managed is cast.
                (_, control) = exit_stack.enter_context(
                    control_info.model_on_device(working_mem_bytes=working_mem_bytes)
                )
                assert isinstance(control, ZImageFunControl)
                control_hooks = exit_stack.enter_context(
                    install_fun_control(transformer, control, control_context, scale=self.control.control_context_scale)
                )

            # Apply LoRA models to the transformer.
            # Note: We apply the LoRA after the transformer has been moved to its target device for faster patching.
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=transformer,
                    patches=self._lora_iterator(context),
                    prefix=Z_IMAGE_LORA_TRANSFORMER_PREFIX,
                    dtype=inference_dtype,
                    cached_weights=cached_weights,
                    force_sidecar_patching=model_is_quantized,
                )
            )

            # Apply regional prompting patch if we have regional masks
            exit_stack.enter_context(
                patch_transformer_for_regional_prompting(
                    transformer=transformer,
                    regional_attn_mask=regional_extension.regional_attn_mask,
                    img_seq_len=img_seq_len,
                    positive_cap_feats=pos_prompt_embeds,
                )
            )

            exit_stack.enter_context(sage_attention_scope())

            # Denoising loop - supports both built-in Euler and diffusers schedulers
            # Track user-facing step for progress (accounts for Heun's double steps)
            user_step = 0

            if use_scheduler and scheduler is not None:
                # LCM re-noises the sample on every step. Without an explicit generator that noise
                # comes from the global RNG and the same seed gives a different image each run.
                # XOR keeps the step noise decorrelated from the initial noise (as in ERNIE-Image).
                step_generator = torch.Generator(device="cpu").manual_seed(self.seed ^ 0xFFFFFFFF)
                # Use diffusers scheduler for stepping
                # Use tqdm with total_steps (user-facing steps) not num_scheduler_steps (internal steps)
                # This ensures progress bar shows 1/8, 2/8, etc. even when scheduler uses more internal steps
                pbar = tqdm(total=total_steps, desc=f"Denoising{TorchDevice.get_session_device_label()}")
                for step_index in range(num_scheduler_steps):
                    sched_timestep = scheduler.timesteps[step_index]
                    # Convert scheduler timestep (0-1000) to normalized sigma (0-1)
                    sigma_curr = sched_timestep.item() / scheduler.config.num_train_timesteps

                    # For Heun scheduler, track if we're in first or second order step
                    in_first_order = scheduler.state_in_first_order if is_heun else True

                    # Timestep tensor for Z-Image model
                    # The model expects t=0 at start (noise) and t=1 at end (clean)
                    model_t = 1.0 - sigma_curr
                    timestep = torch.tensor([model_t], device=device, dtype=inference_dtype).expand(latents.shape[0])

                    # Run transformer for positive prediction
                    # `transformer.dtype` is the float8 *storage* dtype once FP8 storage is on, and
                    # torch has no arithmetic kernels for it (see `get_model_compute_dtype`).
                    latent_model_input = latents.to(get_model_compute_dtype(transformer))
                    latent_model_input = latent_model_input.unsqueeze(2)  # Add frame dimension
                    latent_model_input_list = list(latent_model_input.unbind(dim=0))

                    # Heun's two forwards of a step share its user step, and so its control gate.
                    self._gate_control(control_hooks, total_steps, user_step)

                    model_out_list = transformer(
                        x=latent_model_input_list,
                        t=timestep,
                        cap_feats=[pos_prompt_embeds],
                    )[0]

                    noise_pred_cond = torch.stack([t.float() for t in model_out_list], dim=0)
                    noise_pred_cond = noise_pred_cond.squeeze(2)
                    noise_pred_cond = -noise_pred_cond  # Z-Image uses v-prediction with negation

                    # Apply CFG if enabled
                    if do_classifier_free_guidance and neg_prompt_embeds is not None:
                        model_out_list_uncond = transformer(
                            x=latent_model_input_list,
                            t=timestep,
                            cap_feats=[neg_prompt_embeds],
                        )[0]

                        noise_pred_uncond = torch.stack([t.float() for t in model_out_list_uncond], dim=0)
                        noise_pred_uncond = noise_pred_uncond.squeeze(2)
                        noise_pred_uncond = -noise_pred_uncond
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_cond

                    # Use scheduler.step() for the update
                    step_output = scheduler.step(
                        model_output=noise_pred, timestep=sched_timestep, sample=latents, generator=step_generator
                    )
                    latents = step_output.prev_sample

                    # Get sigma_prev for inpainting (next sigma value)
                    if step_index + 1 < len(scheduler.sigmas):
                        sigma_prev = scheduler.sigmas[step_index + 1].item()
                    else:
                        sigma_prev = 0.0

                    if inpaint_extension is not None:
                        latents = inpaint_extension.merge_intermediate_latents_with_init_latents(latents, sigma_prev)

                    # For Heun, a user step completes with its second-order evaluation. The final
                    # interval has none: Diffusers ends it with a first-order step to the terminal sigma.
                    if is_heun:
                        if not in_first_order or step_index == num_scheduler_steps - 1:
                            user_step += 1
                            # Only call step_callback if we haven't exceeded total_steps
                            if user_step <= total_steps:
                                pbar.update(1)
                                step_callback(
                                    PipelineIntermediateState(
                                        step=user_step,
                                        # Progress over completed user steps, not the solver's order.
                                        order=1,
                                        total_steps=total_steps,
                                        timestep=int(sigma_curr * 1000),
                                        latents=latents,
                                    ),
                                )
                    else:
                        # For first-order schedulers (Euler, LCM)
                        user_step += 1
                        if user_step <= total_steps:
                            pbar.update(1)
                            step_callback(
                                PipelineIntermediateState(
                                    step=user_step,
                                    order=1,
                                    total_steps=total_steps,
                                    timestep=int(sigma_curr * 1000),
                                    latents=latents,
                                ),
                            )
                pbar.close()
            else:
                # Original Euler implementation (default, optimized for Z-Image)
                for step_idx in tqdm(range(total_steps), desc=f"Denoising{TorchDevice.get_session_device_label()}"):
                    sigma_curr = sigmas[step_idx]
                    sigma_prev = sigmas[step_idx + 1]

                    # Timestep tensor for Z-Image model
                    # The model expects t=0 at start (noise) and t=1 at end (clean)
                    # Sigma goes from 1 (noise) to 0 (clean), so model_t = 1 - sigma
                    model_t = 1.0 - sigma_curr
                    timestep = torch.tensor([model_t], device=device, dtype=inference_dtype).expand(latents.shape[0])

                    # Run transformer for positive prediction
                    # Z-Image transformer expects: x as list of [C, 1, H, W] tensors, t, cap_feats as list
                    # Prepare latent input: [B, C, H, W] -> [B, C, 1, H, W] -> list of [C, 1, H, W]
                    # See above: never build tensors from the float8 storage dtype.
                    latent_model_input = latents.to(get_model_compute_dtype(transformer))
                    latent_model_input = latent_model_input.unsqueeze(2)  # Add frame dimension
                    latent_model_input_list = list(latent_model_input.unbind(dim=0))

                    self._gate_control(control_hooks, total_steps, step_idx)

                    model_out_list = transformer(
                        x=latent_model_input_list,
                        t=timestep,
                        cap_feats=[pos_prompt_embeds],
                    )[0]  # The list of output tensors

                    noise_pred_cond = torch.stack([t.float() for t in model_out_list], dim=0)
                    noise_pred_cond = noise_pred_cond.squeeze(2)  # Remove frame dimension
                    noise_pred_cond = -noise_pred_cond  # Z-Image uses v-prediction with negation

                    # Apply CFG if enabled
                    if do_classifier_free_guidance and neg_prompt_embeds is not None:
                        model_out_list_uncond = transformer(
                            x=latent_model_input_list,
                            t=timestep,
                            cap_feats=[neg_prompt_embeds],
                        )[0]

                        noise_pred_uncond = torch.stack([t.float() for t in model_out_list_uncond], dim=0)
                        noise_pred_uncond = noise_pred_uncond.squeeze(2)
                        noise_pred_uncond = -noise_pred_uncond
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_cond

                    # Euler step
                    latents_dtype = latents.dtype
                    latents = latents.to(dtype=torch.float32)
                    latents = latents + (sigma_prev - sigma_curr) * noise_pred
                    latents = latents.to(dtype=latents_dtype)

                    if inpaint_extension is not None:
                        latents = inpaint_extension.merge_intermediate_latents_with_init_latents(latents, sigma_prev)

                    step_callback(
                        PipelineIntermediateState(
                            step=step_idx + 1,
                            order=1,
                            total_steps=total_steps,
                            timestep=int(sigma_curr * 1000),
                            latents=latents,
                        ),
                    )

        return latents

    def _load_control(self, context: InvocationContext) -> LoadedModel | None:
        """The control model in RAM, ahead of the transformer's lock so the reservation can size its hints."""
        if self.control is None:
            return None
        if self.vae is None:
            raise ValueError(
                "Z-Image ControlNet needs the VAE that encodes its control image: connect one to the 'vae' input."
            )
        control_info = context.models.load(self.control.control_model)
        if not isinstance(control_info.model, ZImageFunControl):
            raise TypeError(
                f"Expected a Z-Image ControlNet for '{self.control.control_model.name}', "
                f"got {type(control_info.model).__name__}."
            )
        return control_info

    def _build_control_context(self, context: InvocationContext, control: ZImageFunControl) -> torch.Tensor:
        """The control image at the generation size as the tokens ``control`` embeds.

        VideoX-Fun's context for control alone: the control image's latents, a zero keep mask and zero masked-image
        latents, packed 2x2 as the transformer packs its latents. Encoded to the mode of the VAE's distribution as
        VideoX-Fun encodes it, not to a sample of it.
        """
        assert self.control is not None and self.vae is not None
        field = self.control
        vae_info = context.models.load(self.vae.vae)
        tiled = field.tiled or context.config.get().force_tiled_decode

        def encode(pixels: torch.Tensor) -> torch.Tensor:
            return ZImageImageToLatentsInvocation.vae_encode(
                vae_info=vae_info, image_tensor=pixels, tiled=tiled, tile_size=field.tile_size, deterministic=True
            )

        image = context.images.get_pil(field.image_name).convert("RGB")
        image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        pixels = image_resized_to_grid_as_tensor(image).unsqueeze(0)
        return control.pack(build_fun_control_context(encode=encode, control_image=pixels))

    def _gate_control(self, hooks: FunControlHooks | None, total_steps: int, step_index: int) -> None:
        # One gate per user step: both guidance passes run with the control or neither does.
        if hooks is None:
            return
        assert self.control is not None
        hooks.active = should_apply(
            step_index, total_steps, self.control.begin_step_percent, self.control.end_step_percent
        )

    def _prepare_noise_tensor(
        self, context: InvocationContext, inference_dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if self.noise is not None:
            noise = context.tensors.load(self.noise.latents_name).to(device=device, dtype=inference_dtype)
            validate_noise_tensor_shape(noise, "Z-Image", self.width, self.height)
            return noise

        return self._get_noise(
            batch_size=1,
            num_channels_latents=16,
            height=self.height,
            width=self.width,
            dtype=inference_dtype,
            device=device,
            seed=self.seed,
        )

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, BaseModelType.ZImage)

        return step_callback

    def _lora_iterator(self, context: InvocationContext) -> Iterator[PatchSpec]:
        """Iterate over LoRA models to apply to the transformer."""
        for lora in self.transformer.loras:
            lora_info = context.models.load(lora.lora)
            if not isinstance(lora_info.model, ModelPatchRaw):
                raise TypeError(
                    f"Expected ModelPatchRaw for LoRA '{lora.lora.key}', got {type(lora_info.model).__name__}. "
                    "The LoRA model may be corrupted or incompatible."
                )
            yield (lora_info.model, lora.weight, lora_info.model_in_ram())
