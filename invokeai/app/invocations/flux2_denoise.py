"""Flux2 Klein Denoise Invocation.

Run denoising process with a FLUX.2 Klein transformer model.
Uses Qwen3 conditioning instead of CLIP+T5.
"""

from contextlib import ExitStack
from typing import Callable, Iterator, Optional, Tuple

import torch
import torchvision.transforms as tv_transforms
from torchvision.transforms.functional import resize as tv_resize

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    FluxConditioningField,
    Input,
    InputField,
    LatentsField,
)
from invokeai.app.invocations.model import TransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.sampling_utils import (
    clip_timestep_schedule_fractional,
    get_schedule,
)
from invokeai.backend.flux.schedulers import FLUX_SCHEDULER_LABELS, FLUX_SCHEDULER_MAP, FLUX_SCHEDULER_NAME_VALUES
from invokeai.backend.flux2.denoise import denoise
from invokeai.backend.flux2.sampling_utils import (
    generate_img_ids_flux2,
    get_noise_flux2,
    pack_flux2,
    unpack_flux2,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import FLUXConditioningInfo
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "flux2_denoise",
    title="FLUX2 Denoise",
    tags=["image", "flux", "flux2", "klein", "denoise"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux2DenoiseInvocation(BaseInvocation):
    """Run denoising process with a FLUX.2 Klein transformer model.

    This node is designed for FLUX.2 Klein models which use Qwen3 as the text encoder.
    It does not support ControlNet, IP-Adapters, or regional prompting.
    """

    latents: Optional[LatentsField] = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    denoise_mask: Optional[DenoiseMaskField] = InputField(
        default=None,
        description=FieldDescriptions.denoise_mask,
        input=Input.Connection,
    )
    denoising_start: float = InputField(
        default=0.0,
        ge=0,
        le=1,
        description=FieldDescriptions.denoising_start,
    )
    denoising_end: float = InputField(
        default=1.0,
        ge=0,
        le=1,
        description=FieldDescriptions.denoising_end,
    )
    add_noise: bool = InputField(default=True, description="Add noise based on denoising start.")
    transformer: TransformerField = InputField(
        description=FieldDescriptions.flux_model,
        input=Input.Connection,
        title="Transformer",
    )
    positive_text_conditioning: FluxConditioningField = InputField(
        description=FieldDescriptions.positive_cond,
        input=Input.Connection,
    )
    negative_text_conditioning: Optional[FluxConditioningField] = InputField(
        default=None,
        description="Negative conditioning tensor. Can be None if cfg_scale is 1.0.",
        input=Input.Connection,
    )
    cfg_scale: float = InputField(
        default=1.0,
        description=FieldDescriptions.cfg_scale,
        title="CFG Scale",
    )
    width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")
    num_steps: int = InputField(
        default=28,
        description="Number of diffusion steps. Recommended: 28 for Klein.",
    )
    scheduler: FLUX_SCHEDULER_NAME_VALUES = InputField(
        default="euler",
        description="Scheduler (sampler) for the denoising process. 'euler' is fast and standard. "
        "'heun' is 2nd-order (better quality, 2x slower). 'lcm' is optimized for few steps.",
        ui_choice_labels=FLUX_SCHEDULER_LABELS,
    )
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_diffusion(context)
        latents = latents.detach().to("cpu")

        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)

    def _run_diffusion(self, context: InvocationContext) -> torch.Tensor:
        inference_dtype = torch.bfloat16
        device = TorchDevice.choose_torch_device()

        # Load the input latents, if provided
        init_latents = context.tensors.load(self.latents.latents_name) if self.latents else None
        if init_latents is not None:
            init_latents = init_latents.to(device=device, dtype=inference_dtype)

        # Prepare input noise (FLUX.2 uses 32 channels)
        noise = get_noise_flux2(
            num_samples=1,
            height=self.height,
            width=self.width,
            device=device,
            dtype=inference_dtype,
            seed=self.seed,
        )
        b, _c, latent_h, latent_w = noise.shape
        packed_h = latent_h // 2
        packed_w = latent_w // 2

        # Load the conditioning data
        pos_cond_data = context.conditioning.load(self.positive_text_conditioning.conditioning_name)
        assert len(pos_cond_data.conditionings) == 1
        pos_flux_conditioning = pos_cond_data.conditionings[0]
        assert isinstance(pos_flux_conditioning, FLUXConditioningInfo)
        pos_flux_conditioning = pos_flux_conditioning.to(dtype=inference_dtype, device=device)

        # Qwen3 stacked embeddings (stored in t5_embeds field for compatibility)
        txt = pos_flux_conditioning.t5_embeds
        # Generate text position IDs (FLUX.2 uses 4D coordinates: T, H, W, L)
        txt_ids = torch.zeros(1, txt.shape[1], 4, device=device, dtype=inference_dtype)

        # Load negative conditioning if provided
        neg_txt = None
        neg_txt_ids = None
        if self.negative_text_conditioning is not None:
            neg_cond_data = context.conditioning.load(self.negative_text_conditioning.conditioning_name)
            assert len(neg_cond_data.conditionings) == 1
            neg_flux_conditioning = neg_cond_data.conditionings[0]
            assert isinstance(neg_flux_conditioning, FLUXConditioningInfo)
            neg_flux_conditioning = neg_flux_conditioning.to(dtype=inference_dtype, device=device)
            neg_txt = neg_flux_conditioning.t5_embeds
            neg_txt_ids = torch.zeros(1, neg_txt.shape[1], 4, device=device, dtype=inference_dtype)

        # Validate transformer config
        transformer_config = context.models.get_config(self.transformer.transformer)
        assert transformer_config.base == BaseModelType.Flux2 and transformer_config.type == ModelType.Main

        # Calculate the timestep schedule
        # Klein uses shifted schedule like Dev
        timesteps = get_schedule(
            num_steps=self.num_steps,
            image_seq_len=packed_h * packed_w,
            shift=True,  # Klein uses shifted schedule
        )

        # Clip the timesteps schedule based on denoising_start and denoising_end
        timesteps = clip_timestep_schedule_fractional(timesteps, self.denoising_start, self.denoising_end)

        # Prepare input latent image
        if init_latents is not None:
            if self.add_noise:
                t_0 = timesteps[0]
                x = t_0 * noise + (1.0 - t_0) * init_latents
            else:
                x = init_latents
        else:
            if self.denoising_start > 1e-5:
                raise ValueError("denoising_start should be 0 when initial latents are not provided.")
            x = noise

        # If len(timesteps) == 1, then short-circuit
        if len(timesteps) <= 1:
            return x

        # Generate image position IDs (FLUX.2 uses 4D coordinates)
        img_ids = generate_img_ids_flux2(h=latent_h, w=latent_w, batch_size=b, device=device, dtype=inference_dtype)

        # Prepare inpaint mask
        inpaint_mask = self._prep_inpaint_mask(context, x)

        # Pack all latent tensors
        init_latents_packed = pack_flux2(init_latents) if init_latents is not None else None
        inpaint_mask_packed = pack_flux2(inpaint_mask) if inpaint_mask is not None else None
        noise_packed = pack_flux2(noise)
        x = pack_flux2(x)

        # Verify packed dimensions
        assert packed_h * packed_w == x.shape[1]

        # Prepare inpaint extension
        inpaint_extension: Optional[RectifiedFlowInpaintExtension] = None
        if inpaint_mask_packed is not None:
            assert init_latents_packed is not None
            inpaint_extension = RectifiedFlowInpaintExtension(
                init_latents=init_latents_packed,
                inpaint_mask=inpaint_mask_packed,
                noise=noise_packed,
            )

        # Prepare CFG scale list
        num_steps = len(timesteps) - 1
        cfg_scale_list = [self.cfg_scale] * num_steps

        # Create scheduler if not using default euler
        scheduler = None
        if self.scheduler in FLUX_SCHEDULER_MAP:
            scheduler_class = FLUX_SCHEDULER_MAP[self.scheduler]
            scheduler = scheduler_class(num_train_timesteps=1000)

        with ExitStack() as exit_stack:
            # Load the transformer model
            (cached_weights, transformer) = exit_stack.enter_context(
                context.models.load(self.transformer.transformer).model_on_device()
            )
            config = transformer_config

            # Determine if the model is quantized
            if config.format in [ModelFormat.Diffusers]:
                model_is_quantized = False
            elif config.format in [
                ModelFormat.BnbQuantizedLlmInt8b,
                ModelFormat.BnbQuantizednf4b,
                ModelFormat.GGUFQuantized,
            ]:
                model_is_quantized = True
            else:
                model_is_quantized = False

            # Apply LoRA models to the transformer
            exit_stack.enter_context(
                LayerPatcher.apply_smart_model_patches(
                    model=transformer,
                    patches=self._lora_iterator(context),
                    prefix=FLUX_LORA_TRANSFORMER_PREFIX,
                    dtype=inference_dtype,
                    cached_weights=cached_weights,
                    force_sidecar_patching=model_is_quantized,
                )
            )

            x = denoise(
                model=transformer,
                img=x,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                step_callback=self._build_step_callback(context),
                cfg_scale=cfg_scale_list,
                neg_txt=neg_txt,
                neg_txt_ids=neg_txt_ids,
                scheduler=scheduler,
            )

            # Apply inpainting if enabled
            if inpaint_extension is not None:
                x = inpaint_extension.merge_intermediate_latents_with_init_latents(x, 0.0)

        x = unpack_flux2(x.float(), self.height, self.width)
        return x

    def _prep_inpaint_mask(self, context: InvocationContext, latents: torch.Tensor) -> Optional[torch.Tensor]:
        """Prepare the inpaint mask."""
        if self.denoise_mask is None:
            return None

        mask = context.tensors.load(self.denoise_mask.mask_name)
        mask = 1.0 - mask

        _, _, latent_height, latent_width = latents.shape
        mask = tv_resize(
            img=mask,
            size=[latent_height, latent_width],
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
            antialias=False,
        )

        mask = mask.to(device=latents.device, dtype=latents.dtype)
        return mask.expand_as(latents)

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        """Iterate over LoRA models to apply."""
        for lora in self.transformer.loras:
            lora_info = context.models.load(lora.lora)
            assert isinstance(lora_info.model, ModelPatchRaw)
            yield (lora_info.model, lora.weight)
            del lora_info

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        """Build a callback for step progress updates."""

        def step_callback(state: PipelineIntermediateState) -> None:
            latents = state.latents.float()
            state.latents = unpack_flux2(latents, self.height, self.width).squeeze()
            context.util.flux2_step_callback(state)

        return step_callback
