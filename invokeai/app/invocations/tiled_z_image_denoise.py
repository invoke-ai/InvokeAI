"""Tiled Multi-Diffusion denoising for Z-Image transformer models.

Analogous to TiledFluxDenoiseLatents but for Z-Image's transformer architecture.
Primarily intended for tiled upscaling workflows.
"""

import copy
import math
from contextlib import ExitStack
from typing import Callable, Iterator, Optional, Tuple

import einops
import torch
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    ZImageConditioningField,
)
from invokeai.app.invocations.model import TransformerField, VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.invocations.z_image_control import ZImageControlField
from invokeai.app.invocations.z_image_image_to_latents import ZImageImageToLatentsInvocation
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.schedulers import ZIMAGE_SCHEDULER_LABELS, ZIMAGE_SCHEDULER_NAME_VALUES
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.z_image_lora_constants import Z_IMAGE_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    PipelineIntermediateState,
    image_resized_to_grid_as_tensor,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ZImageConditioningInfo
from invokeai.backend.tiles.tiles import calc_tiles_min_overlap
from invokeai.backend.tiles.utils import TBLR
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.z_image.extensions.regional_prompting_extension import ZImageRegionalPromptingExtension
from invokeai.backend.z_image.text_conditioning import ZImageTextConditioning
from invokeai.backend.z_image.z_image_control_adapter import ZImageControlAdapter
from invokeai.backend.z_image.z_image_controlnet_extension import (
    ZImageControlNetExtension,
    z_image_forward_with_control,
)
from invokeai.backend.z_image.z_image_transformer_patch import patch_transformer_for_regional_prompting


@invocation(
    "tiled_z_image_denoise",
    title="Tiled Multi-Diffusion Denoise - Z-Image",
    tags=["upscale", "denoise", "z-image"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class TiledZImageDenoiseLatents(BaseInvocation):
    """Tiled Multi-Diffusion denoising for Z-Image models.

    This node handles automatically tiling the input latents and is primarily intended for
    tiled upscaling workflows with Z-Image transformer models.
    """

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    positive_conditioning: ZImageConditioningField | list[ZImageConditioningField] = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: ZImageConditioningField | list[ZImageConditioningField] | None = InputField(
        default=None, description=FieldDescriptions.negative_cond, input=Input.Connection
    )
    transformer: TransformerField = InputField(
        description=FieldDescriptions.z_image_model, input=Input.Connection, title="Transformer"
    )
    control: Optional[ZImageControlField] = InputField(
        default=None,
        description="Z-Image control conditioning for spatial control.",
        input=Input.Connection,
    )
    vae: Optional[VAEField] = InputField(
        default=None,
        description=FieldDescriptions.vae + " Required for control conditioning.",
        input=Input.Connection,
    )
    width: int = InputField(default=1024, multiple_of=16, description="Width of the full image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the full image.")
    tile_height: int = InputField(
        default=1024, gt=0, multiple_of=LATENT_SCALE_FACTOR, description="Height of tiles in image space."
    )
    tile_width: int = InputField(
        default=1024, gt=0, multiple_of=LATENT_SCALE_FACTOR, description="Width of tiles in image space."
    )
    tile_overlap: int = InputField(
        default=128,
        gt=0,
        multiple_of=LATENT_SCALE_FACTOR,
        description="Overlap between adjacent tiles in image space.",
    )
    steps: int = InputField(default=8, gt=0, description="Number of denoising steps.")
    guidance_scale: float = InputField(
        default=1.0,
        ge=1.0,
        description="Guidance scale for classifier-free guidance. 1.0 = no CFG.",
        title="Guidance Scale",
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
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")
    scheduler: ZIMAGE_SCHEDULER_NAME_VALUES = InputField(
        default="euler",
        description="Scheduler (sampler) for the denoising process.",
        ui_choice_labels=ZIMAGE_SCHEDULER_LABELS,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_tiled_diffusion(context)
        latents = latents.detach().to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)

    def _calculate_shift(self, image_seq_len: int) -> float:
        """Calculate timestep shift based on image sequence length."""
        base_image_seq_len = 256
        max_image_seq_len = 4096
        base_shift = 0.5
        max_shift = 1.15
        m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
        b = base_shift - m * base_image_seq_len
        return image_seq_len * m + b

    def _get_sigmas(self, mu: float, num_steps: int) -> list[float]:
        """Generate sigma schedule with time shift."""

        def time_shift(mu: float, sigma: float, t: float) -> float:
            if t <= 0:
                return 0.0
            if t >= 1:
                return 1.0
            return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

        sigmas = []
        for i in range(num_steps + 1):
            t = 1.0 - i / num_steps
            sigma = time_shift(mu, 1.0, t)
            sigmas.append(sigma)
        return sigmas

    def _run_tiled_diffusion(self, context: InvocationContext) -> torch.Tensor:
        device = TorchDevice.choose_torch_device()
        inference_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        # Load init latents.
        init_latents = context.tensors.load(self.latents.latents_name)
        init_latents = init_latents.to(device=device, dtype=inference_dtype)
        _, _, latent_h, latent_w = init_latents.shape

        # Generate noise.
        num_channels_latents = 16
        noise = torch.randn(
            1,
            num_channels_latents,
            self.height // LATENT_SCALE_FACTOR,
            self.width // LATENT_SCALE_FACTOR,
            device="cpu",
            dtype=torch.float32,
            generator=torch.Generator(device="cpu").manual_seed(self.seed),
        ).to(device=device, dtype=inference_dtype)

        # Z-Image patch_size = 2
        patch_size = 2
        img_token_height = latent_h // patch_size
        img_token_width = latent_w // patch_size
        img_seq_len = img_token_height * img_token_width

        # Load text conditioning.
        pos_text_conditionings = self._load_text_conditioning(
            context=context,
            cond_field=self.positive_conditioning,
            img_height=img_token_height,
            img_width=img_token_width,
            dtype=inference_dtype,
            device=device,
        )

        do_cfg = not math.isclose(self.guidance_scale, 1.0) and self.negative_conditioning is not None
        neg_prompt_embeds: torch.Tensor | None = None
        if do_cfg:
            assert self.negative_conditioning is not None
            neg_text_conditionings = self._load_text_conditioning(
                context=context,
                cond_field=self.negative_conditioning,
                img_height=img_token_height,
                img_width=img_token_width,
                dtype=inference_dtype,
                device=device,
            )
            neg_prompt_embeds = torch.cat([tc.prompt_embeds for tc in neg_text_conditionings], dim=0)

        # Calculate sigma schedule.
        mu = self._calculate_shift(img_seq_len)
        sigmas = self._get_sigmas(mu, self.steps)

        # Apply denoising_start/end clipping.
        if self.denoising_start > 0 or self.denoising_end < 1:
            total_sigmas = len(sigmas)
            start_idx = int(self.denoising_start * (total_sigmas - 1))
            end_idx = int(self.denoising_end * (total_sigmas - 1)) + 1
            sigmas = sigmas[start_idx:end_idx]

        total_steps = len(sigmas) - 1

        # Noise the init latents.
        s_0 = sigmas[0]
        x = s_0 * noise + (1.0 - s_0) * init_latents

        if total_steps <= 0:
            return x

        # Calculate tile layout in latent space.
        latent_tile_h = self.tile_height // LATENT_SCALE_FACTOR
        latent_tile_w = self.tile_width // LATENT_SCALE_FACTOR
        latent_tile_overlap = self.tile_overlap // LATENT_SCALE_FACTOR

        tiles = calc_tiles_min_overlap(
            image_height=latent_h,
            image_width=latent_w,
            tile_height=latent_tile_h,
            tile_width=latent_tile_w,
            min_overlap=latent_tile_overlap,
        )

        step_callback = self._build_step_callback(context)

        with ExitStack() as exit_stack:
            transformer_config = context.models.get_config(self.transformer.transformer)

            # Prepare control extension if provided.
            control_extension: ZImageControlNetExtension | None = None
            if self.control is not None:
                control_extension = self._prep_control_extension(context, exit_stack, device, inference_dtype)

            # Load transformer.
            (cached_weights, transformer) = exit_stack.enter_context(
                context.models.load(self.transformer.transformer).model_on_device()
            )

            # Determine if quantized.
            if transformer_config.format in [ModelFormat.Diffusers, ModelFormat.Checkpoint]:
                model_is_quantized = False
            elif transformer_config.format in [ModelFormat.GGUFQuantized]:
                model_is_quantized = True
            else:
                raise ValueError(f"Unsupported Z-Image model format: {transformer_config.format}")

            # Apply LoRAs.
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

            # Tiled multi-diffusion denoise loop.
            for step_idx in tqdm(range(total_steps), desc="Tiled Z-Image Denoise"):
                sigma_curr = sigmas[step_idx]
                sigma_prev = sigmas[step_idx + 1]

                model_t = 1.0 - sigma_curr
                timestep = torch.tensor([model_t], device=device, dtype=inference_dtype).expand(x.shape[0])

                # Accumulation buffers.
                merged_pred = torch.zeros_like(x)
                merge_weights = torch.zeros((1, 1, latent_h, latent_w), device=device, dtype=inference_dtype)

                for tile in tiles:
                    tile_h = tile.coords.bottom - tile.coords.top
                    tile_w = tile.coords.right - tile.coords.left

                    # Crop tile from latent space.
                    tile_x = x[:, :, tile.coords.top : tile.coords.bottom, tile.coords.left : tile.coords.right]

                    # Prepare for Z-Image transformer: [B, C, H, W] -> [B, C, 1, H, W] -> list of [C, 1, H, W]
                    tile_model_input = tile_x.to(transformer.dtype)
                    tile_model_input = tile_model_input.unsqueeze(2)
                    tile_model_input_list = list(tile_model_input.unbind(dim=0))

                    # Prepare tile-specific text conditioning.
                    tile_img_token_h = tile_h // patch_size
                    tile_img_token_w = tile_w // patch_size
                    tile_img_seq_len = tile_img_token_h * tile_img_token_w

                    tile_regional_ext = ZImageRegionalPromptingExtension.from_text_conditionings(
                        text_conditionings=pos_text_conditionings,
                        img_seq_len=tile_img_seq_len,
                    )
                    tile_pos_embeds = tile_regional_ext.regional_text_conditioning.prompt_embeds

                    # Apply regional prompting patch for this tile.
                    with patch_transformer_for_regional_prompting(
                        transformer=transformer,
                        regional_attn_mask=tile_regional_ext.regional_attn_mask,
                        img_seq_len=tile_img_seq_len,
                    ):
                        # Prepare tile-specific control extension if needed.
                        tile_control_ext = None
                        if control_extension is not None:
                            tile_control_ext = self._crop_control_extension_to_tile(control_extension, tile.coords)
                            apply_control = tile_control_ext.should_apply(step_idx, total_steps)
                        else:
                            apply_control = False

                        # Run positive prediction.
                        if apply_control and tile_control_ext is not None:
                            model_out_list, _ = z_image_forward_with_control(
                                transformer=transformer,
                                x=tile_model_input_list,
                                t=timestep,
                                cap_feats=[tile_pos_embeds],
                                control_extension=tile_control_ext,
                            )
                        else:
                            model_output = transformer(
                                x=tile_model_input_list,
                                t=timestep,
                                cap_feats=[tile_pos_embeds],
                            )
                            model_out_list = model_output[0]

                        noise_pred_cond = torch.stack([t.float() for t in model_out_list], dim=0)
                        noise_pred_cond = noise_pred_cond.squeeze(2)
                        noise_pred_cond = -noise_pred_cond  # Z-Image v-prediction negation

                        # Apply CFG if needed.
                        if do_cfg and neg_prompt_embeds is not None:
                            if apply_control and tile_control_ext is not None:
                                model_out_uncond, _ = z_image_forward_with_control(
                                    transformer=transformer,
                                    x=tile_model_input_list,
                                    t=timestep,
                                    cap_feats=[neg_prompt_embeds],
                                    control_extension=tile_control_ext,
                                )
                            else:
                                model_out_uncond_result = transformer(
                                    x=tile_model_input_list,
                                    t=timestep,
                                    cap_feats=[neg_prompt_embeds],
                                )
                                model_out_uncond = model_out_uncond_result[0]

                            noise_pred_uncond = torch.stack([t.float() for t in model_out_uncond], dim=0)
                            noise_pred_uncond = noise_pred_uncond.squeeze(2)
                            noise_pred_uncond = -noise_pred_uncond
                            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                        else:
                            noise_pred = noise_pred_cond

                    # Euler step for this tile.
                    tile_result = tile_x.to(dtype=torch.float32) + (sigma_prev - sigma_curr) * noise_pred
                    tile_result = tile_result.to(dtype=x.dtype)

                    # Compute gradient blend weight.
                    tile_weight = self._compute_tile_weight(tile_h, tile_w, device, inference_dtype)

                    # Accumulate.
                    merged_pred[:, :, tile.coords.top : tile.coords.bottom, tile.coords.left : tile.coords.right] += (
                        tile_result * tile_weight
                    )
                    merge_weights[:, :, tile.coords.top : tile.coords.bottom, tile.coords.left : tile.coords.right] += (
                        tile_weight
                    )

                # Normalize.
                x = merged_pred / merge_weights

                step_callback(
                    PipelineIntermediateState(
                        step=step_idx + 1,
                        order=1,
                        total_steps=total_steps,
                        timestep=int(sigma_curr * 1000),
                        latents=x,
                    ),
                )

        return x

    @staticmethod
    def _compute_tile_weight(tile_h: int, tile_w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute a gradient blending weight for smooth tile overlap."""
        h_ramp = torch.linspace(0, 1, tile_h, device=device, dtype=dtype)
        h_ramp = torch.min(h_ramp, 1.0 - h_ramp) * 2.0
        h_ramp = h_ramp.clamp(min=0.01)

        w_ramp = torch.linspace(0, 1, tile_w, device=device, dtype=dtype)
        w_ramp = torch.min(w_ramp, 1.0 - w_ramp) * 2.0
        w_ramp = w_ramp.clamp(min=0.01)

        weight = h_ramp[:, None] * w_ramp[None, :]
        return weight.unsqueeze(0).unsqueeze(0)

    def _crop_control_extension_to_tile(
        self,
        control_ext: ZImageControlNetExtension,
        tile_region: TBLR,
    ) -> ZImageControlNetExtension:
        """Create a copy of the control extension with conditioning cropped to a tile."""
        cropped = copy.copy(control_ext)
        # control_cond is in [C, 1, H, W] format where H,W are latent dims
        cropped._control_cond = control_ext._control_cond[
            :,
            :,
            tile_region.top : tile_region.bottom,
            tile_region.left : tile_region.right,
        ]
        return cropped

    def _prep_control_extension(
        self,
        context: InvocationContext,
        exit_stack: ExitStack,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ZImageControlNetExtension:
        """Prepare Z-Image ControlNet extension."""
        assert self.control is not None

        control_model_info = context.models.load(self.control.control_model)
        (_, control_adapter) = exit_stack.enter_context(control_model_info.model_on_device())
        assert isinstance(control_adapter, ZImageControlAdapter)

        adapter_config = control_adapter.config
        control_in_dim = adapter_config.get("control_in_dim", 16)

        if self.vae is None:
            raise ValueError("VAE is required when using Z-Image Control.")

        control_image = context.images.get_pil(self.control.image_name)
        control_image = control_image.convert("RGB")
        control_image = control_image.resize((self.width, self.height))

        control_image_tensor = image_resized_to_grid_as_tensor(control_image)
        if control_image_tensor.dim() == 3:
            control_image_tensor = einops.rearrange(control_image_tensor, "c h w -> 1 c h w")

        vae_info = context.models.load(self.vae.vae)
        control_latents = ZImageImageToLatentsInvocation.vae_encode(
            vae_info=vae_info,
            image_tensor=control_image_tensor,
        )
        control_latents = control_latents.to(device=device, dtype=dtype)

        # Add frame dimension: [B, C, H, W] -> [C, 1, H, W]
        control_latents = control_latents.squeeze(0).unsqueeze(1)

        # Pad to control_in_dim if needed (V2.0 needs 33 channels).
        c, f, h, w = control_latents.shape
        if c < control_in_dim:
            padding_channels = control_in_dim - c
            if padding_channels == 17:
                ref_padding = torch.zeros((16, f, h, w), device=device, dtype=dtype)
                mask_channel = torch.ones((1, f, h, w), device=device, dtype=dtype)
                control_latents = torch.cat([control_latents, ref_padding, mask_channel], dim=0)
            else:
                zero_padding = torch.zeros((padding_channels, f, h, w), device=device, dtype=dtype)
                control_latents = torch.cat([control_latents, zero_padding], dim=0)

        return ZImageControlNetExtension(
            control_adapter=control_adapter,
            control_cond=control_latents,
            weight=self.control.control_context_scale,
            begin_step_percent=self.control.begin_step_percent,
            end_step_percent=self.control.end_step_percent,
        )

    def _load_text_conditioning(
        self,
        context: InvocationContext,
        cond_field: ZImageConditioningField | list[ZImageConditioningField],
        img_height: int,
        img_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[ZImageTextConditioning]:
        """Load Z-Image text conditioning data."""
        cond_list = [cond_field] if isinstance(cond_field, ZImageConditioningField) else cond_field

        text_conditionings: list[ZImageTextConditioning] = []
        for cond in cond_list:
            cond_data = context.conditioning.load(cond.conditioning_name)
            assert len(cond_data.conditionings) == 1
            z_cond = cond_data.conditionings[0]
            assert isinstance(z_cond, ZImageConditioningInfo)
            z_cond = z_cond.to(dtype=dtype, device=device)

            mask = None
            if cond.mask is not None:
                mask = context.tensors.load(cond.mask.tensor_name)
                mask = mask.to(device=device)
                mask = ZImageRegionalPromptingExtension.preprocess_regional_prompt_mask(
                    mask, img_height, img_width, dtype, device
                )

            text_conditionings.append(ZImageTextConditioning(prompt_embeds=z_cond.prompt_embeds, mask=mask))

        return text_conditionings

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, BaseModelType.ZImage)

        return step_callback

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        for lora in self.transformer.loras:
            lora_info = context.models.load(lora.lora)
            assert isinstance(lora_info.model, ModelPatchRaw)
            yield (lora_info.model, lora.weight)
            del lora_info
