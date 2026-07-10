"""Tiled Multi-Diffusion denoising for FLUX transformer models.

Analogous to TiledMultiDiffusionDenoiseLatents but for FLUX's transformer architecture
instead of UNet. Primarily intended for tiled upscaling workflows.
"""

import copy
import math
from contextlib import ExitStack
from typing import Iterator, Tuple, Union

import torch
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    FluxConditioningField,
    Input,
    InputField,
    LatentsField,
)
from invokeai.app.invocations.flux_controlnet import FluxControlNetField
from invokeai.app.invocations.flux_denoise import FluxDenoiseInvocation
from invokeai.app.invocations.model import TransformerField, VAEField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.controlnet.controlnet_flux_output import ControlNetFluxOutput, sum_controlnet_flux_outputs
from invokeai.backend.flux.controlnet.instantx_controlnet_flux import InstantXControlNetFlux
from invokeai.backend.flux.controlnet.xlabs_controlnet_flux import XLabsControlNetFlux
from invokeai.backend.flux.extensions.instantx_controlnet_extension import InstantXControlNetExtension
from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
from invokeai.backend.flux.extensions.xlabs_controlnet_extension import XLabsControlNetExtension
from invokeai.backend.flux.model import Flux
from invokeai.backend.flux.sampling_utils import (
    clip_timestep_schedule_fractional,
    generate_img_ids,
    get_noise,
    get_schedule,
    pack,
    unpack,
)
from invokeai.backend.flux.schedulers import FLUX_SCHEDULER_LABELS, FLUX_SCHEDULER_NAME_VALUES
from invokeai.backend.flux.text_conditioning import FluxTextConditioning
from invokeai.backend.model_manager.taxonomy import BaseModelType, FluxVariantType, ModelFormat, ModelType
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import FLUXConditioningInfo
from invokeai.backend.tiles.tiles import calc_tiles_min_overlap
from invokeai.backend.tiles.utils import TBLR
from invokeai.backend.util.devices import TorchDevice

# FLUX latent scale factor: the VAE downsamples by 8x, and then packing groups 2x2 patches.
# For tiling purposes, we work in unpacked latent space where the scale factor is 8.
FLUX_LATENT_SCALE_FACTOR = 8


def _crop_controlnet_cond_instantx(
    controlnet_cond_packed: torch.Tensor,
    full_latent_h: int,
    full_latent_w: int,
    tile_region: TBLR,
) -> torch.Tensor:
    """Crop a packed InstantX ControlNet conditioning tensor to a tile region.

    The InstantX controlnet_cond is VAE-encoded and packed: shape (B, H*W/4, C*4).
    We unpack to spatial (B, C, H, W), crop, then repack.
    """
    # Unpack to spatial format
    cond_unpacked = unpack(
        controlnet_cond_packed, full_latent_h * FLUX_LATENT_SCALE_FACTOR, full_latent_w * FLUX_LATENT_SCALE_FACTOR
    )
    # Crop in latent space
    cond_cropped = cond_unpacked[
        :,
        :,
        tile_region.top : tile_region.bottom,
        tile_region.left : tile_region.right,
    ]
    # Repack
    return pack(cond_cropped)


def _crop_controlnet_cond_xlabs(
    controlnet_cond: torch.Tensor,
    tile_region: TBLR,
) -> torch.Tensor:
    """Crop an XLabs ControlNet conditioning tensor to a tile region.

    The XLabs controlnet_cond is in pixel space: shape (B, 3, H, W).
    We crop in pixel space (tile_region is in latent coords, so multiply by scale factor).
    """
    return controlnet_cond[
        :,
        :,
        tile_region.top * FLUX_LATENT_SCALE_FACTOR : tile_region.bottom * FLUX_LATENT_SCALE_FACTOR,
        tile_region.left * FLUX_LATENT_SCALE_FACTOR : tile_region.right * FLUX_LATENT_SCALE_FACTOR,
    ]


@invocation(
    "tiled_flux_denoise",
    title="Tiled Multi-Diffusion Denoise - FLUX",
    tags=["upscale", "denoise", "flux"],
    category="latents",
    version="1.0.0",
)
class TiledFluxDenoiseLatents(BaseInvocation):
    """Tiled Multi-Diffusion denoising for FLUX models.

    This node handles automatically tiling the input latents and is primarily intended for
    tiled upscaling workflows with FLUX transformer models. It applies the same multi-diffusion
    blending approach as the SD1.5/SDXL tiled denoise node.
    """

    latents: LatentsField = InputField(
        description=FieldDescriptions.latents,
        input=Input.Connection,
    )
    positive_text_conditioning: FluxConditioningField | list[FluxConditioningField] = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_text_conditioning: FluxConditioningField | list[FluxConditioningField] | None = InputField(
        default=None,
        description="Negative conditioning tensor. Can be None if cfg_scale is 1.0.",
        input=Input.Connection,
    )
    transformer: TransformerField = InputField(
        description=FieldDescriptions.flux_model,
        input=Input.Connection,
        title="Transformer",
    )
    control: FluxControlNetField | list[FluxControlNetField] | None = InputField(
        default=None, input=Input.Connection, description="ControlNet models."
    )
    controlnet_vae: VAEField | None = InputField(
        default=None,
        description=FieldDescriptions.vae,
        input=Input.Connection,
    )
    width: int = InputField(default=1024, multiple_of=16, description="Width of the full image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the full image.")
    tile_height: int = InputField(default=1024, gt=0, multiple_of=16, description="Height of tiles in image space.")
    tile_width: int = InputField(default=1024, gt=0, multiple_of=16, description="Width of tiles in image space.")
    tile_overlap: int = InputField(
        default=128,
        gt=0,
        multiple_of=16,
        description="Overlap between adjacent tiles in image space.",
    )
    num_steps: int = InputField(default=8, gt=0, description="Number of diffusion steps.")
    guidance: float = InputField(
        default=4.0,
        description="Guidance strength. Higher values adhere more strictly to the prompt.",
    )
    cfg_scale: float | list[float] = InputField(default=1.0, description=FieldDescriptions.cfg_scale, title="CFG Scale")
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
    scheduler: FLUX_SCHEDULER_NAME_VALUES = InputField(
        default="euler",
        description="Scheduler (sampler) for the denoising process.",
        ui_choice_labels=FLUX_SCHEDULER_LABELS,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_tiled_diffusion(context)
        latents = latents.detach().to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)

    def _run_tiled_diffusion(self, context: InvocationContext) -> torch.Tensor:
        inference_dtype = torch.bfloat16
        device = TorchDevice.choose_torch_device()

        # Load the init latents (from VAE encode).
        init_latents = context.tensors.load(self.latents.latents_name)
        init_latents = init_latents.to(device=device, dtype=inference_dtype)
        _, _, latent_h, latent_w = init_latents.shape

        # Generate noise matching the full latent dimensions.
        noise = get_noise(
            num_samples=1,
            height=self.height,
            width=self.width,
            device=device,
            dtype=inference_dtype,
            seed=self.seed,
        )

        # Calculate packed dimensions for text conditioning.
        packed_h = latent_h // 2
        packed_w = latent_w // 2

        # Load text conditioning.
        pos_text_conditionings = self._load_text_conditioning(
            context=context,
            cond_field=self.positive_text_conditioning,
            packed_height=packed_h,
            packed_width=packed_w,
            dtype=inference_dtype,
            device=device,
        )
        neg_text_conditionings: list[FluxTextConditioning] | None = None
        if self.negative_text_conditioning is not None:
            neg_text_conditionings = self._load_text_conditioning(
                context=context,
                cond_field=self.negative_text_conditioning,
                packed_height=packed_h,
                packed_width=packed_w,
                dtype=inference_dtype,
                device=device,
            )

        # Get transformer config.
        transformer_config = context.models.get_config(self.transformer.transformer)
        assert (
            transformer_config.base in (BaseModelType.Flux, BaseModelType.Flux2)
            and transformer_config.type is ModelType.Main
        )
        is_schnell = (
            transformer_config.base is BaseModelType.Flux and transformer_config.variant is FluxVariantType.Schnell
        )

        # Calculate the timestep schedule.
        timesteps = get_schedule(
            num_steps=self.num_steps,
            image_seq_len=packed_h * packed_w,
            shift=not is_schnell,
        )
        timesteps = clip_timestep_schedule_fractional(timesteps, self.denoising_start, self.denoising_end)

        # Prepare cfg_scale schedule.
        cfg_scale = FluxDenoiseInvocation.prep_cfg_scale(
            cfg_scale=self.cfg_scale,
            timesteps=timesteps,
            cfg_scale_start_step=0,
            cfg_scale_end_step=-1,
        )

        # Noise the init latents for the first timestep.
        t_0 = timesteps[0]
        x = t_0 * noise + (1.0 - t_0) * init_latents

        if len(timesteps) <= 1:
            return unpack(pack(x), self.height, self.width)

        # Calculate tile layout in latent space.
        latent_tile_h = self.tile_height // FLUX_LATENT_SCALE_FACTOR
        latent_tile_w = self.tile_width // FLUX_LATENT_SCALE_FACTOR
        latent_tile_overlap = self.tile_overlap // FLUX_LATENT_SCALE_FACTOR

        tiles = calc_tiles_min_overlap(
            image_height=latent_h,
            image_width=latent_w,
            tile_height=latent_tile_h,
            tile_width=latent_tile_w,
            min_overlap=latent_tile_overlap,
        )

        with ExitStack() as exit_stack:
            # Prepare ControlNet extensions (before loading transformer to minimize peak memory).
            controlnet_extensions = self._prep_controlnet_extensions(
                context=context,
                exit_stack=exit_stack,
                latent_height=latent_h,
                latent_width=latent_w,
                dtype=inference_dtype,
                device=device,
            )

            # Load the transformer model.
            (cached_weights, transformer) = exit_stack.enter_context(
                context.models.load(self.transformer.transformer).model_on_device()
            )
            assert isinstance(transformer, Flux)

            # Determine if the model is quantized.
            if transformer_config.format in [ModelFormat.Checkpoint]:
                model_is_quantized = False
            elif transformer_config.format in [
                ModelFormat.BnbQuantizedLlmInt8b,
                ModelFormat.BnbQuantizednf4b,
                ModelFormat.GGUFQuantized,
            ]:
                model_is_quantized = True
            else:
                raise ValueError(f"Unsupported model format: {transformer_config.format}")

            # Apply LoRA models.
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

            guidance_vec = torch.full((x.shape[0],), self.guidance, device=device, dtype=inference_dtype)

            # Run tiled multi-diffusion denoise loop.
            total_steps = len(timesteps) - 1
            for step_index, (t_curr, t_prev) in tqdm(
                list(enumerate(zip(timesteps[:-1], timesteps[1:], strict=True))),
                desc="Tiled FLUX Denoise",
            ):
                t_vec = torch.full((x.shape[0],), t_curr, dtype=inference_dtype, device=device)
                step_cfg_scale = cfg_scale[step_index]

                # Accumulation buffers in unpacked spatial format.
                merged_pred = torch.zeros_like(x)
                merge_weights = torch.zeros((1, 1, latent_h, latent_w), device=device, dtype=inference_dtype)

                for tile in tiles:
                    tile_h = tile.coords.bottom - tile.coords.top
                    tile_w = tile.coords.right - tile.coords.left

                    # Crop the latent tile from unpacked spatial representation.
                    tile_x = x[:, :, tile.coords.top : tile.coords.bottom, tile.coords.left : tile.coords.right]

                    # Pack the tile for FLUX transformer.
                    tile_packed = pack(tile_x)

                    # Generate position IDs for this tile (local coordinates).
                    tile_img_ids = generate_img_ids(
                        h=tile_h, w=tile_w, batch_size=x.shape[0], device=device, dtype=inference_dtype
                    )

                    # Prepare tile-specific text conditioning (use tile dimensions for regional prompt masks).
                    tile_packed_h = tile_h // 2
                    tile_packed_w = tile_w // 2
                    pos_regional_ext = RegionalPromptingExtension.from_text_conditioning(
                        text_conditioning=pos_text_conditionings,
                        redux_conditioning=[],
                        img_seq_len=tile_packed_h * tile_packed_w,
                    )
                    neg_regional_ext = (
                        RegionalPromptingExtension.from_text_conditioning(
                            text_conditioning=neg_text_conditionings,
                            redux_conditioning=[],
                            img_seq_len=tile_packed_h * tile_packed_w,
                        )
                        if neg_text_conditionings
                        else None
                    )

                    # Run ControlNet models for this tile (with cropped conditioning).
                    controlnet_residuals: list[ControlNetFluxOutput] = []
                    for cn_ext in controlnet_extensions:
                        tile_cn_ext = self._crop_controlnet_extension_to_tile(cn_ext, latent_h, latent_w, tile.coords)
                        controlnet_residuals.append(
                            tile_cn_ext.run_controlnet(
                                timestep_index=step_index,
                                total_num_timesteps=total_steps,
                                img=tile_packed,
                                img_ids=tile_img_ids,
                                txt=pos_regional_ext.regional_text_conditioning.t5_embeddings,
                                txt_ids=pos_regional_ext.regional_text_conditioning.t5_txt_ids,
                                y=pos_regional_ext.regional_text_conditioning.clip_embeddings,
                                timesteps=t_vec,
                                guidance=guidance_vec,
                            )
                        )
                    merged_controlnet_residuals = sum_controlnet_flux_outputs(controlnet_residuals)

                    # Run positive transformer prediction.
                    pred = transformer(
                        img=tile_packed,
                        img_ids=tile_img_ids,
                        txt=pos_regional_ext.regional_text_conditioning.t5_embeddings,
                        txt_ids=pos_regional_ext.regional_text_conditioning.t5_txt_ids,
                        y=pos_regional_ext.regional_text_conditioning.clip_embeddings,
                        timesteps=t_vec,
                        guidance=guidance_vec,
                        timestep_index=step_index,
                        total_num_timesteps=total_steps,
                        controlnet_double_block_residuals=merged_controlnet_residuals.double_block_residuals,
                        controlnet_single_block_residuals=merged_controlnet_residuals.single_block_residuals,
                        ip_adapter_extensions=[],
                        regional_prompting_extension=pos_regional_ext,
                    )

                    # Apply CFG if needed.
                    if not math.isclose(step_cfg_scale, 1.0):
                        if neg_regional_ext is None:
                            raise ValueError("Negative text conditioning is required when cfg_scale is not 1.0.")
                        neg_pred = transformer(
                            img=tile_packed,
                            img_ids=tile_img_ids,
                            txt=neg_regional_ext.regional_text_conditioning.t5_embeddings,
                            txt_ids=neg_regional_ext.regional_text_conditioning.t5_txt_ids,
                            y=neg_regional_ext.regional_text_conditioning.clip_embeddings,
                            timesteps=t_vec,
                            guidance=guidance_vec,
                            timestep_index=step_index,
                            total_num_timesteps=total_steps,
                            controlnet_double_block_residuals=None,
                            controlnet_single_block_residuals=None,
                            ip_adapter_extensions=[],
                            regional_prompting_extension=neg_regional_ext,
                        )
                        pred = neg_pred + step_cfg_scale * (pred - neg_pred)

                    # Euler step for this tile (in packed space).
                    tile_result_packed = tile_packed + (t_prev - t_curr) * pred

                    # Unpack back to spatial format.
                    tile_img_h = tile_h * FLUX_LATENT_SCALE_FACTOR
                    tile_img_w = tile_w * FLUX_LATENT_SCALE_FACTOR
                    tile_result = unpack(tile_result_packed, tile_img_h, tile_img_w)

                    # Compute gradient blend weight for this tile.
                    tile_weight = self._compute_tile_weight(tile_h, tile_w, device, inference_dtype)

                    # Accumulate.
                    merged_pred[:, :, tile.coords.top : tile.coords.bottom, tile.coords.left : tile.coords.right] += (
                        tile_result * tile_weight
                    )
                    merge_weights[:, :, tile.coords.top : tile.coords.bottom, tile.coords.left : tile.coords.right] += (
                        tile_weight
                    )

                # Normalize by accumulated weights.
                x = merged_pred / merge_weights

                # Step callback for progress reporting.
                preview_img = x - t_curr * (x - init_latents)  # Approximate preview
                context.util.flux_step_callback(
                    PipelineIntermediateState(
                        step=step_index + 1,
                        order=1,
                        total_steps=total_steps,
                        timestep=int(t_curr),
                        latents=unpack(pack(preview_img.float()), self.height, self.width).squeeze(),
                    ),
                )

        # Return in unpacked format.
        return unpack(pack(x.float()), self.height, self.width)

    @staticmethod
    def _compute_tile_weight(tile_h: int, tile_w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute a gradient blending weight for a tile.

        Creates a ramp weight that goes from 0 at the edges to 1 in the center,
        which provides smooth blending in overlap regions.
        """
        # Create 1D ramps.
        h_ramp = torch.linspace(0, 1, tile_h, device=device, dtype=dtype)
        h_ramp = torch.min(h_ramp, 1.0 - h_ramp) * 2.0
        h_ramp = h_ramp.clamp(min=0.01)

        w_ramp = torch.linspace(0, 1, tile_w, device=device, dtype=dtype)
        w_ramp = torch.min(w_ramp, 1.0 - w_ramp) * 2.0
        w_ramp = w_ramp.clamp(min=0.01)

        # Outer product to create 2D weight.
        weight = h_ramp[:, None] * w_ramp[None, :]
        return weight.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    def _crop_controlnet_extension_to_tile(
        self,
        cn_ext: Union[InstantXControlNetExtension, XLabsControlNetExtension],
        full_latent_h: int,
        full_latent_w: int,
        tile_region: TBLR,
    ) -> Union[InstantXControlNetExtension, XLabsControlNetExtension]:
        """Create a copy of a ControlNet extension with conditioning cropped to a tile region."""
        cropped = copy.copy(cn_ext)
        if isinstance(cn_ext, InstantXControlNetExtension):
            cropped._controlnet_cond = _crop_controlnet_cond_instantx(
                cn_ext._controlnet_cond, full_latent_h, full_latent_w, tile_region
            )
        elif isinstance(cn_ext, XLabsControlNetExtension):
            cropped._controlnet_cond = _crop_controlnet_cond_xlabs(cn_ext._controlnet_cond, tile_region)
        return cropped

    def _load_text_conditioning(
        self,
        context: InvocationContext,
        cond_field: FluxConditioningField | list[FluxConditioningField],
        packed_height: int,
        packed_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[FluxTextConditioning]:
        """Load text conditioning data from FluxConditioningField(s)."""
        cond_list = [cond_field] if isinstance(cond_field, FluxConditioningField) else cond_field

        text_conditionings: list[FluxTextConditioning] = []
        for field in cond_list:
            cond_data = context.conditioning.load(field.conditioning_name)
            assert len(cond_data.conditionings) == 1
            flux_conditioning = cond_data.conditionings[0]
            assert isinstance(flux_conditioning, FLUXConditioningInfo)
            flux_conditioning = flux_conditioning.to(dtype=dtype, device=device)

            mask = None
            if field.mask is not None:
                mask = context.tensors.load(field.mask.tensor_name)
                mask = mask.to(device=device)
                mask = RegionalPromptingExtension.preprocess_regional_prompt_mask(
                    mask, packed_height, packed_width, dtype, device
                )

            text_conditionings.append(
                FluxTextConditioning(flux_conditioning.t5_embeds, flux_conditioning.clip_embeds, mask)
            )

        return text_conditionings

    def _prep_controlnet_extensions(
        self,
        context: InvocationContext,
        exit_stack: ExitStack,
        latent_height: int,
        latent_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[Union[XLabsControlNetExtension, InstantXControlNetExtension]]:
        """Prepare ControlNet extensions. Same logic as FluxDenoiseInvocation._prep_controlnet_extensions."""
        controlnets: list[FluxControlNetField]
        if self.control is None:
            controlnets = []
        elif isinstance(self.control, FluxControlNetField):
            controlnets = [self.control]
        elif isinstance(self.control, list):
            controlnets = self.control
        else:
            raise ValueError(f"Unsupported controlnet type: {type(self.control)}")

        # Prepare conditioning tensors (may require VAE encoding).
        controlnet_conds: list[torch.Tensor] = []
        for controlnet in controlnets:
            image = context.images.get_pil(controlnet.image.image_name)
            controlnet_model = context.models.load(controlnet.control_model)

            if isinstance(controlnet_model.model, InstantXControlNetFlux):
                if self.controlnet_vae is None:
                    raise ValueError("A ControlNet VAE is required when using an InstantX FLUX ControlNet.")
                vae_info = context.models.load(self.controlnet_vae.vae)
                controlnet_conds.append(
                    InstantXControlNetExtension.prepare_controlnet_cond(
                        controlnet_image=image,
                        vae_info=vae_info,
                        latent_height=latent_height,
                        latent_width=latent_width,
                        dtype=dtype,
                        device=device,
                        resize_mode=controlnet.resize_mode,
                    )
                )
            elif isinstance(controlnet_model.model, XLabsControlNetFlux):
                controlnet_conds.append(
                    XLabsControlNetExtension.prepare_controlnet_cond(
                        controlnet_image=image,
                        latent_height=latent_height,
                        latent_width=latent_width,
                        dtype=dtype,
                        device=device,
                        resize_mode=controlnet.resize_mode,
                    )
                )

        # Load ControlNet models and initialize extensions.
        controlnet_extensions: list[Union[XLabsControlNetExtension, InstantXControlNetExtension]] = []
        for controlnet, controlnet_cond in zip(controlnets, controlnet_conds, strict=True):
            model = exit_stack.enter_context(context.models.load(controlnet.control_model))

            if isinstance(model, XLabsControlNetFlux):
                controlnet_extensions.append(
                    XLabsControlNetExtension(
                        model=model,
                        controlnet_cond=controlnet_cond,
                        weight=controlnet.control_weight,
                        begin_step_percent=controlnet.begin_step_percent,
                        end_step_percent=controlnet.end_step_percent,
                    )
                )
            elif isinstance(model, InstantXControlNetFlux):
                instantx_control_mode: torch.Tensor | None = None
                if controlnet.instantx_control_mode is not None and controlnet.instantx_control_mode >= 0:
                    instantx_control_mode = torch.tensor(controlnet.instantx_control_mode, dtype=torch.long)
                    instantx_control_mode = instantx_control_mode.reshape([-1, 1])

                controlnet_extensions.append(
                    InstantXControlNetExtension(
                        model=model,
                        controlnet_cond=controlnet_cond,
                        instantx_control_mode=instantx_control_mode,
                        weight=controlnet.control_weight,
                        begin_step_percent=controlnet.begin_step_percent,
                        end_step_percent=controlnet.end_step_percent,
                    )
                )
            else:
                raise ValueError(f"Unsupported ControlNet model type: {type(model)}")

        return controlnet_extensions

    def _lora_iterator(self, context: InvocationContext) -> Iterator[Tuple[ModelPatchRaw, float]]:
        for lora in self.transformer.loras:
            lora_info = context.models.load(lora.lora)
            assert isinstance(lora_info.model, ModelPatchRaw)
            yield (lora_info.model, lora.weight)
            del lora_info
