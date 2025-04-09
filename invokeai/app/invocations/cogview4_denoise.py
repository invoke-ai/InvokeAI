from typing import Callable, Optional

import torch
import torchvision.transforms as tv_transforms
from diffusers.models.transformers.transformer_cogview4 import CogView4Transformer2DModel
from torchvision.transforms.functional import resize as tv_resize
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.fields import (
    CogView4ConditioningField,
    DenoiseMaskField,
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import TransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.sampling_utils import clip_timestep_schedule_fractional
from invokeai.backend.model_manager.config import BaseModelType
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import CogView4ConditioningInfo
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "cogview4_denoise",
    title="Denoise - CogView4",
    tags=["image", "cogview4"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class CogView4DenoiseInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Run the denoising process with a CogView4 model."""

    # If latents is provided, this means we are doing image-to-image.
    latents: Optional[LatentsField] = InputField(
        default=None, description=FieldDescriptions.latents, input=Input.Connection
    )
    # denoise_mask is used for image-to-image inpainting. Only the masked region is modified.
    denoise_mask: Optional[DenoiseMaskField] = InputField(
        default=None, description=FieldDescriptions.denoise_mask, input=Input.Connection
    )
    denoising_start: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.denoising_start)
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    transformer: TransformerField = InputField(
        description=FieldDescriptions.cogview4_model, input=Input.Connection, title="Transformer"
    )
    positive_conditioning: CogView4ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: CogView4ConditioningField = InputField(
        description=FieldDescriptions.negative_cond, input=Input.Connection
    )
    cfg_scale: float | list[float] = InputField(default=3.5, description=FieldDescriptions.cfg_scale, title="CFG Scale")
    width: int = InputField(default=1024, multiple_of=32, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=32, description="Height of the generated image.")
    steps: int = InputField(default=25, gt=0, description=FieldDescriptions.steps)
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_diffusion(context)
        latents = latents.detach().to("cpu")

        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)

    def _prep_inpaint_mask(self, context: InvocationContext, latents: torch.Tensor) -> torch.Tensor | None:
        """Prepare the inpaint mask.
        - Loads the mask
        - Resizes if necessary
        - Casts to same device/dtype as latents

        Args:
            context (InvocationContext): The invocation context, for loading the inpaint mask.
            latents (torch.Tensor): A latent image tensor. Used to determine the target shape, device, and dtype for the
                inpaint mask.

        Returns:
            torch.Tensor | None: Inpaint mask. Values of 0.0 represent the regions to be fully denoised, and 1.0
                represent the regions to be preserved.
        """
        if self.denoise_mask is None:
            return None
        mask = context.tensors.load(self.denoise_mask.mask_name)

        # The input denoise_mask contains values in [0, 1], where 0.0 represents the regions to be fully denoised, and
        # 1.0 represents the regions to be preserved.
        # We invert the mask so that the regions to be preserved are 0.0 and the regions to be denoised are 1.0.
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
        conditioning_name: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        # Load the conditioning data.
        cond_data = context.conditioning.load(conditioning_name)
        assert len(cond_data.conditionings) == 1
        cogview4_conditioning = cond_data.conditionings[0]
        assert isinstance(cogview4_conditioning, CogView4ConditioningInfo)
        cogview4_conditioning = cogview4_conditioning.to(dtype=dtype, device=device)

        return cogview4_conditioning.glm_embeds

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
        # We always generate noise on the same device and dtype then cast to ensure consistency across devices/dtypes.
        rand_device = "cpu"
        rand_dtype = torch.float16

        return torch.randn(
            batch_size,
            num_channels_latents,
            int(height) // LATENT_SCALE_FACTOR,
            int(width) // LATENT_SCALE_FACTOR,
            device=rand_device,
            dtype=rand_dtype,
            generator=torch.Generator(device=rand_device).manual_seed(seed),
        ).to(device=device, dtype=dtype)

    def _prepare_cfg_scale(self, num_timesteps: int) -> list[float]:
        """Prepare the CFG scale list.

        Args:
            num_timesteps (int): The number of timesteps in the scheduler. Could be different from num_steps depending
            on the scheduler used (e.g. higher order schedulers).

        Returns:
            list[float]: _description_
        """
        if isinstance(self.cfg_scale, float):
            cfg_scale = [self.cfg_scale] * num_timesteps
        elif isinstance(self.cfg_scale, list):
            assert len(self.cfg_scale) == num_timesteps
            cfg_scale = self.cfg_scale
        else:
            raise ValueError(f"Invalid CFG scale type: {type(self.cfg_scale)}")

        return cfg_scale

    def _convert_timesteps_to_sigmas(self, image_seq_len: int, timesteps: torch.Tensor) -> list[float]:
        # The logic to prepare the timestep / sigma schedule is based on:
        # https://github.com/huggingface/diffusers/blob/b38450d5d2e5b87d5ff7088ee5798c85587b9635/src/diffusers/pipelines/cogview4/pipeline_cogview4.py#L575-L595
        # The default FlowMatchEulerDiscreteScheduler configs are based on:
        # https://huggingface.co/THUDM/CogView4-6B/blob/fb6f57289c73ac6d139e8d81bd5a4602d1877847/scheduler/scheduler_config.json
        # This implementation differs slightly from the original for the sake of simplicity (differs in terminal value
        # handling, not quantizing timesteps to integers, etc.).

        def calculate_timestep_shift(
            image_seq_len: int, base_seq_len: int = 256, base_shift: float = 0.25, max_shift: float = 0.75
        ) -> float:
            m = (image_seq_len / base_seq_len) ** 0.5
            mu = m * max_shift + base_shift
            return mu

        def time_shift_linear(mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
            return mu / (mu + (1 / t - 1) ** sigma)

        mu = calculate_timestep_shift(image_seq_len)
        sigmas = time_shift_linear(mu, 1.0, timesteps)
        return sigmas.tolist()

    def _run_diffusion(
        self,
        context: InvocationContext,
    ):
        inference_dtype = torch.bfloat16
        device = TorchDevice.choose_torch_device()

        transformer_info = context.models.load(self.transformer.transformer)
        assert isinstance(transformer_info.model, CogView4Transformer2DModel)

        # Load/process the conditioning data.
        # TODO(ryand): Make CFG optional.
        do_classifier_free_guidance = True
        pos_prompt_embeds = self._load_text_conditioning(
            context=context,
            conditioning_name=self.positive_conditioning.conditioning_name,
            dtype=inference_dtype,
            device=device,
        )
        neg_prompt_embeds = self._load_text_conditioning(
            context=context,
            conditioning_name=self.negative_conditioning.conditioning_name,
            dtype=inference_dtype,
            device=device,
        )

        # Prepare misc. conditioning variables.
        # TODO(ryand): We could expose these as params (like with SDXL). But, we should experiment to see if they are
        # useful first.
        original_size = torch.tensor([(self.height, self.width)], dtype=pos_prompt_embeds.dtype, device=device)
        target_size = torch.tensor([(self.height, self.width)], dtype=pos_prompt_embeds.dtype, device=device)
        crops_coords_top_left = torch.tensor([(0, 0)], dtype=pos_prompt_embeds.dtype, device=device)

        # Prepare the timestep / sigma schedule.
        patch_size = transformer_info.model.config.patch_size  # type: ignore
        assert isinstance(patch_size, int)
        image_seq_len = ((self.height // LATENT_SCALE_FACTOR) * (self.width // LATENT_SCALE_FACTOR)) // (patch_size**2)
        # We add an extra step to the end to account for the final timestep of 0.0.
        timesteps: list[float] = torch.linspace(1, 0, self.steps + 1).tolist()
        # Clip the timesteps schedule based on denoising_start and denoising_end.
        timesteps = clip_timestep_schedule_fractional(timesteps, self.denoising_start, self.denoising_end)
        sigmas = self._convert_timesteps_to_sigmas(image_seq_len, torch.tensor(timesteps))
        total_steps = len(timesteps) - 1

        # Prepare the CFG scale list.
        cfg_scale = self._prepare_cfg_scale(total_steps)

        # Load the input latents, if provided.
        init_latents = context.tensors.load(self.latents.latents_name) if self.latents else None
        if init_latents is not None:
            init_latents = init_latents.to(device=device, dtype=inference_dtype)

        # Generate initial latent noise.
        num_channels_latents = transformer_info.model.config.in_channels  # type: ignore
        assert isinstance(num_channels_latents, int)
        noise = self._get_noise(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=self.height,
            width=self.width,
            dtype=inference_dtype,
            device=device,
            seed=self.seed,
        )

        # Prepare input latent image.
        if init_latents is not None:
            # Noise the init_latents by the appropriate amount for the first timestep.
            s_0 = sigmas[0]
            latents = s_0 * noise + (1.0 - s_0) * init_latents
        else:
            # init_latents are not provided, so we are not doing image-to-image (i.e. we are starting from pure noise).
            if self.denoising_start > 1e-5:
                raise ValueError("denoising_start should be 0 when initial latents are not provided.")
            latents = noise

        # If len(timesteps) == 1, then short-circuit. We are just noising the input latents, but not taking any
        # denoising steps.
        if len(timesteps) <= 1:
            return latents

        # Prepare inpaint extension.
        inpaint_mask = self._prep_inpaint_mask(context, latents)
        inpaint_extension: RectifiedFlowInpaintExtension | None = None
        if inpaint_mask is not None:
            assert init_latents is not None
            inpaint_extension = RectifiedFlowInpaintExtension(
                init_latents=init_latents,
                inpaint_mask=inpaint_mask,
                noise=noise,
            )

        step_callback = self._build_step_callback(context)

        step_callback(
            PipelineIntermediateState(
                step=0,
                order=1,
                total_steps=total_steps,
                timestep=int(timesteps[0]),
                latents=latents,
            ),
        )

        with transformer_info.model_on_device() as (_, transformer):
            assert isinstance(transformer, CogView4Transformer2DModel)

            # Denoising loop
            for step_idx in tqdm(range(total_steps)):
                t_curr = timesteps[step_idx]
                sigma_curr = sigmas[step_idx]
                sigma_prev = sigmas[step_idx + 1]

                # Expand the timestep to match the latent model input.
                # Multiply by 1000 to match the default FlowMatchEulerDiscreteScheduler num_train_timesteps.
                timestep = torch.tensor([t_curr * 1000], device=device).expand(latents.shape[0])

                # TODO(ryand): Support both sequential and batched CFG inference.
                noise_pred_cond = transformer(
                    hidden_states=latents,
                    encoder_hidden_states=pos_prompt_embeds,
                    timestep=timestep,
                    original_size=original_size,
                    target_size=target_size,
                    crop_coords=crops_coords_top_left,
                    return_dict=False,
                )[0]

                # Apply CFG.
                if do_classifier_free_guidance:
                    noise_pred_uncond = transformer(
                        hidden_states=latents,
                        encoder_hidden_states=neg_prompt_embeds,
                        timestep=timestep,
                        original_size=original_size,
                        target_size=target_size,
                        crop_coords=crops_coords_top_left,
                        return_dict=False,
                    )[0]

                    noise_pred = noise_pred_uncond + cfg_scale[step_idx] * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                # Compute the previous noisy sample x_t -> x_t-1.
                latents_dtype = latents.dtype
                # TODO(ryand): Is casting to float32 necessary for precision/stability? I copied this from SD3.
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
                        timestep=int(t_curr),
                        latents=latents,
                    ),
                )

        return latents

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, BaseModelType.CogView4)

        return step_callback
