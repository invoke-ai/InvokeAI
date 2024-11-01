from typing import Callable, Tuple

import torch
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    SD3ConditioningField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import TransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.invocations.sd3_text_encoder import SD3_T5_MAX_SEQ_LEN
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.model_manager.config import BaseModelType
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import SD3ConditioningInfo
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "sd3_denoise",
    title="SD3 Denoise",
    tags=["image", "sd3"],
    category="image",
    version="1.0.0",
    classification=Classification.Prototype,
)
class SD3DenoiseInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Run denoising process with a SD3 model."""

    transformer: TransformerField = InputField(
        description=FieldDescriptions.sd3_model,
        input=Input.Connection,
        title="Transformer",
    )
    positive_conditioning: SD3ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: SD3ConditioningField = InputField(
        description=FieldDescriptions.negative_cond, input=Input.Connection
    )
    cfg_scale: float | list[float] = InputField(default=3.5, description=FieldDescriptions.cfg_scale, title="CFG Scale")
    width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")
    steps: int = InputField(default=10, gt=0, description=FieldDescriptions.steps)
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        latents = self._run_diffusion(context)
        latents = latents.detach().to("cpu")

        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=None)

    def _load_text_conditioning(
        self,
        context: InvocationContext,
        conditioning_name: str,
        joint_attention_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load the conditioning data.
        cond_data = context.conditioning.load(conditioning_name)
        assert len(cond_data.conditionings) == 1
        sd3_conditioning = cond_data.conditionings[0]
        assert isinstance(sd3_conditioning, SD3ConditioningInfo)
        sd3_conditioning = sd3_conditioning.to(dtype=dtype, device=device)

        t5_embeds = sd3_conditioning.t5_embeds
        if t5_embeds is None:
            t5_embeds = torch.zeros(
                (1, SD3_T5_MAX_SEQ_LEN, joint_attention_dim),
                device=device,
                dtype=dtype,
            )

        clip_prompt_embeds = torch.cat([sd3_conditioning.clip_l_embeds, sd3_conditioning.clip_g_embeds], dim=-1)
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_embeds.shape[-1] - clip_prompt_embeds.shape[-1])
        )

        prompt_embeds = torch.cat([clip_prompt_embeds, t5_embeds], dim=-2)
        pooled_prompt_embeds = torch.cat(
            [sd3_conditioning.clip_l_pooled_embeds, sd3_conditioning.clip_g_pooled_embeds], dim=-1
        )

        return prompt_embeds, pooled_prompt_embeds

    def _get_noise(
        self,
        num_samples: int,
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
            num_samples,
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

    def _run_diffusion(
        self,
        context: InvocationContext,
    ):
        inference_dtype = TorchDevice.choose_torch_dtype()
        device = TorchDevice.choose_torch_device()

        transformer_info = context.models.load(self.transformer.transformer)

        # Load/process the conditioning data.
        # TODO(ryand): Make CFG optional.
        do_classifier_free_guidance = True
        pos_prompt_embeds, pos_pooled_prompt_embeds = self._load_text_conditioning(
            context=context,
            conditioning_name=self.positive_conditioning.conditioning_name,
            joint_attention_dim=transformer_info.model.config.joint_attention_dim,
            dtype=inference_dtype,
            device=device,
        )
        neg_prompt_embeds, neg_pooled_prompt_embeds = self._load_text_conditioning(
            context=context,
            conditioning_name=self.negative_conditioning.conditioning_name,
            joint_attention_dim=transformer_info.model.config.joint_attention_dim,
            dtype=inference_dtype,
            device=device,
        )
        # TODO(ryand): Support both sequential and batched CFG inference.
        prompt_embeds = torch.cat([neg_prompt_embeds, pos_prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([neg_pooled_prompt_embeds, pos_pooled_prompt_embeds], dim=0)

        # Prepare the scheduler.
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(num_inference_steps=self.steps, device=device)
        timesteps = scheduler.timesteps
        assert isinstance(timesteps, torch.Tensor)

        # Prepare the CFG scale list.
        cfg_scale = self._prepare_cfg_scale(len(timesteps))

        # Generate initial latent noise.
        num_channels_latents = transformer_info.model.config.in_channels
        assert isinstance(num_channels_latents, int)
        noise = self._get_noise(
            num_samples=1,
            num_channels_latents=num_channels_latents,
            height=self.height,
            width=self.width,
            dtype=inference_dtype,
            device=device,
            seed=self.seed,
        )
        latents: torch.Tensor = noise

        total_steps = len(timesteps)
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

        with transformer_info.model_on_device() as (cached_weights, transformer):
            assert isinstance(transformer, SD3Transformer2DModel)

            # 6. Denoising loop
            for step_idx, t in tqdm(list(enumerate(timesteps))):
                # Expand the latents if we are doing CFG.
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                # Expand the timestep to match the latent model input.
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

                # Apply CFG.
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg_scale[step_idx] * (noise_pred_cond - noise_pred_uncond)

                # Compute the previous noisy sample x_t -> x_t-1.
                latents_dtype = latents.dtype
                latents = scheduler.step(model_output=noise_pred, timestep=t, sample=latents, return_dict=False)[0]

                # TODO(ryand): This MPS dtype handling was copied from diffusers, I haven't tested to see if it's
                # needed.
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                step_callback(
                    PipelineIntermediateState(
                        step=step_idx + 1,
                        order=1,
                        total_steps=total_steps,
                        timestep=int(t),
                        latents=latents,
                    ),
                )

        return latents

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, BaseModelType.StableDiffusion3)

        return step_callback
