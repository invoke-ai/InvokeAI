from contextlib import ExitStack
from typing import Optional

import torch

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    ErnieImageConditioningField,
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
)
from invokeai.app.invocations.model import TransformerField
from invokeai.app.invocations.primitives import LatentsOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ernie_image import sampling_utils
from invokeai.backend.ernie_image.denoise import denoise as ernie_denoise
from invokeai.backend.flux.schedulers import (
    ERNIE_IMAGE_SCHEDULER_LABELS,
    ERNIE_IMAGE_SCHEDULER_MAP,
    ERNIE_IMAGE_SCHEDULER_NAME_VALUES,
)
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ErnieImageConditioningInfo
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "ernie_image_denoise",
    title="Denoise - ERNIE-Image",
    tags=["latents", "denoise", "ernie-image"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class ErnieImageDenoiseInvocation(BaseInvocation):
    """Run the ERNIE-Image denoising loop and emit packed latents."""

    transformer: TransformerField = InputField(
        description=FieldDescriptions.transformer, input=Input.Connection, title="Transformer"
    )
    positive_conditioning: ErnieImageConditioningField = InputField(
        description="Positive prompt conditioning", input=Input.Connection
    )
    negative_conditioning: Optional[ErnieImageConditioningField] = InputField(
        default=None,
        description="Negative prompt conditioning (required when guidance_scale != 1.0)",
        input=Input.Connection,
    )
    latents: Optional[LatentsField] = InputField(
        default=None,
        description="Optional starting latents for img2img (must already be VAE-encoded, BN-normalized, and patchified).",
        input=Input.Connection,
    )

    width: int = InputField(default=1024, multiple_of=16, description="Generation width.")
    height: int = InputField(default=1024, multiple_of=16, description="Generation height.")
    steps: int = InputField(default=50, gt=0, description="Denoising steps. Use 8 for ERNIE-Image-Turbo.")
    guidance_scale: float = InputField(
        default=4.0,
        ge=1.0,
        description="Classifier-free guidance scale. 4.0 for ERNIE-Image, 1.0 (no CFG) for Turbo.",
    )
    denoising_start: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.denoising_start)
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    seed: int = InputField(default=0, description="Random seed for noise generation.")
    scheduler: ERNIE_IMAGE_SCHEDULER_NAME_VALUES = InputField(
        default="euler",
        description="Scheduler used during denoising.",
        ui_choice_labels=ERNIE_IMAGE_SCHEDULER_LABELS,
    )

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsOutput:
        device = TorchDevice.choose_torch_device()
        dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        pos_info = self._load_conditioning(context, self.positive_conditioning, dtype, device)
        neg_info: Optional[ErnieImageConditioningInfo] = None
        do_cfg = self.guidance_scale > 1.0
        if do_cfg:
            if self.negative_conditioning is None:
                raise ValueError("Negative conditioning is required when guidance_scale > 1.0")
            neg_info = self._load_conditioning(context, self.negative_conditioning, dtype, device)

        transformer_info = context.models.load(self.transformer.transformer)

        with ExitStack() as exit_stack:
            (_, transformer) = exit_stack.enter_context(transformer_info.model_on_device())

            text_in_dim = int(transformer.config.text_in_dim)
            in_channels = int(transformer.config.in_channels)  # 128 -- already patched

            text_bth, text_lens = sampling_utils.pad_text(
                [pos_info.prompt_embeds], device=device, dtype=dtype, text_in_dim=text_in_dim
            )
            neg_text_bth = neg_text_lens = None
            if neg_info is not None:
                neg_text_bth, neg_text_lens = sampling_utils.pad_text(
                    [neg_info.prompt_embeds], device=device, dtype=dtype, text_in_dim=text_in_dim
                )

            latent_h = self.height // sampling_utils.VAE_SCALE_FACTOR
            latent_w = self.width // sampling_utils.VAE_SCALE_FACTOR

            if self.latents is not None:
                img = context.tensors.load(self.latents.latents_name).to(device=device, dtype=dtype)
                if img.shape[1] != in_channels:
                    raise ValueError(
                        f"Input latents have {img.shape[1]} channels but transformer expects {in_channels}. "
                        "Pass already-patched latents (use the ERNIE-Image VAE encode node)."
                    )
            else:
                generator = torch.Generator(device=device).manual_seed(self.seed)
                img = torch.randn(
                    (1, in_channels, latent_h, latent_w),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )

            sigmas = sampling_utils.get_schedule(
                self.steps, denoising_start=self.denoising_start, denoising_end=self.denoising_end
            )
            timesteps = sigmas.tolist()
            cfg_scale = [self.guidance_scale] * (len(timesteps) - 1)

            scheduler_cls = ERNIE_IMAGE_SCHEDULER_MAP[self.scheduler]
            scheduler = scheduler_cls()

            def _step_callback(state: PipelineIntermediateState) -> None:
                context.util.signal_progress(f"ERNIE-Image step {state.step}/{state.total_steps}")

            img = ernie_denoise(
                model=transformer,
                img=img,
                text_bth=text_bth,
                text_lens=text_lens,
                timesteps=timesteps,
                step_callback=_step_callback,
                cfg_scale=cfg_scale,
                neg_text_bth=neg_text_bth,
                neg_text_lens=neg_text_lens,
                scheduler=scheduler,
            )

        latents = img.detach().to("cpu")
        name = context.tensors.save(tensor=latents)
        return LatentsOutput.build(latents_name=name, latents=latents, seed=self.seed)

    def _load_conditioning(
        self,
        context: InvocationContext,
        cond_field: ErnieImageConditioningField,
        dtype: torch.dtype,
        device: torch.device,
    ) -> ErnieImageConditioningInfo:
        cond_data = context.conditioning.load(cond_field.conditioning_name)
        if len(cond_data.conditionings) != 1:
            raise ValueError(f"Expected exactly one conditioning, got {len(cond_data.conditionings)}")
        info = cond_data.conditionings[0]
        if not isinstance(info, ErnieImageConditioningInfo):
            raise TypeError(
                f"Expected ErnieImageConditioningInfo, got {type(info).__name__}. "
                "Connect an ERNIE-Image text encoder to this input."
            )
        return info.to(device=device, dtype=dtype)
