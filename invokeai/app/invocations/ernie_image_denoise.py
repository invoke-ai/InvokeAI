from contextlib import ExitStack
from typing import Optional

import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin

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
from invokeai.backend.model_manager.taxonomy import BaseModelType
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
                # Always generate noise on the same device and dtype, then cast, so a given seed
                # produces the same image regardless of the execution device (CUDA/ROCm/MPS/CPU).
                rand_device = "cpu"
                rand_dtype = torch.float32
                img = torch.randn(
                    (1, in_channels, latent_h, latent_w),
                    generator=torch.Generator(device=rand_device).manual_seed(self.seed),
                    device=rand_device,
                    dtype=rand_dtype,
                ).to(device=device, dtype=dtype)

            sigmas = sampling_utils.get_schedule(
                self.steps, denoising_start=self.denoising_start, denoising_end=self.denoising_end
            )
            timesteps = sigmas.tolist()
            cfg_scale = [self.guidance_scale] * (len(timesteps) - 1)

            scheduler = self._build_scheduler(context)

            def _step_callback(state: PipelineIntermediateState) -> None:
                context.util.sd_step_callback(state, BaseModelType.ErnieImage)

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
        # `LatentsOutput.build` assumes unpatched latents at 1/8 scale. ERNIE's latents are
        # 2x2-patchified on top of the 8x VAE downscale, so report the requested size directly.
        return LatentsOutput(
            latents=LatentsField(latents_name=name, seed=self.seed),
            width=self.width,
            height=self.height,
        )

    def _build_scheduler(self, context: InvocationContext) -> SchedulerMixin:
        """Instantiate the selected scheduler from the pipeline's own `scheduler/` config.

        The diffusers defaults (`shift=1.0`, `num_train_timesteps=1000`) are not necessarily what
        the checkpoint ships, and both feed the sampling math (`set_timesteps` applies `shift` to
        the sigmas, and `num_train_timesteps` scales the timestep fed to the transformer). Falling
        back to a default-constructed scheduler would silently diverge from the reference pipeline.
        """
        scheduler_cls = ERNIE_IMAGE_SCHEDULER_MAP[self.scheduler]
        config = context.models.get_config(self.transformer.transformer)
        # Model paths in the record store are relative to `models_path` for Invoke-managed models.
        model_path = (context.config.get().models_path / config.path).resolve()
        scheduler_dir = model_path / "scheduler"
        if not scheduler_dir.is_dir():
            context.logger.warning(
                f"No scheduler config found at {scheduler_dir}; falling back to {scheduler_cls.__name__} defaults."
            )
            return scheduler_cls()
        try:
            return scheduler_cls.from_pretrained(model_path, subfolder="scheduler")
        except Exception as e:
            context.logger.warning(
                f"Failed to load scheduler config from {scheduler_dir} ({e}); "
                f"falling back to {scheduler_cls.__name__} defaults."
            )
            return scheduler_cls()

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
