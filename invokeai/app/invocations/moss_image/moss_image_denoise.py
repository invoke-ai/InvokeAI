import copy
from contextlib import ExitStack
from typing import Any, Optional

import torch
import torchvision.transforms as tv_transforms
from torchvision.transforms.functional import resize as tv_resize

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    Input,
    InputField,
    LatentsField,
    MetadataField,
    MossImageConditioningField,
    WithMetadata,
)
from invokeai.app.invocations.metadata import (
    LatentsMetaOutput,
    build_denoise_metadata,
    serialize_lora_metadata,
)
from invokeai.app.invocations.model import TransformerField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.ernie_image.sampling_utils import patchify_latents
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.moss_image.denoise import denoise
from invokeai.backend.moss_image.sampling import get_moss_noise, get_moss_sigmas, to_model_space, to_raw_latent
from invokeai.backend.moss_image.working_memory import estimate_moss_image_denoise_working_memory
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import MossImageConditioningInfo
from invokeai.backend.util.devices import TorchDevice

# The FLUX.2 VAE's raw latent, which this node takes and returns: 32 channels at 1/8 of the image size.
MOSS_IMAGE_RAW_LATENT_CHANNELS = 32
MOSS_IMAGE_RAW_LATENT_SCALE = 8


@invocation(
    "moss_image_denoise",
    title="Denoise - MOSS-Image",
    tags=["image", "moss_image", "moss-image"],
    category="image",
    version="1.1.0",
    classification=Classification.Prototype,
)
class MossImageDenoiseInvocation(BaseInvocation, WithMetadata):
    """Run the MOSS-Image denoising loop.

    Takes and returns latents in the FLUX.2 VAE's raw layout, so the FLUX.2 VAE encode and decode nodes work on either
    side. The schedule is the model's own shifted Euler schedule; with CFG the negative branch skips the transformer
    blocks in `uncond_skip_layers`, as the reference pipeline does.
    """

    latents: Optional[LatentsField] = InputField(
        default=None,
        description="Starting latents for img2img or inpainting: a FLUX.2 VAE encode of the source image.",
        input=Input.Connection,
    )
    denoise_mask: Optional[DenoiseMaskField] = InputField(
        default=None,
        description=FieldDescriptions.denoise_mask,
        input=Input.Connection,
    )
    denoising_start: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.denoising_start)
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    transformer: TransformerField = InputField(
        description=FieldDescriptions.transformer, input=Input.Connection, title="Transformer"
    )
    positive_conditioning: MossImageConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: Optional[MossImageConditioningField] = InputField(
        default=None,
        description="Negative conditioning tensor. Required when cfg_scale > 1; an empty prompt is the reference default.",
        input=Input.Connection,
    )
    cfg_scale: float = InputField(default=3.0, ge=1.0, description=FieldDescriptions.cfg_scale, title="CFG Scale")
    # `multiple_of` must equal FeaturesFacet.dimension_grid.
    width: int = InputField(default=1024, gt=0, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, gt=0, multiple_of=16, description="Height of the generated image.")
    steps: int = InputField(default=50, gt=0, description=FieldDescriptions.steps)
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")
    vae: VAEField = InputField(
        description="The FLUX.2 VAE whose BatchNorm statistics define the latent space the transformer denoises in.",
        input=Input.Connection,
    )
    uncond_skip_layers: list[int] = InputField(
        default=[22], description=FieldDescriptions.moss_image_uncond_skip_layers, title="Uncond Skip Layers"
    )

    def _build_metadata(self) -> MetadataField:
        updates: dict[str, Any] = {
            "generation_mode": (
                "moss_image_inpaint"
                if getattr(self, "denoise_mask", None) is not None
                else ("moss_image_img2img" if getattr(self, "latents", None) is not None else "moss_image_txt2img")
            ),
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "denoising_start": self.denoising_start,
            "denoising_end": self.denoising_end,
            "seed": self.seed,
            "model": self.transformer.transformer,
            "moss_image_uncond_skip_layers": list(self.uncond_skip_layers),
        }
        if getattr(self, "vae", None) is not None:
            updates["vae"] = self.vae.vae
        loras = serialize_lora_metadata(getattr(self.transformer, "loras", None))
        if loras:
            updates["loras"] = loras

        return build_denoise_metadata(getattr(self, "metadata", None), updates)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsMetaOutput:
        latents = self._run_diffusion(context)
        name = context.tensors.save(tensor=latents)
        return LatentsMetaOutput.build(
            latents_name=name,
            latents=latents,
            seed=None,
            metadata=self._build_metadata(),
            width=self.width,
            height=self.height,
        )

    def _run_diffusion(self, context: InvocationContext) -> torch.Tensor:
        use_cfg = self.cfg_scale > 1.0
        if use_cfg and self.negative_conditioning is None:
            raise ValueError(
                "MOSS-Image needs negative conditioning when cfg_scale > 1. Connect a negative prompt (an empty one is "
                "the reference default), or set cfg_scale to 1."
            )

        init_latents = self._load_init_latents(context)
        if init_latents is None:
            if self.denoise_mask is not None:
                raise ValueError("A denoise mask needs the starting latents it inpaints. Connect latents.")
            if self.denoising_start > 0.0:
                raise ValueError("denoising_start must be 0 when no starting latents are connected.")

        # Nothing to step: hand the starting latents back as they came. The schedule refuses an empty window.
        if self.denoising_start >= self.denoising_end:
            if init_latents is None:
                raise ValueError(
                    f"The denoising window [{self.denoising_start}, {self.denoising_end}] is empty and there are no "
                    "starting latents to return."
                )
            return init_latents

        transformer_info = context.models.load(self.transformer.transformer)
        transformer_config = transformer_info.model.config
        skip_layers = self._validated_skip_layers(int(transformer_config.num_layers)) if use_cfg else []

        device = transformer_info.compute_device
        dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        bn = self._load_vae_bn(context)
        sigmas = get_moss_sigmas(self.steps, self.denoising_start, self.denoising_end)

        # Drawn on the CPU and already in model space: the reference's noise, the same for a seed on every device.
        noise = get_moss_noise(self.seed, self.height, self.width).to(device)
        inpaint_extension: Optional[RectifiedFlowInpaintExtension] = None
        if init_latents is None:
            x = noise
        else:
            init_model_space = to_model_space(init_latents, bn).to(device)
            x = sigmas[0] * noise + (1.0 - sigmas[0]) * init_model_space
            inpaint_mask = self._prep_inpaint_mask(context, device)
            if inpaint_mask is not None:
                inpaint_extension = RectifiedFlowInpaintExtension(
                    init_latents=init_model_space, inpaint_mask=inpaint_mask, noise=noise
                )

        positive = self._load_conditioning(context, self.positive_conditioning, device, dtype)
        negative = None
        if use_cfg:
            assert self.negative_conditioning is not None
            negative = self._load_conditioning(context, self.negative_conditioning, device, dtype)
        text_dim = int(transformer_config.text_dim)
        for name, embeds in (("positive", positive), ("negative", negative)):
            if embeds is not None and embeds.shape[-1] != text_dim:
                raise ValueError(
                    f"The {name} conditioning is {embeds.shape[-1]} wide, but this MOSS-Image transformer reads "
                    f"{text_dim}-wide text: encode the prompt with the Qwen3.5 9B encoder."
                )

        # Read from the unlocked model, before the lock the reservation applies to.
        working_memory = estimate_moss_image_denoise_working_memory(
            transformer_info.model,
            height=self.height,
            width=self.width,
            text_tokens=max(e.shape[1] for e in (positive, negative) if e is not None),
            device=device,
            dtype=dtype,
        )

        def step_callback(state: PipelineIntermediateState) -> None:
            context.util.sd_step_callback(state, BaseModelType.MossImage)

        with ExitStack() as exit_stack:
            (_, transformer) = exit_stack.enter_context(transformer_info.model_on_device(working_memory=working_memory))
            result = denoise(
                transformer=transformer,
                latents=x,
                sigmas=sigmas,
                positive=positive,
                negative=negative,
                cfg_scale=self.cfg_scale,
                uncond_skip_layers=skip_layers,
                step_callback=step_callback,
                inpaint_extension=inpaint_extension,
            )

        return to_raw_latent(result, bn).cpu()

    def _load_init_latents(self, context: InvocationContext) -> Optional[torch.Tensor]:
        """The starting latents in the FLUX.2 raw layout, float32 on the CPU, checked against the requested size."""
        if self.latents is None:
            return None
        latents = context.tensors.load(self.latents.latents_name)
        expected = (
            1,
            MOSS_IMAGE_RAW_LATENT_CHANNELS,
            self.height // MOSS_IMAGE_RAW_LATENT_SCALE,
            self.width // MOSS_IMAGE_RAW_LATENT_SCALE,
        )
        if tuple(latents.shape) != expected:
            raise ValueError(
                f"The starting latents have shape {tuple(latents.shape)}, but a {self.width}x{self.height} MOSS-Image "
                f"generation takes FLUX.2 VAE latents of shape {expected}. Encode the image with the FLUX.2 VAE at "
                "the generation size."
            )
        return latents.to(device="cpu", dtype=torch.float32)

    def _validated_skip_layers(self, num_layers: int) -> list[int]:
        """`uncond_skip_layers`, refused with the transformer's real depth before any weight moves to the device."""
        out_of_range = sorted({index for index in self.uncond_skip_layers if not 0 <= index < num_layers})
        if out_of_range:
            raise ValueError(
                f"uncond_skip_layers {out_of_range} out of range: this MOSS-Image transformer has {num_layers} blocks, "
                f"numbered 0 to {num_layers - 1}."
            )
        return list(self.uncond_skip_layers)

    def _load_vae_bn(self, context: InvocationContext) -> torch.nn.BatchNorm2d:
        """A float32 CPU copy of the VAE's BatchNorm, whose running statistics map the raw latent to model space."""
        with context.models.load(self.vae.vae).model_in_ram() as vae:
            bn = getattr(vae, "bn", None)
            if not isinstance(bn, torch.nn.BatchNorm2d) or bn.running_mean is None or bn.running_var is None:
                raise ValueError(
                    f"MOSS-Image needs a FLUX.2 VAE, whose BatchNorm statistics define its latent space; got a "
                    f"{type(vae).__name__} without them."
                )
            return copy.deepcopy(bn).to(device="cpu", dtype=torch.float32)

    def _prep_inpaint_mask(self, context: InvocationContext, device: torch.device) -> Optional[torch.Tensor]:
        """The inpaint mask in model space, 1 where the image is regenerated.

        Resized at the raw latent's resolution and packed like a latent, as FLUX.2 does, so each of a patch's 128
        channels follows the pixel it came from. Not normalized: it is a weight, not a latent.
        """
        if self.denoise_mask is None:
            return None
        mask = 1.0 - context.tensors.load(self.denoise_mask.mask_name).float()
        mask = tv_resize(
            img=mask,
            size=[self.height // MOSS_IMAGE_RAW_LATENT_SCALE, self.width // MOSS_IMAGE_RAW_LATENT_SCALE],
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
            antialias=False,
        )
        mask = mask.expand(1, MOSS_IMAGE_RAW_LATENT_CHANNELS, *mask.shape[-2:])
        return patchify_latents(mask.contiguous()).to(device)

    def _load_conditioning(
        self,
        context: InvocationContext,
        field: MossImageConditioningField,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """One prompt's embeddings as a batch of one, (1, L, text_dim)."""
        conditioning = context.conditioning.load(field.conditioning_name)
        info = conditioning.conditionings[0]
        if not isinstance(info, MossImageConditioningInfo):
            raise TypeError(f"Expected MossImageConditioningInfo, got {type(info).__name__}.")
        return info.prompt_embeds.to(device=device, dtype=dtype).unsqueeze(0)
