from contextlib import ExitStack
from typing import Any, Callable, Optional

import torch
import torchvision.transforms as tv_transforms
from torchvision.transforms.functional import resize as tv_resize

from invokeai.app.invocations.baseinvocation import BaseInvocation, Classification, invocation
from invokeai.app.invocations.fields import (
    DenoiseMaskField,
    FieldDescriptions,
    Flux3ConditioningField,
    FluxKontextConditioningField,
    Input,
    InputField,
    LatentsField,
    MetadataField,
    WithMetadata,
)
from invokeai.app.invocations.metadata import (
    LatentsMetaOutput,
    build_denoise_metadata,
    serialize_lora_metadata,
)
from invokeai.app.invocations.model import TransformerField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.flux.sampling_utils import clip_timestep_schedule_fractional
from invokeai.backend.flux2.latent_norm import bn_denormalize, bn_normalize, read_vae_bn_stats
from invokeai.backend.flux2.sampling_utils import pack_flux2, unpack_flux2
from invokeai.backend.flux3.denoise import denoise
from invokeai.backend.flux3.denoise_working_memory import estimate_flux3_denoise_working_memory
from invokeai.backend.flux3.ref_images import FLUX3_MAX_REFERENCE_IMAGES, encode_flux3_references
from invokeai.backend.flux3.sampling import (
    get_noise_flux3,
    get_schedule_flux3,
    image_position_ids,
    redensify_schedule_flux3,
    text_position_ids,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, Flux3VariantType, ModelType
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Flux3ConditioningInfo
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "flux3_denoise",
    title="Denoise - FLUX.3",
    tags=["image", "flux3", "flux", "denoise"],
    category="latents",
    version="1.0.0",
    classification=Classification.Prototype,
)
class Flux3DenoiseInvocation(BaseInvocation, WithMetadata):
    """Run the denoising process with a FLUX.3 Image transformer (Klein or Klein Base).

    Klein is step-distilled (a flow map) and samples without CFG; Klein Base takes CFG with a negative
    conditioning. Reference images condition an edit: describe the result in the prompt and refer to them as
    <ref_image_0>, <ref_image_1>, ... in the order they are connected.
    """

    latents: Optional[LatentsField] = InputField(
        default=None, description=FieldDescriptions.latents, input=Input.Connection
    )
    denoise_mask: Optional[DenoiseMaskField] = InputField(
        default=None, description=FieldDescriptions.denoise_mask, input=Input.Connection
    )
    denoising_start: float = InputField(default=0.0, ge=0, le=1, description=FieldDescriptions.denoising_start)
    denoising_end: float = InputField(default=1.0, ge=0, le=1, description=FieldDescriptions.denoising_end)
    add_noise: bool = InputField(default=True, description="Add noise based on denoising start.")
    transformer: TransformerField = InputField(
        description=FieldDescriptions.flux3_model, input=Input.Connection, title="Transformer"
    )
    positive_conditioning: Flux3ConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: Optional[Flux3ConditioningField] = InputField(
        default=None,
        description="Negative conditioning. Used by Klein Base when CFG Scale is above 1.",
        input=Input.Connection,
    )
    cfg_scale: float = InputField(
        default=1.0,
        ge=1.0,
        description="Classifier-free guidance scale for Klein Base (3.5 recommended); 1 disables it. Klein takes "
        "no CFG.",
        title="CFG Scale",
    )
    # `multiple_of` must equal FeaturesFacet.dimension_grid.
    width: int = InputField(default=1024, gt=0, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, gt=0, multiple_of=16, description="Height of the generated image.")
    steps: int = InputField(
        default=16, gt=0, description="Number of denoising steps: 16 for Klein (4-16 work), 50 for Klein Base."
    )
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")
    vae: VAEField = InputField(
        description="The FLUX.2 VAE: its latent statistics, and the encoder for reference images.",
        input=Input.Connection,
    )
    kontext_conditioning: FluxKontextConditioningField | list[FluxKontextConditioningField] | None = InputField(
        default=None,
        description="Reference images to edit or combine (up to 10).",
        input=Input.Connection,
        title="Reference Images",
    )

    def _build_metadata(self) -> MetadataField:
        updates: dict[str, Any] = {
            "generation_mode": (
                "flux3_inpaint"
                if getattr(self, "denoise_mask", None) is not None
                else ("flux3_img2img" if getattr(self, "latents", None) is not None else "flux3_txt2img")
            ),
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "denoising_start": self.denoising_start,
            "denoising_end": self.denoising_end,
            "add_noise": self.add_noise,
            "seed": self.seed,
            "model": self.transformer.transformer,
            "vae": self.vae.vae,
        }
        loras = serialize_lora_metadata(getattr(self.transformer, "loras", None))
        if loras:
            updates["loras"] = loras
        return build_denoise_metadata(getattr(self, "metadata", None), updates)

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> LatentsMetaOutput:
        latents = self._run_diffusion(context).detach().to("cpu")
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
        device = TorchDevice.choose_torch_device()
        # The dtype the loaders build the transformer in, so the activations match its weights.
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(device)

        transformer_config = context.models.get_config(self.transformer.transformer)
        if transformer_config.base is not BaseModelType.Flux3 or transformer_config.type is not ModelType.Main:
            raise ValueError(f"'{transformer_config.name}' is not a FLUX.3 transformer.")

        references = self._reference_fields()
        if len(references) > FLUX3_MAX_REFERENCE_IMAGES:
            raise ValueError(f"FLUX.3 takes at most {FLUX3_MAX_REFERENCE_IMAGES} reference images.")
        use_cfg = self.cfg_scale > 1.0
        # Checked before anything loads; `denoise` refuses the combination too, keyed on the loaded weights.
        if use_cfg and getattr(transformer_config, "variant", None) is Flux3VariantType.Klein:
            raise ValueError(
                "FLUX.3 Klein is step-distilled and samples without CFG. Set CFG Scale to 1, or use FLUX.3 Klein "
                "Base for CFG and a negative prompt."
            )
        if use_cfg and self.negative_conditioning is None:
            raise ValueError("A CFG Scale above 1 needs a negative conditioning.")

        vae_info = context.models.load(self.vae.vae)
        with vae_info.model_on_device() as (_, vae):
            bn_stats = read_vae_bn_stats(vae)
        if bn_stats is None:
            raise ValueError("FLUX.3 needs the FLUX.2 VAE, whose latent statistics this VAE does not carry.")
        bn_mean, bn_std = bn_stats

        txt = self._load_conditioning(context, self.positive_conditioning, device, compute_dtype)
        neg_txt = (
            self._load_conditioning(context, self.negative_conditioning, device, compute_dtype)
            if use_cfg and self.negative_conditioning is not None
            else None
        )

        noise = get_noise_flux3(self.height, self.width, self.seed).to(device)
        init_latents = None
        if self.latents is not None:
            init_latents = context.tensors.load(self.latents.latents_name).to(device=device, dtype=torch.float32)
            if init_latents.shape[-2:] != noise.shape[-2:]:
                raise ValueError(
                    f"The input latents are {init_latents.shape[-1] * 8}x{init_latents.shape[-2] * 8} pixels, but "
                    f"the node generates {self.width}x{self.height}."
                )
        elif self.denoising_start > 1e-5:
            raise ValueError("denoising_start must be 0 when no input latents are given.")

        # The schedule is clipped by sigma value to the denoising window. A window that starts late (img2img at
        # low strength) is re-spaced so it still takes the requested number of steps (see
        # redensify_schedule_flux3); one that starts at 0 is the head of a split chain and keeps its own steps.
        timesteps = get_schedule_flux3(self.steps)
        timesteps = clip_timestep_schedule_fractional(timesteps, self.denoising_start, self.denoising_end)
        if self.denoising_start > 0:
            timesteps = redensify_schedule_flux3(timesteps, self.steps)

        # The transformer works on packed tokens in the VAE's BN-normalized space. Only the init latents are
        # normalized: the noise is already N(0, 1), the distribution the transformer expects there, so the
        # img2img blend is built from the normalized init latents rather than normalizing the mixture (#8964).
        noise_packed = pack_flux2(noise)
        init_packed = bn_normalize(pack_flux2(init_latents), bn_mean, bn_std) if init_latents is not None else None
        if init_packed is None:
            x = noise_packed
        elif self.add_noise:
            x = timesteps[0] * noise_packed + (1.0 - timesteps[0]) * init_packed
        else:
            x = init_packed

        if len(timesteps) <= 1:
            return self._to_latents(x, bn_mean, bn_std)

        inpaint_extension = None
        if self.denoise_mask is not None:
            if init_packed is None:
                raise ValueError("Inpainting needs input latents.")
            inpaint_extension = RectifiedFlowInpaintExtension(
                init_latents=init_packed,
                inpaint_mask=pack_flux2(self._prep_inpaint_mask(context, noise)),
                noise=noise_packed,
            )

        ref_tokens, ref_ids = None, None
        if references:
            ref_tokens, ref_ids = encode_flux3_references(
                images=[context.images.get_pil(field.image.image_name) for field in references],
                vae_info=vae_info,
                bn_mean=bn_mean,
                bn_std=bn_std,
                device=device,
                dtype=compute_dtype,
            )

        batch_size = x.shape[0]
        txt = txt.expand(batch_size, -1, -1)
        neg_txt = neg_txt.expand(batch_size, -1, -1) if neg_txt is not None else None
        ref_tokens = ref_tokens.expand(batch_size, -1, -1) if ref_tokens is not None else None
        packed_h, packed_w = self.height // 16, self.width // 16

        with ExitStack() as exit_stack:
            transformer_info = context.models.load(self.transformer.transformer)
            # Ask the model cache to keep free what one transformer forward needs. Reference images are the dominant
            # term: their tokens join the attended sequence, so ten 1 MP references add ten times the tokens of a
            # 1 MP image. Priced on the unlocked model, whose GGUF or fp8-stored layers add a transient per forward,
            # before the VRAM lock the reservation applies to. CFG runs its two forwards one after the other, so only
            # the longer caption counts.
            working_memory = estimate_flux3_denoise_working_memory(
                transformer_info.model,
                image_seq_len=packed_h * packed_w,
                ref_image_seq_len=ref_tokens.shape[1] if ref_tokens is not None else 0,
                text_seq_len=max(txt.shape[1], neg_txt.shape[1] if neg_txt is not None else 0),
                batch_size=batch_size,
                device=device,
                dtype=compute_dtype,
            )
            (_, transformer) = exit_stack.enter_context(transformer_info.model_on_device(working_memory=working_memory))
            x = denoise(
                model=transformer,
                img=x,
                img_ids=image_position_ids(packed_h, packed_w, l_coord=0, device=device),
                txt=txt,
                txt_ids=text_position_ids(txt.shape[1], device=device),
                timesteps=timesteps,
                step_callback=self._build_step_callback(context),
                flow_map=bool(transformer.config.flow_map),
                compute_dtype=compute_dtype,
                cfg_scale=self.cfg_scale,
                neg_txt=neg_txt,
                neg_txt_ids=text_position_ids(neg_txt.shape[1], device=device) if neg_txt is not None else None,
                ref_tokens=ref_tokens,
                ref_ids=ref_ids,
                inpaint_extension=inpaint_extension,
            )

        return self._to_latents(x, bn_mean, bn_std)

    def _to_latents(self, x: torch.Tensor, bn_mean: torch.Tensor, bn_std: torch.Tensor) -> torch.Tensor:
        """Packed, normalized tokens -> the unpacked, un-normalized latents FLUX.2-VAE nodes take."""
        return unpack_flux2(bn_denormalize(x.float(), bn_mean, bn_std), self.height, self.width)

    def _reference_fields(self) -> list[FluxKontextConditioningField]:
        if self.kontext_conditioning is None:
            return []
        if isinstance(self.kontext_conditioning, list):
            return self.kontext_conditioning
        return [self.kontext_conditioning]

    @staticmethod
    def _load_conditioning(
        context: InvocationContext, field: Flux3ConditioningField, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        cond_data = context.conditioning.load(field.conditioning_name)
        if len(cond_data.conditionings) != 1 or not isinstance(cond_data.conditionings[0], Flux3ConditioningInfo):
            raise ValueError("The conditioning is not a FLUX.3 conditioning. Connect a 'Prompt - FLUX.3' node.")
        return cond_data.conditionings[0].to(device=device, dtype=dtype).prompt_embeds

    def _prep_inpaint_mask(self, context: InvocationContext, latents: torch.Tensor) -> torch.Tensor:
        """The denoise mask, inverted to "1 = regenerate" and resized to the unpacked latent grid."""
        assert self.denoise_mask is not None
        mask = 1.0 - context.tensors.load(self.denoise_mask.mask_name)
        _, _, latent_height, latent_width = latents.shape
        mask = tv_resize(
            img=mask,
            size=[latent_height, latent_width],
            interpolation=tv_transforms.InterpolationMode.BILINEAR,
            antialias=False,
        )
        return mask.to(device=latents.device, dtype=latents.dtype).expand_as(latents)

    def _build_step_callback(self, context: InvocationContext) -> Callable[[PipelineIntermediateState], None]:
        def step_callback(state: PipelineIntermediateState) -> None:
            # The x0 estimate in the normalized space, unpacked: the space FLUX2_32's preview factors expect.
            state.latents = unpack_flux2(state.latents.float(), self.height, self.width).squeeze()
            context.util.sd_step_callback(state, BaseModelType.Flux3)

        return step_callback
