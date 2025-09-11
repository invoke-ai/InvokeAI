# Copyright (c) 2024, Brandon W. Rising and the InvokeAI Development Team
"""Qwen-Image denoising invocation using diffusers pipeline."""

from typing import Optional

import torch
from diffusers.pipelines import QwenImagePipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import (
    FieldDescriptions,
    Input,
    InputField,
    UIType,
    DenoiseMaskField,
    LatentsField,
    QwenImageConditioningField,
    WithBoard,
    WithMetadata,
)
from invokeai.app.invocations.model import ModelIdentifierField, Qwen2_5VLField, TransformerField, VAEField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "qwen_image_denoise",
    title="Qwen-Image Denoise",
    tags=["image", "qwen"],
    category="image",
    version="1.0.0",
)
class QwenImageDenoiseInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Run text-to-image generation with a Qwen-Image diffusion model."""

    # Model components
    transformer: TransformerField = InputField(
        description=FieldDescriptions.transformer,
        input=Input.Connection,
        title="Transformer",
    )
    
    vae: VAEField = InputField(
        description=FieldDescriptions.vae,
        input=Input.Connection,
        title="VAE",
    )
    
    # Qwen2.5-VL text encoder (tokenizer + text_encoder)
    qwen2_5_vl: Qwen2_5VLField = InputField(
        description="Qwen2.5-VL vision-language model",
        input=Input.Connection,
        title="Qwen2.5-VL",
    )
    # Optional scheduler passed from model loader. If not provided, a default FlowMatch scheduler is used.
    scheduler_model: Optional[ModelIdentifierField] = InputField(
        default=None,
        description="Qwen-Image scheduler (optional; loaded from model if omitted)",
        input=Input.Connection,
        ui_type=UIType.Scheduler,
        title="Scheduler",
    )
    
    # Text conditioning
    positive_conditioning: QwenImageConditioningField = InputField(
        description=FieldDescriptions.positive_cond, input=Input.Connection
    )
    negative_conditioning: Optional[QwenImageConditioningField] = InputField(
        default=None,
        description=FieldDescriptions.negative_cond,
        input=Input.Connection,
        ui_hidden=True,
    )
    
    # Generation parameters
    width: int = InputField(default=1024, multiple_of=16, description="Width of the generated image.")
    height: int = InputField(default=1024, multiple_of=16, description="Height of the generated image.")
    num_inference_steps: int = InputField(
        default=50, gt=0, description="Number of denoising steps."
    )
    guidance_scale: float = InputField(
        default=7.5, gt=1.0, description="Guidance-distilled scale (if model supports)."
    )
    true_cfg_scale: float = InputField(
        default=4.0, ge=1.0, description="Classifier-free guidance scale (true CFG)."
    )
    # Optional initial latents for img2img
    latents: Optional[LatentsField] = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
        title="Init Latents",
    )
    # Optional inpaint mask
    denoise_mask: Optional[DenoiseMaskField] = InputField(
        default=None,
        description=FieldDescriptions.denoise_mask,
        input=Input.Connection,
        title="Inpaint Mask",
    )
    # Optional initial latents for img2img
    latents: Optional[LatentsField] = InputField(
        default=None,
        description=FieldDescriptions.latents,
        input=Input.Connection,
        title="Init Latents",
    )
    seed: int = InputField(default=0, description="Randomness seed for reproducibility.")

    @torch.no_grad()
    def invoke(self, context: InvocationContext) -> ImageOutput:
        """Generate image using Qwen-Image pipeline (Diffusers)."""

        device = TorchDevice.choose_torch_device()
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Load components
        with context.models.load(self.transformer.transformer) as transformer_info, \
             context.models.load(self.vae.vae) as vae_info, \
             context.models.load(self.qwen2_5_vl.tokenizer) as tokenizer_info, \
             context.models.load(self.qwen2_5_vl.text_encoder) as text_encoder_info:

            # Load conditioning data (prompt)
            conditioning_data = context.conditioning.load(self.positive_conditioning.conditioning_name)
            assert len(conditioning_data.conditionings) == 1
            conditioning_info = conditioning_data.conditionings[0]
            prompt = getattr(conditioning_info, 'prompt', "A high-quality image")

            negative_prompt = None
            if self.negative_conditioning is not None:
                neg_data = context.conditioning.load(self.negative_conditioning.conditioning_name)
                if len(neg_data.conditionings) > 0:
                    neg_info = neg_data.conditionings[0]
                    negative_prompt = getattr(neg_info, 'prompt', None)

            try:
                # Build pipeline from loaded components
                # Load scheduler from model if provided; otherwise use default
                scheduler = None
                if self.scheduler_model is not None:
                    try:
                        with context.models.load(self.scheduler_model) as scheduler_info:
                            scheduler = scheduler_info.model
                    except Exception:
                        scheduler = None
                if scheduler is None:
                    scheduler = FlowMatchEulerDiscreteScheduler()

                pipe = QwenImagePipeline(
                    scheduler=scheduler,
                    transformer=transformer_info.model.to(device=device, dtype=dtype),
                    vae=vae_info.model.to(device=device, dtype=dtype),
                    tokenizer=tokenizer_info.model,  # hf tokenizer
                    text_encoder=text_encoder_info.model.to(device=device, dtype=dtype),
                )

                # Reproducibility
                generator = torch.Generator(device=device)
                generator.manual_seed(self.seed)

                # Prefer embeddings from conditioning if available
                prompt_embeds = getattr(conditioning_info, 'text_embeds', None)
                prompt_embeds_mask = getattr(conditioning_info, 'text_embeds_mask', None)
                neg_embeds = None
                neg_mask = None
                if self.negative_conditioning is not None:
                    try:
                        neg_data = context.conditioning.load(self.negative_conditioning.conditioning_name)
                        if len(neg_data.conditionings) > 0:
                            neg_info = neg_data.conditionings[0]
                            neg_embeds = getattr(neg_info, 'text_embeds', None)
                            neg_mask = getattr(neg_info, 'text_embeds_mask', None)
                    except Exception:
                        neg_embeds = None

                call_kwargs = dict(
                    width=self.width,
                    height=self.height,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    true_cfg_scale=self.true_cfg_scale,
                    generator=generator,
                    output_type="pil",
                )

                # Load init latents if provided and pack for the pipeline
                init_latents = None
                if self.latents is not None:
                    try:
                        init_latents = context.tensors.load(self.latents.latents_name)
                        # Ensure BCHW; some VAEs output [B, C, 1, H, W]
                        if init_latents.ndim == 5 and init_latents.shape[2] == 1:
                            init_latents = init_latents[:, :, 0, :, :]
                        # Pack latents to patches as expected by Qwen pipeline
                        b, c, h, w = init_latents.shape
                        init_latents = init_latents.view(b, 1, c, h, w)
                        # Pack to [B, (H/2)*(W/2), C*4]
                        init_latents = init_latents.view(b, 1, c, h // 2, 2, w // 2, 2)
                        init_latents = init_latents.permute(0, 3, 5, 2, 4, 6).contiguous()
                        init_latents = init_latents.view(b, (h // 2) * (w // 2), c * 4)
                    except Exception:
                        init_latents = None

                # If doing inpaint: attempt stepwise masked denoising with the transformer
                if self.denoise_mask is not None:
                    try:
                        # Helper: pack/unpack latents (ported from Qwen pipeline logic)
                        def pack_latents(x: torch.Tensor) -> torch.Tensor:
                            # x: [B, C, H, W] -> [B, (H/2)*(W/2), C*4]
                            b, c, h, w = x.shape
                            x = x.view(b, c, h // 2, 2, w // 2, 2)
                            x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
                            x = x.view(b, (h // 2) * (w // 2), c * 4)
                            return x

                        def unpack_latents(x: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
                            # Inverse of pack
                            b, num_patches, channels = x.shape
                            h = 2 * (int(height) // (vae_scale_factor * 2))
                            w = 2 * (int(width) // (vae_scale_factor * 2))
                            x = x.view(b, h // 2, w // 2, channels // 4, 2, 2)
                            x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
                            x = x.view(b, channels // 4 // (2 * 2), 1, h, w)
                            return x

                        # Prepare embeddings & masks
                        if prompt_embeds is None or prompt_embeds_mask is None:
                            raise ValueError("Qwen inpainting requires embeddings and masks")

                        # Load init latents (required for inpaint)
                        if self.latents is None:
                            raise ValueError("Inpainting requires initial latents")
                        init_unpacked = context.tensors.load(self.latents.latents_name)  # [B, C, H, W] or [B,C,1,H,W]
                        if init_unpacked.ndim == 5 and init_unpacked.shape[2] == 1:
                            init_unpacked = init_unpacked[:, :, 0, :, :]
                        init_unpacked = init_unpacked.to(device=device, dtype=dtype)
                        init_packed = pack_latents(init_unpacked)

                        # Prepare noise in packed shape
                        b, npatches, c4 = init_packed.shape
                        gen = torch.Generator(device=device)
                        gen.manual_seed(self.seed)
                        noise = torch.randn((b, npatches, c4), device=device, dtype=dtype, generator=gen)

                        # Prepare sigma/timestep schedules via scheduler with mu shift
                        import numpy as np
                        sigmas_np = np.linspace(1.0, 1.0 / self.num_inference_steps, self.num_inference_steps)
                        def _cfg(key: str, default):
                            try:
                                return scheduler.config.get(key, default)  # type: ignore[attr-defined]
                            except Exception:
                                return default
                        base_seq_len = _cfg('base_image_seq_len', 256)
                        max_seq_len = _cfg('max_image_seq_len', 4096)
                        base_shift = _cfg('base_shift', 0.5)
                        max_shift = _cfg('max_shift', 1.15)
                        m = (max_shift - base_shift) / max(1, (max_seq_len - base_seq_len))
                        b_mu = base_shift - m * base_seq_len
                        mu = npatches * m + b_mu
                        scheduler.set_timesteps(sigmas=sigmas_np.tolist(), device=device, mu=mu)  # type: ignore[arg-type]
                        timesteps = scheduler.timesteps  # type: ignore[attr-defined]
                        sigmas = torch.tensor(sigmas_np, device=device, dtype=dtype)

                        # Initialize latents by noising init latents for the first sigma
                        s0 = sigmas[0].item()
                        latents = s0 * noise + (1.0 - s0) * init_packed

                        # Prepare inpaint mask in packed format
                        mask_tensor = context.tensors.load(self.denoise_mask.mask_name)  # [1,H,W] or [B,1,H,W]
                        mask_tensor = mask_tensor.to(device=device, dtype=dtype)
                        # invert semantics: mask=1 preserve -> we want denoise area =1, so invert later when merging
                        # Resize to latent spatial dims before packing
                        _, _, h_u, w_u = init_unpacked.shape
                        import torchvision.transforms.functional as tvf
                        mask_resized = tvf.resize(mask_tensor, [h_u, w_u], antialias=False)
                        # Expand to BCHW like latents
                        if mask_resized.ndim == 3:
                            mask_resized = mask_resized.unsqueeze(0)
                        mask_resized = mask_resized.expand_as(init_unpacked)
                        # Pack mask to match latents
                        mask_packed = pack_latents(mask_resized)
                        # Build inpaint extension
                        from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import (
                            RectifiedFlowInpaintExtension,
                        )
                        inpaint_ext = RectifiedFlowInpaintExtension(
                            init_latents=init_packed,
                            inpaint_mask=1.0 - mask_packed,  # convert to denoise-area mask (1=regenerate)
                            noise=noise,
                        )

                        # True CFG setup
                        do_true_cfg = self.true_cfg_scale > 1.0 and (neg_embeds is not None and neg_mask is not None)

                        # Denoising loop (rectified flow update)
                        with transformer_info.model_on_device() as (_, transformer):
                            total_steps = len(timesteps)
                            for i in range(total_steps):
                                t = timesteps[i]
                                sigma_prev = sigmas[i + 1] if i + 1 < len(sigmas) else torch.tensor(0.0, device=device, dtype=dtype)
                                timestep_in = (t / 1000).expand(latents.shape[0]).to(latents.dtype)

                                noise_pred_cond = transformer(
                                    hidden_states=latents,
                                    timestep=timestep_in,
                                    guidance=None,
                                    encoder_hidden_states_mask=prompt_embeds_mask.to(device=device, dtype=dtype),
                                    encoder_hidden_states=prompt_embeds.to(device=device, dtype=dtype),
                                    return_dict=False,
                                )[0]

                                if do_true_cfg:
                                    noise_pred_uncond = transformer(
                                        hidden_states=latents,
                                        timestep=timestep_in,
                                        guidance=None,
                                        encoder_hidden_states_mask=neg_mask.to(device=device, dtype=dtype),
                                        encoder_hidden_states=neg_embeds.to(device=device, dtype=dtype),
                                        return_dict=False,
                                    )[0]
                                    comb_pred = noise_pred_uncond + self.true_cfg_scale * (noise_pred_cond - noise_pred_uncond)
                                    cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
                                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                                    noise_pred = comb_pred * (cond_norm / (noise_norm + 1e-8))
                                else:
                                    noise_pred = noise_pred_cond

                                latents_dtype = latents.dtype
                                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]  # type: ignore[arg-type]
                                latents = latents.to(dtype=latents_dtype)
                                latents = inpaint_ext.merge_intermediate_latents_with_init_latents(latents, float(sigma_prev.item()))

                        # Unpack and decode
                        vae_scale_factor = 8
                        latents_unpacked = unpack_latents(latents, self.height, self.width, vae_scale_factor)
                        with vae_info.model_on_device() as (_, vae_dec):
                            # If VAE config has mean/std, apply inverse scaling
                            try:
                                latents_mean = torch.tensor(vae_dec.config.latents_mean).view(1, vae_dec.config.z_dim, 1, 1, 1).to(latents_unpacked.device, latents_unpacked.dtype)  # type: ignore[attr-defined]
                                latents_std = 1.0 / torch.tensor(vae_dec.config.latents_std).view(1, vae_dec.config.z_dim, 1, 1, 1).to(latents_unpacked.device, latents_unpacked.dtype)  # type: ignore[attr-defined]
                                latents_dec = latents_unpacked / latents_std + latents_mean
                            except Exception:
                                latents_dec = latents_unpacked
                            decoded = vae_dec.decode(latents_dec).sample
                        img = (decoded[:, :, 0] if decoded.ndim == 5 else decoded)
                        img = (img / 2 + 0.5).clamp(0, 1)
                        img = img.cpu().permute(0, 2, 3, 1).float().numpy()
                        if img.ndim == 4:
                            img = img[0]
                        from PIL import Image
                        pil_image = Image.fromarray((img * 255).round().astype("uint8"))
                        images = [pil_image]
                    except Exception as e:
                        context.logger.error(f"Qwen stepwise inpaint failed, falling back to composite: {e}")
                        # Fall through to composite or standard path below
                        self.denoise_mask = None

                # Only use embeddings if both embeds and masks are present and we are not inpainting
                if self.denoise_mask is None and prompt_embeds is not None and prompt_embeds_mask is not None:
                    # Use embeddings path
                    images = pipe(
                        prompt_embeds=prompt_embeds,
                        prompt_embeds_mask=prompt_embeds_mask,
                        negative_prompt_embeds=neg_embeds,
                        negative_prompt_embeds_mask=neg_mask,
                        latents=init_latents,
                        **call_kwargs,
                    ).images
                else:
                    # Fallback: standard or composite postprocess if mask provided
                    images_full = pipe(
                        prompt=prompt if (prompt_embeds is None or prompt_embeds_mask is None) else None,
                        negative_prompt=negative_prompt if (neg_embeds is None or neg_mask is None) else None,
                        prompt_embeds=None if self.denoise_mask is not None else prompt_embeds,
                        prompt_embeds_mask=None if self.denoise_mask is not None else prompt_embeds_mask,
                        negative_prompt_embeds=None if self.denoise_mask is not None else neg_embeds,
                        negative_prompt_embeds_mask=None if self.denoise_mask is not None else neg_mask,
                        latents=init_latents,
                        **call_kwargs,
                    ).images

                    pil_image = images_full[0] if images_full else None
                    if pil_image is None:
                        from PIL import Image
                        pil_image = Image.new('RGB', (self.width, self.height), color='gray')

                    if self.denoise_mask is not None and self.latents is not None:
                        # Decode the source (init) latents to an image for compositing
                        init_unpacked = context.tensors.load(self.latents.latents_name)
                        with vae_info.model_on_device() as (_, vae_dec):
                            if init_unpacked.ndim == 5 and init_unpacked.shape[2] == 1:
                                init_unpacked = init_unpacked[:, :, 0, :, :]
                            init_unpacked = init_unpacked.to(device=vae_dec.device, dtype=vae_dec.dtype)
                            src_img_tensor = vae_dec.decode(init_unpacked).sample
                        src_img = (src_img_tensor / 2 + 0.5).clamp(0, 1)
                        src_img = src_img.cpu().permute(0, 2, 3, 1).float().numpy()
                        if src_img.ndim == 4:
                            src_img = src_img[0]
                        from PIL import Image
                        src_pil = Image.fromarray((src_img * 255).round().astype('uint8')).resize((self.width, self.height), Image.BILINEAR)

                        # Load and resize mask
                        mask_tensor = context.tensors.load(self.denoise_mask.mask_name)
                        mask_np = mask_tensor.squeeze().cpu().numpy()
                        mask_pil = Image.fromarray((mask_np * 255).astype('uint8')).resize((self.width, self.height), Image.BILINEAR)
                        # Invert mask semantics: DenoiseMaskField uses 1.0=preserve; composite needs preserve from src
                        mask_inv = Image.eval(mask_pil.convert('L'), lambda x: 255 - x)

                        pil_image = Image.composite(pil_image, src_pil, mask_inv)

                    images = [pil_image]

                pil_image = images[0] if images else None
                if pil_image is None:
                    from PIL import Image
                    pil_image = Image.new('RGB', (self.width, self.height), color='gray')
            except Exception as e:
                context.logger.error(f"Error during Qwen-Image generation: {e}")
                from PIL import Image
                pil_image = Image.new('RGB', (self.width, self.height), color='gray')

            image_dto = context.images.save(image=pil_image)
            return ImageOutput.build(image_dto)
