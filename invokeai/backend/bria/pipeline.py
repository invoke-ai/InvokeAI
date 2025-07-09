#!/usr/bin/env python
"""
Bria TextΓÇætoΓÇæImage Pipeline (GPUΓÇæready)
Using your local Bria checkpoints.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Your bria_utils imports
from .bria_utils import get_original_sigmas, get_t5_prompt_embeds, is_ng_none
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from PIL import Image
from tqdm import tqdm  # add this at the top of your file

# Your custom transformer import
from .transformer_bria import BriaTransformer2DModel
from transformers import T5EncoderModel, T5TokenizerFast


# -----------------------------------------------------------------------------
# 1. Model Loader
# -----------------------------------------------------------------------------
class BriaModelLoader:
    def __init__(
        self,
        transformer_ckpt: str,
        vae_ckpt: str,
        text_encoder_ckpt: str,
        tokenizer_ckpt: str,
        device: torch.device,
    ):
        self.device = device

        # print("Loading Bria Transformer from", transformer_ckpt)
        # self.transformer = BriaTransformer2DModel.from_pretrained(transformer_ckpt, torch_dtype=torch.bfloat16).to(device)

        # print("Loading VAE from", vae_ckpt)
        # self.vae = AutoencoderKL.from_pretrained(vae_ckpt, torch_dtype=torch.float32).to(device)

        # print("Loading T5 Encoder from", text_encoder_ckpt)
        # self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_ckpt, torch_dtype=torch.float16).to(device)

        # print("Loading Tokenizer from", tokenizer_ckpt)
        # self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer_ckpt, legacy=False)
        self.transformer = BriaTransformer2DModel.from_pretrained(transformer_ckpt, torch_dtype=torch.float16).to(
            device
        )
        self.vae = AutoencoderKL.from_pretrained(vae_ckpt, torch_dtype=torch.float16).to(device)
        self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_ckpt, torch_dtype=torch.float16).to(device)
        self.tokenizer = T5TokenizerFast.from_pretrained(tokenizer_ckpt)

    def get(self):
        return {
            "transformer": self.transformer,
            "vae": self.vae,
            "text_encoder": self.text_encoder,
            "tokenizer": self.tokenizer,
        }


# -----------------------------------------------------------------------------
# 2. Text Encoder (uses bria_utils)
# -----------------------------------------------------------------------------
class BriaTextEncoder:
    def __init__(
        self,
        text_encoder: T5EncoderModel,
        tokenizer: T5TokenizerFast,
        device: torch.device,
        max_length: int = 128,
    ):
        self.model = text_encoder.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def encode(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images_per_prompt: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1) get positive embeddings
        pos = get_t5_prompt_embeds(
            tokenizer=self.tokenizer,
            text_encoder=self.model,
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=self.max_length,
            device=self.device,
        )
        # 2) get negative or zeros
        if negative_prompt is None or is_ng_none(negative_prompt):
            neg = torch.zeros_like(pos)
        else:
            neg = get_t5_prompt_embeds(
                tokenizer=self.tokenizer,
                text_encoder=self.model,
                prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=self.max_length,
                device=self.device,
            )

        # 3) build text_ids: shape [S_text, 3]
        #    S_text = number of tokens = pos.shape[1]
        S_text = pos.shape[1]
        text_ids = torch.zeros((1, S_text, 3), device=self.device, dtype=torch.long)
        text_ids = torch.zeros((S_text, 3), device=self.device, dtype=torch.long)

        print(f"Text embeds shapes ΓåÆ pos: {pos.shape}, neg: {neg.shape}, text_ids: {text_ids.shape}")
        return pos, neg, text_ids


# -----------------------------------------------------------------------------
# 3. Latent Sampler
# -----------------------------------------------------------------------------
class BriaLatentSampler:
    def __init__(self, transformer: BriaTransformer2DModel, vae: AutoencoderKL, device: torch.device):
        self.device = device
        self.latent_channels = transformer.config.in_channels
        # self.latent_height = vae.config.sample_size
        # self.latent_width  = vae.config.sample_size
        self.latent_height = 128
        self.latent_width = 128

    @staticmethod
    def _prepare_latent_image_ids(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype):
        # Build the same img_ids FluxPipeline.prepare_latents would use
        latent_image_ids = torch.zeros((height, width, 3), device=device, dtype=dtype)
        latent_image_ids[..., 1] = torch.arange(height, device=device)[:, None]
        latent_image_ids[..., 2] = torch.arange(width, device=device)[None, :]
        # reshape to [1, height*width, 3] then repeat for batch
        latent_image_ids = latent_image_ids.view(1, height * width, 3)
        return latent_image_ids.repeat(batch_size, 1, 1)

    def sample(self, batch_size: int = 1, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator(device=self.device).manual_seed(seed)

        # 1) sample & pack the noise exactly as before
        shrunk = self.latent_channels // 4
        noise4d = torch.randn(
            (batch_size, shrunk, self.latent_height, self.latent_width),
            device=self.device,
            generator=gen,
        )
        latents = (
            noise4d.view(batch_size, shrunk, self.latent_height // 2, 2, self.latent_width // 2, 2)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch_size, (self.latent_height // 2) * (self.latent_width // 2), shrunk * 4)
        )

        # 2) build the matching latent_image_ids
        latent_image_ids = self._prepare_latent_image_ids(
            batch_size,
            self.latent_height // 2,
            self.latent_width // 2,
            device=self.device,
            dtype=torch.long,
        )
        if latent_image_ids.ndim == 3 and latent_image_ids.shape[0] == 1:
            latent_image_ids = latent_image_ids[0]  # [S_img , 3]

        latent_image_ids = latent_image_ids.squeeze(0)

        print(f"Sampled & packed latents: {latents.shape}")
        return latents, latent_image_ids


# -----------------------------------------------------------------------------
# 4. Denoising Loop (uses bria_utils for ╧â schedule)
# -----------------------------------------------------------------------------
class BriaDenoise:
    def __init__(
        self,
        transformer: nn.Module,
        scheduler_name: str,
        device: torch.device,
        num_train_timesteps: int,
        num_inference_steps: int,
        **sched_kwargs,
    ):
        self.transformer = transformer.to(device)
        self.device = device

        # Build scheduler
        if scheduler_name == "flow_match":
            from diffusers import FlowMatchEulerDiscreteScheduler

            self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(transformer.config, **sched_kwargs)
        else:
            from diffusers import DDIMScheduler

            self.scheduler = DDIMScheduler(**sched_kwargs)

        # Use your exact ╧â schedule from bria_utils
        from bria_utils import get_original_sigmas

        sigmas = get_original_sigmas(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
        )
        self.scheduler.set_timesteps(
            num_inference_steps=None,
            timesteps=None,
            sigmas=sigmas,
            device=device,
        )

        # allow early exit
        self.interrupt = False
        # will be set in denoise()
        self._guidance_scale = 1.0
        self._joint_attention_kwargs = {}
        self.transformer = transformer.to(device)
        self.device = device

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self.guidance_scale > 1.0

    @property
    def joint_attention_kwargs(self) -> dict:
        return self._joint_attention_kwargs

    @torch.no_grad()
    def denoise(
        self,
        latents: torch.Tensor,  # [B, seq_len, C_hidden]
        latent_image_ids: torch.Tensor,  # [B, seq_len, 3]
        prompt_embeds: torch.Tensor,  # [B, S_text, D]
        negative_prompt_embeds: torch.Tensor,  # [B, S_text, D]
        text_ids: torch.Tensor,  # [B, S_text, 3]
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        normalize: bool = False,
        clip_value: float | None = None,
        seed: int = 0,
    ) -> torch.Tensor:
        # 0) Quick cast & setup
        device = self.device
        # ensure dtype matches transformer
        target_dtype = next(self.transformer.parameters()).dtype
        latents = latents.to(device, dtype=target_dtype)
        prompt_embeds = prompt_embeds.to(device, dtype=target_dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(device, dtype=target_dtype)

        # replicate reference encode_prompt behaviour
        if negative_prompt_embeds is None:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        if guidance_scale > 1.0:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        self._guidance_scale = guidance_scale

        # 1) Prepare FlowΓÇæMatch timesteps identical to reference pipeline
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler) and getattr(
            self.scheduler.config, "use_dynamic_shifting", False
        ):
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            image_seq_len = latents.shape[1]
            mu = calculate_shift(image_seq_len, 256, 16_384, 0.25, 0.75)
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, None, sigmas, mu=mu
            )
        else:
            sigmas = get_original_sigmas(
                num_train_timesteps=self.scheduler.config.num_train_timesteps,
                num_inference_steps=num_inference_steps,
            )
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, None, sigmas
            )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 2) Loop with progress bar

        with tqdm(total=num_inference_steps, desc="Denoising", unit="step") as progress_bar:
            for i, t in enumerate(timesteps):
                # a) expand for CFG?
                latent_model_input = torch.cat([latents] * 2, dim=0) if self.do_classifier_free_guidance else latents

                # b) scale model input if needed
                if not isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # c) broadcast timestep
                timestep = t.expand(latent_model_input.shape[0])

                # d) predict noise
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                )[0]

                # e) classifierΓÇæfree guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    cfg_noise_pred_text = noise_pred_text.std()
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # f) optional normalize/clip
                if normalize:
                    noise_pred = noise_pred * (0.7 * (cfg_noise_pred_text / noise_pred.std())) + 0.3 * noise_pred

                if clip_value:
                    noise_pred = noise_pred.clamp(-clip_value, clip_value)

                # g) scheduler step, inΓÇæplace
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if (i + 1) % 5 == 0 or i == len(timesteps) - 1:
                    progress_bar.update(5 if i + 1 < len(timesteps) else (len(timesteps) % 5))

                # # j) XLA sync
                # if XLA_AVAILABLE:
                #     xm.mark_step()

        # 3) Return the final packed latents (still [B, seq_len, C_hidden])
        return latents


# -----------------------------------------------------------------------------
# 5. Latents ΓåÆ Image
# -----------------------------------------------------------------------------
class BriaLatentsToImage:
    def __init__(self, vae: AutoencoderKL, device: torch.device):
        self.vae = vae.to(device)
        self.device = device

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> list[Image.Image]:
        """
        Accepts either of the two packed shapes that come out of the denoiser

        ΓÇó [B ,  S , 16]          ΓÇô 3ΓÇæD, where S = H┬▓   (e.g. 16┬á384 for 1024├ù1024)
        ΓÇó [B , 1 ,  S , 16]      ΓÇô 4ΓÇæD misΓÇæordered (what caused your crash)

        Converts them to the VAEΓÇÖs expected shape [B , 4 , H , W] before decoding.
        """
        # ---- 1. UnΓÇæpack to (B , 4 , H , W) ----------------------------------
        if latents.ndim == 3:  # (B,S,16)
            B, S, C = latents.shape
            H2 = int(S**0.5)  # 128 for 1024├ù1024
            latents = (
                latents.view(B, H2, H2, 4, 2, 2)  # split channels into 4├ù(2├ù2)
                .permute(0, 3, 1, 4, 2, 5)  # (B,4,H2,2,W2,2)
                .reshape(B, 4, H2 * 2, H2 * 2)  # (B,4,H,W)
            )

        elif latents.ndim == 4 and latents.shape[1] == 1:  # (B,1,S,16)
            B, _, S, C = latents.shape
            H2 = int(S**0.5)
            latents = (
                latents.squeeze(1)  # -> (B,S,16)
                .view(B, H2, H2, 4, 2, 2)
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(B, 4, H2 * 2, H2 * 2)
            )
        # else: already (B,4,H,W)

        # ---- 2. Standard VAE decode -----------------------------------------
        shift = 0 if self.vae.config.shift_factor is None else self.vae.config.shift_factor
        latents = (latents / self.vae.config.scaling_factor) + shift

        # 1. temporarily move VAE to fp32 for the forward pass
        self.vae.to(dtype=torch.float32)
        images = self.vae.decode(latents.to(torch.float32)).sample  # fullΓÇæprecision decode
        self.vae.to(dtype=torch.bfloat16)  # cast to fp32 **after** decode
        images = (images.clamp(-1, 1) + 1) / 2  # [0,1] fp32
        images = (images.cpu().permute(0, 2, 3, 1).numpy() * 255).astype("uint8")

        return [Image.fromarray(img) for img in images]


# -----------------------------------------------------------------------------
# Main: Assemble & Run
# -----------------------------------------------------------------------------
def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ΓöÇΓöÇΓöÇ Use your actual checkpoint locations ΓöÇΓöÇΓöÇ
    transformer_ckpt = "/home/ubuntu/invoke_local_nodes/bria_3_1/transformer"
    vae_ckpt = "/home/ubuntu/invoke_local_nodes/bria_3_1/vae"
    text_encoder_ckpt = "/home/ubuntu/invoke_local_nodes/bria_3_1/text_encoder"
    tokenizer_ckpt = "/home/ubuntu/invoke_local_nodes/bria_3_1/tokenizer"

    # 1. Load models
    loader = BriaModelLoader(
        transformer_ckpt,
        vae_ckpt,
        text_encoder_ckpt,
        tokenizer_ckpt,
        device,
    )
    mdl = loader.get()
    # if diffusers.__version__ >= "0.27.0":
    #     mdl["transformer"].enable_xformers_memory_efficient_attention()  # now safe
    # else:
    #     mdl["transformer"].disable_xformers_memory_efficient_attention() # keep quality

    # 2. Encode prompt ΓÇö now capture text_ids as well
    text_enc = BriaTextEncoder(mdl["text_encoder"], mdl["tokenizer"], device)
    pos_embeds, neg_embeds, text_ids = text_enc.encode(
        prompt="3d rendered image, landscape made out of ice cream, rich ice cream textures, ice cream-valley , with a milky ice cream river, the ice cream has rich texture with visible chocolate chunks and intricate details, in the background an air balloon floats over the vally, in the sky visible dramatic like clouds, brown-chocolate color white and pink pallet, drama, beautiful surreal landscape, polarizing lens, very high contrast, 3d rendered realistic",
        negative_prompt=None,
        num_images_per_prompt=1,
    )

    # 3. Sample initial noise ΓåÆ get both latents & latent_image_ids
    sampler = BriaLatentSampler(mdl["transformer"], mdl["vae"], device)
    init_latents, latent_image_ids = sampler.sample(batch_size=1, seed=1249141701)

    # 4. Denoise ΓÇö now passing latent_image_ids and text_ids
    denoiser = BriaDenoise(
        transformer=mdl["transformer"],
        scheduler_name="flow_match",
        device=device,
        num_train_timesteps=1000,
        num_inference_steps=30,
        base_shift=0.5,
        max_shift=1.15,
    )
    final_latents = denoiser.denoise(
        init_latents,
        latent_image_ids,
        pos_embeds,
        neg_embeds,
        text_ids,
        num_inference_steps=30,
        guidance_scale=5.0,
        seed=1249141701,
    )

    # 5. Decode
    decoder = BriaLatentsToImage(mdl["vae"], device)
    images = decoder.decode(final_latents)
    for i, img in enumerate(images):
        img.save(f"bria_output_{i}.png")


if __name__ == "__main__":
    main()
