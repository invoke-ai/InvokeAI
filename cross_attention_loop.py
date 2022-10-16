import random
import traceback

import numpy as np
import torch

from diffusers import (LMSDiscreteScheduler)
from PIL import Image
from torch import autocast
from tqdm.auto import tqdm

import .ldm.models.diffusion.cross_attention


@torch.no_grad()
def stablediffusion(
                    clip,
                    clip_tokenizer,
                    device,
                    vae,
                    unet,
                    prompt='',
                    prompt_edit=None,
                    prompt_edit_token_weights=None,
                    prompt_edit_tokens_start=0.0,
                    prompt_edit_tokens_end=1.0,
                    prompt_edit_spatial_start=0.0,
                    prompt_edit_spatial_end=1.0,
                    guidance_scale=7.5,
                    steps=50,
                    seed=None,
                    width=512,
                    height=512,
                    init_image=None,
                    init_image_strength=0.5,
                    ):
    if prompt_edit_token_weights is None:
        prompt_edit_token_weights = []
    # Change size to multiple of 64 to prevent size mismatches inside model
    width = width - width % 64
    height = height - height % 64

    # If seed is None, randomly select seed from 0 to 2^32-1
    if seed is None: seed = random.randrange(2**32 - 1)
    generator = torch.manual_seed(seed)

    # Set inference timesteps to scheduler
    scheduler = LMSDiscreteScheduler(beta_start=0.00085,
                                     beta_end=0.012,
                                     beta_schedule='scaled_linear',
                                     num_train_timesteps=1000,
                                     )
    scheduler.set_timesteps(steps)

    # Preprocess image if it exists (img2img)
    if init_image is not None:
        # Resize and transpose for numpy b h w c -> torch b c h w
        init_image = init_image.resize((width, height), resample=Image.Resampling.LANCZOS)
        init_image = np.array(init_image).astype(np.float32) / 255.0 * 2.0 - 1.0
        init_image = torch.from_numpy(init_image[np.newaxis, ...].transpose(0, 3, 1, 2))

        # If there is alpha channel, composite alpha for white, as the diffusion
        # model does not support alpha channel
        if init_image.shape[1] > 3:
            init_image = init_image[:, :3] * init_image[:, 3:] + (1 - init_image[:, 3:])

        # Move image to GPU
        init_image = init_image.to(device)

        # Encode image
        with autocast(device):
            init_latent = (vae.encode(init_image)
                           .latent_dist
                           .sample(generator=generator)
                           * 0.18215)

        t_start = steps - int(steps * init_image_strength)

    else:
        init_latent = torch.zeros((1, unet.in_channels, height // 8, width // 8),
                                  device=device)
        t_start = 0

    # Generate random normal noise
    noise = torch.randn(init_latent.shape, generator=generator, device=device)
    latent = scheduler.add_noise(init_latent,
                                 noise,
                                 torch.tensor([scheduler.timesteps[t_start]], device=device)
                                 ).to(device)

    # Process clip
    with autocast(device):
        tokens_uncond = clip_tokenizer('', padding='max_length',
                                       max_length=clip_tokenizer.model_max_length,
                                       truncation=True, return_tensors='pt',
                                       return_overflowing_tokens=True
                                       )
        embedding_uncond = clip(tokens_uncond.input_ids.to(device)).last_hidden_state

        tokens_cond = clip_tokenizer(prompt, padding='max_length',
                                     max_length=clip_tokenizer.model_max_length,
                                     truncation=True, return_tensors='pt',
                                     return_overflowing_tokens=True
                                     )
        embedding_cond = clip(tokens_cond.input_ids.to(device)).last_hidden_state

        # Process prompt editing
        if prompt_edit is not None:
            tokens_cond_edit = clip_tokenizer(prompt_edit, padding='max_length',
                                              max_length=clip_tokenizer.model_max_length,
                                              truncation=True, return_tensors='pt',
                                              return_overflowing_tokens=True
                                              )
            embedding_cond_edit = clip(tokens_cond_edit.input_ids.to(device)).last_hidden_state

            c_a_c.init_attention_edit(tokens_cond, tokens_cond_edit)

        c_a_c.init_attention_func()
        c_a_c.init_attention_weights(prompt_edit_token_weights)

        timesteps = scheduler.timesteps[t_start:]

        for idx, timestep in tqdm(enumerate(timesteps), total=len(timesteps)):
            t_index = t_start + idx

            latent_model_input = latent
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

            # Predict the unconditional noise residual
            noise_pred_uncond = unet(latent_model_input,
                                     timestep,
                                     encoder_hidden_states=embedding_uncond
                                     ).sample

            # Prepare the Cross-Attention layers
            if prompt_edit is not None:
                c_a_c.save_last_tokens_attention()
                c_a_c.save_last_self_attention()
            else:
                # Use weights on non-edited prompt when edit is None
                c_a_c.use_last_tokens_attention_weights()

            # Predict the conditional noise residual and save the
            # cross-attention layer activations
            noise_pred_cond = unet(latent_model_input,
                                   timestep,
                                   encoder_hidden_states=embedding_cond
                                   ).sample

            # Edit the Cross-Attention layer activations
            if prompt_edit is not None:
                t_scale = timestep / scheduler.num_train_timesteps
                if (t_scale >= prompt_edit_tokens_start
                        and t_scale <= prompt_edit_tokens_end):
                    c_a_c.use_last_tokens_attention()
                if (t_scale >= prompt_edit_spatial_start
                        and t_scale <= prompt_edit_spatial_end):
                    c_a_c.use_last_self_attention()

                # Use weights on edited prompt
                c_a_c.use_last_tokens_attention_weights()

                # Predict the edited conditional noise residual using the
                # cross-attention masks
                noise_pred_cond = unet(latent_model_input,
                                       timestep,
                                       encoder_hidden_states=embedding_cond_edit
                                       ).sample

            # Perform guidance
            noise_pred = (noise_pred_uncond + guidance_scale
                          * (noise_pred_cond - noise_pred_uncond))

            latent = scheduler.step(noise_pred,
                                    t_index,
                                    latent
                                    ).prev_sample

        # scale and decode the image latents with vae
        latent = latent / 0.18215
        image = vae.decode(latent.to(vae.dtype)).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype('uint8')
    return Image.fromarray(image)
