Our application generates images from text prompts. Part of this process involves using VAE to encode images into latent space or decode latents into image space.

The application runs on consumer GPUs with limited VRAM and different capabilities. Models may run at different precisisons.

The app has a model manager which dynamically on/off-loads models from VRAM as needed. It also has the ability to reserve working memory for computation. For example, when we VAE decode, we reserve some "working memory" in the model manager for the data that we operate on. The model manager then handles model weights on/off-loading as if this working memory is unavailable.

Your task is to do a review of this working memory estimation. Write scripts using real models at a variety of resolutions and fp16/fp32 precision to get empirical numbers for the working memory required for VAE encode and decode operations.

Use @agent-ai-engineer for this task.

Notes:
- There is a venv at /home/bat/Documents/Code/InvokeAI/.venv which you can use to run the scripts.
- You are running on a Linux machine w/ an RTX 4090 GPU with 24GB of VRAM. 32 GB of RAM.
- We are reserving working memory for VAE decode, but not for VAE encode, but the encode operation _does_ use working memory.
- Our estimations use magic numbers. I suspect they may be too high.
- The required working memory may depend on the model precision.
- Some models may operate in a mixed precision.
- In https://github.com/invoke-ai/InvokeAI/pull/7674, we increased the magic numbers to prevent OOMs. The author notes that torch _reserves_ more VRAM than it allocates, and the numbers reflect this. Please investigate further.
- In https://github.com/invoke-ai/InvokeAI/issues/6981, SD1.5 seems to require more working memory than SDXL, and our estimations may be too low.
- In https://github.com/invoke-ai/InvokeAI/issues/8405, FLUX Kontext uses VAE encode and is causing an OOM. The encode is done in /home/bat/Documents/Code/InvokeAI/invokeai/backend/flux/extensions/kontext_extension.py
- The application services have complex interdependencies. You'll need to extract the model loading logic (which is fairly simple) to load the models instead of using the existing service classes. Inference code is modularized so you can use the existing classes.

- Code references & models (models may be in diffusers or single-file formats):
  - FLUX:
    - VAE decode: /home/bat/Documents/Code/InvokeAI/invokeai/app/invocations/flux_vae_decode.py
    - VAE encode: /home/bat/Documents/Code/InvokeAI/invokeai/app/invocations/flux_vae_encode.py
    - VAE model: /home/bat/invokeai-4.0.0/models/flux/vae/FLUX.1-schnell_ae.safetensors
  - SD1.5, SDXL:
    - VAE decode: /home/bat/Documents/Code/InvokeAI/invokeai/app/invocations/latents_to_image.py
    - VAE encode: /home/bat/Documents/Code/InvokeAI/invokeai/app/invocations/image_to_latents.py
    - SDXL VAE model (fp16): /home/bat/invokeai-4.0.0/models/sdxl/vae/sdxl-vae-fp16-fix
    - SD1.5 VAE model: /home/bat/invokeai-4.0.0/models/sd-1/vae/sd-vae-ft-mse
  - CogView4:
    - VAE encode: /home/bat/Documents/Code/InvokeAI/invokeai/app/invocations/cogview4_image_to_latents.py
    - VAE decode: /home/bat/Documents/Code/InvokeAI/invokeai/app/invocations/cogview4_latents_to_image.py
    - VAE model: /home/bat/invokeai-4.0.0/models/cogview4/main/CogView4/vae
  - SD3:
    - VAE decode: /home/bat/Documents/Code/InvokeAI/invokeai/app/invocations/sd3_image_to_latents.py
    - VAE encode: /home/bat/Documents/Code/InvokeAI/invokeai/app/invocations/sd3_latents_to_image.py
    - VAE model: /home/bat/invokeai-4.0.0/models/sd-3/main/SD3.5-medium/vae
