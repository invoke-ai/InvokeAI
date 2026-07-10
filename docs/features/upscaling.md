---
title: Tiled Upscaling (Multi-Diffusion)
---

# Tiled Upscaling (Multi-Diffusion)

Invoke's Upscale tab uses a tiled multi-diffusion pipeline to enlarge images while adding fine detail. It works with **SD1.5 / SDXL**, **FLUX**, and **Z-Image** models.

## How It Works

The pipeline runs in five stages:

1. **Spandrel upscale** — A neural-network upscaler (e.g. RealESRGAN, SwinIR) scales the image to the target resolution.
2. **Sharpening** — An unsharp mask refines edges introduced by the upscaler.
3. **Encode to latents** — The sharpened image is encoded via the model's VAE.
4. **Tiled multi-diffusion denoise** — The latent is split into overlapping tiles. Each tile is denoised independently, then tiles are blended back together with gradient weights for seamless results. A **Tile ControlNet** guides each tile to preserve the original structure.
5. **Decode** — The refined latents are decoded back to a full-resolution image.

## Required Models per Architecture

| Component | SD1.5 / SDXL | FLUX | Z-Image |
|---|---|---|---|
| **Main model** | Any SD1.5 or SDXL checkpoint | FLUX.1 Dev / Schnell | Z-Image model |
| **Spandrel upscaler** | Any Spandrel model | Any Spandrel model | Any Spandrel model |
| **Tile ControlNet** | SD1.5 or SDXL Tile ControlNet | FLUX Tile or Union ControlNet | Z-Image Tile ControlNet |
| **Text encoder** | Built-in CLIP | T5 Encoder + CLIP Embed | Qwen3 Encoder (standalone or via Qwen3 Source) |
| **VAE** | Built-in / model default | FLUX VAE | FLUX VAE |

## Parameters

- **Scale** — Target scale factor for the upscale (e.g. 2× or 4×).
- **Creativity** — Controls how much new detail the denoiser adds. Higher values produce more creative (but less faithful) results.
- **Structure** — Controls how strongly the Tile ControlNet preserves the original composition. Higher values keep the image closer to the input.
- **Tile Size** — Size of each tile in pixels. Larger tiles use more VRAM but may produce more coherent results.
- **Tile Overlap** — How much adjacent tiles overlap, in pixels. More overlap improves blending at the cost of speed.

## VRAM Requirements

Tiled upscaling requires the full model plus enough VRAM for a single tile's denoising pass. On a 24 GB GPU, all three architectures work comfortably. Low-VRAM mode (`enable_partial_loading: true` in `invokeai.yaml`) can help on smaller GPUs — see [Low-VRAM mode](low-vram.md).

## Tips

- Start with the default **Creativity** and **Structure** settings, then adjust to taste.
- Use a descriptive positive prompt — the denoiser uses it to guide detail generation in each tile.
- If you see visible tile seams, increase **Tile Overlap**.
- For maximum quality, use a high-quality Spandrel upscaler as the initial upscale step.
