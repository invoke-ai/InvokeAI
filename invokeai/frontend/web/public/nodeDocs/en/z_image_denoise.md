# Z Image Denoise Node

The Z Image Denoise node performs diffusion-based denoising on an image using the Z-Image (Zunderbolts) model architecture. This node is designed for high-quality image generation and modification workflows.

## Overview

This node takes a source image, encodes it into latent space, applies denoising with a specified strength, and then decodes the result back into an image. The denoising process allows for controlled transformation of the input image while preserving core structure.

## Inputs

### Required Inputs

- **Model**: The Z-Image main model to use for denoising
- **Positive Prompt**: Text description of what you want to see in the output
- **Negative Prompt**: Text description of what you want to avoid in the output
- **Image**: The source image to be processed

### Optional Inputs

- **Denoising Strength**: Controls how much of the original image is preserved (0.0 = no change, 1.0 = full regeneration)
- **Steps**: Number of denoising steps (more steps = higher quality but slower)
- **CFG Scale**: How strongly the model should follow your prompt
- **Scheduler**: The noise scheduling algorithm to use
- **Seed**: Random seed for reproducible results

## Outputs

- **Image**: The denoised/transformed output image

## Tips

1. **Lower denoising strength** (0.2-0.5) preserves more of the original image structure
2. **Higher denoising strength** (0.7-1.0) allows for more creative reinterpretation
3. Use **negative prompts** to steer the model away from unwanted artifacts or styles
4. If results are too noisy, try increasing the number of steps

## Example Use Cases

- Image-to-image style transfer
- Photo restoration and enhancement
- Creative image modifications
- Consistent character regeneration with slight variations
