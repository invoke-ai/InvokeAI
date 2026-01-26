# Add Image Noise

Adds noise to an image, either Gaussian or Salt-and-Pepper. Optionally restrict noise to regions using a mask. Alpha is preserved.

## Inputs

- `image`: The image to add noise to.
- `mask` (optional): Grayscale mask determining where to apply noise (black=noise, white=no noise).
- `seed`: Random seed for reproducible noise.
- `noise_type`: `gaussian` (default) or `salt_and_pepper`.
- `amount`: Strength of the noise (0–1, default `0.1`).
- `noise_color`: If `true`, produce colored noise; otherwise use monochrome noise.
- `size`: Size of noise points (default `1`).

## Outputs

- `image`: The noisy image with original alpha restored.

## Example Usage

### Add film grain

Use `Add Image Noise` to add subtle grain for realism or to break up banding artifacts.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Add subtle Gaussian noise for natural texture.

## Notes

- For `salt_and_pepper`, noise is applied probabilistically per-pixel according to `amount`.
- The node respects the provided `mask` by inverting it internally before compositing the noisy region.