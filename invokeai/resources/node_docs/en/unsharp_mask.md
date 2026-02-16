# Unsharp Mask

Applies an unsharp mask filter to enhance perceived sharpness by boosting high-frequency components. Preserves alpha channels when present.

## Inputs

- `image`: The image to sharpen.
- `radius`: Radius of the Gaussian blur used to create the unsharp mask (default `2`).
- `strength`: Strength of the effect as a percentage (default `50`).

## Outputs

- `image`: The sharpened image.

## Example Usage

### Increase clarity

Use `Unsharp Mask` to bring out edge detail and improve perceived sharpness after upscaling or denoising.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Sharpen fine details while preserving transparency.

## Notes

- The node handles images with alpha by temporarily working in RGB and restoring the original alpha channel.