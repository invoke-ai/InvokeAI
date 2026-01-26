# Mask from Alpha

Extracts the alpha channel from an image and returns it as a grayscale mask. Optionally inverts the mask.

## Inputs

- `image`: The image containing an alpha channel to extract.
- `invert`: If `true`, the extracted alpha mask will be inverted.

## Outputs

- `image`: A grayscale mask image (white/black) representing the alpha channel.

## Example Usage

### Create an edit mask

Use `Mask from Alpha` to turn an image's transparency into a mask you can use for inpainting or compositing.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Extract alpha to create a mask for selective edits.

## Notes

- The output is saved as a mask category image for use in inpainting and other mask-aware nodes.