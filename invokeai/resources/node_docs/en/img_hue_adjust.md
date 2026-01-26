# Adjust Image Hue

Rotates the hue of an image by a specified number of degrees. Useful for creative color shifts or quick recoloring.

## Inputs

- `image`: The image to adjust.
- `hue`: Degrees to rotate hue (0–360). Positive values rotate the hue forward.

## Outputs

- `image`: The hue-adjusted image (converted back to RGBA and saved).

## Example Usage

### Recolor elements

Use `Adjust Image Hue` to quickly shift the overall color palette of an image or to create variations.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Shift hues for stylistic variation.

## Notes

- The node internally converts the image to HSV, adjusts the hue channel, and converts back to RGBA.