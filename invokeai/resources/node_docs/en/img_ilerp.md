# Inverse Lerp Image

Inverse-linear remaps pixel values by mapping an input range `[min..max]` back to `[0..255]`. Helpful for normalizing or preparing images where a known intensity window should be stretched to full range.

## Inputs

- `image`: The image to remap.
- `min`: Input minimum value (default `0`).
- `max`: Input maximum value (default `255`).

## Outputs

- `image`: The remapped image.

## Example Usage

### Normalize window

Use `Inverse Lerp Image` when you want to stretch a specific intensity window to the full 0-255 range for visualization or further processing.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Stretch a specific intensity window to full range.

## Notes

- Values outside the `[min..max]` input range are clipped to 0 or 255 respectively.