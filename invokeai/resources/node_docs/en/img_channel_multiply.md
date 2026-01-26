# Multiply Image Channel

Scales a particular color channel by a factor and optionally inverts it. Works across multiple colorspaces and restores original alpha.

## Inputs

- `image`: The image to adjust.
- `channel`: Which channel to scale (e.g., `Green (RGBA)`, `Cb (YCbCr)`).
- `scale`: Multiplicative factor to apply (default `1.0`).
- `invert_channel`: If `true`, the channel is inverted after scaling.

## Outputs

- `image`: The image after channel scaling.

## Example Usage

### Desaturate by scaling saturation

Use `Multiply Image Channel` to reduce the saturation channel in `HSV` by scaling it below 1.0.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Scale a single channel to affect color/contrast.

## Notes

- The node clips values to the valid 0–255 range and restores the original alpha channel when appropriate.