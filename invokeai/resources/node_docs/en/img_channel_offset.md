# Offset Image Channel

Adds or subtracts a value from a chosen color channel in various colorspaces (RGBA, CMYK, HSV, LAB, YCbCr). Useful for fine-tuning specific channels like red, hue, or luminance.

## Inputs

- `image`: The image to adjust.
- `channel`: Which channel to modify (e.g., `Red (RGBA)`, `Hue (HSV)`, `Luminosity (LAB)`).
- `offset`: Integer amount to add (or subtract if negative) to the channel (range `-255..255`).

## Outputs

- `image`: The adjusted image with the channel offset applied.

## Example Usage

### Boost red channel

Use `Offset Image Channel` to increase or decrease a specific channel, such as warming an image by boosting red.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Adjust a single channel without affecting others.

## Notes

- When adjusting the hue channel, values wrap around rather than clamp.