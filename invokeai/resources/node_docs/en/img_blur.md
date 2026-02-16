# Blur Image

Applies a blur to an image while correctly handling premultiplied alpha so transparent edges don't darken. Supports Gaussian and Box blur modes.

## Inputs

- `image`: The image to blur.
- `radius`: The blur radius (default `8.0`).
- `blur_type`: The blur algorithm to use: `gaussian` (default) or `box`.