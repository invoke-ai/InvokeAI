# Expand Mask with Fade

Expands a binary mask outward by a specified fade distance and applies a smooth fade from black to white. Black indicates areas to keep from the generated image and white indicates areas to discard.

## Inputs

- `mask`: The mask to expand (grayscale).
- `threshold`: Threshold used to binarize the input mask (default `0`).
- `fade_size_px`: Fade distance in pixels (default `32`). If `0`, the mask is returned unchanged.