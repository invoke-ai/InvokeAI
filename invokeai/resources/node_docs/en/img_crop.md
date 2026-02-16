# Crop Image

Crops an input image to a rectangular region. The crop rectangle can extend outside the boundaries of the original image; areas outside the image will be transparent.

## Inputs

- `image`: The image to crop.
- `x`: Left x coordinate of the crop rectangle (default `0`).
- `y`: Top y coordinate of the crop rectangle (default `0`).
- `width`: Width of the crop rectangle in pixels (default `512`).
- `height`: Height of the crop rectangle in pixels (default `512`).

## Notes

- If the crop rectangle extends beyond the original image, transparent padding is added to reach the requested size.