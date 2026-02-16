# Center Pad or Crop Image

Pad or crop an image from the center by specifying pixel amounts for each side. Positive values add padding (transparent area) outward; negative values crop inward.

## Inputs

- `image`: The image to modify.
- `left`: Pixels to pad/crop on the left (positive pads, negative crops).
- `right`: Pixels to pad/crop on the right.
- `top`: Pixels to pad/crop on the top.
- `bottom`: Pixels to pad/crop on the bottom.

## Outputs

- `image`: The padded or cropped image.

## Example Usage

### Expand canvas for composition

Use `Center Pad or Crop Image` to add border space around an image for compositing or to crop equally from both sides.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Center-pad an image to prepare space for new elements.

## Notes

- The operation centers the original image within the new dimensions, so padding/cropping applies equally relative to the original center.