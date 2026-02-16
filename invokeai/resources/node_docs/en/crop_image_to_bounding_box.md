# Crop Image to Bounding Box

Crops an image to a provided bounding box. If no bounding box is supplied, the node crops to the image's non-transparent bounding box.

## Inputs

- `image`: The image to crop.
- `bounding_box` (optional): The bounding box to crop to; if omitted, the image's non-transparent extents are used.

## Outputs

- `image`: The cropped image.

## Example Usage

### Trim transparent edges

Use `Crop Image to Bounding Box` to remove surrounding transparent pixels or to crop to a specific box.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Crop to the non-transparent content of an image.

## Notes

- If a bounding box is provided it must be compatible with the image dimensions.