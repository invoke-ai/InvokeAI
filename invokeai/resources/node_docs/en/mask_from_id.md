# Mask from Segmented Image

Generate a binary mask isolating a particular ID color from a segmented/ID map image. Use this to extract regions corresponding to a specific object or class in an ID map.

## Inputs

- `image`: The ID map image (typically a segmented image with distinct colors representing classes).
- `color`: The target ID color to isolate.
- `threshold`: Distance threshold for color matching (default `100`).
- `invert`: If `true`, the resulting mask will be inverted.

## Outputs

- `image`: A binary mask (category: mask) highlighting the matched ID region.

## Example Usage

### Extract object selection

Use `Mask from Segmented Image` to create a mask of a single object class from a segmentation output.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Generate a mask for a class by color.

## Notes

- The node computes Euclidean color distance in RGBA space and thresholds it to build the mask.