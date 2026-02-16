# Combine Masks

Combines two masks by multiplying them together so that only areas present in both masks remain. Useful for intersecting mask regions.

## Inputs

- `mask1`: The first grayscale mask.
- `mask2`: The second grayscale mask.

## Outputs

- `image`: The combined mask (saved as mask category) resulting from pixel-wise multiplication.

## Example Usage

### Intersect selections

Use `Combine Masks` to intersect two separate selections so downstream operations only affect the overlap region.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Multiply masks to compute their intersection.

## Notes

- Uses `PIL.ImageChops.multiply()` for combination.