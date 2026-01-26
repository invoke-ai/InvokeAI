# Multiply Images

Multiplies two images together using pixel-wise multiplication. This is useful for combining masks, applying multiply blend modes, or modulating brightness.

## Inputs

- `image1`: The first image to multiply.
- `image2`: The second image to multiply.

## Outputs

- `image`: The resulting image after multiplication.

## Example Usage

### Combine masks

Use `Multiply Images` to combine two masks so that only regions present in both remain.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Multiply two masks to intersect their regions.

## Notes

- Uses `PIL.ImageChops.multiply()` under the hood.