# Lerp Image

Linearly remaps pixel values of an image from [0..255] to a specified `[min..max]` range. Useful for adjusting contrast or re-normalizing image data.

## Inputs

- `image`: The image to remap.
- `min`: Output minimum value (default `0`).
- `max`: Output maximum value (default `255`).

## Outputs

- `image`: The remapped image.

## Example Usage

### Adjust output range

Use `Lerp Image` to expand or compress the numeric range of pixel values before further processing.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Remap pixel intensities to a new range.

## Notes

- Works on all channels and preserves image dimensions.