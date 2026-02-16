# Resize Image

Resize an image to exact pixel dimensions using configurable resampling modes. Choose the resampling filter for quality or speed trade-offs.

## Inputs

- `image`: The image to resize.
- `width`: Destination width in pixels (default `512`).
- `height`: Destination height in pixels (default `512`).
- `resample_mode`: Resampling filter to use (`nearest`, `box`, `bilinear`, `hamming`, `bicubic`, `lanczos`). Default is `bicubic`.

## Outputs

- `image`: The resized image.

## Example Usage

### Prepare a model input

Use `Resize Image` to make sure images are the correct size before passing to models or other nodes that require specific dimensions.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Resize for model inputs or exports.

## Notes

- Choosing `lanczos` or `bicubic` yields higher-quality downsamples, while `nearest` is fastest.