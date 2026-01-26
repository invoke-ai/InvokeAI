# Scale Image

Scale an image by a multiplicative factor, preserving the original aspect ratio. Useful for quick upscales or downscales where only a factor is known.

## Inputs

- `image`: The image to scale.
- `scale_factor`: Multiplicative factor to scale dimensions by (default `2.0`).
- `resample_mode`: Resampling filter to use (`nearest`, `box`, `bilinear`, `hamming`, `bicubic`, `lanczos`). Default is `bicubic`.

## Outputs

- `image`: The scaled image.

## Example Usage

### Quick upscale

Use `Scale Image` to double the size of an image before further processing or for preview purposes.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Double the size of an image while preserving sharpness.

## Notes

- The node multiplies both width and height by the same factor so aspect ratio is preserved.