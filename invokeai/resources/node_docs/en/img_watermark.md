# Add Invisible Watermark

Embeds an invisible watermark into an image using a text key. This watermark is not visible but can be detected by compatible tools to assert provenance.

## Inputs

- `image`: The image to watermark.
- `text`: Watermark text/key (default `InvokeAI`).

## Outputs

- `image`: The watermarked image.

## Example Usage

### Mark generated images

Use `Add Invisible Watermark` to embed a hidden provenance tag into images before saving or sharing.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Embed an invisible watermark for provenance.

## Notes

- The watermark is not visible in the pixel data but can be detected by compatible watermark readers.