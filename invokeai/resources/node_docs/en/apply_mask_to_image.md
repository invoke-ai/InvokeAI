# Apply Mask to Image

Extracts a region from an image defined by a mask (black=keep, white=discard) and uses the mask as the alpha channel so the extracted region can be composited elsewhere.

## Inputs

- `image`: The source image to extract from (RGBA expected).
- `mask`: The mask defining the region (black=keep, white=discard).
- `invert_mask`: Whether to invert the mask before applying it.

## Outputs

- `image`: The resulting image where the mask has been applied as the alpha channel.

## Example Usage

### Extract masked region

Use `Apply Mask to Image` to produce an RGBA image where only the masked area remains visible for pasting into another composition.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Turn a mask into an image alpha to extract a region.

## Notes

- The mask is used directly as the alpha channel; black areas become opaque.