# Paste Image into Bounding Box

Pastes a source image into a target image at the area defined by a bounding box. The source image must match the bounding box size.

## Inputs

- `source_image`: The image to paste (must match bounding box dimensions).
- `target_image`: The image to paste into.
- `bounding_box`: The bounding box (x, y, width, height) defining where to paste the source.

## Outputs

- `image`: The resulting composited image.

## Example Usage

### Tile into target

Use `Paste Image into Bounding Box` to place a patch into a larger image at an explicit rectangle (useful for tiled generation or compositing).

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Paste a prepared patch into a target area on a canvas.

## Notes

- The bounding box must fit inside the target image and the source must match the box size.