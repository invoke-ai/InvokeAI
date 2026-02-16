# Save Image

Saves a copy of an image to the image store. Unlike primitive image outputs, this invocation explicitly stores a persistent copy that can be reused later.

## Inputs

- `image`: The image to save.

## Outputs

- `image`: The saved image entry passed downstream.

## Example Usage

### Persist a result

Use `Save Image` when you want to persist the current image state before further modifying it.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Save a copy of the current image for later use.

## Notes

- The node always writes a new image record to the image store.