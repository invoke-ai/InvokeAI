# Blur NSFW Image

Checks an image for NSFW content and applies a blur if the image is considered NSFW. Use this node to automatically obfuscate images that may violate content policies.

## Inputs

- `image`: The image to check and blur if needed.

## Outputs

- `image`: The possibly blurred image. If the image is not flagged, it is returned unchanged.

## Example Usage

### Protect displaying content

Place `Blur NSFW Image` before nodes that display or export images to ensure potentially sensitive content is safely blurred.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Automatically blur potential NSFW content.

## Notes

- Uses an internal safety checker to determine whether to blur; behavior depends on the configured safety model.