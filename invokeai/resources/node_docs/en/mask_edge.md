# Mask Edge

Creates an edge-based mask from an image by combining gradient and Canny edge detection, dilating the result, and optionally blurring. The final mask is inverted so black indicates areas to keep when inpainting.

## Inputs

- `image`: The image to compute edges from (converted to grayscale internally).
- `edge_size`: Pixel size used to dilate the detected edges.
- `edge_blur`: Amount of blur to apply to the resulting mask.
- `low_threshold`: Lower threshold for Canny edge detection.
- `high_threshold`: Upper threshold for Canny edge detection.

## Outputs

- `image`: A grayscale mask image (category: mask) where black indicates areas to retain and white indicates areas to discard.

## Example Usage

### Create inpainting edge mask

Use `Mask Edge` to generate masks that preserve the interior of shapes while isolating edges for inpainting or blending.

![IMAGE_PLACEHOLDER](./images/IMAGE_PLACEHOLDER.png)  
Generate an edge mask to guide inpainting along object boundaries.

## Notes

- The node leverages OpenCV for Canny detection and dilation; performance depends on image size.