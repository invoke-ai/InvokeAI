# Canvas Paste Back

Combines two images using a mask, used in several of the Unified Canvas workflows. This pastes the `target_image` onto the `source_image` using a prepared mask. The node dilates (expands) and thenblurs the mask before using it to blend the images. This reduces the visible seam when pasting inpaint results back into the original canvas.

## Inputs

- `source_image`: The image onto which the target will be pasted.
- `target_image`: The image to paste into the source.
- `mask`: The mask controlling where the paste occurs.
- `mask_blur`: Amount of Gaussian blur to apply to the mask (default `0`).