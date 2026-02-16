# Paste Image

Pastes one image into another at a specified offset, optionally using a mask to control transparency. The base image can be cropped to its original size after pasting.

## Inputs

- `base_image`: The target image to paste into.
- `image`: The image to paste on top of the base image.
- `mask` (optional): A mask controlling the paste (white=keep/paste areas). If supplied, the mask is inverted internally before use.
- `x`: Left x coordinate where the image is pasted (default `0`).
- `y`: Top y coordinate where the image is pasted (default `0`).
- `crop`: If `true`, the resulting composite will be cropped to the dimensions of the base image in case the pasted image extends beyond its borders (default `false`).

---

## Example Usage

### Composite elements

![image](images/paste_images_blank_background.jpg)

In this example, the `Blank Image` node creates a solid color background. Multiple images are then pasted onto this background to create a composite image with a border effect.  

![image](images/paste_images_blank_background_result.jpg)

---

## Notes

- The coordinates `x` and `y` determine where the top-left corner of the pasted image will be placed on the base image.