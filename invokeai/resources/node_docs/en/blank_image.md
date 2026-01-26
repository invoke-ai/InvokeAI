# Blank Image

Creates a new blank image with the specified dimensions, color, and mode, then forwards it downstream. Useful for creating a canvas for compositing, drawing, or testing.

## Inputs

- `width`: The width of the image in pixels (default `512`).
- `height`: The height of the image in pixels (default `512`).
- `mode`: The image mode (`RGB` or `RGBA`) determining channels and transparency.
- `color`: The background color to fill the image. Supports RGBA values.

---

## Example Usage

### Pasting images onto a solid background border

![image](images/paste_images_blank_background.jpg)

In this example, the `Blank Image` node creates a solid color background. Multiple images are then pasted onto this background to create a composite image with a border effect.  

![image](images/paste_images_blank_background_result.jpg)

### Generating random color noise for Text2Img workflows

![image](images/blank_image_random_color_noise.jpg)

In this example, a blank image is created with middle gray before being adjusted for saturation, hue, and brightness to create a random color input for a Text2Img workflow.