The Unified Canvas is a tool designed to facilitate composition, enabling artists to use all available generation modes (TextToImage, ImageToImage, Inpainting, and Outpainting) in a single unified workflow. The flexibility of the tool allows for users to correct image generations, extend images, and create new content within (or outside) existing images. 

This documentation will help explain the basics of using the Unified Canvas, as well as some advanced tools available to power users of the canvas.

# Basics
The Unified Canvas consists of two layers: The **Base Layer** and the **Mask Layer**, which can be toggled using the (Q) hotkey.

### Base Layer
The **Base Layer** is the image content currently managed by the canvas, and can be exported at any time to the gallery by using the **Save to Gallery** option. When the Base Layer is selected, the Brush (B) and Eraser (E) tools will directly manipulate the base layer. Any images uploaded to the canvas (or sent to the unified canvas from the gallery) will reset the canvas, and add the image as your initial base layer image.

### Staging Area
When you generate images, they will display in the canvas **Staging Area**, alongside the Staging Area toolbar buttons. While the Staging Area is active, you cannot interact with the canvas.

<figure markdown>
![staging area](../assets/canvas/staging_area.png)
</figure>

Accepting generations will commit the new generation to the **Base Layer**. You can review all generated images using the Prev/Next arrows, save any individual generations to your gallery (without committing to the Base layer) or discard generations. While you can Undo a discard in an individual Canvas session, any generations that are not saved will be lost when the canvas resets.

### Mask Layer
The **Mask Layer** consists of any masked sections that have been created to inform Inpainting generations. You can generate a new mask by using the Brush tool with the Mask layer set as your Active layer. Any masked areas will only affect generation inside of the current bounding box.

### Bounding Box
When generating a new image, Invoke will process and apply new images within the area denoted by the **Bounding Box**. New invocations will be generated and applied based on the Width & Height settings of the Bounding Box. The Bounding Box can be moved and resized using the Move (V) tool - It can also be resized using the Bounding Box options in the Options Panel. This allows users to generate larger or smaller images, control which sections of the image are being processed, as well as control Bounding Box tools like the Bounding Box fill/erase. 

### Inpainting & Outpainting
"Inpainting" means asking the AI to refine part of an image while leaving the rest alone. For example, updating a portrait of your grandmother to have her wear a biker's jacket.

<figure markdown>
![granny with a mask applied](../assets/canvas/mask_granny.png)
</figure>

<figure markdown>
![just like magic, granny with a biker's jacket](../assets/canvas/biker_jacket_granny.png)
</figure>

"Outpainting" means asking the AI to expand the original image beyond its original borders, making a bigger image that's still based on the original. For example, extending the above image of your Grandmother in a biker's jacket to include her wearing jeans (and while we're at it, a motorcycle!)

<figure markdown>
![more magic - granny with a tattooed arm, denim pants, and an obscured motorcycle](../assets/canvas/biker_jacket_granny.png)
</figure>

# Getting Started

To get started with the Unified Canvas, you will want to generate a new base layer using Txt2Img or importing an initial image - We'll refer to either of these methods as the "initial image" in the below guide.

From there, you can consider the following techniques to augment your image:
* **New Images**: Move the bounding box to an empty area of the canvas, type in your prompt, and Invoke, to generate a new image using the Text to Image function.
* **Image Correction**: Use the color picker and brush tool to paint corrections on the image, switch to the Mask layer, and brush a mask over your painted area to use **Inpainting**. You can also use the **Img2Img** generation method to invoke new interpretations of the image. 
* **Image Expansion**: Move the bounding box to include a portion of your initial image, and a portion of transparent canvas, then invoke using a prompt that describes what you'd like to see in that area - This will outpaint the image. You'll typically find more coherent results if you keep about 50-60% of the original image in the bounding box, and make sure that you Image to Image Strength is high. 
* **New Content on Existing Images**: If you want to add new details or objects into your image, use the brush tool to paint a sketch of what you'd like to see on the image, switch to the Mask layer, and brush a mask over your painted area to use **Inpainting** - If the masked area is small, consider using a smaller bounding box to take advantage of Scaling features which produce better details. 
* **And more**: There are a number of creative ways to use the Canvas, and the above are just starting points. We're excited to see what you come up with!


# Generation Methods
The Canvas can use all generation methods available (Txt2Img, Img2Img, Inpainting, and Outpainting), and these will be automatically selected and used based on the current selection area within the Bounding Box.

### Text to Image -
If the Bounding Box is placed over an area of canvas with an **empty Base Layer**, invoking a new image will use **Text to Image**. This generates an entirely new image based on your prompt.

### Image to Image - 
If the Bounding Box is placed over an area of canvas with an **existing Base Layer area with no transparent pixels or masks**, invoking a new image will use **Image to Image**. This uses the image within the bounding box and your prompt to interpret a new image. The image will be closer to your original image at lower Image to Image strengths.

### Inpainting - 
If the Bounding Box is placed over an area of canvas with an **existing Base Layer and any pixels selected using the Mask layer**, invoking a new image will use **Inpainting**. Inpainting uses the existing colors/forms in the masked area in order to generate a new image for the masked area only. The unmasked portion of the image will remain the same. Image to Image strength applies to the inpainted area.

If you desire something completely different than the original image in your new generation (i.e., ignoring existing colors/forms), consider toggling the Inpaint Replace setting on, and use high values for both that and Image to Image strength.

Note: By default, the **Scale Before Processing** option will be set to automatically generate at a larger resolution and scale down to inpaint more coherent details. In order to leverage this feature, the best inpainting results will be found by resizing the bounding box to a smaller area that contains your mask, and updating your prompt to describe *just* the area within the bounding box.

### Outpainting - 
If the Bounding Box is placed over an area of canvas partially filled by an existing Base Layer area and partially by transparent pixels or masks, invoking a new image will use **Outpainting**, as well as **Inpainting** any masked areas.
 

# Advanced Features

Features with non-obvious behavior are detailed below, in order to provide clarity on the intent and common use cases we expect for utilizing them.

## Toolbar

### Mask Options
* **Enable Mask** - This flag can be used to Enable or Disable the currently painted mask. This allows you the option to maintain your mask layer without it affecting current invocations.
* **Preserve Masked Area** - When enabled, Preserve Masked Area inverts the inpainting process, maintaining the masked areas and regenerating unmasked areas.

### Creative Tools
* **Brush - Base/Mask Modes** - Without user intervention, the Brush tool will operate between two modes for Base and Mask Layers respectively. On Base Layer mode, the brush will directly paint on the canvas using the color selected on the Brush Options menu. On Mask Layer mode, the brush will create a new mask. The color of the mask created is controlled by the color selector on the Mask Options dropdown.
* **Erase Bounding Box** - Erases all content on the base layer within the current bounding box.
* **Fill Bounding Box** - Fill the base layer within the current bounding box with the currently selected color

### Canvas Tools
* **Move Tool** - Allows for manipulation of the canvas view (by dragging on the canvas, outside the bounding box), the Bounding Box (using the edges of the box), or the Width/Height of the Bounding Box (using the 9-directional handles).
* **Reset View** - Will reorient the user to the image within the Bounding box.
* **Merge Visible** - In the event your computer is demonstrating performance issues, using this feature will consolidate all information currently being rendered in your browser into a merged copy of the image, improving the amount of resources available.

## Seam Correction
In this context, the term `seam` is used to describe the area that Invoke will process on the boundary between existing and newly created image areas during the final generation step of inpainting or outpainting. This process generates a mask and then inpaints over the edge between original image and transparent regions. These controls determine how that seam will be generated. 

A wider seam and a blur setting of about 1/3 of the seam have been noted as producing consistently strong results (e.g. 96 wide and 16 blur - adds up to 32 blur with both sides). Seam strength of 0.7 is best for reducing hard seams.
* **Seam Size** - This setting controls the size of the seam masked area
* **Seam Blur** - This setting controls the size of the blur on *each* side of the masked area
* **Seam Strength** - This setting controls the Image2Image strength applied to the seam area
* **Seam Steps** - This setting controls the number of steps applied to the seam.

## Infill & Scaling
* **Scale Before Processing & W/H**: When generating images with a bounding box smaller than the optimized W/H of the model (e.g., 512x512 for SD1.5), this feature first generates at a larger size with the same aspect ratio, and then scales that image down to fill the selected area. It is particularly useful when inpainting very small details. This feature is an optional feature that is enabled by default.
* **Inpaint Replace**: When inpainting, the default method is to utilize the existing RGB values of the base layer to inform the generation process. This feature instead fills the area with noise (completely replacing the original RGB values at an Inpaint Replace value of 1), which can help generate more significant variations from the original image. You should always use a higher ImageToImage strength value when using Inpaint replace, especially at high Inpaint Replace values
* **Infill Method**: Patchmatch and Tile are two methods for producing RGB values for use in the outpainting process. We believe that Patchmatch is a superior method, however provide support for Tile in the event that Patchmatch cannot be installed or is unavailable.
* **Tile Size**: When using the Tile method for generation, Tile will source small portions of the original image, and randomly place these in the area being outpainted. This value sets the size of those tiles.

# Hot Keys
The Unified Canvas is a tool that excels when users utilize hotkeys. You can view the full list of keyboard shortcuts, updated with all new features, by clicking the Keyboard Shortcuts icon at the top right of the InvokeAI WebUI.
