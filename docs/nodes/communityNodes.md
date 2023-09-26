# Community Nodes

These are nodes that have been developed by the community, for the community. If you're not sure what a node is, you can learn more about nodes [here](overview.md).

If you'd like to submit a node for the community, please refer to the [node creation overview](contributingNodes.md).

To download a node, simply download the `.py` node file from the link and add it to the `invokeai/app/invocations` folder in your Invoke AI install location. If you used the automated installation, this can be found inside the `.venv` folder. Along with the node, an example node graph should be provided to help you get started with the node. 

To use a community workflow, download the the `.json` node graph file and load it into Invoke AI via the **Load Workflow** button in the Workflow Editor. 

## Community Nodes

--------------------------------
### Ideal Size

**Description:** This node calculates an ideal image size for a first pass of a multi-pass upscaling. The aim is to avoid duplication that results from choosing a size larger than the model is capable of.

**Node Link:** https://github.com/JPPhoto/ideal-size-node

--------------------------------
### Film Grain

**Description:** This node adds a film grain effect to the input image based on the weights, seeds, and blur radii parameters. It works with RGB input images only.

**Node Link:** https://github.com/JPPhoto/film-grain-node

--------------------------------
### Image Picker

**Description:** This InvokeAI node takes in a collection of images and randomly chooses one. This can be useful when you have a number of poses to choose from for a ControlNet node, or a number of input images for another purpose.

**Node Link:** https://github.com/JPPhoto/image-picker-node

--------------------------------
### Retroize

**Description:** Retroize is a collection of nodes for InvokeAI to "Retroize" images. Any image can be given a fresh coat of retro paint with these nodes, either from your gallery or from within the graph itself. It includes nodes to pixelize, quantize, palettize, and ditherize images; as well as to retrieve palettes from existing images.

**Node Link:** https://github.com/Ar7ific1al/invokeai-retroizeinode/

**Retroize Output Examples**

![image](https://github.com/Ar7ific1al/InvokeAI_nodes_retroize/assets/2306586/de8b4fa6-324c-4c2d-b36c-297600c73974)

--------------------------------
### GPT2RandomPromptMaker

**Description:** A node for InvokeAI utilizes the GPT-2 language model to generate random prompts based on a provided seed and context.

**Node Link:** https://github.com/mickr777/GPT2RandomPromptMaker

**Output Examples** 

Generated Prompt: An enchanted weapon will be usable by any character regardless of their alignment.

![9acf5aef-7254-40dd-95b3-8eac431dfab0 (1)](https://github.com/mickr777/InvokeAI/assets/115216705/8496ba09-bcdd-4ff7-8076-ff213b6a1e4c)

--------------------------------
### Load Video Frame

**Description:** This is a video frame image provider + indexer/video creation nodes for hooking up to iterators and ranges and ControlNets and such for invokeAI node experimentation. Think animation + ControlNet outputs.

**Node Link:** https://github.com/helix4u/load_video_frame

**Example Node Graph:**  https://github.com/helix4u/load_video_frame/blob/main/Example_Workflow.json

**Output Example:** 
=======
![Example animation](https://github.com/helix4u/load_video_frame/blob/main/testmp4_embed_converted.gif)
[Full mp4 of Example Output test.mp4](https://github.com/helix4u/load_video_frame/blob/main/test.mp4)

--------------------------------

### Oobabooga

**Description:** asks a local LLM running in Oobabooga's Text-Generation-Webui to write a prompt based on the user input.

**Link:** https://github.com/sammyf/oobabooga-node


**Example:**

"describe a new mystical  creature in its natural environment"

*can return*

"The mystical creature I am describing to you is called the "Glimmerwing". It is a majestic, iridescent being that inhabits the depths of the most enchanted forests and glimmering lakes. Its body is covered in shimmering scales that reflect every color of the rainbow, and it has delicate, translucent wings that sparkle like diamonds in the sunlight. The Glimmerwing's home is a crystal-clear lake, surrounded by towering trees with leaves that shimmer like jewels. In this serene environment, the Glimmerwing spends its days swimming gracefully through the water, chasing schools of glittering fish and playing with the gentle ripples of the lake's surface.
As the sun sets, the Glimmerwing perches on a branch of one of the trees, spreading its wings to catch the last rays of light. The creature's scales glow softly, casting a rainbow of colors across the forest floor. The Glimmerwing sings a haunting melody, its voice echoing through the stillness of the night air. Its song is said to have the power to heal the sick and bring peace to troubled souls. Those who are lucky enough to hear the Glimmerwing's song are forever changed by its beauty and grace."

![glimmerwing_small](https://github.com/sammyf/oobabooga-node/assets/42468608/cecdd820-93dd-4c35-abbf-607e001fb2ed)

**Requirement**

a Text-Generation-Webui instance (might work remotely too, but I never tried it) and obviously InvokeAI 3.x

**Note**

This node works best with SDXL models, especially as the style can be described independantly of the LLM's output.

--------------------------------
### Depth Map from Wavefront OBJ

**Description:** Render depth maps from Wavefront .obj files (triangulated) using this simple 3D renderer utilizing numpy and matplotlib to compute and color the scene. There are simple parameters to change the FOV, camera position, and model orientation.

To be imported, an .obj must use triangulated meshes, so make sure to enable that option if exporting from a 3D modeling program. This renderer makes each triangle a solid color based on its average depth, so it will cause anomalies if your .obj has large triangles. In Blender, the Remesh modifier can be helpful to subdivide a mesh into small pieces that work well given these limitations.

**Node Link:** https://github.com/dwringer/depth-from-obj-node

**Example Usage:**
![depth from obj usage graph](https://raw.githubusercontent.com/dwringer/depth-from-obj-node/main/depth_from_obj_usage.jpg)

--------------------------------
### Generative Grammar-Based Prompt Nodes

**Description:** This set of 3 nodes generates prompts from simple user-defined grammar rules (loaded from custom files - examples provided below). The prompts are made by recursively expanding a special template string, replacing nonterminal "parts-of-speech" until no more nonterminal terms remain in the string.

This includes 3 Nodes:
- *Lookup Table from File* - loads a YAML file "prompt" section (or of a whole folder of YAML's) into a JSON-ified dictionary (Lookups output)
- *Lookups Entry from Prompt* - places a single entry in a new Lookups output under the specified heading
- *Prompt from Lookup Table* - uses a Collection of Lookups as grammar rules from which to randomly generate prompts.

**Node Link:** https://github.com/dwringer/generative-grammar-prompt-nodes

**Example Usage:**
![lookups usage example graph](https://raw.githubusercontent.com/dwringer/generative-grammar-prompt-nodes/main/lookuptables_usage.jpg)

--------------------------------
### Image and Mask Composition Pack

**Description:** This is a pack of nodes for composing masks and images, including a simple text mask creator and both image and latent offset nodes. The offsets wrap around, so these can be used in conjunction with the Seamless node to progressively generate centered on different parts of the seamless tiling.

This includes 14 Nodes:
- *Adjust Image Hue Plus* - Rotate the hue of an image in one of several different color spaces.
- *Blend Latents/Noise (Masked)* - Use a mask to blend part of one latents tensor [including Noise outputs] into another. Can be used to "renoise" sections during a multi-stage [masked] denoising process.
- *Enhance Image* - Boost or reduce color saturation, contrast, brightness, sharpness, or invert colors of any image at any stage with this simple wrapper for pillow [PIL]'s ImageEnhance module.
- *Equivalent Achromatic Lightness* - Calculates image lightness accounting for Helmholtz-Kohlrausch effect based on a method described by High, Green, and Nussbaum (2023).
- *Text to Mask (Clipseg)* - Input a prompt and an image to generate a mask representing areas of the image matched by the prompt.
- *Text to Mask Advanced (Clipseg)* - Output up to four prompt masks combined with logical "and", logical "or", or as separate channels of an RGBA image.
- *Image Layer Blend* - Perform a layered blend of two images using alpha compositing. Opacity of top layer is selectable, with optional mask and several different blend modes/color spaces.
- *Image Compositor* - Take a subject from an image with a flat backdrop and layer it on another image using a chroma key or flood select background removal.
- *Image Dilate or Erode* - Dilate or expand a mask (or any image!). This is equivalent to an expand/contract operation.
- *Image Value Thresholds* - Clip an image to pure black/white beyond specified thresholds.
- *Offset Latents* - Offset a latents tensor in the vertical and/or horizontal dimensions, wrapping it around.
- *Offset Image* - Offset an image in the vertical and/or horizontal dimensions, wrapping it around.
- *Shadows/Highlights/Midtones* - Extract three masks (with adjustable hard or soft thresholds) representing shadows, midtones, and highlights regions of an image.
- *Text Mask (simple 2D)* - create and position a white on black (or black on white) line of text using any font locally available to Invoke.

**Node Link:** https://github.com/dwringer/composition-nodes

**Nodes and Output Examples:**
![composition nodes usage graph](https://raw.githubusercontent.com/dwringer/composition-nodes/main/composition_pack_overview.jpg)

--------------------------------
### Size Stepper Nodes

**Description:** This is a set of nodes for calculating the necessary size increments for doing upscaling workflows. Use the *Final Size & Orientation* node to enter your full size dimensions and orientation (portrait/landscape/random), then plug that and your initial generation dimensions into the *Ideal Size Stepper* and get 1, 2, or 3 intermediate pairs of dimensions for upscaling. Note this does not output the initial size or full size dimensions: the 1, 2, or 3 outputs of this node are only the intermediate sizes.

A third node is included, *Random Switch (Integers)*, which is just a generic version of Final Size with no orientation selection.

**Node Link:** https://github.com/dwringer/size-stepper-nodes

**Example Usage:**
![size stepper usage graph](https://raw.githubusercontent.com/dwringer/size-stepper-nodes/main/size_nodes_usage.jpg)

--------------------------------

### Text font to Image

**Description:** text font to text image node for InvokeAI, download a font to use (or if in font cache uses it from there), the text is always resized to the image size, but can control that with padding, optional 2nd line

**Node Link:** https://github.com/mickr777/textfontimage

**Output Examples**

![a3609d48-d9b7-41f0-b280-063d857986fb](https://github.com/mickr777/InvokeAI/assets/115216705/c21b0af3-d9c6-4c16-9152-846a23effd36)

Results after using the depth controlnet

![9133eabb-bcda-4326-831e-1b641228b178](https://github.com/mickr777/InvokeAI/assets/115216705/915f1a53-968e-43eb-aa61-07cd8f1a733a)
![4f9a3fa8-9be9-4236-8a3e-fcec66decd2a](https://github.com/mickr777/InvokeAI/assets/115216705/821ef89e-8a60-44f5-b94e-471a9d8690cc)
![babd69c4-9d60-4a55-a834-5e8397f62610](https://github.com/mickr777/InvokeAI/assets/115216705/2befcb6d-49f4-4bfd-b5fc-1fee19274f89)

--------------------------------

### Prompt Tools 

**Description:** A set of InvokeAI nodes that add general prompt manipulation tools.  These where written to accompany the PromptsFromFile node and other prompt generation nodes.

1. PromptJoin - Joins to prompts into one.
2. PromptReplace - performs a search and replace on a prompt. With the option of using regex.
3. PromptSplitNeg - splits a prompt into positive and negative using the old V2 method of [] for negative.
4. PromptToFile - saves a prompt or collection of prompts to a file. one per line. There is an append/overwrite option.
5. PTFieldsCollect - Converts image generation fields into a Json format string that can be passed to Prompt to file. 
6. PTFieldsExpand - Takes Json string and converts it to individual generation parameters This can be fed from the Prompts to file node.
7. PromptJoinThree -  Joins 3 prompt together.
8. PromptStrength - This take a string and float and outputs another string in the format of (string)strength like the weighted format of compel. 
9. PromptStrengthCombine - This takes a collection of prompt strength strings and outputs a string in the .and() or .blend() format that can be fed into a proper prompt node.

See full docs here: https://github.com/skunkworxdark/Prompt-tools-nodes/edit/main/README.md

**Node Link:** https://github.com/skunkworxdark/Prompt-tools-nodes

--------------------------------

### XY Image to Grid and Images to Grids nodes

**Description:** Image to grid nodes and supporting tools.

1. "Images To Grids" node - Takes a collection of images and creates a grid(s) of images. If there are more images than the size of a single grid then mutilple grids will be created until it runs out of images.
2. "XYImage To Grid" node - Converts a collection of XYImages into a labeled Grid of images.  The XYImages collection has to be built using the supporoting nodes. See example node setups for more details.
   

See full docs here: https://github.com/skunkworxdark/XYGrid_nodes/edit/main/README.md

**Node Link:** https://github.com/skunkworxdark/XYGrid_nodes

--------------------------------

### Example Node Template

**Description:** This node allows you to do super cool things with InvokeAI.

**Node Link:** https://github.com/invoke-ai/InvokeAI/fake_node.py

**Example Node Graph:**  https://github.com/invoke-ai/InvokeAI/fake_node_graph.json

**Output Examples** 

![Example Image](https://invoke-ai.github.io/InvokeAI/assets/invoke_ai_banner.png){: style="height:115px;width:240px"}


## Disclaimer

The nodes linked have been developed and contributed by members of the Invoke AI community. While we strive to ensure the quality and safety of these contributions, we do not guarantee the reliability or security of the nodes. If you have issues or concerns with any of the nodes below, please raise it on GitHub or in the Discord.


## Help
If you run into any issues with a node, please post in the [InvokeAI Discord](https://discord.gg/ZmtBAhwWhy). 

