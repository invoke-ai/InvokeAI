# Community Nodes

These are nodes that have been developed by the community, for the community. If you're not sure what a node is, you can learn more about nodes [here](overview.md).

If you'd like to submit a node for the community, please refer to the [node creation overview](contributingNodes.md).

To download a node, simply download the `.py` node file from the link and add it to the `invokeai/app/invocations` folder in your Invoke AI install location. Along with the node, an example node graph should be provided to help you get started with the node. 

To use a community node graph, download the the `.json` node graph file and load it into Invoke AI via the **Load Nodes** button on the Node Editor. 

## Community Nodes

### FaceTools

**Description:** FaceTools is a collection of nodes created to manipulate faces as you would in Unified Canvas. It includes FaceMask, FaceOff, and FacePlace. FaceMask autodetects a face in the image using MediaPipe and creates a mask from it. FaceOff similarly detects a face, then takes the face off of the image by adding a square bounding box around it and cropping/scaling it. FacePlace puts the bounded face image from FaceOff back onto the original image. Using these nodes with other inpainting node(s), you can put new faces on existing things, put new things around existing faces, and work closer with a face as a bounded image. Additionally, you can supply X and Y offset values to scale/change the shape of the mask for finer control on FaceMask and FaceOff. See GitHub repository below for usage examples.

**Node Link:** https://github.com/ymgenesis/FaceTools/

**FaceMask Output Examples** 

![5cc8abce-53b0-487a-b891-3bf94dcc8960](https://github.com/invoke-ai/InvokeAI/assets/25252829/43f36d24-1429-4ab1-bd06-a4bedfe0955e)
![b920b710-1882-49a0-8d02-82dff2cca907](https://github.com/invoke-ai/InvokeAI/assets/25252829/7660c1ed-bf7d-4d0a-947f-1fc1679557ba)
![71a91805-fda5-481c-b380-264665703133](https://github.com/invoke-ai/InvokeAI/assets/25252829/f8f6a2ee-2b68-4482-87da-b90221d5c3e2)

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

**Node Link:** https://github.com/JPPhoto/film-grain-node

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

**Description:** Render depth maps from Wavefront .obj files (triangulated) using this simple 3D renderer utilizing numpy and matplotlib to compute and color the scene. 

To be imported, an .obj must use triangulated meshes, so make sure to enable that option if exporting from a 3D modeling program. This renderer makes each triangle a solid color based on its average depth, so it will cause anomalies if your .obj has large triangles. In Blender, the Remesh modifier can be helpful to subdivide a mesh into small pieces that work well given these limitations.

There are simple parameters to change the FOV, camera position, and model orientation.

Additional parameters like different image sizes will probably be added, and things like more sophisticated rotations are planned but this node is experimental and may or may not change much.

**Node Link:** https://github.com/dwringer/invoke-nodes/blob/main/depth_from_obj.py

--------------------------------
### Enhance Image

**Description:** Boost or reduce color saturation, contrast, brightness, sharpness, or invert colors of any image at any stage with this simple wrapper for pillow [PIL]'s ImageEnhance module.

Color inversion is toggled with a simple switch, while each of the four enhancer modes are activated by entering a value other than 1 in each corresponding input field. Values less than 1 will reduce the corresponding property, while values greater than 1 will enhance it.

**Node Link:** https://github.com/dwringer/invoke-nodes/blob/main/image_enhance.py

--------------------------------
### Generative Grammar-Based Prompt Nodes

**Description:** This generates prompts from simple user-defined grammar rules (loaded from custom files - examples provided below). The prompts are made by recursively expanding a special template string, replacing nonterminal "parts-of-speech" until no more nonterminal terms remain in the string.

**Three nodes are included:**
- *Lookup Table from File* - loads a YAML file "prompt" section (or of a whole folder of YAML's) into a JSON-ified dictionary (Lookups output)
- *Lookups Entry from Prompt* - places a single entry in a new Lookups output under the specified heading
- *Prompt from LookupTable* - uses a Collection of Lookups as grammar rules from which to randomly generate prompts.

**Node Link:** https://github.com/dwringer/invoke-nodes/blob/main/lookuptables.py

**Example Templates:** 
- https://github.com/dwringer/invoke-nodes/blob/main/example_template.yaml
- https://github.com/dwringer/invoke-nodes/blob/main/movies.yaml
- https://github.com/dwringer/invoke-nodes/blob/main/photograph.yaml

**Example Usage:**
![lookups usage example graph](https://raw.githubusercontent.com/dwringer/invoke-nodes/main/lookuptables_usage.jpg)

--------------------------------
### Ideal Size Stepper

**Description:** Plug in your full size dimensions as well as JPPhoto's Ideal Size node's output dimensions, and get 1, 2, or 3 intermediate pairs of dimensions for upscaling based on the natural log of the image area. Thus, each successive generation (from ideal size to full size) adds approximately the same percentage of new pixels to the image. Note this does not output the ideal size or full size dimensions. The 1, 2, or 3 outputs of this node are only the intermediate step sizes.

There are up to three stages which determine how many intermediate sizes to compute and output. With Tapers B and C disabled, outputs A, B, and C will be the same (inactive outputs yield copies of the previous pair). With a taper assigned to B, Width/Height A and B will both be active with progressively larger intermediate resolutions, and if Taper C is activated, active outputs will be calculated with even finer gradations.

**Node Link:** https://github.com/dwringer/invoke-nodes/blob/main/ideal_size_stepper.py

--------------------------------
### Image Compositor

**Description:** Take a subject from an image with a flat backdrop and layer it on another image using a chroma key to specify a color value/threshold to remove backdrop pixels, or leave the color blank and a "flood select" will be used from the image corners.

The subject image may be scaled using the fill X and fill Y options (enable both to stretch-fit).  Final subject position may also be adjusted with X offset and Y offset. If used, chroma key may be specified either as an (R, G, B) tuple, or a CSS-3 color string.

**Node Link:** https://github.com/dwringer/invoke-nodes/blob/main/image_composite.py

--------------------------------
### Final Size & Orientation / Random Switch (Integers)

**Description:** Input two integers, get two out in random, landscape, or portrait orientations.

Pretty self explanatory. You can just enter your height/width in the two inputs and then get the requested configurations of WxH or HxW from the two outputs. 

Contains two nodes: Final Size & Orientation, and Random Switch (Integer) which was the original, slightly more generalized version.

**Node Link:** https://github.com/dwringer/invoke-nodes/blob/main/random_switch.py

--------------------------------
### Text Mask (simple 2D)

**Description:** Create a white on black (or black on white) text image for use with controlnets or further processing in other nodes. Specify any TTF/OTF font file available to Invoke and control parameters to resize, rotate, and reposition the text.

Currently this only generates one line of text, but it can be layered with other images using the Image Compositor node or any other such tool.

**Node Link:** https://github.com/dwringer/invoke-nodes/blob/main/text_mask.py

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

