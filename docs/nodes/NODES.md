# Using the Node Editor

The nodes editor is a blank canvas allowing for the use of individual functions and image transformations to control the image generation workflow. The node processing flow is usually done from left (inputs) to right (outputs), but is not always the case the more complex the node graph becomes. Nodes inputs and outputs are connected by dragging connectors from node to node.

To better understand how nodes are used, think of how an electric power bar works. It takes in one input (electricity from a wall outlet) and passes it to multiple devices through multiple outputs. Similarly, a node could have multiple inputs and outputs functioning at the same (or different) time, but all node outputs pass information onward like a power bar passes electricity. Not all outputs are compatible with all inputs, however - Each node has different constraints on how it is expecting to input/output information. In general, node outputs are colour-coded to match compatible inputs of other nodes.


If you're not familiar with Diffusion, take a look at our [Diffusion Overview.](../help/diffusion.md) Understanding how diffusion works will enable you to more easily use the Nodes Editor and build workflows to suit your needs.

## Important Concepts

There are several node grouping concepts that can be examined with a narrow focus. These (and other) groupings can be pieced together to make up functional graph setups, and are important to understanding how groups of nodes work together as part of a whole. Note that the screenshots below aren't examples of complete functioning node graphs (see Examples).

### Noise

An initial noise tensor is necessary for the latent diffusion process. As a result, the Denoising node requires a noise node input.  

![groupsnoise](../assets/nodes/groupsnoise.png)

### Conditioning

Conditioning is necessary for the latent diffusion process, whether empty or not. As a result, the Denoising node requires positive and negative conditioning inputs. Conditioning is reliant on a CLIP text encoder provided by the Model Loader node.

![groupsconditioning](../assets/nodes/groupsconditioning.png)

### Image Space & VAE

The ImageToLatents node doesn't require a noise node input, but requires a VAE input to convert the image from image space into latent space. In reverse, the LatentsToImage node requires a VAE input to convert from latent space back into image space.

![groupsimgvae](../assets/nodes/groupsimgvae.png)

### Defined & Random Seeds

It is common to want to use both the same seed (for continuity) and random seeds (for variance). To define a seed, simply enter it into the 'Seed' field on a noise node. Conversely, the RandomInt node generates a random integer between 'Low' and 'High', and can be used as input to the 'Seed' edge point on a noise node to randomize your seed.

![groupsrandseed](../assets/nodes/groupsrandseed.png)

### Control

Control means to guide the diffusion process to adhere to a defined input or structure. Control can be provided as input to non-image *ToLatents nodes from ControlNet nodes. ControlNet nodes usually require an image processor which converts an input image for use with ControlNet.

![groupscontrol](../assets/nodes/groupscontrol.png)

### LoRA

The Lora Loader node lets you load a LoRA (say that ten times fast) and pass it as output to both the Prompt (Compel) and non-image *ToLatents nodes. A model's CLIP tokenizer is passed through the LoRA into Prompt (Compel), where it affects conditioning. A model's U-Net is also passed through the LoRA into a non-image *ToLatents node, where it affects noise prediction.

![groupslora](../assets/nodes/groupslora.png)

### Scaling

Use the ImageScale, ScaleLatents, and Upscale nodes to upscale images and/or latent images. The chosen method differs across contexts. However, be aware that latents are already noisy and compressed at their original resolution; scaling an image could produce more detailed results.

![groupsallscale](../assets/nodes/groupsallscale.png)

### Iteration + Multiple Images as Input

Iteration is a common concept in any processing, and means to repeat a process with given input. In nodes, you're able to use the Iterate node to iterate through collections usually gathered by the Collect node. The Iterate node has many potential uses, from processing a collection of images one after another, to varying seeds across multiple image generations and more. This screenshot demonstrates how to collect several images and pass them out one at a time.

![groupsiterate](../assets/nodes/groupsiterate.png)

### Multiple Image Generation + Random Seeds

Multiple image generation in the node editor is done using the RandomRange node. In this case, the 'Size' field represents the number of images to generate. As RandomRange produces a collection of integers, we need to add the Iterate node to iterate through the collection. 

To control seeds across generations takes some care. The first row in the screenshot will generate multiple images with different seeds, but using the same RandomRange parameters across invocations will result in the same group of random seeds being used across the images, producing repeatable results. In the second row, adding the RandomInt node as input to RandomRange's 'Seed' edge point will ensure that seeds are varied across all images across invocations, producing varied results.

![groupsmultigenseeding](../assets/nodes/groupsmultigenseeding.png)
