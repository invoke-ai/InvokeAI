# Using the Node Editor

The nodes editor is a blank canvas allowing for the use of individual functions and image transformations to control the image generation workflow. Nodes take in inputs on the left side of the node, and return an output on the right side of the node. A node graph is composed of multiple nodes that are connected together to create a workflow. Nodes' inputs and outputs are connected by dragging connectors from node to node. Inputs and outputs are color coded for ease of use.

To better understand how nodes are used, think of how an electric power bar works. It takes in one input (electricity from a wall outlet) and passes it to multiple devices through multiple outputs. Similarly, a node could have multiple inputs and outputs functioning at the same (or different) time, but all node outputs pass information onward like a power bar passes electricity. Not all outputs are compatible with all inputs, however - Each node has different constraints on how it is expecting to input/output information. In general, node outputs are colour-coded to match compatible inputs of other nodes.


If you're not familiar with Diffusion, take a look at our [Diffusion Overview.](../help/diffusion.md) Understanding how diffusion works will enable you to more easily use the Nodes Editor and build workflows to suit your needs.

## Important Concepts

There are several node grouping concepts that can be examined with a narrow focus. These (and other) groupings can be pieced together to make up functional graph setups, and are important to understanding how groups of nodes work together as part of a whole. Note that the screenshots below aren't examples of complete functioning node graphs (see Examples).

### Noise

An initial noise tensor is necessary for the latent diffusion process. As a result, the Denoising node requires a noise node input.  

![groupsnoise](../assets/nodes/groupsnoise.png)

### Text Prompt Conditioning

Conditioning is necessary for the latent diffusion process, whether empty or not. As a result, the Denoising node requires positive and negative conditioning inputs. Conditioning is reliant on a CLIP text encoder provided by the Model Loader node.

![groupsconditioning](../assets/nodes/groupsconditioning.png)

### Image to Latents & VAE

The ImageToLatents node takes in a pixel image and a VAE and outputs a latents. The LatentsToImage node does the opposite, taking in a latents and a VAE and outpus a pixel image. 

![groupsimgvae](../assets/nodes/groupsimgvae.png)

### Defined & Random Seeds

It is common to want to use both the same seed (for continuity) and random seeds (for variety). To define a seed, simply enter it into the 'Seed' field on a noise node. Conversely, the RandomInt node generates a random integer between 'Low' and 'High', and can be used as input to the 'Seed' edge point on a noise node to randomize your seed.

![groupsrandseed](../assets/nodes/groupsrandseed.png)

### ControlNet

The ControlNet node outputs a Control, which can be provided as input to non-image *ToLatents nodes. Depending on the type of ControlNet desired, ControlNet nodes usually require an image processor node, such as a Canny Processor or Depth Processor, which prepares an input image for use with ControlNet.

![groupscontrol](../assets/nodes/groupscontrol.png)

### LoRA

The Lora Loader node lets you load a LoRA and pass it as output.A LoRA provides fine-tunes to the UNet and text encoder weights that augment the base model’s image and text vocabularies.

![groupslora](../assets/nodes/groupslora.png)

### Scaling

Use the ImageScale, ScaleLatents, and Upscale nodes to upscale images and/or latent images. Upscaling is the process of enlarging an image and adding more detail. The chosen method differs across contexts. However, be aware that latents are already noisy and compressed at their original resolution; scaling an image could produce more detailed results.

![groupsallscale](../assets/nodes/groupsallscale.png)

### Iteration + Multiple Images as Input

Iteration is a common concept in any processing, and means to repeat a process with given input. In nodes, you're able to use the Iterate node to iterate through collections usually gathered by the Collect node. The Iterate node has many potential uses, from processing a collection of images one after another, to varying seeds across multiple image generations and more. This screenshot demonstrates how to collect several images and use them in an image generation workflow.

![groupsiterate](../assets/nodes/groupsiterate.png)

### Multiple Image Generation + Random Seeds

Multiple image generation in the node editor is done using the RandomRange node. In this case, the 'Size' field represents the number of images to generate. As RandomRange produces a collection of integers, we need to add the Iterate node to iterate through the collection. 

To control seeds across generations takes some care. The first row in the screenshot will generate multiple images with different seeds, but using the same RandomRange parameters across invocations will result in the same group of random seeds being used across the images, producing repeatable results. In the second row, adding the RandomInt node as input to RandomRange's 'Seed' edge point will ensure that seeds are varied across all images across invocations, producing varied results.

![groupsmultigenseeding](../assets/nodes/groupsmultigenseeding.png)
