# Nodes Editor

The nodes editor is a blank canvas where you add modular node windows for image generation. The node processing flow is usually done from left to right, though linearity can become abstracted the more complex the node graph becomes. Nodes are connected via wires/noodles.

To better understand how nodes are used, think of how an electric power bar works. It takes in one input (electricity from a wall outlet) and passes it to multiple devices through multiple outputs. Similarly, a node could have multiple inputs and outputs functioning at the same (or different) time, but all node outputs pass information onwards like a power bar passes electricity. Not all outputs are compatible with all inputs, however, much like a power bar can’t take in spaghetti noodles instead of electricity. In general, node outputs are colour-coded to match compatible inputs of other nodes.

## Anatomy of a Node

Individual nodes are made up of the following:

- Inputs: Edge points on the left side of the node window where you connect outputs from other nodes.
- Outputs: Edge points on the right side of the node window where you connect to inputs on other nodes.
- Options: Various options which are either manually configured, or overridden by connecting an output from another node to the input.

## Diffusion Overview

Taking the time to understand the diffusion process will help you to understand how to setup your nodes in the nodes editor. 

There are two main spaces Stable Diffusion works in: image space and latent space.

Image space represents images in pixel-form that you look at. Latent space represents compressed inputs. It’s in latent space that Stable Diffusion processes images. A VAE (Variational Auto Encoder) is responsible for compressing and encoding inputs into latent space, as well as decoding outputs back into image space.

When you generate an image using text-to-image, multiple steps occur in latent space:
1. Random noise is generated at the chosen height and width. The noise’s characteristics are dictated by the chosen (or not chosen) seed. This noise tensor is passed into latent space. We’ll call this noise A.
1. Using a model’s U-Net, a noise predictor examines noise A, and the words tokenized by CLIP from your prompt (conditioning). It generates its own noise tensor to predict what the final image might look like in latent space. We’ll call this noise B.
1. Noise B is subtracted from noise A in an attempt to create a final latent image indicative of the inputs. This step is repeated for the number of sampler steps chosen.
1. The VAE decodes the final latent image from latent space into image space.

image-to-image is a similar process, with only step 1 being different:
1. The input image is decoded from image space into latent space by the VAE. Noise is then added to the input latent image. Denoising Strength dictates how much noise is added, 0 being none, 1 being all-encompassing. We’ll call this noise A. The process is then the same as steps 2-4 in the text-to-image explanation above. 

Furthermore, a model provides the CLIP prompt tokenizer, the VAE, and a U-Net (where noise prediction occurs given a prompt and initial noise tensor).

A noise scheduler (eg. DPM++ 2M Karras) schedules the subtraction of noise from the latent image across the sampler steps chosen (step 3 above). Less noise is usually subtracted at higher sampler steps. 

## Node Types

_List all nodes with a short explanation of each_

## Examples

With our knowledge on the diffusion process, let’s break down some basic graphs in the nodes editor. Note that a node's options can be overridden by inputs from other nodes. These examples aren't strict rules to follow, and only demonstrate some basic configurations.

### Basic text-to-image Node Graph

<img width="875" alt="nodest2i" src="https://github.com/ymgenesis/InvokeAI/assets/25252829/17c67720-c376-4db8-94f0-5e00381a61ee">

- Model Loader: A necessity to generating images (as we’ve read above). We choose our model from the dropdown. It outputs a U-Net, CLIP tokenizer, and VAE.
- Prompt (Compel): Another necessity. Two prompt nodes are created. One will output positive conditioning (what you want, ‘dog’), one will output negative (what you don’t want, ‘cat’). They both input the CLIP tokenizer that the Model Loader node outputs.
- Noise: Consider this noise A from step one of the text-to-image explanation above. Choose a seed number, width, and height.
- TextToLatents: This node takes many inputs for converting and processing text & noise from image space into latent space, hence the name TextTo**Latents**. In this setup, it inputs positive and negative conditioning from the prompt nodes for processing (step 2 above). It inputs noise from the noise node for processing (steps 2 & 3 above). Lastly, it inputs a U-Net from the Model Loader node for processing (step 2 above). It outputs latents for use in the next LatentsToImage node. Choose number of sampler steps, CFG scale, and scheduler.
- LatentsToImage: This node takes in processed latents from the TextToLatents node, and the model’s VAE from the Model Loader node which is responsible for decoding latents back into the image space, hence the name LatentsTo**Image**. This node is the last stop, and once the image is decoded, it is saved to the gallery.

### Basic image-to-image Node Graph

<img width="998" alt="nodesi2i" src="https://github.com/ymgenesis/InvokeAI/assets/25252829/3f2c95d5-cee7-4415-9b79-b46ee60a92fe">

- Model Loader: Choose a model from the dropdown.
- Prompt (Compel): Two prompt nodes. One positive (dog), one negative (dog). Same CLIP inputs from the Model Loader node as before.
- ImageToLatents: Upload a a source image directly in the node window, via drag'n'drop from the gallery, or passed in as input. The ImageToLatents node inputs the VAE from the Model Loader node to decode the chosen image from image space into latent space, hence the name ImageTo**Latents**. It outputs latents for use in the next LatentsToLatents node. It also outputs the source image's width and height for use in the next Noise node if the final image is to be the same dimensions as the source image.
- Noise: A noise tensor is created with the width and height of the source image, and connected to the next LatentsToLatents node. Notice the width and height fields are overridden by the input from the ImageToLatents width and height outputs.
- LatentsToLatents: The inputs and options are nearly identical to TextToLatents, except that LatentsToLatents also takes latents as an input. Considering our source image is already converted to latents in the last ImageToLatents node, and text + noise are no longer the only inputs to process, we use the LatentsToLatents node. 
- LatentsToImage: Like previously, the LatentsToImage node will use the VAE from the Model Loader as input to decode the latents from LatentsToLatents into image space, and save it to the gallery.

### Basic ControlNet Node Graph

<img width="703" alt="nodescontrol" src="https://github.com/ymgenesis/InvokeAI/assets/25252829/b02ded86-ceb4-44a2-9910-e19ad184d471">

- Model Loader
- Prompt (Compel)
- Noise: Width and height of the CannyImageProcessor ControlNet image is passed in to set the dimensions of the noise passed to TextToLatents.
- CannyImageProcessor: The CannyImageProcessor node is used to process the source image being used as a ControlNet. Each ControlNet processor node applies control in different ways, and has some different options to configure. Width and height are passed to noise, as mentioned. The processed ControlNet image is output to the ControlNet node.
- ControlNet: Select the type of control model. In this case, canny is chosen as the CannyImageProcessor was used to generate the ControlNet image. Configure the control node options, and pass the control output to TextToLatents.
- TextToLatents: Similar to the basic text-to-image example, except ControlNet is passed to the control input edge point.
- LatentsToImage
