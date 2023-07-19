---
title: ControlNet
---

# :material-loupe: ControlNet

## ControlNet

ControlNet

ControlNet is a powerful set of features developed by the open-source
community (notably, Stanford researcher
[**@ilyasviel**](https://github.com/lllyasviel)) that allows you to
apply a secondary neural network model to your image generation
process in Invoke.

With ControlNet, you can get more control over the output of your
image generation, providing you with a way to direct the network
towards generating images that better fit your desired style or
outcome.


### How it works

ControlNet works by analyzing an input image, pre-processing that
image to identify relevant information that can be interpreted by each
specific ControlNet model, and then inserting that control information
into the generation process. This can be used to adjust the style,
composition, or other aspects of the image to better achieve a
specific result.


### Models

InvokeAI provides access to a series of ControlNet models that provide
different effects or styles in your generated images.  Currently
InvokeAI only supports "diffuser" style ControlNet models. These are
folders that contain the files `config.json` and/or
`diffusion_pytorch_model.safetensors` and
`diffusion_pytorch_model.fp16.safetensors`. The name of the folder is
the name of the model.

***InvokeAI does not currently support checkpoint-format
ControlNets. These come in the form of a single file with the
extension `.safetensors`.***

Diffuser-style ControlNet models are available at HuggingFace
(http://huggingface.co) and accessed via their repo IDs (identifiers
in the format "author/modelname"). The easiest way to install them is
to use the InvokeAI model installer application. Use the
`invoke.sh`/`invoke.bat` launcher to select item [5] and then navigate
to the CONTROLNETS section. Select the models you wish to install and
press "APPLY CHANGES". You may also enter additional HuggingFace
repo_ids in the "Additional models" textbox:

![Model Installer -
Controlnetl](../assets/installing-models/model-installer-controlnet.png){:width="640px"}

Command-line users can launch the model installer using the command
`invokeai-model-install`.

_Be aware that some ControlNet models require additional code
functionality in order to work properly, so just installing a
third-party ControlNet model may not have the desired effect._ Please
read and follow the documentation for installing a third party model
not currently included among InvokeAI's default list.

The models currently supported include:

**Canny**:

When the Canny model is used in ControlNet, Invoke will attempt to generate images that match the edges detected. 

Canny edge detection works by detecting the edges in an image by looking for abrupt changes in intensity. It is known for its ability to detect edges accurately while reducing noise and false edges, and the preprocessor can identify more information by decreasing the thresholds.

**M-LSD**: 

M-LSD is another edge detection algorithm used in ControlNet. It stands for Multi-Scale Line Segment Detector. 

It detects straight line segments in an image by analyzing the local structure of the image at multiple scales.  It can be useful for architectural imagery, or anything where straight-line structural information is needed for the resulting output. 

**Lineart**: 

The Lineart model in ControlNet generates line drawings from an input image. The resulting pre-processed image is a simplified version of the original, with only the outlines of objects visible.The Lineart model in ControlNet is known for its ability to accurately capture the contours of the objects in an input sketch. 

**Lineart Anime**: 

A variant of the Lineart model that generates line drawings with a distinct style inspired by anime and manga art styles.

**Depth**: 
A model that generates depth maps of images, allowing you to create more realistic 3D models or to simulate depth effects in post-processing.

**Normal Map (BAE):** 
A model that generates normal maps from input images, allowing for more realistic lighting effects in 3D rendering.
		
**Image Segmentation**: 
A model that divides input images into segments or regions, each of which corresponds to a different object or part of the image. (More details coming soon)


**Openpose**: 
The OpenPose control model allows for the identification of the general pose of a character by pre-processing an existing image with a clear human structure. With advanced options, Openpose can also detect the face or hands in the image. 

**Mediapipe Face**:

The MediaPipe Face identification processor is able to clearly identify facial features in order to capture vivid expressions of human faces.

**Tile (experimental)**:

The Tile model fills out details in the image to match the image, rather than the prompt. The Tile Model is a versatile tool that offers a range of functionalities. Its primary capabilities can be boiled down to two main behaviors:

- It can reinterpret specific details within an image and create fresh, new elements.
- It has the ability to disregard global instructions if there's a discrepancy between them and the local context or specific parts of the image. In such cases, it uses the local context to guide the process.

The Tile Model can be a powerful tool in your arsenal for enhancing image quality and details. If there are undesirable elements in your images, such as blurriness caused by resizing, this model can effectively eliminate these issues, resulting in cleaner, crisper images. Moreover, it can generate and add refined details to your images, improving their overall quality and appeal. 

**Pix2Pix (experimental)**

With Pix2Pix, you can input an image into the controlnet, and then "instruct" the model to change it using your prompt. For example, you can say "Make it winter" to add more wintry elements to a scene.

**Inpaint**: Coming Soon - Currently this model is available but not functional on the Canvas. An upcoming release will provide additional capabilities for using this model when inpainting.

Each of these models can be adjusted and combined with other ControlNet models to achieve different results, giving you even more control over your image generation process.


## Using ControlNet

To use ControlNet, you can simply select the desired model and adjust both the ControlNet and Pre-processor settings to achieve the desired result. You can also use multiple ControlNet models at the same time, allowing you to achieve even more complex effects or styles in your generated images.


Each ControlNet has two settings that are applied to the ControlNet.

Weight - Strength of the Controlnet model applied to the generation for the section, defined by start/end.

Start/End  - 0 represents the start of the generation, 1 represents the end. The Start/end setting controls what steps during the generation process have the ControlNet applied.

Additionally, each ControlNet section can be expanded in order to manipulate settings for the image pre-processor that adjusts your uploaded image before using it in when you Invoke.
