---
title: Control Adapters
---

# :material-loupe: Control Adapters

## ControlNet

ControlNet is a powerful set of features developed by the open-source
community (notably, Stanford researcher
[**@ilyasviel**](https://github.com/lllyasviel)) that allows you to
apply a secondary neural network model to your image generation
process in Invoke.

With ControlNet, you can get more control over the output of your
image generation, providing you with a way to direct the network
towards generating images that better fit your desired style or
outcome.

ControlNet works by analyzing an input image, pre-processing that
image to identify relevant information that can be interpreted by each
specific ControlNet model, and then inserting that control information
into the generation process. This can be used to adjust the style,
composition, or other aspects of the image to better achieve a
specific result.

#### Installation

InvokeAI provides access to a series of ControlNet models that provide
different effects or styles in your generated images.

To install ControlNet Models:

1. The easiest way to install them is
to use the InvokeAI model installer application. Use the
`invoke.sh`/`invoke.bat` launcher to select item [4] and then navigate
to the CONTROLNETS section. Select the models you wish to install and
press "APPLY CHANGES". You may also enter additional HuggingFace
repo_ids in the "Additional models" textbox. 
2. Using the "Add Model" function of the  model manager, enter the HuggingFace Repo ID of the ControlNet. The ID is in the format "author/repoName"


_Be aware that some ControlNet models require additional code
functionality in order to work properly, so just installing a
third-party ControlNet model may not have the desired effect._ Please
read and follow the documentation for installing a third party model
not currently included among InvokeAI's default list.

Currently InvokeAI **only** supports 🤗 Diffusers-format ControlNet models. These are
folders that contain the files `config.json` and/or
`diffusion_pytorch_model.safetensors` and
`diffusion_pytorch_model.fp16.safetensors`. The name of the folder is
the name of the model.

🤗 Diffusers-format ControlNet models are available at HuggingFace
(http://huggingface.co) and accessed via their repo IDs (identifiers
in the format "author/modelname").

#### ControlNet Models
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

**QR Code Monster**:
A model that helps generate creative QR codes that still scan. Can also be used to create images with text, logos or shapes within them. 

**Openpose**: 
The OpenPose control model allows for the identification of the general pose of a character by pre-processing an existing image with a clear human structure. With advanced options, Openpose can also detect the face or hands in the image. 

*Note:* The DWPose Processor has replaced the OpenPose processor in Invoke. Workflows and generations that relied on the OpenPose Processor will need to be updated to use the DWPose Processor instead.

**Mediapipe Face**:

The MediaPipe Face identification processor is able to clearly identify facial features in order to capture vivid expressions of human faces.

**Tile**:

The Tile model fills out details in the image to match the image, rather than the prompt. The Tile Model is a versatile tool that offers a range of functionalities. Its primary capabilities can be boiled down to two main behaviors:

- It can reinterpret specific details within an image and create fresh, new elements.
- It has the ability to disregard global instructions if there's a discrepancy between them and the local context or specific parts of the image. In such cases, it uses the local context to guide the process.

The Tile Model can be a powerful tool in your arsenal for enhancing image quality and details. If there are undesirable elements in your images, such as blurriness caused by resizing, this model can effectively eliminate these issues, resulting in cleaner, crisper images. Moreover, it can generate and add refined details to your images, improving their overall quality and appeal. 

**Pix2Pix (experimental)**

With Pix2Pix, you can input an image into the controlnet, and then "instruct" the model to change it using your prompt. For example, you can say "Make it winter" to add more wintry elements to a scene.

Each of these models can be adjusted and combined with other ControlNet models to achieve different results, giving you even more control over your image generation process.


### Using ControlNet

To use ControlNet, you can simply select the desired model and adjust both the ControlNet and Pre-processor settings to achieve the desired result. You can also use multiple ControlNet models at the same time, allowing you to achieve even more complex effects or styles in your generated images.


Each ControlNet has two settings that are applied to the ControlNet.

Weight - Strength of the Controlnet model applied to the generation for the section, defined by start/end.

Start/End  - 0 represents the start of the generation, 1 represents the end. The Start/end setting controls what steps during the generation process have the ControlNet applied.

Additionally, each ControlNet section can be expanded in order to manipulate settings for the image pre-processor that adjusts your uploaded image before using it in when you Invoke.

## T2I-Adapter
[T2I-Adapter](https://github.com/TencentARC/T2I-Adapter) is a tool similar to ControlNet that allows for control over the generation process by providing control information during the generation process. T2I-Adapter models tend to be smaller and more efficient than ControlNets. 

##### Installation
To install T2I-Adapter Models:

1. The easiest way to install models is
to use the InvokeAI model installer application. Use the
`invoke.sh`/`invoke.bat` launcher to select item [5] and then navigate
to the T2I-Adapters section. Select the models you wish to install and
press "APPLY CHANGES". You may also enter additional HuggingFace
repo_ids in the "Additional models" textbox. 
2. Using the "Add Model" function of the  model manager, enter the HuggingFace Repo ID of the T2I-Adapter. The ID is in the format "author/repoName"

#### Usage
Each T2I Adapter has two settings that are applied.

Weight - Strength of the model applied to the generation for the section, defined by start/end.

Start/End  - 0 represents the start of the generation, 1 represents the end. The Start/end setting controls what steps during the generation process have the ControlNet applied.

Additionally, each  section can be expanded with the "Show Advanced" button in order to manipulate settings for the image pre-processor that adjusts your uploaded image before using it in during the generation process.


## IP-Adapter

[IP-Adapter](https://ip-adapter.github.io) is a tooling that allows for image prompt capabilities with text-to-image diffusion models. IP-Adapter works by analyzing the given image prompt to extract features, then passing those features to the UNet along with any other conditioning provided. 

![IP-Adapter + T2I](https://github.com/tencent-ailab/IP-Adapter/raw/main/assets/demo/ip_adpter_plus_multi.jpg)

![IP-Adapter + IMG2IMG](https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/main/assets/demo/image-to-image.jpg)

#### Installation
There are several ways to install IP-Adapter models with an existing InvokeAI installation:

1. Through the command line interface launched from the invoke.sh / invoke.bat scripts, option [4] to download models.
2. Through the Model Manager UI with models from the *Tools* section of [www.models.invoke.ai](https://www.models.invoke.ai). To do this, copy the repo ID from the desired model page, and paste it in the Add Model field of the model manager. **Note** Both the IP-Adapter and the Image Encoder must be installed for IP-Adapter to work. For example, the [SD 1.5 IP-Adapter](https://models.invoke.ai/InvokeAI/ip_adapter_plus_sd15) and [SD1.5 Image Encoder](https://models.invoke.ai/InvokeAI/ip_adapter_sd_image_encoder) must be installed to use IP-Adapter with SD1.5 based models.  
3. **Advanced -- Not recommended ** Manually downloading the IP-Adapter and Image Encoder files - Image Encoder folders shouid be placed in the `models\any\clip_vision` folders. IP Adapter Model folders should be placed in the relevant `ip-adapter` folder of relevant base model folder of Invoke root directory. For example, for the SDXL IP-Adapter, files should be added to the `model/sdxl/ip_adapter/` folder. 

#### Using IP-Adapter

IP-Adapter can be used by navigating to the *Control Adapters* options and enabling IP-Adapter. 

IP-Adapter requires an image to be used as the Image Prompt. It can also be used in conjunction with text prompts, Image-to-Image, Inpainting, Outpainting, ControlNets and LoRAs.


Each IP-Adapter has two settings that are applied to the IP-Adapter:

* Weight - Strength of the IP-Adapter model applied to the generation for the section, defined by start/end
* Start/End  - 0 represents the start of the generation, 1 represents the end. The Start/end setting controls what steps during the generation process have the IP-Adapter applied.
