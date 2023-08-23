# ComfyUI to InvokeAI

If you're coming to InvokeAI from ComfyUI, welcome! You'll find things are similar but different - the good news is that you already know how things should work, and it's just a matter of wiring them up! 

Some things to note: 

- InvokeAI's nodes tend to be more granular than default nodes in Comfy. This means each node in Invoke will do a specific task and you might need to use multiple nodes to achieve the same result. The added granularity improves the control you have have over your workflows. 
- InvokeAI's backend and ComfyUI's backend are very different which means Comfy workflows are not able to be imported into InvokeAI. However, we have created a [list of popular workflows](exampleWorkflows.md) for you to get started with Nodes in InvokeAI!

## Node Equivalents:

| Comfy UI Category | ComfyUI Node | Invoke Equivalent      |
|:---------------------------------- |:---------------------------------- | :----------------------------------|
| Sampling |KSampler |Denoise Latents|
| Sampling |Ksampler Advanced|Denoise Latents |
| Loaders |Load Checkpoint | Main Model Loader _or_ SDXL Main Model Loader|
| Loaders |Load VAE | VAE Loader |
| Loaders |Load Lora | LoRA Loader _or_ SDXL Lora Loader|
| Loaders |Load ControlNet Model | ControlNet|
| Loaders |Load ControlNet Model (diff) | ControlNet|
| Loaders |Load Style Model | Reference Only ControlNet will be coming in a future version of InvokeAI|
| Loaders |unCLIPCheckpointLoader | N/A |
| Loaders |GLIGENLoader | N/A |
| Loaders |Hypernetwork Loader | N/A |
| Loaders |Load Upscale Model | Occurs within "Upscale (RealESRGAN)"|
|Conditioning |CLIP Text Encode (Prompt) | Compel (Prompt) or SDXL Compel (Prompt) |
|Conditioning |CLIP Set Last Layer | CLIP Skip|
|Conditioning |Conditioning (Average) | Use the .blend() feature of prompts |
|Conditioning |Conditioning (Combine) | N/A |
|Conditioning |Conditioning (Concat) | See the Prompt Tools Community Node|
|Conditioning |Conditioning (Set Area) | N/A |
|Conditioning |Conditioning (Set Mask) | Mask Edge |
|Conditioning |CLIP Vision Encode | N/A |
|Conditioning |unCLIPConditioning | N/A |
|Conditioning |Apply ControlNet | ControlNet |
|Conditioning |Apply ControlNet (Advanced) | ControlNet |
|Latent |VAE Decode | Latents to Image|
|Latent |VAE Encode | Image to Latents |
|Latent |Empty Latent Image | Noise |
|Latent |Upscale Latent |Resize Latents |
|Latent |Upscale Latent By |Scale Latents |
|Latent |Latent Composite | Blend Latents |
|Latent |LatentCompositeMasked | N/A |
|Image |Save Image | Image |
|Image |Preview Image |Current |
|Image |Load Image | Image|
|Image |Empty Image| Blank Image |
|Image |Invert Image | Invert Lerp Image |
|Image |Batch Images | Link "Image" nodes into an "Image Collection" node |
|Image |Pad Image for Outpainting | Outpainting is easily accomplished in the Unified Canvas |
|Image |ImageCompositeMasked | Paste Image |
|Image | Upscale Image | Resize Image |
|Image | Upscale Image By | Upscale Image |
|Image | Upscale Image (using Model) | Upscale Image |
|Image | ImageBlur | Blur Image |
|Image | ImageQuantize | N/A |
|Image | ImageSharpen | N/A |
|Image | Canny | Canny Processor |
|Mask |Load Image (as Mask) | Image |
|Mask |Convert Mask to Image | Image|
|Mask |Convert Image to Mask | Image |
|Mask |SolidMask | N/A |
|Mask |InvertMask |Invert Lerp Image |
|Mask |CropMask | Crop Image |
|Mask |MaskComposite | Combine Mask |
|Mask |FeatherMask | Blur Image |
|Advanced | Load CLIP | Main Model Loader _or_ SDXL Main Model Loader|
|Advanced | UNETLoader | Main Model Loader _or_ SDXL Main Model Loader|
|Advanced | DualCLIPLoader | Main Model Loader _or_ SDXL Main Model Loader|
|Advanced | Load Checkpoint | Main Model Loader _or_ SDXL Main Model Loader |
|Advanced | ConditioningZeroOut | N/A |
|Advanced | ConditioningSetTimestepRange | N/A |
|Advanced | CLIPTextEncodeSDXLRefiner | Compel (Prompt) or SDXL Compel (Prompt) |
|Advanced | CLIPTextEncodeSDXL |Compel (Prompt) or SDXL Compel (Prompt) |
|Advanced | ModelMergeSimple | Model Merging is available in the Model Manager |
|Advanced | ModelMergeBlocks | Model Merging is available in the Model Manager|
|Advanced | CheckpointSave | Model saving is available in the Model Manager|
|Advanced | CLIPMergeSimple | N/A |


