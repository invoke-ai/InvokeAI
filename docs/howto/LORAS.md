---
title: LoRAs & LCM-LoRAs
---

# :material-library-shelves: LoRAs & LCM-LoRAs

With the advances in research, many new capabilities are available to customize the knowledge and understanding of novel concepts not originally contained in the base model. 

## LoRAs

Low-Rank Adaptation (LoRA) files are models that customize the output of Stable Diffusion
image generation.  Larger than embeddings, but much smaller than full
models, they augment SD with improved understanding of subjects and
artistic styles.

Unlike TI files, LoRAs do not introduce novel vocabulary into the
model's known tokens. Instead, LoRAs augment the model's weights that
are applied to generate imagery. LoRAs may be supplied with a
"trigger" word that they have been explicitly trained on, or may
simply apply their effect without being triggered.

LoRAs are typically stored in .safetensors files, which are the most
secure way to store and transmit these types of weights.

To use these when generating, open the LoRA menu item in the options
panel, select the LoRAs you want to apply and ensure that they have
the appropriate weight recommended by the model provider. Typically,
most LoRAs perform best at a weight of .75-1.


## LCM-LoRAs
Latent Consistency Models (LCMs) allowed a reduced number of steps to be used to generate images with Stable Diffusion. These are created by distilling base models, creating models that only require a small number of steps to generate images. However, LCMs require that any fine-tune of a base model be distilled to be used as an LCM. 

LCM-LoRAs are models that provide the benefit of LCMs but are able to be used as LoRAs and applied to any fine tune of a base model. LCM-LoRAs are created by training a small number of adapters, rather than distilling the entire fine-tuned base model. The resulting LoRA can be used the same way as a standard LoRA, but with a greatly reduced step count. This enables SDXL images to be generated up to 10x faster than without the use of LCM-LoRAs. 


**Using LCM-LoRAs**

LCM-LoRAs are natively supported in InvokeAI throughout the application. To get started, install any diffusers format LCM-LoRAs using the model manager and select it in the LoRA field.

There are a number parameter differences when using LCM-LoRAs and standard generation: 

- When using LCM-LoRAs, the LoRA strength should be lower than if using a standard LoRA, with 0.35 recommended as a starting point.  
- The LCM scheduler should be used for generation
- CFG-Scale should be reduced to ~1
- Steps should be reduced in the range of 4-8

Standard LoRAs can also be used alongside LCM-LoRAs, but will also require a lower strength, with 0.45 being recommended as a starting point. 

More information can be found here: https://huggingface.co/blog/lcm_lora#fast-inference-with-sdxl-lcm-loras
