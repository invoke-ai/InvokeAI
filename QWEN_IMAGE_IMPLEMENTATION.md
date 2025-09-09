# Qwen-Image Implementation for InvokeAI

## Overview

This implementation adds basic Qwen-Image support to InvokeAI, enabling text-to-image generation using Qwen-Image diffusion models through the diffusers library.

## Components Implemented

### 1. Model Taxonomy (`/invokeai/backend/model_manager/taxonomy.py`)
- Added `QwenImage = "qwen-image"` to `BaseModelType` enum

### 2. Model Configuration (`/invokeai/backend/model_manager/config.py`)
- Added `QwenImageConfig` class that extends `DiffusersConfigBase`, `MainConfigBase`, and `ModelConfigBase`
- Implements model detection by checking for "QwenImage" in the pipeline class name
- Added to the `AnyModelConfig` union type

### 3. Model Loader (`/invokeai/backend/model_manager/load/model_loaders/qwen_image.py`)
- `QwenImageLoader` class registered for `BaseModelType.QwenImage`, `ModelType.Main`, `ModelFormat.Diffusers`
- Loads Qwen-Image models as diffusers pipelines
- Supports loading both full pipelines and individual submodels

### 4. Conditioning Fields (`/invokeai/app/invocations/fields.py`)
- Added `QwenImageConditioningField` for handling Qwen-Image specific conditioning data

### 5. Conditioning Output (`/invokeai/app/invocations/primitives.py`)
- Added `QwenImageConditioningOutput` class for text encoder output
- Added import for `QwenImageConditioningField`

### 6. Text Encoder Node (`/invokeai/app/invocations/qwen_image_text_encoder.py`)
- `QwenImageTextEncoderInvocation` for encoding text prompts
- Attempts to extract tokenizer and text encoder from the pipeline
- Falls back to simple text storage if direct access fails

### 7. Denoising Node (`/invokeai/app/invocations/qwen_image_denoise.py`)
- `QwenImageDenoiseInvocation` for image generation
- Uses diffusers pipeline directly for generation
- Supports standard parameters: width, height, steps, guidance_scale, seed

## Dependencies Updated

- Updated `pyproject.toml` to use `diffusers[torch]==0.35.0` (from 0.33.0) to support Qwen-Image models

## Usage Workflow

1. **Load Model**: Use the model manager to load a Qwen-Image model (diffusers format)
2. **Encode Text**: Use `QwenImageTextEncoderInvocation` to encode your text prompt
3. **Generate Image**: Use `QwenImageDenoiseInvocation` to generate images using the encoded conditioning

## Current Limitations

1. **Basic Text Encoding**: The text encoder implementation is simplified and may not fully utilize Qwen-Image's text encoding capabilities
2. **No Advanced Features**: Currently only supports basic text-to-image generation
3. **Conditioning Handling**: Text conditioning handling could be more sophisticated
4. **Model Detection**: Model detection relies on pipeline class name containing "QwenImage"

## Future Enhancements

1. **Advanced Text Encoding**: Implement proper Qwen-Image text encoder extraction and usage
2. **Image-to-Image**: Add support for image-to-image generation
3. **Inpainting**: Add inpainting capabilities if supported by Qwen-Image
4. **ControlNet**: Add ControlNet support for better control
5. **LoRA Support**: Add LoRA fine-tuning support
6. **Advanced Conditioning**: Better handling of conditioning data and embeddings

## Testing

The implementation has been tested for basic imports and should work with Qwen-Image models that follow the standard diffusers pipeline format. To use:

1. Install a Qwen-Image model in diffusers format
2. Use InvokeAI's model manager to load it as a QwenImage type model
3. Create a workflow using the Qwen-Image text encoder and denoising nodes

## Files Modified/Created

- `/invokeai/backend/model_manager/taxonomy.py` (modified)
- `/invokeai/backend/model_manager/config.py` (modified)
- `/invokeai/backend/model_manager/load/model_loaders/qwen_image.py` (created)
- `/invokeai/app/invocations/fields.py` (modified)
- `/invokeai/app/invocations/primitives.py` (modified)
- `/invokeai/app/invocations/qwen_image_text_encoder.py` (created)
- `/invokeai/app/invocations/qwen_image_denoise.py` (created)
- `/pyproject.toml` (modified)