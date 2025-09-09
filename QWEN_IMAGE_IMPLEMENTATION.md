# Qwen-Image Implementation for InvokeAI

## Overview

This implementation adds support for the Qwen-Image family of models to InvokeAI. Qwen-Image is a 20B parameter Multimodal Diffusion Transformer (MMDiT) model that excels at complex text rendering and precise image editing.

## Model Setup

### 1. Download the Qwen-Image Model
```bash
# Option 1: Using git (recommended for large models)
git clone https://huggingface.co/Qwen/Qwen-Image invokeai/models/qwen-image/Qwen-Image

# Option 2: Using huggingface-cli
huggingface-cli download Qwen/Qwen-Image --local-dir invokeai/models/qwen-image/Qwen-Image
```

### 2. Download Qwen2.5-VL Text Encoder
Qwen-Image uses Qwen2.5-VL-7B as its text encoder (not CLIP):
```bash
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct invokeai/models/qwen-image/Qwen2.5-VL-7B-Instruct
```

## Model Architecture

### Components
1. **Transformer**: QwenImageTransformer2DModel (MMDiT architecture, 20B parameters)
2. **Text Encoder**: Qwen2.5-VL-7B-Instruct (7B parameter vision-language model)
3. **VAE**: AutoencoderKLQwenImage (bundled with main model in `/vae` subdirectory)
4. **Scheduler**: FlowMatchEulerDiscreteScheduler

### Key Features
- **Complex Text Rendering**: Superior ability to render text accurately in images
- **Bundled VAE**: The model includes its own custom VAE (no separate download needed)
- **Large Text Encoder**: Uses a 7B parameter VLM instead of traditional CLIP
- **Optional VAE Override**: Can use custom VAE models if desired

## Components Implemented

### Backend Components
1. **Model Taxonomy** (`taxonomy.py`): Added `QwenImage = "qwen-image"` base model type
2. **Model Configuration** (`config.py`): Uses MainDiffusersConfig for Qwen-Image models
3. **Model Loader** (`qwen_image.py`): Loads models and submodels via diffusers
4. **Model Loader Node** (`qwen_image_model_loader.py`): Loads transformer, text encoder, and VAE
5. **Text Encoder Node** (`qwen_image_text_encoder.py`): Encodes prompts using Qwen2.5-VL
6. **Denoising Node** (`qwen_image_denoise.py`): Generates images using QwenImagePipeline

### Frontend Components
1. **UI Types**: Added QwenImageMainModel, Qwen2_5VLModel field types
2. **Field Components**: Created input components for model selection
3. **Type Guards**: Added model detection and filtering functions
4. **Hooks**: Model loading hooks for UI dropdowns

## Dependencies Updated

- Updated `pyproject.toml` to use `diffusers[torch]==0.35.0` (from 0.33.0) to support Qwen-Image models

## Usage in InvokeAI

### Node Graph Setup
1. Add a **"Main Model - Qwen-Image"** loader node
2. Select your Qwen-Image model from the dropdown
3. Select the Qwen2.5-VL model for text encoding
4. VAE field is optional (uses bundled VAE if left empty)
5. Connect to **Qwen-Image Text Encoder** node
6. Connect to **Qwen-Image Denoise** node
7. Add **VAE Decode** node to convert latents to images

### Model Selection
- **Main Model**: Select from models with base type "qwen-image"
- **Text Encoder**: Select Qwen2.5-VL-7B-Instruct
- **VAE**: Optional - leave empty to use bundled VAE, or select a custom VAE

## Troubleshooting

### VAE Not Showing Up
The Qwen-Image VAE is bundled with the main model. You don't need to download or select a separate VAE - just leave the VAE field empty to use the bundled one.

### Memory Issues
- Use bfloat16 precision for reduced memory usage
- Consider quantization options (e.g., qwen-image-nf4 from diffusers)
- Recommended: 24GB+ VRAM for full model

### Model Not Loading
- Ensure the model is in the correct directory structure
- Check that both Qwen-Image and Qwen2.5-VL models are downloaded
- Verify diffusers version is 0.35.0 or higher

## Future Enhancements

1. **Image Editing**: Support for Qwen-Image-Edit variant
2. **LoRA Support**: Fine-tuning capabilities
3. **Optimizations**: Quantization and speed improvements (Qwen-Image-Lightning)
4. **Advanced Features**: Image-to-image, inpainting, controlnet support

## Files Modified/Created

- `/invokeai/backend/model_manager/taxonomy.py` (modified)
- `/invokeai/backend/model_manager/config.py` (modified)
- `/invokeai/backend/model_manager/load/model_loaders/qwen_image.py` (created)
- `/invokeai/app/invocations/fields.py` (modified)
- `/invokeai/app/invocations/primitives.py` (modified)
- `/invokeai/app/invocations/qwen_image_text_encoder.py` (created)
- `/invokeai/app/invocations/qwen_image_denoise.py` (created)
- `/pyproject.toml` (modified)