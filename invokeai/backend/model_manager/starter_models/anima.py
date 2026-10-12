"""Anima starter models."""

from invokeai.backend.model_manager.starter_models.common import anima_qwen3_encoder
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    AnimaVariantType,
    BaseModelType,
    ModelFormat,
    ModelType,
    Qwen35VariantType,
)

anima_vae = StarterModel(
    name="Anima QwenImage VAE",
    base=BaseModelType.Anima,
    source="https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/vae/qwen_image_vae.safetensors",
    description="QwenImage VAE for Anima (fine-tuned Wan 2.1 VAE, 16 latent channels). ~200MB",
    type=ModelType.VAE,
    format=ModelFormat.Checkpoint,
)

anima_base = StarterModel(
    name="Anima Base 1.0",
    base=BaseModelType.Anima,
    source="https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima-base-v1.0.safetensors",
    description="Anima Base 1.0 - 2B parameter anime-focused text-to-image model built on Cosmos Predict2 DiT. ~4.5GB",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    variant=AnimaVariantType.Qwen3,
    dependencies=[anima_qwen3_encoder, anima_vae],
)

_DERIVATIVE_LICENSE = (
    "A derivative of Anima under the CircleStone Labs Non-Commercial License: the model may not be used "
    "commercially (generated images may)."
)

anima_2_9b = StarterModel(
    name="Anima-2.9B (Preview v1)",
    base=BaseModelType.Anima,
    source="https://huggingface.co/Gazingstars123/Anima-2.9B/resolve/main/Anima-2.9B-preview-v1.safetensors",
    description="Community finetune of Anima Base, depth-expanded from 28 to 40 blocks and trained on 1.7M more "
    f"anime/illustration images (knowledge cutoff July 2026). Recommended: 28-50 steps, CFG 3.5-5. "
    f"{_DERIVATIVE_LICENSE} ~5.8GB",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    variant=AnimaVariantType.Qwen3,
    dependencies=[anima_qwen3_encoder, anima_vae],
)

anima_2_9b_int8 = StarterModel(
    name="Anima-2.9B (Preview v1, int8)",
    base=BaseModelType.Anima,
    source="https://huggingface.co/Gazingstars123/Anima-2.9B/resolve/main/Anima-2.9B-preview-v1_int8_convrot.safetensors",
    description="Anima-2.9B in ComfyUI's int8_tensorwise format: half the memory of the bf16 build, dequantized "
    f"per forward (no speedup). {_DERIVATIVE_LICENSE} ~3.1GB",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    variant=AnimaVariantType.Qwen3,
    dependencies=[anima_qwen3_encoder, anima_vae],
)

anima_qwen3_5_encoder = StarterModel(
    name="Anima-3.8B Qwen3.5 4B Encoder",
    base=BaseModelType.Any,
    source="https://huggingface.co/lylogummy/Anima-3.8B/resolve/main/text_encoders/qwen35_4b.safetensors",
    description="Qwen3.5 4B text encoder (raw fp8) that Anima-3.8B's semantic connector reads. Apache-2.0. ~4.8GB",
    type=ModelType.Qwen35Encoder,
    format=ModelFormat.Checkpoint,
    variant=Qwen35VariantType.Qwen35_4B,
)

anima_3_8b = StarterModel(
    name="Anima-3.8B (v1.1)",
    base=BaseModelType.Anima,
    source="https://huggingface.co/lylogummy/Anima-3.8B/resolve/main/difussion_models/Anima-3.8B-v1.1.safetensors",
    description="Community expansion of Anima-2.9B to 52 blocks with a bundled Qwen3.5 semantic connector, for "
    "prompt adherence, multi-character binding and mixed natural-language/tag prompts. Needs both the Qwen3 0.6B "
    f"and the Qwen3.5 4B encoder. Recommended: 28-50 steps, CFG 4-7. {_DERIVATIVE_LICENSE} ~8.8GB",
    type=ModelType.Main,
    format=ModelFormat.Checkpoint,
    variant=AnimaVariantType.Qwen35,
    dependencies=[anima_qwen3_encoder, anima_qwen3_5_encoder, anima_vae],
)

anima_lllite_inpainting = StarterModel(
    name="Anima LLLite Inpainting",
    base=BaseModelType.Anima,
    source="https://huggingface.co/kohya-ss/Anima-LLLite/resolve/main/anima-lllite-inpainting-v2.safetensors",
    description="ControlNet-LLLite inpainting adapter for Anima by kohya-ss. Conditions the model on the masked image content during inpainting/outpainting. ~66MB",
    type=ModelType.ControlNet,
    format=ModelFormat.Checkpoint,
)

anima_lllite_sketch = StarterModel(
    name="Anima LLLite Sketch",
    base=BaseModelType.Anima,
    source="https://huggingface.co/kohya-ss/Anima-LLLite/resolve/main/anima-lllite-any-test-like-v2.safetensors",
    description="ControlNet-LLLite control adapter for Anima by kohya-ss. Trained on mixed scribble/HED/lineart/grayscale conditioning images. ~16MB",
    type=ModelType.ControlNet,
    format=ModelFormat.Checkpoint,
)

anima_lllite_depth_preview3 = StarterModel(
    name="Anima LLLite Depth (Preview3)",
    base=BaseModelType.Anima,
    source="https://huggingface.co/kohya-ss/Anima-LLLite/resolve/main/anima-lllite-depth-1.safetensors",
    description="ControlNet-LLLite depth adapter for Anima by kohya-ss. Trained on the Preview3 build; reduced quality on Anima Base 1.0. ~8MB",
    type=ModelType.ControlNet,
    format=ModelFormat.Checkpoint,
)

anima_lllite_scribble_preview3 = StarterModel(
    name="Anima LLLite Scribble (Preview3)",
    base=BaseModelType.Anima,
    source="https://huggingface.co/kohya-ss/Anima-LLLite/resolve/main/anima-lllite-scribble-1.safetensors",
    description="ControlNet-LLLite scribble adapter for Anima by kohya-ss. Trained on the Preview3 build; reduced quality on Anima Base 1.0. ~8MB",
    type=ModelType.ControlNet,
    format=ModelFormat.Checkpoint,
)

anima_lllite_lineart_preview3 = StarterModel(
    name="Anima LLLite Lineart (Preview3)",
    base=BaseModelType.Anima,
    source="https://huggingface.co/kohya-ss/Anima-LLLite/resolve/main/anima-lllite-lineart-1.safetensors",
    description="ControlNet-LLLite lineart adapter for Anima by kohya-ss. Trained on the Preview3 build; reduced quality on Anima Base 1.0. ~8MB",
    type=ModelType.ControlNet,
    format=ModelFormat.Checkpoint,
)

anima_lllite_pose_preview3 = StarterModel(
    name="Anima LLLite Pose (Preview3)",
    base=BaseModelType.Anima,
    source="https://huggingface.co/kohya-ss/Anima-LLLite/resolve/main/anima-lllite-pose-1.safetensors",
    description="ControlNet-LLLite pose adapter for Anima by kohya-ss. Trained on the Preview3 build; notably weak on Anima Base 1.0. ~23MB",
    type=ModelType.ControlNet,
    format=ModelFormat.Checkpoint,
)
