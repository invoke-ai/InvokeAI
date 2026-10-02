"""Anima starter models."""

from invokeai.backend.model_manager.starter_models.common import anima_qwen3_encoder
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
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
    dependencies=[anima_qwen3_encoder, anima_vae],
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
