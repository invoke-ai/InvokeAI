"""MOSS-Image starter models."""

from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType

# Both are complete diffusers pipelines. Their text encoder and VAE are byte-identical, but each install keeps its own
# copy, so the descriptions say what installing both costs.
_MOSS_IMAGE_REQUIREMENTS = (
    "Needs ~27GB of RAM or VRAM for its weights: the Qwen3.5-9B text encoder (~17GB in bf16) and the ~10GB "
    "transformer, which low-VRAM mode loads one after the other. Apache-2.0."
)

moss_image_base = StarterModel(
    name="MOSS-Image Base",
    base=BaseModelType.MossImage,
    source="OpenMOSS-Team/MOSS-Image-Base",
    description=(
        "OpenMOSS MOSS-Image: a 5B single-stream DiT conditioned by Qwen3.5-9B, decoding with the FLUX.2 VAE. "
        "Takes natural-language prompts, tag sequences and JSON layouts. Defaults to 50 steps with CFG 3. "
        f"{_MOSS_IMAGE_REQUIREMENTS} Installing the SFT model as well downloads the same encoder again. ~27GB"
    ),
    type=ModelType.Main,
)

moss_image_sft = StarterModel(
    name="MOSS-Image SFT",
    base=BaseModelType.MossImage,
    source="OpenMOSS-Team/MOSS-Image-SFT",
    description=(
        "The supervised fine-tune of MOSS-Image Base: the same encoder and VAE with a fine-tuned transformer, which "
        f"groups text and lays out scenes more cleanly. {_MOSS_IMAGE_REQUIREMENTS} Installing the Base model as well "
        "downloads the same encoder again. ~27GB"
    ),
    type=ModelType.Main,
)
