"""ERNIE-Image starter models."""

from invokeai.backend.model_manager.starter_models.flux2 import flux2_vae
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelType,
)

# Comfy-Org's single-file releases: the transformer alone, paired with a standalone encoder and
# VAE. The prompt enhancer is only in the diffusers pipelines above.
ernie_image_mistral_encoder = StarterModel(
    name="ERNIE-Image Ministral Encoder",
    base=BaseModelType.Any,
    source="https://huggingface.co/Comfy-Org/ERNIE-Image/resolve/main/text_encoders/ministral-3-3b.safetensors",
    description=(
        "Ministral 3B text encoder for the single-file ERNIE-Image transformers. A different "
        "architecture from the Mistral Small 3 encoder FLUX.2 uses; the two are not "
        "interchangeable. ~7.7GB"
    ),
    type=ModelType.MistralEncoder,
)

# region ERNIE-Image
ernie_image = StarterModel(
    name="ERNIE-Image",
    base=BaseModelType.ErnieImage,
    source="baidu/ERNIE-Image",
    description=(
        "Baidu ERNIE-Image: 8B single-stream DiT with Mistral3 text encoder, AutoencoderKLFlux2 VAE, "
        "and bundled Ministral3 prompt enhancer. Defaults to 50 steps with CFG 4.0."
    ),
    type=ModelType.Main,
)

ernie_image_turbo = StarterModel(
    name="ERNIE-Image Turbo",
    base=BaseModelType.ErnieImage,
    source="baidu/ERNIE-Image-Turbo",
    description=(
        "ERNIE-Image-Turbo: distilled variant of ERNIE-Image. Same architecture as ERNIE-Image but "
        "tuned for fast inference at 8 steps with CFG disabled (1.0)."
    ),
    type=ModelType.Main,
)

ernie_image_single_file = StarterModel(
    name="ERNIE-Image (single file)",
    base=BaseModelType.ErnieImage,
    source="https://huggingface.co/Comfy-Org/ERNIE-Image/resolve/main/diffusion_models/ernie-image.safetensors",
    description=(
        "Comfy-Org single-file ERNIE-Image transformer. Installs with the Ministral 3B encoder and the "
        "FLUX.2 VAE; no prompt enhancer (the diffusers pipeline carries that). ~16GB"
    ),
    type=ModelType.Main,
    dependencies=[ernie_image_mistral_encoder, flux2_vae],
)

ernie_image_turbo_single_file = StarterModel(
    name="ERNIE-Image Turbo (single file)",
    base=BaseModelType.ErnieImage,
    source="https://huggingface.co/Comfy-Org/ERNIE-Image/resolve/main/diffusion_models/ernie-image-turbo.safetensors",
    description=(
        "Comfy-Org single-file ERNIE-Image-Turbo transformer, tuned for 8 steps with CFG disabled. "
        "Installs with the Ministral 3B encoder and the FLUX.2 VAE. ~16GB"
    ),
    type=ModelType.Main,
    dependencies=[ernie_image_mistral_encoder, flux2_vae],
)
