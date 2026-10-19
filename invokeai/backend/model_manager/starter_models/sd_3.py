"""Stable Diffusion 3.5 starter models."""

from invokeai.backend.model_manager.starter_models.common import gemma2_2b_encoder
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
    PiDDecoderVariantType,
)

# SD3 uses a 16-channel latent, architecturally identical to FLUX.1. The config probe disambiguates via the
# checkpoint's directory name (`…official_sd3_distill…`); if the HF single-file download drops that name, the
# explicit base=StableDiffusion3 override the installer sends is trusted instead (see pid_decoder.py::_validate_base).
pid_decoder_sd3_2k = StarterModel(
    name="PiD Decoder SD3 (2K)",
    base=BaseModelType.StableDiffusion3,
    source="nvidia/PiD::checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth",
    description="NVIDIA PiD 4x super-resolution decoder for SD3 latents, 2K target preset (e.g. 512 -> 2048). ~5GB",
    type=ModelType.PiDDecoder,
    format=ModelFormat.Checkpoint,
    variant=PiDDecoderVariantType.Res2k_Sr4x,
    dependencies=[gemma2_2b_encoder],
)

pid_decoder_sd3_2kto4k = StarterModel(
    name="PiD Decoder SD3 (2K to 4K)",
    base=BaseModelType.StableDiffusion3,
    source="nvidia/PiD::checkpoints/PiD_res2kto4k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth",
    description="NVIDIA PiD 4x super-resolution decoder for SD3 latents, 2K-to-4K preset for higher-resolution output. ~5GB",
    type=ModelType.PiDDecoder,
    format=ModelFormat.Checkpoint,
    variant=PiDDecoderVariantType.Res2kTo4k_Sr4x,
    dependencies=[gemma2_2b_encoder],
)

sd35_medium = StarterModel(
    name="SD3.5 Medium",
    base=BaseModelType.StableDiffusion3,
    source="stabilityai/stable-diffusion-3.5-medium",
    description="Medium SD3.5 Model: ~16GB",
    type=ModelType.Main,
    dependencies=[],
)

sd35_large = StarterModel(
    name="SD3.5 Large",
    base=BaseModelType.StableDiffusion3,
    source="stabilityai/stable-diffusion-3.5-large",
    description="Large SD3.5 Model: ~28GB",
    type=ModelType.Main,
    dependencies=[],
)
