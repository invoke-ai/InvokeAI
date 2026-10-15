"""What the qwen-image architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import WAN21_16, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.vae import VaeCompatibility, VaeFacet
from invokeai.backend.architectures.facets.variant import VariantFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelType,
    PiDDecoderVariantType,
    QwenImageVariantType,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import QwenImageConditioningInfo

# Qwen-Image uses the Wan 2.1 VAE.
register(
    BaseModelType.QwenImage,
    LatentSpaceFacet(WAN21_16),
    ConditioningFacet(QwenImageConditioningInfo),
    DefaultSettingsFacet(
        {None: MainModelDefaultSettings(scheduler="euler_a", steps=40, cfg_scale=4.0, width=1024, height=1024)}
    ),
    ModalityFacet(frozenset({"txt2img", "img2img", "inpaint", "outpaint"}), metadata_slug="qwen_image"),
    FeaturesFacet(
        negative_prompt=NegativePrompt(visible=True, usage="cfg-gated"),
        dimension_grid=16,
        guidance_label="CFG",
        scheduler_set="standard",
        max_reference_images=5,
        reference_images_require_variant="edit",
    ),
    VaeFacet(
        frozenset(
            {
                # The same 16-channel checkpoint is registered under either base depending on which
                # family it was installed for.
                VaeCompatibility(BaseModelType.QwenImage),
                VaeCompatibility(BaseModelType.Anima),
            }
        )
    ),
    VariantFacet(
        {
            ModelType.Main: QwenImageVariantType,
            ModelType.PiDDecoder: PiDDecoderVariantType,
        }
    ),
)
