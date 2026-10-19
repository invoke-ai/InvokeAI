"""What the z-image architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import FLUX_16, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.vae import VaeCompatibility, VaeFacet
from invokeai.backend.architectures.facets.variant import VariantFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, ZImageVariantType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ZImageConditioningInfo

# Z-Image decodes with a FLUX-compatible 16-channel VAE.
register(
    BaseModelType.ZImage,
    LatentSpaceFacet(FLUX_16),
    ConditioningFacet(ZImageConditioningInfo),
    DefaultSettingsFacet(
        {
            # The undistilled base needs more steps and supports CFG.
            ZImageVariantType.ZBase: MainModelDefaultSettings(
                scheduler="euler", steps=50, cfg_scale=4.0, width=1024, height=1024
            ),
            # Turbo (distilled): fewer steps, no CFG.
            None: MainModelDefaultSettings(scheduler="euler", steps=9, cfg_scale=1.0, width=1024, height=1024),
        }
    ),
    ModalityFacet(frozenset({"txt2img", "img2img", "inpaint", "outpaint"}), metadata_slug="z_image"),
    FeaturesFacet(
        negative_prompt=NegativePrompt(visible=True, usage="cfg-gated"),
        dimension_grid=16,
        guidance_label="CFG",
        # z_image_denoise.guidance_scale is ge=1.0; 1.0 is CFG off, which is what Turbo runs at.
        guidance_min=1.0,
        scheduler_set="flow",
        # `z_image_denoise.scheduler` documents LCM as working with Turbo only, not Base.
        scheduler_set_by_variant={ZImageVariantType.ZBase: "flow-no-lcm"},
        scheduler_applies_to_graph=True,
        control_kinds=frozenset({"z_image_control"}),
        # The text encoder masks positive conditioning; the denoiser takes a negative list but
        # discards its masks, so a regional negative would act globally and is rejected instead.
        supports_regional_guidance=True,
    ),
    VaeFacet(
        frozenset(
            {
                # Z-Image decodes with a FLUX-compatible VAE; `z_image_model_loader` says so.
                VaeCompatibility(BaseModelType.Flux),
            }
        )
    ),
    VariantFacet(
        {
            ModelType.Main: ZImageVariantType,
            ModelType.LoRA: ZImageVariantType,
        }
    ),
)
