"""What the flux2 architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import FLUX2_32, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.variant import VariantFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType, Flux2VariantType, ModelType, PiDDecoderVariantType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import FLUXConditioningInfo

register(
    BaseModelType.Flux2,
    LatentSpaceFacet(FLUX2_32),
    # FLUX.2 encodes to the same conditioning shape as FLUX.1, both [dev] and Klein.
    ConditioningFacet(FLUXConditioningInfo),
    DefaultSettingsFacet(
        {
            # [dev] is guidance-distilled: guidance 3.5, 28 steps, CFG off.
            Flux2VariantType.Dev: MainModelDefaultSettings(
                scheduler="euler", steps=28, cfg_scale=1.0, guidance=3.5, width=1024, height=1024
            ),
            # The undistilled Klein bases need the steps but not the guidance.
            Flux2VariantType.Klein4BBase: MainModelDefaultSettings(
                scheduler="euler", steps=28, cfg_scale=1.0, width=1024, height=1024
            ),
            Flux2VariantType.Klein9BBase: MainModelDefaultSettings(
                scheduler="euler", steps=28, cfg_scale=1.0, width=1024, height=1024
            ),
            # Distilled Klein 4B / 9B.
            None: MainModelDefaultSettings(scheduler="euler", steps=4, cfg_scale=1.0, width=1024, height=1024),
        }
    ),
    ModalityFacet(frozenset({"txt2img", "img2img", "inpaint", "outpaint"}), metadata_slug="flux2"),
    FeaturesFacet(
        negative_prompt=NegativePrompt(visible=False, usage="never"),
        dimension_grid=16,
        guidance_label="Guidance",
        # flux2_denoise.guidance is le=20 -- unlike flux_denoise.guidance, which is unbounded and
        # where FLUX Fill wants 30.
        guidance_max=20.0,
        scheduler_set="flow",
        scheduler_applies_to_graph=True,
        max_reference_images=5,
        supports_regional_guidance=True,
    ),
    VariantFacet(
        {
            # FLUX.2 LoRAs are labelled with the same variant enum as the mains they target.
            ModelType.Main: Flux2VariantType,
            ModelType.LoRA: Flux2VariantType,
            ModelType.PiDDecoder: PiDDecoderVariantType,
        }
    ),
)
