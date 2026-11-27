"""What the sdxl-refiner architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import SDXL_4, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.variant import VariantFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, ModelVariantType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import SDXLConditioningInfo

# The refiner shares SDXL's VAE.
register(
    BaseModelType.StableDiffusionXLRefiner,
    LatentSpaceFacet(SDXL_4),
    ConditioningFacet(SDXLConditioningInfo),
    # Same canvas as SDXL, which it refines. Steps and CFG are deliberately absent: the refiner
    # is a second pass over an SDXL latent and the UI drives it with its own parameters, so
    # there is nothing here for them to prefill.
    DefaultSettingsFacet({None: MainModelDefaultSettings(width=1024, height=1024)}),
    # Generates nothing on its own; it refines an SDXL latent, so it writes no mode string.
    ModalityFacet(frozenset()),
    # Not selected as a generation model — it declares no modes — but it is an SDXL pass and
    # answers as one, so a UI that does surface it does not have to special-case it.
    FeaturesFacet(
        negative_prompt=NegativePrompt(visible=True, usage="always"),
        dimension_grid=8,
        guidance_label="CFG",
        scheduler_set="standard",
        scheduler_applies_to_graph=True,
        sd_vae_override=True,
        color_compensation=True,
        vae_precision=True,
    ),
    VariantFacet({ModelType.Main: ModelVariantType}),
)
