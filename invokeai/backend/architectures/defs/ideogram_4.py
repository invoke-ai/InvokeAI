"""What the ideogram-4 architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import FLUX2_32, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.vae import VaeCompatibility, VaeFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Ideogram4ConditioningInfo

# Ideogram 4 also uses a FLUX.2-style 32-channel VAE. It was the one architecture missing
# from the old preview dispatch entirely — its node carried a second copy of the logic.
register(
    BaseModelType.Ideogram4,
    LatentSpaceFacet(FLUX2_32),
    ConditioningFacet(Ideogram4ConditioningInfo),
    # Ideogram 4 samples from presets (V4_QUALITY_48 by default) with a dual-branch guidance
    # schedule; the step count is a sensible UI default rather than the sampler's own number.
    #
    # cfg_scale is 1.0 because the model is CFG-distilled: `ideogram4_denoise` has no `cfg_scale`
    # input at all, only `guidance_scale`, and the FeaturesFacet below already says so with
    # `negative_prompt: never` and `guidance_label: "Guidance"`. Any other value would be a number
    # the UI shows for a control the sampler does not have.
    DefaultSettingsFacet(
        {None: MainModelDefaultSettings(scheduler="euler", steps=48, cfg_scale=1.0, width=1024, height=1024)}
    ),
    VaeFacet(
        frozenset(
            {
                # Ideogram 4 decodes with the same 32-channel autoencoder as FLUX.2, and Comfy-Org
                # ships that exact file (`vae/flux2-vae.safetensors`) next to the single-file
                # transformers. It is the only entry because no VAE config class produces
                # `ideogram-4`: the released file identifies as a FLUX.2 VAE, and a record forced to
                # this base by hand would find no loader at all.
                VaeCompatibility(BaseModelType.Flux2),
            }
        )
    ),
    # Text-to-image only.
    ModalityFacet(frozenset({"txt2img"}), metadata_slug="ideogram4"),
    FeaturesFacet(
        negative_prompt=NegativePrompt(visible=False, usage="never"),
        dimension_grid=16,
        guidance_label="Guidance",
        scheduler_set="flow",
    ),
)
