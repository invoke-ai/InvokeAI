"""What the anima architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import WAN21_16, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.vae import VaeCompatibility, VaeFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import AnimaConditioningInfo

# Anima uses the Wan 2.1 VAE.
register(
    BaseModelType.Anima,
    LatentSpaceFacet(WAN21_16),
    ConditioningFacet(AnimaConditioningInfo),
    DefaultSettingsFacet(
        {None: MainModelDefaultSettings(scheduler="euler", steps=35, cfg_scale=4.5, width=1024, height=1024)}
    ),
    ModalityFacet(frozenset({"txt2img", "img2img", "inpaint", "outpaint"}), metadata_slug="anima"),
    FeaturesFacet(
        negative_prompt=NegativePrompt(visible=True, usage="cfg-gated"),
        dimension_grid=8,
        guidance_label="CFG",
        # anima_denoise.guidance_scale is ge=1.0; 1.0 is CFG off.
        guidance_min=1.0,
        scheduler_set="anima",
        scheduler_applies_to_graph=True,
        # Same shape as Z-Image: masked positive conditioning only.
        supports_regional_guidance=True,
    ),
    VaeFacet(
        frozenset(
            {
                # The Wan 2.1 VAE, registered under whichever base it was installed for -- all
                # three point at the same 194-tensor checkpoint.
                #
                # Not FLUX. `anima_l2i` takes a FluxAutoEncoder without raising, but that branch
                # skips the Wan denormalisation and decodes a WAN21_16 latent in FLUX's basis:
                # measured on a real denoised latent against the Anima VAE's decode of it, 8.67 dB
                # PSNR, a magenta moire in place of the subject. Accepting it is a silent corruption,
                # not a fallback. See `tests/app/invocations/test_anima_vae.py`.
                VaeCompatibility(BaseModelType.Anima),
                VaeCompatibility(BaseModelType.QwenImage),
                VaeCompatibility(BaseModelType.Wan, latent_channels=16),
            }
        )
    ),
)
