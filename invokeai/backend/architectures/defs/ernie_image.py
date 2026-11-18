"""What the ernie-image architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import FLUX2_32, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.vae import VaeCompatibility, VaeFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ErnieImageConditioningInfo

# ERNIE-Image uses AutoencoderKLFlux2. The shapes line up because the denoise loop unpatches
# before previewing; the values are approximate, because ERNIE denoises in BN-normalized
# latent space and the BN stats live on the VAE, which is not loaded to draw a preview.
register(
    BaseModelType.ErnieImage,
    LatentSpaceFacet(FLUX2_32),
    ConditioningFacet(ErnieImageConditioningInfo),
    DefaultSettingsFacet(
        {None: MainModelDefaultSettings(scheduler="euler", steps=50, cfg_scale=4.0, width=1024, height=1024)},
        # Turbo and the base model share an architecture and a config, so there is nothing on
        # disk to discriminate on and no variant is modeled. The name is the only signal.
        by_name_hint={
            "turbo": MainModelDefaultSettings(scheduler="euler", steps=8, cfg_scale=1.0, width=1024, height=1024)
        },
    ),
    VaeFacet(
        frozenset(
            {
                # ERNIE-Image decodes with AutoencoderKLFlux2 -- its pipeline says so, and Comfy-Org
                # ships the FLUX.2 VAE file alongside the single-file transformer, byte-identical to
                # the one FLUX.2 Klein ships. It is the only entry because no VAE config class can
                # produce `ernie-image`: the released file identifies as a FLUX.2 VAE, and a record
                # forced to this base by hand would fall through to the generic VAE loader, which
                # cannot serve a `vae` submodel at all. Listing it would advertise a broken choice.
                VaeCompatibility(BaseModelType.Flux2),
            }
        )
    ),
    # Text-to-image only.
    ModalityFacet(frozenset({"txt2img"}), metadata_slug="ernie_image"),
    # ernie_image_denoise takes a negative_conditioning that is 'required when
    # guidance_scale != 1.0' — cfg-gated, exactly like the other distilled models. Its
    # scheduler set is ERNIE_IMAGE_SCHEDULER_MAP, a flow family.
    FeaturesFacet(
        negative_prompt=NegativePrompt(visible=True, usage="cfg-gated"),
        dimension_grid=16,
        guidance_label="CFG",
        # ernie_image_denoise.guidance_scale is ge=1.0; 1.0 is CFG off.
        guidance_min=1.0,
        scheduler_set="flow",
        # `ernie_image_denoise` takes a `scheduler` field and builds the sampler from it
        # (ERNIE_IMAGE_SCHEDULER_MAP), so the choice reaches the graph rather than being a
        # UI affordance. Omitting this defaulted it to False, which would have hidden the
        # dropdown and pinned every generation to the first entry.
        scheduler_applies_to_graph=True,
    ),
)
