"""What the flux3 architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import FLUX2_32, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.vae import VaeCompatibility, VaeFacet
from invokeai.backend.architectures.facets.variant import VariantFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType, Flux3VariantType, ModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import Flux3ConditioningInfo

# FLUX.3 Image decodes with the FLUX.2 autoencoder: the released VAE is byte-identical to the one FLUX.2
# ships, so it shares that latent space. Its denoise node previews the same unpacked, BN-normalized x0
# estimate FLUX.2's does.
register(
    BaseModelType.Flux3,
    LatentSpaceFacet(FLUX2_32),
    ConditioningFacet(Flux3ConditioningInfo),
    DefaultSettingsFacet(
        {
            # The reference sampler's settings for each checkpoint (BFL's flux-3-image-klein README and
            # pipeline.py). No scheduler: there is none to choose, see `scheduler_set` below.
            Flux3VariantType.KleinBase: MainModelDefaultSettings(steps=50, cfg_scale=3.5, width=1024, height=1024),
            # Klein, the step-distilled flow map: 16 steps (it is trained for 4 to 16), no CFG.
            None: MainModelDefaultSettings(steps=16, cfg_scale=1.0, width=1024, height=1024),
        }
    ),
    ModalityFacet(frozenset({"txt2img", "img2img", "inpaint", "outpaint"}), metadata_slug="flux3"),
    FeaturesFacet(
        # Klein Base takes classifier-free guidance with a negative prompt; Klein (the flow map) takes neither,
        # and `flux3_denoise` refuses a CFG scale above 1 for it.
        negative_prompt=NegativePrompt(visible=True, usage="cfg-gated"),
        dimension_grid=16,  # equals `multiple_of` on flux3_denoise.width/height
        guidance_label="CFG",
        # FLUX.3 drives its own sampler: Euler on a fixed shifted schedule, and for Klein a flow-map step that
        # also tells the transformer the timestep it jumps to. A different scheduler would step to sigmas the
        # flow map was not told about, so there is no choice to offer.
        scheduler_set=None,
        max_reference_images=10,
    ),
    VaeFacet(
        frozenset(
            {
                # The FLUX.2 VAE file is what FLUX.3 ships (`ae.safetensors`, and Comfy-Org's
                # `vae/flux2-vae.safetensors`); it identifies as a FLUX.2 VAE, and no VAE config produces
                # `flux3`.
                VaeCompatibility(BaseModelType.Flux2),
            }
        )
    ),
    # The values are flux3_klein / flux3_klein_base: variant strings are resolved without their base, and
    # `klein_*` already belongs to FLUX.2.
    VariantFacet({ModelType.Main: Flux3VariantType}),
)
