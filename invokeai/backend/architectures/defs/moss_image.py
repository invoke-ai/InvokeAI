"""What the moss-image architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import FLUX2_32, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.vae import VaeCompatibility, VaeFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import MossImageConditioningInfo

register(
    BaseModelType.MossImage,
    # MOSS-Image denoises the FLUX.2 VAE latent (its VAE's encoder and BatchNorm are FLUX.2's, rounded to bf16, with
    # a smaller decoder), and between nodes its latents travel in FLUX.2's raw layout.
    LatentSpaceFacet(FLUX2_32),
    ConditioningFacet(MossImageConditioningInfo),
    # The reference pipeline's defaults (`MOSSImagePipeline.__call__`): 50 steps, CFG 3. The schedule is the model's
    # own shifted Euler schedule, which `moss_image_denoise` builds; "euler" names it for the UI.
    DefaultSettingsFacet(
        {None: MainModelDefaultSettings(scheduler="euler", steps=50, cfg_scale=3.0, width=1024, height=1024)}
    ),
    VaeFacet(
        frozenset(
            {
                # The MOSS `vae/` and its single-file export identify as FLUX.2 VAEs, and a stock FLUX.2 VAE decodes
                # MOSS latents. No VAE config class produces `moss-image`.
                VaeCompatibility(BaseModelType.Flux2),
            }
        )
    ),
    ModalityFacet(frozenset({"txt2img", "img2img", "inpaint", "outpaint"}), metadata_slug="moss_image"),
    FeaturesFacet(
        negative_prompt=NegativePrompt(visible=True, usage="cfg-gated"),
        dimension_grid=16,  # equals `multiple_of` on moss_image_denoise.width/height
        guidance_label="CFG",
        # The denoise node takes no scheduler: the set only names the default for the UI, which keeps the picker
        # hidden (`scheduler_applies_to_graph` stays False), as for Ideogram 4.
        scheduler_set="flow",
        # Regions are not masked: each becomes a box in the JSON layout prompt (`moss_image_layout_prompt`), which
        # has no negative counterpart, so `regional_negative` keeps its False default.
        supports_regional_guidance=True,
    ),
)
