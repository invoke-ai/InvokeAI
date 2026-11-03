"""What the cogview4 architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import COGVIEW4_16, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import CogView4ConditioningInfo

register(
    BaseModelType.CogView4,
    LatentSpaceFacet(COGVIEW4_16),
    ConditioningFacet(CogView4ConditioningInfo),
    # THUDM/CogView4-6B's own example: 50 steps at guidance 3.5, 1024x1024. This is true
    # classifier-free guidance, so it belongs in cfg_scale — and the denoise node already
    # defaults to 3.5, which nothing was propagating to the sliders.
    DefaultSettingsFacet(
        {None: MainModelDefaultSettings(scheduler="euler_a", steps=50, cfg_scale=3.5, width=1024, height=1024)}
    ),
    ModalityFacet(frozenset({"txt2img", "img2img", "inpaint", "outpaint"}), metadata_slug="cogview4"),
    FeaturesFacet(
        negative_prompt=NegativePrompt(visible=True, usage="always"),
        dimension_grid=32,
        guidance_label="CFG",
        scheduler_set="standard",
    ),
)
