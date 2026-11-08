"""What the sd-3 architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import SD3_16, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.variant import VariantFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, PiDDecoderVariantType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import SD3ConditioningInfo

register(
    BaseModelType.StableDiffusion3,
    LatentSpaceFacet(SD3_16),
    ConditioningFacet(SD3ConditioningInfo),
    # stable-diffusion-3.5-medium's example: 40 steps at guidance 4.5. Medium rather than Large
    # (28/3.5) because there is one `sd-3` row and no variant to tell them apart, and Medium is the
    # smaller, more commonly run model.
    DefaultSettingsFacet(
        {None: MainModelDefaultSettings(scheduler="euler_a", steps=40, cfg_scale=4.5, width=1024, height=1024)},
        # Any SD3 main whose name or file/folder name says "turbo", Diffusers or GGUF: Turbo and Large share an
        # architecture and a config, so the name is the only signal. These follow Stability's step count and
        # guidance for stable-diffusion-3.5-large-turbo (4 steps, guidance 0, i.e. no CFG -- CFG 1 here, where
        # the denoise always mixes the two predictions). Every SD3.5 reference also shifts the noise schedule by
        # 3.0, which the SD3 denoise (`sd3_denoise.py`) applies for no SD3 model; that matters most at 4 steps.
        # Measured on city96's Large Turbo Q4_0: 4 unshifted steps are usable but break objects in places, 8 look
        # about as clean as 4 shifted ones, so the docs advise about 8 steps for undercooked output. The card names
        # no scheduler; SD3 always samples with its own flow sampler.
        by_name_hint={
            "turbo": MainModelDefaultSettings(scheduler="euler_a", steps=4, cfg_scale=1.0, width=1024, height=1024)
        },
    ),
    ModalityFacet(frozenset({"txt2img", "img2img", "inpaint", "outpaint"}), metadata_slug="sd3"),
    FeaturesFacet(
        negative_prompt=NegativePrompt(visible=True, usage="always"),
        dimension_grid=16,
        guidance_label="CFG",
        scheduler_set="standard",
    ),
    # SD3 mains carry no variant; only its PiD decoder checkpoints do.
    VariantFacet({ModelType.PiDDecoder: PiDDecoderVariantType}),
)
