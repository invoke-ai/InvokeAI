"""What the sd-3 architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import SD3_16, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.variant import VariantFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, PiDDecoderVariantType, SD3VariantType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import SD3ConditioningInfo

register(
    BaseModelType.StableDiffusion3,
    LatentSpaceFacet(SD3_16),
    ConditioningFacet(SD3ConditioningInfo),
    # stable-diffusion-3.5-medium's example: 40 steps at guidance 4.5, for every SD 3 model. Medium
    # rather than Large (28/3.5) because Medium is the smaller, more commonly run model; defaults per
    # size, keyed on the variant, are a separate change.
    DefaultSettingsFacet(
        {None: MainModelDefaultSettings(scheduler="euler_a", steps=40, cfg_scale=4.5, width=1024, height=1024)},
        # Any SD3 main whose name or file/folder name says "turbo", Diffusers or GGUF: Turbo and Large share an
        # architecture and a config, so the name is the only signal. These follow Stability's step count and
        # guidance for stable-diffusion-3.5-large-turbo (4 steps, guidance 0, i.e. no CFG -- CFG 1 here, where
        # the denoise always mixes the two predictions). The card names no scheduler; SD3 always samples with its
        # own flow sampler, on the reference's shifted schedule (`sd3_denoise.py`).
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
    # Mains and LoRAs carry the transformer size (Medium / Large), which decides the LoRAs that fit; records
    # installed before it existed carry none. Its PiD decoder checkpoints carry their own variant.
    VariantFacet(
        {
            ModelType.Main: SD3VariantType,
            ModelType.LoRA: SD3VariantType,
            ModelType.PiDDecoder: PiDDecoderVariantType,
        }
    ),
)
