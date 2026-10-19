"""What the qwen-image-2-1 architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import QWEN_IMAGE21_64, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.variant import VariantFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, QwenImage21VariantType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import QwenImage21ConditioningInfo

register(
    BaseModelType.QwenImage21,
    LatentSpaceFacet(QWEN_IMAGE21_64),
    ConditioningFacet(QwenImage21ConditioningInfo),
    DefaultSettingsFacet(
        {
            # Qwen-Image-2.1-Turbo's model card: 8 steps on the schedule the checkpoint ships, CFG 1.
            QwenImage21VariantType.Turbo: MainModelDefaultSettings(
                scheduler="euler", steps=8, cfg_scale=1.0, width=1024, height=1024
            ),
            # Qwen-Image-2.1's model card and pipeline defaults: 40 steps, sampled without guidance. The
            # card shows 2K sizes, but 1024x1024 is the pipeline's default and roughly six times faster.
            None: MainModelDefaultSettings(scheduler="euler", steps=40, cfg_scale=1.0, width=1024, height=1024),
        }
    ),
    ModalityFacet(frozenset({"txt2img", "img2img", "inpaint", "outpaint"}), metadata_slug="qwen_image_2_1"),
    FeaturesFacet(
        # True CFG only runs when a negative prompt is given and the scale is above 1.
        negative_prompt=NegativePrompt(visible=True, usage="cfg-gated"),
        # The VAE downscales 16x and a VL image slot covers 2x2 latents, so sizes are multiples of 32.
        dimension_grid=32,
        guidance_label="CFG",
        scheduler_set="flow",
    ),
    # Values are qwen_image_21_* because variant strings resolve without the base and must be unique.
    VariantFacet({ModelType.Main: QwenImage21VariantType}),
)
