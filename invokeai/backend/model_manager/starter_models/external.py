"""Starter entries for externally hosted models, and the provider-specific presets they carry."""

from invokeai.backend.model_manager.configs.external_api import (
    ExternalApiModelDefaultSettings,
    ExternalImageSize,
    ExternalModelCapabilities,
    ExternalModelPanelSchema,
    ExternalResolutionPreset,
)
from invokeai.backend.model_manager.starter_models.types import StarterModel
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
)

# region External API
GEMINI_3_IMAGE_ALLOWED_ASPECT_RATIOS = [
    "1:1",
    "1:4",
    "1:8",
    "2:3",
    "3:2",
    "3:4",
    "4:1",
    "4:3",
    "4:5",
    "5:4",
    "8:1",
    "9:16",
    "16:9",
    "21:9",
]

GEMINI_3_IMAGE_MAX_SIZE = ExternalImageSize(width=4096, height=4096)


def _gemini_3_resolution_presets(
    image_sizes: list[str],
    aspect_ratios: list[str] | None = None,
) -> list[ExternalResolutionPreset]:
    """Build resolution presets for Gemini 3 models.

    Each preset combines an aspect ratio with an image size preset (512/1K/2K/4K).
    Pixel dimensions are approximations based on the preset name (longest side).
    """
    if aspect_ratios is None:
        aspect_ratios = GEMINI_3_IMAGE_ALLOWED_ASPECT_RATIOS
    base_pixels = {"512": 512, "1K": 1024, "2K": 2048, "4K": 4096}
    presets: list[ExternalResolutionPreset] = []
    for image_size in image_sizes:
        base = base_pixels[image_size]
        for ratio_str in aspect_ratios:
            w_part, h_part = (int(x) for x in ratio_str.split(":"))
            if w_part >= h_part:
                w = base
                h = max(1, round(base * h_part / w_part))
            else:
                h = base
                w = max(1, round(base * w_part / h_part))
            presets.append(
                ExternalResolutionPreset(
                    label=f"{ratio_str} ({image_size}) — {w}\u00d7{h}",
                    aspect_ratio=ratio_str,
                    image_size=image_size,
                    width=w,
                    height=h,
                )
            )
    return presets


GEMINI_3_PRO_RESOLUTION_PRESETS = _gemini_3_resolution_presets(["1K", "2K", "4K"])

GEMINI_3_1_FLASH_RESOLUTION_PRESETS = _gemini_3_resolution_presets(["512", "1K", "2K", "4K"])

gemini_flash_image = StarterModel(
    name="Gemini 2.5 Flash Image",
    base=BaseModelType.External,
    source="external://gemini/gemini-2.5-flash-image",
    description="Google Gemini 2.5 Flash image generation model (external API). Requires a configured Gemini API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img"],
        supports_seed=True,
        supports_reference_images=True,
        max_images_per_request=1,
        allowed_aspect_ratios=[
            "1:1",
            "2:3",
            "3:2",
            "3:4",
            "4:3",
            "4:5",
            "5:4",
            "9:16",
            "16:9",
            "21:9",
        ],
        aspect_ratio_sizes={
            "1:1": ExternalImageSize(width=1024, height=1024),
            "2:3": ExternalImageSize(width=832, height=1248),
            "3:2": ExternalImageSize(width=1248, height=832),
            "3:4": ExternalImageSize(width=864, height=1184),
            "4:3": ExternalImageSize(width=1184, height=864),
            "4:5": ExternalImageSize(width=896, height=1152),
            "5:4": ExternalImageSize(width=1152, height=896),
            "9:16": ExternalImageSize(width=768, height=1344),
            "16:9": ExternalImageSize(width=1344, height=768),
            "21:9": ExternalImageSize(width=1536, height=672),
        },
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1),
    panel_schema=ExternalModelPanelSchema(prompts=[{"name": "reference_images"}], image=[{"name": "dimensions"}]),
)

gemini_pro_image_preview = StarterModel(
    name="Gemini 3 Pro Image Preview",
    base=BaseModelType.External,
    source="external://gemini/gemini-3-pro-image-preview",
    description="Google Gemini 3 Pro image generation preview model (external API). Supports up to 14 reference images, including up to 6 object references and up to 5 character references. Supports 1K/2K/4K resolution presets. Requires a configured Gemini API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img"],
        supports_seed=True,
        supports_reference_images=True,
        max_reference_images=14,
        max_images_per_request=1,
        max_image_size=GEMINI_3_IMAGE_MAX_SIZE,
        allowed_aspect_ratios=GEMINI_3_IMAGE_ALLOWED_ASPECT_RATIOS,
        resolution_presets=GEMINI_3_PRO_RESOLUTION_PRESETS,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1),
    panel_schema=ExternalModelPanelSchema(prompts=[{"name": "reference_images"}], image=[{"name": "dimensions"}]),
)

gemini_3_1_flash_image_preview = StarterModel(
    name="Gemini 3.1 Flash Image Preview",
    base=BaseModelType.External,
    source="external://gemini/gemini-3.1-flash-image-preview",
    description="Google Gemini 3.1 Flash image generation preview model (external API). Supports up to 14 reference images, including up to 10 object references and up to 4 character references. Supports 512/1K/2K/4K resolution presets. Requires a configured Gemini API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img"],
        supports_seed=True,
        supports_reference_images=True,
        max_reference_images=14,
        max_images_per_request=1,
        max_image_size=GEMINI_3_IMAGE_MAX_SIZE,
        allowed_aspect_ratios=GEMINI_3_IMAGE_ALLOWED_ASPECT_RATIOS,
        resolution_presets=GEMINI_3_1_FLASH_RESOLUTION_PRESETS,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1),
    panel_schema=ExternalModelPanelSchema(prompts=[{"name": "reference_images"}], image=[{"name": "dimensions"}]),
)

QWEN_IMAGE_2_ALLOWED_ASPECT_RATIOS = ["1:1", "4:3", "3:4", "16:9", "9:16"]

QWEN_IMAGE_MAX_ALLOWED_ASPECT_RATIOS = ["1:1", "4:3", "3:4", "16:9", "9:16"]

WAN_V2_ALLOWED_ASPECT_RATIOS = ["1:1", "4:3", "3:4", "16:9", "9:16"]

alibabacloud_qwen_image_2_pro = StarterModel(
    name="Qwen Image 2.0 Pro",
    base=BaseModelType.External,
    source="external://alibabacloud/qwen-image-2.0-pro",
    description="Alibaba Cloud Qwen Image 2.0 Pro model (external API). Best quality text-to-image with excellent bilingual text rendering. Requires a configured Alibaba Cloud DashScope API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img"],
        supports_negative_prompt=False,
        supports_seed=True,
        max_images_per_request=4,
        allowed_aspect_ratios=QWEN_IMAGE_2_ALLOWED_ASPECT_RATIOS,
        aspect_ratio_sizes={
            "1:1": ExternalImageSize(width=2048, height=2048),
            "4:3": ExternalImageSize(width=2368, height=1728),
            "3:4": ExternalImageSize(width=1728, height=2368),
            "16:9": ExternalImageSize(width=2688, height=1536),
            "9:16": ExternalImageSize(width=1536, height=2688),
        },
    ),
    default_settings=ExternalApiModelDefaultSettings(width=2048, height=2048, num_images=1),
    panel_schema=ExternalModelPanelSchema(image=[{"name": "dimensions"}]),
)

alibabacloud_qwen_image_2 = StarterModel(
    name="Qwen Image 2.0",
    base=BaseModelType.External,
    source="external://alibabacloud/qwen-image-2.0",
    description="Alibaba Cloud Qwen Image 2.0 model (external API). Fast text-to-image with good bilingual text rendering. Requires a configured Alibaba Cloud DashScope API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img"],
        supports_negative_prompt=False,
        supports_seed=True,
        max_images_per_request=4,
        allowed_aspect_ratios=QWEN_IMAGE_2_ALLOWED_ASPECT_RATIOS,
        aspect_ratio_sizes={
            "1:1": ExternalImageSize(width=2048, height=2048),
            "4:3": ExternalImageSize(width=2368, height=1728),
            "3:4": ExternalImageSize(width=1728, height=2368),
            "16:9": ExternalImageSize(width=2688, height=1536),
            "9:16": ExternalImageSize(width=1536, height=2688),
        },
    ),
    default_settings=ExternalApiModelDefaultSettings(width=2048, height=2048, num_images=1),
    panel_schema=ExternalModelPanelSchema(image=[{"name": "dimensions"}]),
)

alibabacloud_qwen_image_max = StarterModel(
    name="Qwen Image Max",
    base=BaseModelType.External,
    source="external://alibabacloud/qwen-image-max",
    description="Alibaba Cloud Qwen Image Max model (external API). High quality text-to-image generation. Requires a configured Alibaba Cloud DashScope API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img"],
        supports_negative_prompt=False,
        supports_seed=True,
        max_images_per_request=4,
        allowed_aspect_ratios=QWEN_IMAGE_MAX_ALLOWED_ASPECT_RATIOS,
        aspect_ratio_sizes={
            "1:1": ExternalImageSize(width=1328, height=1328),
            "4:3": ExternalImageSize(width=1472, height=1104),
            "3:4": ExternalImageSize(width=1104, height=1472),
            "16:9": ExternalImageSize(width=1664, height=928),
            "9:16": ExternalImageSize(width=928, height=1664),
        },
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1328, height=1328, num_images=1),
    panel_schema=ExternalModelPanelSchema(image=[{"name": "dimensions"}]),
)

alibabacloud_wan26_t2i = StarterModel(
    name="Wan 2.6 Text-to-Image",
    base=BaseModelType.External,
    source="external://alibabacloud/wan2.6-t2i",
    description="Alibaba Cloud Wan 2.6 text-to-image model (external API). Photorealistic image generation. Requires a configured Alibaba Cloud DashScope API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img"],
        supports_negative_prompt=False,
        supports_seed=True,
        max_images_per_request=4,
        allowed_aspect_ratios=WAN_V2_ALLOWED_ASPECT_RATIOS,
        aspect_ratio_sizes={
            "1:1": ExternalImageSize(width=1024, height=1024),
            "4:3": ExternalImageSize(width=1440, height=1080),
            "3:4": ExternalImageSize(width=1080, height=1440),
            "16:9": ExternalImageSize(width=1440, height=810),
            "9:16": ExternalImageSize(width=810, height=1440),
        },
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1),
    panel_schema=ExternalModelPanelSchema(image=[{"name": "dimensions"}]),
)

alibabacloud_qwen_image_edit_max = StarterModel(
    name="Qwen Image Edit Max",
    base=BaseModelType.External,
    source="external://alibabacloud/qwen-image-edit-max",
    description="Alibaba Cloud Qwen Image Edit Max model (external API). Image editing with industrial design and geometric reasoning, driven by up to 3 reference images. Requires a configured Alibaba Cloud DashScope API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img"],
        supports_negative_prompt=False,
        supports_reference_images=True,
        supports_seed=True,
        max_reference_images=3,
        max_images_per_request=4,
        allowed_aspect_ratios=QWEN_IMAGE_2_ALLOWED_ASPECT_RATIOS,
        aspect_ratio_sizes={
            "1:1": ExternalImageSize(width=2048, height=2048),
            "4:3": ExternalImageSize(width=2368, height=1728),
            "3:4": ExternalImageSize(width=1728, height=2368),
            "16:9": ExternalImageSize(width=2688, height=1536),
            "9:16": ExternalImageSize(width=1536, height=2688),
        },
    ),
    default_settings=ExternalApiModelDefaultSettings(width=2048, height=2048, num_images=1),
    panel_schema=ExternalModelPanelSchema(prompts=[{"name": "reference_images"}], image=[{"name": "dimensions"}]),
)

OPENAI_GPT_IMAGE_ASPECT_RATIOS = ["1:1", "3:2", "2:3"]

OPENAI_GPT_IMAGE_ASPECT_RATIO_SIZES = {
    "1:1": ExternalImageSize(width=1024, height=1024),
    "3:2": ExternalImageSize(width=1536, height=1024),
    "2:3": ExternalImageSize(width=1024, height=1536),
}

OPENAI_GPT_IMAGE_PANEL_SCHEMA = ExternalModelPanelSchema(
    prompts=[{"name": "reference_images"}], image=[{"name": "dimensions"}]
)

openai_gpt_image_2 = StarterModel(
    name="GPT Image 2",
    base=BaseModelType.External,
    source="external://openai/gpt-image-2",
    description="OpenAI GPT-Image-2 image generation model. State-of-the-art image generation and editing with flexible sizing and high-fidelity image inputs. Does not support transparent backgrounds or configurable input fidelity. Requires a configured OpenAI API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img", "img2img"],
        supports_reference_images=True,
        max_images_per_request=10,
        allowed_aspect_ratios=OPENAI_GPT_IMAGE_ASPECT_RATIOS,
        aspect_ratio_sizes=OPENAI_GPT_IMAGE_ASPECT_RATIO_SIZES,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1),
    panel_schema=OPENAI_GPT_IMAGE_PANEL_SCHEMA,
)

openai_gpt_image_1_5 = StarterModel(
    name="GPT Image 1.5",
    base=BaseModelType.External,
    source="external://openai/gpt-image-1.5",
    description="OpenAI GPT-Image-1.5 image generation model. Fastest and most affordable GPT image model. Requires a configured OpenAI API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img", "img2img"],
        supports_reference_images=True,
        max_images_per_request=10,
        allowed_aspect_ratios=OPENAI_GPT_IMAGE_ASPECT_RATIOS,
        aspect_ratio_sizes=OPENAI_GPT_IMAGE_ASPECT_RATIO_SIZES,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1),
    panel_schema=OPENAI_GPT_IMAGE_PANEL_SCHEMA,
)

openai_gpt_image_1 = StarterModel(
    name="GPT Image 1",
    base=BaseModelType.External,
    source="external://openai/gpt-image-1",
    description="OpenAI GPT-Image-1 image generation model. High quality image generation. Requires a configured OpenAI API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img", "img2img"],
        supports_reference_images=True,
        max_images_per_request=10,
        allowed_aspect_ratios=OPENAI_GPT_IMAGE_ASPECT_RATIOS,
        aspect_ratio_sizes=OPENAI_GPT_IMAGE_ASPECT_RATIO_SIZES,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1),
    panel_schema=OPENAI_GPT_IMAGE_PANEL_SCHEMA,
)

openai_gpt_image_1_mini = StarterModel(
    name="GPT Image 1 Mini",
    base=BaseModelType.External,
    source="external://openai/gpt-image-1-mini",
    description="OpenAI GPT-Image-1-Mini image generation model. Cost-efficient option, 80%% cheaper than GPT-Image-1. Requires a configured OpenAI API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img", "img2img"],
        supports_reference_images=True,
        max_images_per_request=10,
        allowed_aspect_ratios=OPENAI_GPT_IMAGE_ASPECT_RATIOS,
        aspect_ratio_sizes=OPENAI_GPT_IMAGE_ASPECT_RATIO_SIZES,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1),
    panel_schema=OPENAI_GPT_IMAGE_PANEL_SCHEMA,
)

openai_dall_e_3 = StarterModel(
    name="DALL-E 3",
    base=BaseModelType.External,
    source="external://openai/dall-e-3",
    description="OpenAI DALL-E 3 image generation model. Supports vivid and natural styles. Only text-to-image, no editing. Requires a configured OpenAI API key and may incur provider usage costs.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img"],
        max_images_per_request=1,
        allowed_aspect_ratios=["1:1", "7:4", "4:7"],
        aspect_ratio_sizes={
            "1:1": ExternalImageSize(width=1024, height=1024),
            "7:4": ExternalImageSize(width=1792, height=1024),
            "4:7": ExternalImageSize(width=1024, height=1792),
        },
    ),
    default_settings=ExternalApiModelDefaultSettings(width=1024, height=1024, num_images=1),
    panel_schema=ExternalModelPanelSchema(image=[{"name": "dimensions"}]),
)

SEEDREAM_ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]

SEEDREAM_2K_SIZES = {
    "1:1": ExternalImageSize(width=2048, height=2048),
    "3:4": ExternalImageSize(width=1728, height=2304),
    "4:3": ExternalImageSize(width=2304, height=1728),
    "16:9": ExternalImageSize(width=2848, height=1600),
    "9:16": ExternalImageSize(width=1600, height=2848),
    "3:2": ExternalImageSize(width=2496, height=1664),
    "2:3": ExternalImageSize(width=1664, height=2496),
    "21:9": ExternalImageSize(width=3136, height=1344),
}

SEEDREAM_1K_SIZES = {
    "1:1": ExternalImageSize(width=1024, height=1024),
    "3:4": ExternalImageSize(width=864, height=1152),
    "4:3": ExternalImageSize(width=1152, height=864),
    "16:9": ExternalImageSize(width=1312, height=736),
    "9:16": ExternalImageSize(width=736, height=1312),
    "2:3": ExternalImageSize(width=832, height=1248),
    "3:2": ExternalImageSize(width=1248, height=832),
    "21:9": ExternalImageSize(width=1568, height=672),
}

SEEDREAM_PANEL_SCHEMA = ExternalModelPanelSchema(prompts=[{"name": "reference_images"}], image=[{"name": "dimensions"}])

seedream_5_0 = StarterModel(
    name="Seedream 5.0",
    base=BaseModelType.External,
    source="external://seedream/seedream-5-0-260128",
    description="BytePlus Seedream 5.0 flagship image generation model (external API). Supports 2K and 4K resolutions, txt2img and img2img with multi-image reference input.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img", "img2img"],
        supports_reference_images=True,
        max_reference_images=14,
        max_images_per_request=15,
        allowed_aspect_ratios=SEEDREAM_ASPECT_RATIOS,
        aspect_ratio_sizes=SEEDREAM_2K_SIZES,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=2048, height=2048, num_images=1),
    panel_schema=SEEDREAM_PANEL_SCHEMA,
)

seedream_5_0_lite = StarterModel(
    name="Seedream 5.0 Lite",
    base=BaseModelType.External,
    source="external://seedream/seedream-5-0-lite-260128",
    description="BytePlus Seedream 5.0 Lite image generation model (external API). Supports 2K and 4K resolutions, txt2img and img2img with multi-image reference input.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img", "img2img"],
        supports_reference_images=True,
        max_reference_images=14,
        max_images_per_request=15,
        allowed_aspect_ratios=SEEDREAM_ASPECT_RATIOS,
        aspect_ratio_sizes=SEEDREAM_2K_SIZES,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=2048, height=2048, num_images=1),
    panel_schema=SEEDREAM_PANEL_SCHEMA,
)

seedream_4_5 = StarterModel(
    name="Seedream 4.5",
    base=BaseModelType.External,
    source="external://seedream/seedream-4-5-251128",
    description="BytePlus Seedream 4.5 image generation model (external API). Supports 2K and 4K resolutions, txt2img, img2img, batch generation, and multi-image reference input.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img", "img2img"],
        supports_reference_images=True,
        max_reference_images=14,
        max_images_per_request=15,
        allowed_aspect_ratios=SEEDREAM_ASPECT_RATIOS,
        aspect_ratio_sizes=SEEDREAM_2K_SIZES,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=2048, height=2048, num_images=1),
    panel_schema=SEEDREAM_PANEL_SCHEMA,
)

seedream_4_0 = StarterModel(
    name="Seedream 4.0",
    base=BaseModelType.External,
    source="external://seedream/seedream-4-0-250828",
    description="BytePlus Seedream 4.0 image generation model (external API). Supports 1K, 2K, and 4K resolutions, txt2img, img2img, batch generation, and multi-image reference input.",
    type=ModelType.ExternalImageGenerator,
    format=ModelFormat.ExternalApi,
    capabilities=ExternalModelCapabilities(
        modes=["txt2img", "img2img"],
        supports_reference_images=True,
        max_reference_images=14,
        max_images_per_request=15,
        allowed_aspect_ratios=SEEDREAM_ASPECT_RATIOS,
        aspect_ratio_sizes=SEEDREAM_2K_SIZES,
    ),
    default_settings=ExternalApiModelDefaultSettings(width=2048, height=2048, num_images=1),
    panel_schema=SEEDREAM_PANEL_SCHEMA,
)
