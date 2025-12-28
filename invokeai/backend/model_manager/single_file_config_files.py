from dataclasses import dataclass

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelType,
    ModelVariantType,
    SchedulerPredictionType,
)


@dataclass(frozen=True)
class LegacyConfigKey:
    type: ModelType
    base: BaseModelType
    variant: ModelVariantType | None = None
    pred: SchedulerPredictionType | None = None

    @classmethod
    def from_model_config(cls, config: AnyModelConfig) -> "LegacyConfigKey":
        variant = getattr(config, "variant", None)
        pred = getattr(config, "prediction_type", None)
        return cls(type=config.type, base=config.base, variant=variant, pred=pred)


LEGACY_CONFIG_MAP: dict[LegacyConfigKey, str] = {
    LegacyConfigKey(
        ModelType.Main,
        BaseModelType.StableDiffusion1,
        ModelVariantType.Normal,
        SchedulerPredictionType.Epsilon,
    ): "stable-diffusion/v1-inference.yaml",
    LegacyConfigKey(
        ModelType.Main,
        BaseModelType.StableDiffusion1,
        ModelVariantType.Normal,
        SchedulerPredictionType.VPrediction,
    ): "stable-diffusion/v1-inference-v.yaml",
    LegacyConfigKey(
        ModelType.Main,
        BaseModelType.StableDiffusion1,
        ModelVariantType.Inpaint,
    ): "stable-diffusion/v1-inpainting-inference.yaml",
    LegacyConfigKey(
        ModelType.Main,
        BaseModelType.StableDiffusion2,
        ModelVariantType.Normal,
        SchedulerPredictionType.Epsilon,
    ): "stable-diffusion/v2-inference.yaml",
    LegacyConfigKey(
        ModelType.Main,
        BaseModelType.StableDiffusion2,
        ModelVariantType.Normal,
        SchedulerPredictionType.VPrediction,
    ): "stable-diffusion/v2-inference-v.yaml",
    LegacyConfigKey(
        ModelType.Main,
        BaseModelType.StableDiffusion2,
        ModelVariantType.Inpaint,
        SchedulerPredictionType.Epsilon,
    ): "stable-diffusion/v2-inpainting-inference.yaml",
    LegacyConfigKey(
        ModelType.Main,
        BaseModelType.StableDiffusion2,
        ModelVariantType.Inpaint,
        SchedulerPredictionType.VPrediction,
    ): "stable-diffusion/v2-inpainting-inference-v.yaml",
    LegacyConfigKey(
        ModelType.Main,
        BaseModelType.StableDiffusion2,
        ModelVariantType.Depth,
    ): "stable-diffusion/v2-midas-inference.yaml",
    LegacyConfigKey(
        ModelType.Main,
        BaseModelType.StableDiffusionXL,
        ModelVariantType.Normal,
    ): "stable-diffusion/sd_xl_base.yaml",
    LegacyConfigKey(
        ModelType.Main,
        BaseModelType.StableDiffusionXL,
        ModelVariantType.Inpaint,
    ): "stable-diffusion/sd_xl_inpaint.yaml",
    LegacyConfigKey(
        ModelType.Main,
        BaseModelType.StableDiffusionXLRefiner,
        ModelVariantType.Normal,
    ): "stable-diffusion/sd_xl_refiner.yaml",
    LegacyConfigKey(ModelType.ControlNet, BaseModelType.StableDiffusion1): "controlnet/cldm_v15.yaml",
    LegacyConfigKey(ModelType.ControlNet, BaseModelType.StableDiffusion2): "controlnet/cldm_v21.yaml",
    LegacyConfigKey(ModelType.VAE, BaseModelType.StableDiffusion1): "stable-diffusion/v1-inference.yaml",
    LegacyConfigKey(ModelType.VAE, BaseModelType.StableDiffusion2): "stable-diffusion/v2-inference.yaml",
    LegacyConfigKey(ModelType.VAE, BaseModelType.StableDiffusionXL): "stable-diffusion/sd_xl_base.yaml",
}
