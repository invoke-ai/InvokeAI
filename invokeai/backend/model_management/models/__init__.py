from .base import BaseModelType, ModelType, SubModelType, ModelBase, ModelConfigBase, ModelVariantType, SchedulerPredictionType, ModelError, SilenceWarnings
from .stable_diffusion import StableDiffusion1Model, StableDiffusion2Model
from .vae import VaeModel
from .lora import LoRAModel
from .controlnet import ControlNetModel # TODO:
from .textual_inversion import TextualInversionModel

MODEL_CLASSES = {
    BaseModelType.StableDiffusion1: {
        ModelType.Pipeline: StableDiffusion1Model,
        ModelType.Vae: VaeModel,
        ModelType.Lora: LoRAModel,
        ModelType.ControlNet: ControlNetModel,
        ModelType.TextualInversion: TextualInversionModel,
    },
    BaseModelType.StableDiffusion2: {
        ModelType.Pipeline: StableDiffusion2Model,
        ModelType.Vae: VaeModel,
        ModelType.Lora: LoRAModel,
        ModelType.ControlNet: ControlNetModel,
        ModelType.TextualInversion: TextualInversionModel,
    },
    #BaseModelType.Kandinsky2_1: {
    #    ModelType.Pipeline: Kandinsky2_1Model,
    #    ModelType.MoVQ: MoVQModel,
    #    ModelType.Lora: LoRAModel,
    #    ModelType.ControlNet: ControlNetModel,
    #    ModelType.TextualInversion: TextualInversionModel,
    #},
}

def _get_all_model_configs():
    configs = set()
    for models in MODEL_CLASSES.values():
        for _, model in models.items():
            configs.update(model._get_configs().values())
    configs.discard(None)
    return list(configs)

MODEL_CONFIGS = _get_all_model_configs()
OPENAPI_MODEL_CONFIGS = list()

for cfg in MODEL_CONFIGS:
    model_name, cfg_name = cfg.__qualname__.split('.')[-2:]
    openapi_cfg_name = model_name + cfg_name
    name_wrapper = type(openapi_cfg_name, (cfg,), {})

    #globals()[name] = value
    vars()[openapi_cfg_name] = name_wrapper
    OPENAPI_MODEL_CONFIGS.append(name_wrapper)
