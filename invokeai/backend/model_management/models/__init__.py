from .base import BaseModelType, ModelType, SubModelType, ModelBase, ModelConfigBase, VariantType
from .stable_diffusion import StableDiffusion15Model, StableDiffusion2Model, StableDiffusion2BaseModel
from .vae import VaeModel
from .lora import LoRAModel
#from .controlnet import ControlNetModel # TODO:
from .textual_inversion import TextualInversionModel

# TODO:
class ControlNetModel:
    pass

MODEL_CLASSES = {
    BaseModelType.StableDiffusion1_5: {
        ModelType.Pipeline: StableDiffusion15Model,
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
    BaseModelType.StableDiffusion2Base: {
        ModelType.Pipeline: StableDiffusion2BaseModel,
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

# TODO: check with openapi annotation
def get_all_model_configs():
    configs = []
    for models in MODEL_CLASSES.values():
        for model in models.values():
            configs.extend(model._get_configs())
    return configs
