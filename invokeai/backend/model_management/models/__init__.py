import inspect
from enum import Enum
from pydantic import BaseModel
from typing import Literal, get_origin
from .base import (  # noqa: F401
    BaseModelType,
    ModelType,
    SubModelType,
    ModelBase,
    ModelConfigBase,
    ModelVariantType,
    SchedulerPredictionType,
    ModelError,
    SilenceWarnings,
    ModelNotFoundException,
    InvalidModelException,
    DuplicateModelException,
)
from .stable_diffusion import StableDiffusion1Model, StableDiffusion2Model
from .sdxl import StableDiffusionXLModel
from .vae import VaeModel
from .lora import LoRAModel
from .controlnet import ControlNetModel  # TODO:
from .textual_inversion import TextualInversionModel

from .stable_diffusion_onnx import ONNXStableDiffusion1Model, ONNXStableDiffusion2Model

MODEL_CLASSES = {
    BaseModelType.StableDiffusion1: {
        ModelType.ONNX: ONNXStableDiffusion1Model,
        ModelType.Main: StableDiffusion1Model,
        ModelType.Vae: VaeModel,
        ModelType.Lora: LoRAModel,
        ModelType.ControlNet: ControlNetModel,
        ModelType.TextualInversion: TextualInversionModel,
    },
    BaseModelType.StableDiffusion2: {
        ModelType.ONNX: ONNXStableDiffusion2Model,
        ModelType.Main: StableDiffusion2Model,
        ModelType.Vae: VaeModel,
        ModelType.Lora: LoRAModel,
        ModelType.ControlNet: ControlNetModel,
        ModelType.TextualInversion: TextualInversionModel,
    },
    BaseModelType.StableDiffusionXL: {
        ModelType.Main: StableDiffusionXLModel,
        ModelType.Vae: VaeModel,
        # will not work until support written
        ModelType.Lora: LoRAModel,
        ModelType.ControlNet: ControlNetModel,
        ModelType.TextualInversion: TextualInversionModel,
        ModelType.ONNX: ONNXStableDiffusion2Model,
    },
    BaseModelType.StableDiffusionXLRefiner: {
        ModelType.Main: StableDiffusionXLModel,
        ModelType.Vae: VaeModel,
        # will not work until support written
        ModelType.Lora: LoRAModel,
        ModelType.ControlNet: ControlNetModel,
        ModelType.TextualInversion: TextualInversionModel,
        ModelType.ONNX: ONNXStableDiffusion2Model,
    },
    # BaseModelType.Kandinsky2_1: {
    #    ModelType.Main: Kandinsky2_1Model,
    #    ModelType.MoVQ: MoVQModel,
    #    ModelType.Lora: LoRAModel,
    #    ModelType.ControlNet: ControlNetModel,
    #    ModelType.TextualInversion: TextualInversionModel,
    # },
}

MODEL_CONFIGS = list()
OPENAPI_MODEL_CONFIGS = list()


class OpenAPIModelInfoBase(BaseModel):
    model_name: str
    base_model: BaseModelType
    model_type: ModelType


for base_model, models in MODEL_CLASSES.items():
    for model_type, model_class in models.items():
        model_configs = set(model_class._get_configs().values())
        model_configs.discard(None)
        MODEL_CONFIGS.extend(model_configs)

        # LS: sort to get the checkpoint configs first, which makes
        # for a better template in the Swagger docs
        for cfg in sorted(model_configs, key=lambda x: str(x)):
            model_name, cfg_name = cfg.__qualname__.split(".")[-2:]
            openapi_cfg_name = model_name + cfg_name
            if openapi_cfg_name in vars():
                continue

            api_wrapper = type(
                openapi_cfg_name,
                (cfg, OpenAPIModelInfoBase),
                dict(
                    __annotations__=dict(
                        model_type=Literal[model_type.value],
                    ),
                ),
            )

            # globals()[openapi_cfg_name] = api_wrapper
            vars()[openapi_cfg_name] = api_wrapper
            OPENAPI_MODEL_CONFIGS.append(api_wrapper)


def get_model_config_enums():
    enums = list()

    for model_config in MODEL_CONFIGS:
        if hasattr(inspect, "get_annotations"):
            fields = inspect.get_annotations(model_config)
        else:
            fields = model_config.__annotations__
        try:
            field = fields["model_format"]
        except Exception:
            raise Exception("format field not found")

        # model_format: None
        # model_format: SomeModelFormat
        # model_format: Literal[SomeModelFormat.Diffusers]
        # model_format: Literal[SomeModelFormat.Diffusers, SomeModelFormat.Checkpoint]

        if isinstance(field, type) and issubclass(field, str) and issubclass(field, Enum):
            enums.append(field)

        elif get_origin(field) is Literal and all(
            isinstance(arg, str) and isinstance(arg, Enum) for arg in field.__args__
        ):
            enums.append(type(field.__args__[0]))

        elif field is None:
            pass

        else:
            raise Exception(f"Unsupported format definition in {model_configs.__qualname__}")

    return enums
