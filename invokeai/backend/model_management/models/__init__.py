import inspect
from enum import Enum
from typing import Literal, get_origin

from pydantic import BaseModel, ConfigDict, create_model

from .base import (  # noqa: F401
    BaseModelType,
    DuplicateModelException,
    InvalidModelException,
    ModelBase,
    ModelConfigBase,
    ModelError,
    ModelNotFoundException,
    ModelType,
    ModelVariantType,
    SchedulerPredictionType,
    SilenceWarnings,
    SubModelType,
)
from .clip_vision import CLIPVisionModel
from .controlnet import ControlNetModel  # TODO:
from .ip_adapter import IPAdapterModel
from .lora import LoRAModel
from .sdxl import StableDiffusionXLModel
from .stable_diffusion import StableDiffusion1Model, StableDiffusion2Model
from .stable_diffusion_onnx import ONNXStableDiffusion1Model, ONNXStableDiffusion2Model
from .t2i_adapter import T2IAdapterModel
from .textual_inversion import TextualInversionModel
from .vae import VaeModel

MODEL_CLASSES = {
    BaseModelType.StableDiffusion1: {
        ModelType.ONNX: ONNXStableDiffusion1Model,
        ModelType.Main: StableDiffusion1Model,
        ModelType.Vae: VaeModel,
        ModelType.Lora: LoRAModel,
        ModelType.ControlNet: ControlNetModel,
        ModelType.TextualInversion: TextualInversionModel,
        ModelType.IPAdapter: IPAdapterModel,
        ModelType.CLIPVision: CLIPVisionModel,
        ModelType.T2IAdapter: T2IAdapterModel,
    },
    BaseModelType.StableDiffusion2: {
        ModelType.ONNX: ONNXStableDiffusion2Model,
        ModelType.Main: StableDiffusion2Model,
        ModelType.Vae: VaeModel,
        ModelType.Lora: LoRAModel,
        ModelType.ControlNet: ControlNetModel,
        ModelType.TextualInversion: TextualInversionModel,
        ModelType.IPAdapter: IPAdapterModel,
        ModelType.CLIPVision: CLIPVisionModel,
        ModelType.T2IAdapter: T2IAdapterModel,
    },
    BaseModelType.StableDiffusionXL: {
        ModelType.Main: StableDiffusionXLModel,
        ModelType.Vae: VaeModel,
        # will not work until support written
        ModelType.Lora: LoRAModel,
        ModelType.ControlNet: ControlNetModel,
        ModelType.TextualInversion: TextualInversionModel,
        ModelType.ONNX: ONNXStableDiffusion2Model,
        ModelType.IPAdapter: IPAdapterModel,
        ModelType.CLIPVision: CLIPVisionModel,
        ModelType.T2IAdapter: T2IAdapterModel,
    },
    BaseModelType.StableDiffusionXLRefiner: {
        ModelType.Main: StableDiffusionXLModel,
        ModelType.Vae: VaeModel,
        # will not work until support written
        ModelType.Lora: LoRAModel,
        ModelType.ControlNet: ControlNetModel,
        ModelType.TextualInversion: TextualInversionModel,
        ModelType.ONNX: ONNXStableDiffusion2Model,
        ModelType.IPAdapter: IPAdapterModel,
        ModelType.CLIPVision: CLIPVisionModel,
        ModelType.T2IAdapter: T2IAdapterModel,
    },
    BaseModelType.Any: {
        ModelType.CLIPVision: CLIPVisionModel,
        # The following model types are not expected to be used with BaseModelType.Any.
        ModelType.ONNX: ONNXStableDiffusion2Model,
        ModelType.Main: StableDiffusion2Model,
        ModelType.Vae: VaeModel,
        ModelType.Lora: LoRAModel,
        ModelType.ControlNet: ControlNetModel,
        ModelType.TextualInversion: TextualInversionModel,
        ModelType.IPAdapter: IPAdapterModel,
        ModelType.T2IAdapter: T2IAdapterModel,
    },
    # BaseModelType.Kandinsky2_1: {
    #    ModelType.Main: Kandinsky2_1Model,
    #    ModelType.MoVQ: MoVQModel,
    #    ModelType.Lora: LoRAModel,
    #    ModelType.ControlNet: ControlNetModel,
    #    ModelType.TextualInversion: TextualInversionModel,
    # },
}

MODEL_CONFIGS = []
OPENAPI_MODEL_CONFIGS = []


class OpenAPIModelInfoBase(BaseModel):
    model_name: str
    base_model: BaseModelType
    model_type: ModelType

    model_config = ConfigDict(protected_namespaces=())


for _base_model, models in MODEL_CLASSES.items():
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

            api_wrapper = create_model(
                openapi_cfg_name,
                __base__=(cfg, OpenAPIModelInfoBase),
                model_type=(Literal[model_type], model_type),  # type: ignore
            )
            vars()[openapi_cfg_name] = api_wrapper
            OPENAPI_MODEL_CONFIGS.append(api_wrapper)


def get_model_config_enums():
    enums = []

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
