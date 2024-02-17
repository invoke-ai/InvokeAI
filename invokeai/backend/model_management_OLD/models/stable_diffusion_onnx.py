from enum import Enum
from typing import Literal

from diffusers import OnnxRuntimeModel

from .base import (
    BaseModelType,
    DiffusersModel,
    IAIOnnxRuntimeModel,
    ModelConfigBase,
    ModelType,
    ModelVariantType,
    SchedulerPredictionType,
    classproperty,
)


class StableDiffusionOnnxModelFormat(str, Enum):
    Olive = "olive"
    Onnx = "onnx"


class ONNXStableDiffusion1Model(DiffusersModel):
    class Config(ModelConfigBase):
        model_format: Literal[StableDiffusionOnnxModelFormat.Onnx]
        variant: ModelVariantType

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert base_model == BaseModelType.StableDiffusion1
        assert model_type == ModelType.ONNX
        super().__init__(
            model_path=model_path,
            base_model=BaseModelType.StableDiffusion1,
            model_type=ModelType.ONNX,
        )

        for child_name, child_type in self.child_types.items():
            if child_type is OnnxRuntimeModel:
                self.child_types[child_name] = IAIOnnxRuntimeModel

            # TODO: check that no optimum models provided

    @classmethod
    def probe_config(cls, path: str, **kwargs):
        model_format = cls.detect_format(path)
        in_channels = 4  # TODO:

        if in_channels == 9:
            variant = ModelVariantType.Inpaint
        elif in_channels == 4:
            variant = ModelVariantType.Normal
        else:
            raise Exception("Unkown stable diffusion 1.* model format")

        return cls.create_config(
            path=path,
            model_format=model_format,
            variant=variant,
        )

    @classproperty
    def save_to_config(cls) -> bool:
        return True

    @classmethod
    def detect_format(cls, model_path: str):
        # TODO: Detect onnx vs olive
        return StableDiffusionOnnxModelFormat.Onnx

    @classmethod
    def convert_if_required(
        cls,
        model_path: str,
        output_path: str,
        config: ModelConfigBase,
        base_model: BaseModelType,
    ) -> str:
        return model_path


class ONNXStableDiffusion2Model(DiffusersModel):
    # TODO: check that configs overwriten properly
    class Config(ModelConfigBase):
        model_format: Literal[StableDiffusionOnnxModelFormat.Onnx]
        variant: ModelVariantType
        prediction_type: SchedulerPredictionType
        upcast_attention: bool

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert base_model == BaseModelType.StableDiffusion2
        assert model_type == ModelType.ONNX
        super().__init__(
            model_path=model_path,
            base_model=BaseModelType.StableDiffusion2,
            model_type=ModelType.ONNX,
        )

        for child_name, child_type in self.child_types.items():
            if child_type is OnnxRuntimeModel:
                self.child_types[child_name] = IAIOnnxRuntimeModel
            # TODO: check that no optimum models provided

    @classmethod
    def probe_config(cls, path: str, **kwargs):
        model_format = cls.detect_format(path)
        in_channels = 4  # TODO:

        if in_channels == 9:
            variant = ModelVariantType.Inpaint
        elif in_channels == 5:
            variant = ModelVariantType.Depth
        elif in_channels == 4:
            variant = ModelVariantType.Normal
        else:
            raise Exception("Unkown stable diffusion 2.* model format")

        if variant == ModelVariantType.Normal:
            prediction_type = SchedulerPredictionType.VPrediction
            upcast_attention = True

        else:
            prediction_type = SchedulerPredictionType.Epsilon
            upcast_attention = False

        return cls.create_config(
            path=path,
            model_format=model_format,
            variant=variant,
            prediction_type=prediction_type,
            upcast_attention=upcast_attention,
        )

    @classproperty
    def save_to_config(cls) -> bool:
        return True

    @classmethod
    def detect_format(cls, model_path: str):
        # TODO: Detect onnx vs olive
        return StableDiffusionOnnxModelFormat.Onnx

    @classmethod
    def convert_if_required(
        cls,
        model_path: str,
        output_path: str,
        config: ModelConfigBase,
        base_model: BaseModelType,
    ) -> str:
        return model_path
