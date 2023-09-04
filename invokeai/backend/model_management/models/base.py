import json
import os
import sys
import typing
import inspect
import warnings
from abc import ABCMeta, abstractmethod
from contextlib import suppress
from enum import Enum
from pathlib import Path
from picklescan.scanner import scan_file_path

import torch
import numpy as np
import onnx
import safetensors.torch
from diffusers import DiffusionPipeline, ConfigMixin
from onnx import numpy_helper
from onnxruntime import (
    InferenceSession,
    SessionOptions,
    get_available_providers,
)
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Type, Literal, TypeVar, Generic, Callable, Any, Union
from diffusers import logging as diffusers_logging
from transformers import logging as transformers_logging


class DuplicateModelException(Exception):
    pass


class InvalidModelException(Exception):
    pass


class ModelNotFoundException(Exception):
    pass


class BaseModelType(str, Enum):
    StableDiffusion1 = "sd-1"
    StableDiffusion2 = "sd-2"
    StableDiffusionXL = "sdxl"
    StableDiffusionXLRefiner = "sdxl-refiner"
    # Kandinsky2_1 = "kandinsky-2.1"


class ModelType(str, Enum):
    ONNX = "onnx"
    Main = "main"
    Vae = "vae"
    Lora = "lora"
    ControlNet = "controlnet"  # used by model_probe
    TextualInversion = "embedding"


class SubModelType(str, Enum):
    UNet = "unet"
    TextEncoder = "text_encoder"
    TextEncoder2 = "text_encoder_2"
    Tokenizer = "tokenizer"
    Tokenizer2 = "tokenizer_2"
    Vae = "vae"
    VaeDecoder = "vae_decoder"
    VaeEncoder = "vae_encoder"
    Scheduler = "scheduler"
    SafetyChecker = "safety_checker"
    # MoVQ = "movq"


class ModelVariantType(str, Enum):
    Normal = "normal"
    Inpaint = "inpaint"
    Depth = "depth"


class SchedulerPredictionType(str, Enum):
    Epsilon = "epsilon"
    VPrediction = "v_prediction"
    Sample = "sample"


class ModelError(str, Enum):
    NotFound = "not_found"


class ModelConfigBase(BaseModel):
    path: str  # or Path
    description: Optional[str] = Field(None)
    model_format: Optional[str] = Field(None)
    error: Optional[ModelError] = Field(None)

    class Config:
        use_enum_values = True


class EmptyConfigLoader(ConfigMixin):
    @classmethod
    def load_config(cls, *args, **kwargs):
        cls.config_name = kwargs.pop("config_name")
        return super().load_config(*args, **kwargs)


T_co = TypeVar("T_co", covariant=True)


class classproperty(Generic[T_co]):
    def __init__(self, fget: Callable[[Any], T_co]) -> None:
        self.fget = fget

    def __get__(self, instance: Optional[Any], owner: Type[Any]) -> T_co:
        return self.fget(owner)

    def __set__(self, instance: Optional[Any], value: Any) -> None:
        raise AttributeError("cannot set attribute")


class ModelBase(metaclass=ABCMeta):
    # model_path: str
    # base_model: BaseModelType
    # model_type: ModelType

    def __init__(
        self,
        model_path: str,
        base_model: BaseModelType,
        model_type: ModelType,
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.model_type = model_type

    def _hf_definition_to_type(self, subtypes: List[str]) -> Type:
        if len(subtypes) < 2:
            raise Exception("Invalid subfolder definition!")
        if all(t is None for t in subtypes):
            return None
        elif any(t is None for t in subtypes):
            raise Exception(f"Unsupported definition: {subtypes}")

        if subtypes[0] in ["diffusers", "transformers"]:
            res_type = sys.modules[subtypes[0]]
            subtypes = subtypes[1:]

        else:
            res_type = sys.modules["diffusers"]
            res_type = getattr(res_type, "pipelines")

        for subtype in subtypes:
            res_type = getattr(res_type, subtype)
        return res_type

    @classmethod
    def _get_configs(cls):
        with suppress(Exception):
            return cls.__configs

        configs = dict()
        for name in dir(cls):
            if name.startswith("__"):
                continue

            value = getattr(cls, name)
            if not isinstance(value, type) or not issubclass(value, ModelConfigBase):
                continue

            if hasattr(inspect, "get_annotations"):
                fields = inspect.get_annotations(value)
            else:
                fields = value.__annotations__
            try:
                field = fields["model_format"]
            except Exception:
                raise Exception(f"Invalid config definition - format field not found({cls.__qualname__})")

            if isinstance(field, type) and issubclass(field, str) and issubclass(field, Enum):
                for model_format in field:
                    configs[model_format.value] = value

            elif typing.get_origin(field) is Literal and all(
                isinstance(arg, str) and isinstance(arg, Enum) for arg in field.__args__
            ):
                for model_format in field.__args__:
                    configs[model_format.value] = value

            elif field is None:
                configs[None] = value

            else:
                raise Exception(f"Unsupported format definition in {cls.__qualname__}")

        cls.__configs = configs
        return cls.__configs

    @classmethod
    def create_config(cls, **kwargs) -> ModelConfigBase:
        if "model_format" not in kwargs:
            raise Exception("Field 'model_format' not found in model config")

        configs = cls._get_configs()
        return configs[kwargs["model_format"]](**kwargs)

    @classmethod
    def probe_config(cls, path: str, **kwargs) -> ModelConfigBase:
        return cls.create_config(
            path=path,
            model_format=cls.detect_format(path),
        )

    @classmethod
    @abstractmethod
    def detect_format(cls, path: str) -> str:
        raise NotImplementedError()

    @classproperty
    @abstractmethod
    def save_to_config(cls) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_size(self, child_type: Optional[SubModelType] = None) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_model(
        self,
        torch_dtype: Optional[torch.dtype],
        child_type: Optional[SubModelType] = None,
    ) -> Any:
        raise NotImplementedError()


class DiffusersModel(ModelBase):
    # child_types: Dict[str, Type]
    # child_sizes: Dict[str, int]

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        super().__init__(model_path, base_model, model_type)

        self.child_types: Dict[str, Type] = dict()
        self.child_sizes: Dict[str, int] = dict()

        try:
            config_data = DiffusionPipeline.load_config(self.model_path)
            # config_data = json.loads(os.path.join(self.model_path, "model_index.json"))
        except Exception:
            raise Exception("Invalid diffusers model! (model_index.json not found or invalid)")

        config_data.pop("_ignore_files", None)

        # retrieve all folder_names that contain relevant files
        child_components = [k for k, v in config_data.items() if isinstance(v, list)]

        for child_name in child_components:
            child_type = self._hf_definition_to_type(config_data[child_name])
            self.child_types[child_name] = child_type
            self.child_sizes[child_name] = calc_model_size_by_fs(self.model_path, subfolder=child_name)

    def get_size(self, child_type: Optional[SubModelType] = None):
        if child_type is None:
            return sum(self.child_sizes.values())
        else:
            return self.child_sizes[child_type]

    def get_model(
        self,
        torch_dtype: Optional[torch.dtype],
        child_type: Optional[SubModelType] = None,
    ):
        # return pipeline in different function to pass more arguments
        if child_type is None:
            raise Exception("Child model type can't be null on diffusers model")
        if child_type not in self.child_types:
            return None  # TODO: or raise

        if torch_dtype == torch.float16:
            variants = ["fp16", None]
        else:
            variants = [None, "fp16"]

        # TODO: better error handling(differentiate not found from others)
        for variant in variants:
            try:
                # TODO: set cache_dir to /dev/null to be sure that cache not used?
                model = self.child_types[child_type].from_pretrained(
                    self.model_path,
                    subfolder=child_type.value,
                    torch_dtype=torch_dtype,
                    variant=variant,
                    local_files_only=True,
                )
                break
            except Exception as e:
                if not str(e).startswith("Error no file"):
                    print("====ERR LOAD====")
                    print(f"{variant}: {e}")
                pass
        else:
            raise Exception(f"Failed to load {self.base_model}:{self.model_type}:{child_type} model")

        # calc more accurate size
        self.child_sizes[child_type] = calc_model_size_by_data(model)
        return model

    # def convert_if_required(model_path: str, cache_path: str, config: Optional[dict]) -> str:


def calc_model_size_by_fs(model_path: str, subfolder: Optional[str] = None, variant: Optional[str] = None):
    if subfolder is not None:
        model_path = os.path.join(model_path, subfolder)

    # this can happen when, for example, the safety checker
    # is not downloaded.
    if not os.path.exists(model_path):
        return 0

    all_files = os.listdir(model_path)
    all_files = [f for f in all_files if os.path.isfile(os.path.join(model_path, f))]

    fp16_files = set([f for f in all_files if ".fp16." in f or ".fp16-" in f])
    bit8_files = set([f for f in all_files if ".8bit." in f or ".8bit-" in f])
    other_files = set(all_files) - fp16_files - bit8_files

    if variant is None:
        files = other_files
    elif variant == "fp16":
        files = fp16_files
    elif variant == "8bit":
        files = bit8_files
    else:
        raise NotImplementedError(f"Unknown variant: {variant}")

    # try read from index if exists
    index_postfix = ".index.json"
    if variant is not None:
        index_postfix = f".index.{variant}.json"

    for file in files:
        if not file.endswith(index_postfix):
            continue
        try:
            with open(os.path.join(model_path, file), "r") as f:
                index_data = json.loads(f.read())
            return int(index_data["metadata"]["total_size"])
        except Exception:
            pass

    # calculate files size if there is no index file
    formats = [
        (".safetensors",),  # safetensors
        (".bin",),  # torch
        (".onnx", ".pb"),  # onnx
        (".msgpack",),  # flax
        (".ckpt",),  # tf
        (".h5",),  # tf2
    ]

    for file_format in formats:
        model_files = [f for f in files if f.endswith(file_format)]
        if len(model_files) == 0:
            continue

        model_size = 0
        for model_file in model_files:
            file_stats = os.stat(os.path.join(model_path, model_file))
            model_size += file_stats.st_size
        return model_size

    # raise NotImplementedError(f"Unknown model structure! Files: {all_files}")
    return 0  # scheduler/feature_extractor/tokenizer - models without loading to gpu


def calc_model_size_by_data(model) -> int:
    if isinstance(model, DiffusionPipeline):
        return _calc_pipeline_by_data(model)
    elif isinstance(model, torch.nn.Module):
        return _calc_model_by_data(model)
    elif isinstance(model, IAIOnnxRuntimeModel):
        return _calc_onnx_model_by_data(model)
    else:
        return 0


def _calc_pipeline_by_data(pipeline) -> int:
    res = 0
    for submodel_key in pipeline.components.keys():
        submodel = getattr(pipeline, submodel_key)
        if submodel is not None and isinstance(submodel, torch.nn.Module):
            res += _calc_model_by_data(submodel)
    return res


def _calc_model_by_data(model) -> int:
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs  # in bytes
    return mem


def _calc_onnx_model_by_data(model) -> int:
    tensor_size = model.tensors.size() * 2  # The session doubles this
    mem = tensor_size  # in bytes
    return mem


def _fast_safetensors_reader(path: str):
    checkpoint = dict()
    device = torch.device("meta")
    with open(path, "rb") as f:
        definition_len = int.from_bytes(f.read(8), "little")
        definition_json = f.read(definition_len)
        definition = json.loads(definition_json)

        if "__metadata__" in definition and definition["__metadata__"].get("format", "pt") not in {
            "pt",
            "torch",
            "pytorch",
        }:
            raise Exception("Supported only pytorch safetensors files")
        definition.pop("__metadata__", None)

        for key, info in definition.items():
            dtype = {
                "I8": torch.int8,
                "I16": torch.int16,
                "I32": torch.int32,
                "I64": torch.int64,
                "F16": torch.float16,
                "F32": torch.float32,
                "F64": torch.float64,
            }[info["dtype"]]

            checkpoint[key] = torch.empty(info["shape"], dtype=dtype, device=device)

    return checkpoint


def read_checkpoint_meta(path: Union[str, Path], scan: bool = False):
    if str(path).endswith(".safetensors"):
        try:
            checkpoint = _fast_safetensors_reader(path)
        except Exception:
            # TODO: create issue for support "meta"?
            checkpoint = safetensors.torch.load_file(path, device="cpu")
    else:
        if scan:
            scan_result = scan_file_path(path)
            if scan_result.infected_files != 0:
                raise Exception(f'The model file "{path}" is potentially infected by malware. Aborting import.')
        checkpoint = torch.load(path, map_location=torch.device("meta"))
    return checkpoint


class SilenceWarnings(object):
    def __init__(self):
        self.transformers_verbosity = transformers_logging.get_verbosity()
        self.diffusers_verbosity = diffusers_logging.get_verbosity()

    def __enter__(self):
        transformers_logging.set_verbosity_error()
        diffusers_logging.set_verbosity_error()
        warnings.simplefilter("ignore")

    def __exit__(self, type, value, traceback):
        transformers_logging.set_verbosity(self.transformers_verbosity)
        diffusers_logging.set_verbosity(self.diffusers_verbosity)
        warnings.simplefilter("default")


ONNX_WEIGHTS_NAME = "model.onnx"


class IAIOnnxRuntimeModel:
    class _tensor_access:
        def __init__(self, model):
            self.model = model
            self.indexes = dict()
            for idx, obj in enumerate(self.model.proto.graph.initializer):
                self.indexes[obj.name] = idx

        def __getitem__(self, key: str):
            value = self.model.proto.graph.initializer[self.indexes[key]]
            return numpy_helper.to_array(value)

        def __setitem__(self, key: str, value: np.ndarray):
            new_node = numpy_helper.from_array(value)
            # set_external_data(new_node, location="in-memory-location")
            new_node.name = key
            # new_node.ClearField("raw_data")
            del self.model.proto.graph.initializer[self.indexes[key]]
            self.model.proto.graph.initializer.insert(self.indexes[key], new_node)
            # self.model.data[key] = OrtValue.ortvalue_from_numpy(value)

        # __delitem__

        def __contains__(self, key: str):
            return self.indexes[key] in self.model.proto.graph.initializer

        def items(self):
            raise NotImplementedError("tensor.items")
            # return [(obj.name, obj) for obj in self.raw_proto]

        def keys(self):
            return self.indexes.keys()

        def values(self):
            raise NotImplementedError("tensor.values")
            # return [obj for obj in self.raw_proto]

        def size(self):
            bytesSum = 0
            for node in self.model.proto.graph.initializer:
                bytesSum += sys.getsizeof(node.raw_data)
            return bytesSum

    class _access_helper:
        def __init__(self, raw_proto):
            self.indexes = dict()
            self.raw_proto = raw_proto
            for idx, obj in enumerate(raw_proto):
                self.indexes[obj.name] = idx

        def __getitem__(self, key: str):
            return self.raw_proto[self.indexes[key]]

        def __setitem__(self, key: str, value):
            index = self.indexes[key]
            del self.raw_proto[index]
            self.raw_proto.insert(index, value)

        # __delitem__

        def __contains__(self, key: str):
            return key in self.indexes

        def items(self):
            return [(obj.name, obj) for obj in self.raw_proto]

        def keys(self):
            return self.indexes.keys()

        def values(self):
            return [obj for obj in self.raw_proto]

    def __init__(self, model_path: str, provider: Optional[str]):
        self.path = model_path
        self.session = None
        self.provider = provider
        """
        self.data_path = self.path + "_data"
        if not os.path.exists(self.data_path):
            print(f"Moving model tensors to separate file: {self.data_path}")
            tmp_proto = onnx.load(model_path, load_external_data=True)
            onnx.save_model(tmp_proto, self.path, save_as_external_data=True, all_tensors_to_one_file=True, location=os.path.basename(self.data_path), size_threshold=1024, convert_attribute=False)
            del tmp_proto
            gc.collect()

        self.proto = onnx.load(model_path, load_external_data=False)
        """

        self.proto = onnx.load(model_path, load_external_data=True)
        # self.data = dict()
        # for tensor in self.proto.graph.initializer:
        #     name = tensor.name

        #     if tensor.HasField("raw_data"):
        #         npt = numpy_helper.to_array(tensor)
        #         orv = OrtValue.ortvalue_from_numpy(npt)
        #         # self.data[name] = orv
        #         # set_external_data(tensor, location="in-memory-location")
        #         tensor.name = name
        #         # tensor.ClearField("raw_data")

        self.nodes = self._access_helper(self.proto.graph.node)
        # self.initializers = self._access_helper(self.proto.graph.initializer)
        # print(self.proto.graph.input)
        # print(self.proto.graph.initializer)

        self.tensors = self._tensor_access(self)

    # TODO: integrate with model manager/cache
    def create_session(self, height=None, width=None):
        if self.session is None or self.session_width != width or self.session_height != height:
            # onnx.save(self.proto, "tmp.onnx")
            # onnx.save_model(self.proto, "tmp.onnx", save_as_external_data=True, all_tensors_to_one_file=True, location="tmp.onnx_data", size_threshold=1024, convert_attribute=False)
            # TODO: something to be able to get weight when they already moved outside of model proto
            # (trimmed_model, external_data) = buffer_external_data_tensors(self.proto)
            sess = SessionOptions()
            # self._external_data.update(**external_data)
            # sess.add_external_initializers(list(self.data.keys()), list(self.data.values()))
            # sess.enable_profiling = True

            # sess.intra_op_num_threads = 1
            # sess.inter_op_num_threads = 1
            # sess.execution_mode = ExecutionMode.ORT_SEQUENTIAL
            # sess.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
            # sess.enable_cpu_mem_arena = True
            # sess.enable_mem_pattern = True
            # sess.add_session_config_entry("session.intra_op.use_xnnpack_threadpool", "1") ########### It's the key code
            self.session_height = height
            self.session_width = width
            if height and width:
                sess.add_free_dimension_override_by_name("unet_sample_batch", 2)
                sess.add_free_dimension_override_by_name("unet_sample_channels", 4)
                sess.add_free_dimension_override_by_name("unet_hidden_batch", 2)
                sess.add_free_dimension_override_by_name("unet_hidden_sequence", 77)
                sess.add_free_dimension_override_by_name("unet_sample_height", self.session_height)
                sess.add_free_dimension_override_by_name("unet_sample_width", self.session_width)
                sess.add_free_dimension_override_by_name("unet_time_batch", 1)
            providers = []
            if self.provider:
                providers.append(self.provider)
            else:
                providers = get_available_providers()
            if "TensorrtExecutionProvider" in providers:
                providers.remove("TensorrtExecutionProvider")
            try:
                self.session = InferenceSession(self.proto.SerializeToString(), providers=providers, sess_options=sess)
            except Exception as e:
                raise e
            # self.session = InferenceSession("tmp.onnx", providers=[self.provider], sess_options=self.sess_options)
            # self.io_binding = self.session.io_binding()

    def release_session(self):
        self.session = None
        import gc

        gc.collect()
        return

    def __call__(self, **kwargs):
        if self.session is None:
            raise Exception("You should call create_session before running model")

        inputs = {k: np.array(v) for k, v in kwargs.items()}
        # output_names = self.session.get_outputs()
        # for k in inputs:
        #     self.io_binding.bind_cpu_input(k, inputs[k])
        # for name in output_names:
        #     self.io_binding.bind_output(name.name)
        # self.session.run_with_iobinding(self.io_binding, None)
        # return self.io_binding.copy_outputs_to_cpu()
        return self.session.run(None, inputs)

    # compatability with diffusers load code
    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        subfolder: Union[str, Path] = None,
        file_name: Optional[str] = None,
        provider: Optional[str] = None,
        sess_options: Optional["SessionOptions"] = None,
        **kwargs,
    ):
        file_name = file_name or ONNX_WEIGHTS_NAME

        if os.path.isdir(model_id):
            model_path = model_id
            if subfolder is not None:
                model_path = os.path.join(model_path, subfolder)
            model_path = os.path.join(model_path, file_name)

        else:
            model_path = model_id

        # load model from local directory
        if not os.path.isfile(model_path):
            raise Exception(f"Model not found: {model_path}")

        # TODO: session options
        return cls(model_path, provider=provider)
