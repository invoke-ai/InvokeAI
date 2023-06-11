import sys
import typing
import inspect
from enum import Enum
import torch
from diffusers import DiffusionPipeline, ConfigMixin

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Type, Literal

class BaseModelType(str, Enum):
    #StableDiffusion1_5 = "stable_diffusion_1_5"
    #StableDiffusion2 = "stable_diffusion_2"
    #StableDiffusion2Base = "stable_diffusion_2_base"
    # TODO: maybe then add sample size(512/768)?
    StableDiffusion1_5 = "sd-1.5"
    StableDiffusion2Base = "sd-2-base"   # 512 pixels; this will have epsilon parameterization
    StableDiffusion2 = "sd-2"            # 768 pixels; this will have v-prediction parameterization
    #Kandinsky2_1 = "kandinsky_2_1"

class ModelType(str, Enum):
    Pipeline = "pipeline"
    Vae = "vae"
    Lora = "lora"
    ControlNet = "controlnet" # used by model_probe
    TextualInversion = "embedding"

class SubModelType(str, Enum):
    UNet = "unet"
    TextEncoder = "text_encoder"
    Tokenizer = "tokenizer"
    Vae = "vae"
    Scheduler = "scheduler"
    SafetyChecker = "safety_checker"
    #MoVQ = "movq"

class VariantType(str, Enum):
    Normal = "normal"
    Inpaint = "inpaint"
    Depth = "depth"
    
class ModelError(str, Enum):
    NotFound = "not_found"

class ModelConfigBase(BaseModel):
    path: str # or Path
    #name: str # not included as present in model key
    description: Optional[str] = Field(None)
    format: Optional[str] = Field(None)
    default: Optional[bool] = Field(False)
    # do not save to config
    error: Optional[ModelError] = Field(None, exclude=True)


class EmptyConfigLoader(ConfigMixin):
    @classmethod
    def load_config(cls, *args, **kwargs):
        cls.config_name = kwargs.pop("config_name")
        return super().load_config(*args, **kwargs)

class ModelBase:
    #model_path: str
    #base_model: BaseModelType
    #model_type: ModelType

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
        if not hasattr(cls, "__configs"):
            configs = dict()
            for name in dir(cls):
                if name.startswith("__"):
                    continue

                value = getattr(cls, name)
                if not isinstance(value, type) or not issubclass(value, ModelConfigBase):
                    continue

                fields = inspect.get_annotations(value)
                if "format" not in fields or typing.get_origin(fields["format"]) != Literal:
                    raise Exception("Invalid config definition - format field not found")

                format_type = typing.get_origin(fields["format"])
                if format_type not in {None, Literal}:
                    raise Exception(f"Invalid config definition - unknown format type: {fields['format']}")

                if format_type is Literal:
                    format = fields["format"].__args__[0]
                else:
                    format = None
                configs[format] = value # TODO: error when override(multiple)?

            cls.__configs = configs

        return cls.__configs

    @classmethod
    def build_config(cls, **kwargs):
        if "format" not in kwargs:
            kwargs["format"] = cls.detect_format(kwargs["path"])
            
        configs = cls._get_configs()
        return configs[kwargs["format"]](**kwargs)

    @classmethod
    def detect_format(cls, path: str) -> str:
        raise NotImplementedError()



class DiffusersModel(ModelBase):
    #child_types: Dict[str, Type]
    #child_sizes: Dict[str, int]

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        super().__init__(model_path, base_model, model_type)

        self.child_types: Dict[str, Type] = dict()
        self.child_sizes: Dict[str, int] = dict()

        try:
            config_data = DiffusionPipeline.load_config(self.model_path)
            #config_data = json.loads(os.path.join(self.model_path, "model_index.json"))
        except:
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
            return None # TODO: or raise

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
                print("====ERR LOAD====")
                print(f"{variant}: {e}")

        # calc more accurate size
        self.child_sizes[child_type] = calc_model_size_by_data(model)
        return model

    #def convert_if_required(model_path: str, cache_path: str, config: Optional[dict]) -> str:



def calc_model_size_by_fs(
    model_path: str,
    subfolder: Optional[str] = None,
    variant: Optional[str] = None
):
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
        except:
            pass

    # calculate files size if there is no index file
    formats = [
        (".safetensors",), # safetensors
        (".bin",), # torch
        (".onnx", ".pb"), # onnx
        (".msgpack",), # flax
        (".ckpt",), # tf
        (".h5",), # tf2
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
    
    #raise NotImplementedError(f"Unknown model structure! Files: {all_files}")
    return 0 # scheduler/feature_extractor/tokenizer - models without loading to gpu


def calc_model_size_by_data(model) -> int:
    if isinstance(model, DiffusionPipeline):
        return _calc_pipeline_by_data(model)
    elif isinstance(model, torch.nn.Module):
        return _calc_model_by_data(model)
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
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    return mem
