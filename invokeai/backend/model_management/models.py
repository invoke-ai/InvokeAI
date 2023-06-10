import sys
from enum import Enum
import torch
import safetensors.torch
from diffusers.utils import is_safetensors_available

class BaseModelType(str, Enum):
    #StableDiffusion1_5 = "stable_diffusion_1_5"
    #StableDiffusion2 = "stable_diffusion_2"
    #StableDiffusion2Base = "stable_diffusion_2_base"
    # TODO: maybe then add sample size(512/768)?
    StableDiffusion1_5 = "SD-1"
    StableDiffusion2Base = "SD-2-base"   # 512 pixels; this will have epsilon parameterization
    StableDiffusion2 = "SD-2"            # 768 pixels; this will have v-prediction parameterization
    #Kandinsky2_1 = "kandinsky_2_1"

class ModelType(str, Enum):
    Pipeline = "pipeline"
    Classifier = "classifier"
    Vae = "vae"

    Lora = "lora"
    ControlNet = "controlnet"
    TextualInversion = "embedding"

class SubModelType:
    UNet = "unet"
    TextEncoder = "text_encoder"
    Tokenizer = "tokenizer"
    Vae = "vae"
    Scheduler = "scheduler"
    SafetyChecker = "safety_checker"
    #MoVQ = "movq"

MODEL_CLASSES = {
    BaseModel.StableDiffusion1_5: {
        ModelType.Pipeline: StableDiffusionModel,
        ModelType.Classifier: ClassifierModel,
        ModelType.Vae: VaeModel,
        ModelType.Lora: LoraModel,
        ModelType.ControlNet: ControlNetModel,
        ModelType.TextualInversion: TextualInversionModel,
    },
    BaseModel.StableDiffusion2: {
        ModelType.Pipeline: StableDiffusionModel,
        ModelType.Classifier: ClassifierModel,
        ModelType.Vae: VaeModel,
        ModelType.Lora: LoraModel,
        ModelType.ControlNet: ControlNetModel,
        ModelType.TextualInversion: TextualInversionModel,
    },
    BaseModel.StableDiffusion2Base: {
        ModelType.Pipeline: StableDiffusionModel,
        ModelType.Classifier: ClassifierModel,
        ModelType.Vae: VaeModel,
        ModelType.Lora: LoraModel,
        ModelType.ControlNet: ControlNetModel,
        ModelType.TextualInversion: TextualInversionModel,
    },
    #BaseModel.Kandinsky2_1: {
    #    ModelType.Pipeline: Kandinsky2_1Model,
    #    ModelType.Classifier: ClassifierModel,
    #    ModelType.MoVQ: MoVQModel,
    #    ModelType.Lora: LoraModel,
    #    ModelType.ControlNet: ControlNetModel,
    #    ModelType.TextualInversion: TextualInversionModel,
    #},
}

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

    #def convert_if_required(model_path: Union[str, Path], cache_path: str, config: Optional[dict]) -> Path:


class StableDiffusionModel(DiffusersModel):
    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert base_model in {
            BaseModelType.StableDiffusion1_5,
            BaseModelType.StableDiffusion2,
            BaseModelType.StableDiffusion2Base,
        }
        assert model_type == ModelType.Pipeline
        super().__init__(model_path, base_model, model_type)

    @staticmethod
    def convert_if_required(model_path: Union[str, Path], dst_path: str, config: Optional[dict]) -> Path:
        if not isinstance(model_path, Path):
            model_path = Path(model_path)

        # TODO: args
        # TODO: set model_path, to config? pass dst_path as arg?
        # TODO: check
        return _convert_ckpt_and_cache(config)

class classproperty(object):  # pylint: disable=invalid-name
  """Class property decorator.

  Example usage:

  class MyClass(object):

    @classproperty
    def value(cls):
      return '123'

  > print MyClass.value
  123
  """

  def __init__(self, func):
    self._func = func

  def __get__(self, owner_self, owner_cls):
    return self._func(owner_cls)

class ModelConfigBase(BaseModel):
    path: str # or Path
    name: str
    description: Optional[str]


class StableDiffusionDModel(DiffusersModel):
    class Config(ModelConfigBase):
        format: str
        vae: Optional[str] = Field(None)
        config: Optional[str] = Field(None)

        @root_validator
        def validator(cls, values):
            if values["format"] not in {"checkpoint", "diffusers"}:
                raise ValueError(f"Unkown stable diffusion model format: {values['format']}")
            if values["config"] is not None and values["format"] != "checkpoint":
                raise ValueError(f"Custom config field allowed only in checkpoint stable diffusion model")
            return values

        # return config only for checkpoint format
        def dict(self, *args, **kwargs):
            result = super().dict(*args, **kwargs)
            if self.format != "checkpoint":
                result.pop("config", None)
            return result

    @classproperty
    def has_config(self):
        return True
    
    def build_config(self, **kwargs) -> dict:
        try:
            res = dict(
                path=kwargs["path"],
                name=kwargs["name"],
                description=kwargs.get("description", None),

                format=kwargs["format"],
                vae=kwargs.get("vae", None),
            )
            if res["format"] not in {"checkpoint", "diffusers"}:
                raise Exception(f"Unkonwn stable diffusion model format: {res['format']}")
            if res["format"] == "checkpoint":
                res["config"] = kwargs.get("config", None)
            # TODO: raise if config specified for diffusers?

            return res

        except KeyError as e:
            raise Exception(f"Field \"{e.args[0]}\" not found!")


    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert base_model == BaseModelType.StableDiffusion1_5
        assert model_type == ModelType.Pipeline
        super().__init__(model_path, base_model, model_type)

    @classmethod
    def convert_if_required(cls, model_path: str, dst_path: str, config: Optional[dict]) -> str:
        model_config = cls.Config(
            **config,
            path=model_path,
            name="",
        )

        if hasattr(model_config, "config"):
            convert_ckpt_and_cache(
                model_path=model_path,
                dst_path=dst_path,
                config=config,
            )
            return dst_path

        else:
            return model_path

class StableDiffusion15CheckpointModel(DiffusersModel):
    class Cnfig(ModelConfigBase):
        vae: Optional[str] = Field(None)
        config: Optional[str] = Field(None)

class StableDiffusion2BaseDiffusersModel(DiffusersModel):
    class Config(ModelConfigBase):
        vae: Optional[str] = Field(None)

class StableDiffusion2BaseCheckpointModel(DiffusersModel):
    class Cnfig(ModelConfigBase):
        vae: Optional[str] = Field(None)
        config: Optional[str] = Field(None)

class StableDiffusion2DiffusersModel(DiffusersModel):
    class Config(ModelConfigBase):
        vae: Optional[str] = Field(None)
        attention_upscale: bool = Field(True)

class StableDiffusion2CheckpointModel(DiffusersModel):
    class Config(ModelConfigBase):
        vae: Optional[str] = Field(None)
        config: Optional[str] = Field(None)
        attention_upscale: bool = Field(True)


class ClassifierModel(ModelBase):
    #child_types: Dict[str, Type]
    #child_sizes: Dict[str, int]

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert model_type == SDModelType.Classifier
        super().__init__(model_path, base_model, model_type)

        self.child_types: Dict[str, Type] = dict()
        self.child_sizes: Dict[str, int] = dict()

        try:
            main_config = EmptyConfigLoader.load_config(self.model_path, config_name="config.json")
            #main_config = json.loads(os.path.join(self.model_path, "config.json"))
        except:
            raise Exception("Invalid classifier model! (config.json not found or invalid)")

        self._load_tokenizer(main_config)
        self._load_text_encoder(main_config)
        self._load_feature_extractor(main_config)


    def _load_tokenizer(self, main_config: dict):
        try:
            tokenizer_config = EmptyConfigLoader.load_config(self.model_path, config_name="tokenizer_config.json")
            #tokenizer_config = json.loads(os.path.join(self.model_path, "tokenizer_config.json"))
        except:
            raise Exception("Invalid classifier model! (Failed to load tokenizer_config.json)")

        if "tokenizer_class" in tokenizer_config:
            tokenizer_class_name = tokenizer_config["tokenizer_class"]
        elif "model_type" in main_config:
            tokenizer_class_name = transformers.models.auto.tokenization_auto.TOKENIZER_MAPPING_NAMES[main_config["model_type"]]
        else:
            raise Exception("Invalid classifier model! (Failed to detect tokenizer type)")

        self.child_types[SDModelType.Tokenizer] = self._hf_definition_to_type(["transformers", tokenizer_class_name])
        self.child_sizes[SDModelType.Tokenizer] = 0


    def _load_text_encoder(self, main_config: dict):
        if "architectures" in main_config and len(main_config["architectures"]) > 0:
            text_encoder_class_name = main_config["architectures"][0]
        elif "model_type" in main_config:
            text_encoder_class_name = transformers.models.auto.modeling_auto.MODEL_FOR_PRETRAINING_MAPPING_NAMES[main_config["model_type"]]
        else:
            raise Exception("Invalid classifier model! (Failed to detect text_encoder type)")

        self.child_types[SDModelType.TextEncoder] = self._hf_definition_to_type(["transformers", text_encoder_class_name])
        self.child_sizes[SDModelType.TextEncoder] = calc_model_size_by_fs(self.model_path)


    def _load_feature_extractor(self, main_config: dict):
        self.child_sizes[SDModelType.FeatureExtractor] = 0
        try:
            feature_extractor_config = EmptyConfigLoader.load_config(self.model_path, config_name="preprocessor_config.json")
        except:
            return # feature extractor not passed with t5

        try:
            feature_extractor_class_name = feature_extractor_config["feature_extractor_type"]
            self.child_types[SDModelType.FeatureExtractor] = self._hf_definition_to_type(["transformers", feature_extractor_class_name])
        except:
            raise Exception("Invalid classifier model! (Unknown feature_extrator type)")


    def get_size(self, child_type: Optional[SDModelType] = None):
        if child_type is None:
            return sum(self.child_sizes.values())
        else:
            return self.child_sizes[child_type]


    def get_model(
        self,
        torch_dtype: Optional[torch.dtype],
        child_type: Optional[SDModelType] = None,
    ):
        if child_type is None:
            raise Exception("Child model type can't be null on classififer model")
        if child_type not in self.child_types:
            return None # TODO: or raise
        
        model = self.child_types[child_type].from_pretrained(
            self.model_path,
            subfolder=child_type.value,
            torch_dtype=torch_dtype,
        )
        # calc more accurate size
        self.child_sizes[child_type] = calc_model_size_by_data(model)
        return model

    @staticmethod
    def convert_if_required(model_path: Union[str, Path], cache_path: str, config: Optional[dict]) -> Path:
        if not isinstance(model_path, Path):
            model_path = Path(model_path)
        return model_path



class VaeModel(ModelBase):
    #vae_class: Type
    #model_size: int

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert model_type == ModelType.Vae
        super().__init__(model_path, base_model, model_type)

        try:
            config = EmptyConfigLoader.load_config(self.model_path, config_name="config.json")
            #config = json.loads(os.path.join(self.model_path, "config.json"))
        except:
            raise Exception("Invalid vae model! (config.json not found or invalid)")

        try:
            vae_class_name = config.get("_class_name", "AutoencoderKL")
            self.vae_class = self._hf_definition_to_type(["diffusers", vae_class_name])
            self.model_size = calc_model_size_by_fs(self.model_path)
        except:
            raise Exception("Invalid vae model! (Unkown vae type)")

    def get_size(self, child_type: Optional[SDModelType] = None):
        if child_type is not None:
            raise Exception("There is no child models in vae model")
        return self.model_size

    def get_model(
        self,
        torch_dtype: Optional[torch.dtype],
        child_type: Optional[SDModelType] = None,
    ):
        if child_type is not None:
            raise Exception("There is no child models in vae model")

        model = self.vae_class.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
        )
        # calc more accurate size
        self.model_size = calc_model_size_by_data(model)
        return model

    @staticmethod
    def convert_if_required(model_path: Union[str, Path], cache_path: str, config: Optional[dict]) -> Path:
        if not isinstance(model_path, Path):
            model_path = Path(model_path)
        # TODO:
        #_convert_vae_ckpt_and_cache
        raise Exception("TODO: ")


class LoRAModel(ModelBase):
    #model_size: int

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert model_type == ModelType.Lora
        super().__init__(model_path, base_model, model_type)

        self.model_size = os.path.getsize(self.model_path)

    def get_size(self, child_type: Optional[SDModelType] = None):
        if child_type is not None:
            raise Exception("There is no child models in lora")
        return self.model_size

    def get_model(
        self,
        torch_dtype: Optional[torch.dtype],
        child_type: Optional[SDModelType] = None,
    ):
        if child_type is not None:
            raise Exception("There is no child models in lora")

        model = LoRAModel.from_checkpoint(
            file_path=self.model_path,
            dtype=torch_dtype,
        )

        self.model_size = model.calc_size()
        return model

    @staticmethod
    def convert_if_required(model_path: Union[str, Path], cache_path: str, config: Optional[dict]) -> Path:
        if not isinstance(model_path, Path):
            model_path = Path(model_path)

        # TODO: add diffusers lora when it stabilizes a bit
        return model_path


class TextualInversionModel(ModelBase):
    #model_size: int

    def __init__(self, model_path: str, base_model: BaseModelType, model_type: ModelType):
        assert model_type == ModelType.TextualInversion
        super().__init__(model_path, base_model, model_type)

        self.model_size = os.path.getsize(self.model_path)

    def get_size(self, child_type: Optional[SDModelType] = None):
        if child_type is not None:
            raise Exception("There is no child models in textual inversion")
        return self.model_size

    def get_model(
        self,
        torch_dtype: Optional[torch.dtype],
        child_type: Optional[SDModelType] = None,
    ):
        if child_type is not None:
            raise Exception("There is no child models in textual inversion")

        model = TextualInversionModel.from_checkpoint(
            file_path=self.model_path,
            dtype=torch_dtype,
        )

        self.model_size = model.embedding.nelement() * model.embedding.element_size()
        return model

    @staticmethod
    def convert_if_required(model_path: Union[str, Path], cache_path: str, config: Optional[dict]) -> Path:
        if not isinstance(model_path, Path):
            model_path = Path(model_path)
        return model_path









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


def _convert_ckpt_and_cache(self, mconfig: DictConfig) -> Path:
    """
    Convert the checkpoint model indicated in mconfig into a
    diffusers, cache it to disk, and return Path to converted
    file. If already on disk then just returns Path.
    """
    app_config = InvokeAIAppConfig.get_config()
    weights = app_config.root_dir / mconfig.path
    config_file = app_config.root_dir / mconfig.config
    diffusers_path = app_config.converted_ckpts_dir / weights.stem

    # return cached version if it exists
    if diffusers_path.exists():
        return diffusers_path

    # TODO: I think that it more correctly to convert with embedded vae
    #       as if user will delete custom vae he will got not embedded but also custom vae
    #vae_ckpt_path, vae_model = self._get_vae_for_conversion(weights, mconfig)
    vae_ckpt_path, vae_model = None, None

    # to avoid circular import errors
    from .convert_ckpt_to_diffusers import convert_ckpt_to_diffusers
    with SilenceWarnings():        
        convert_ckpt_to_diffusers(
            weights,
            diffusers_path,
            extract_ema=True,
            original_config_file=config_file,
            vae=vae_model,
            vae_path=str(app_config.root_dir / vae_ckpt_path) if vae_ckpt_path else None,
            scan_needed=True,
        )
    return diffusers_path

def _convert_vae_ckpt_and_cache(self, mconfig: DictConfig) -> Path:
    """
    Convert the VAE indicated in mconfig into a diffusers AutoencoderKL
    object, cache it to disk, and return Path to converted
    file. If already on disk then just returns Path.
    """
    app_config = InvokeAIAppConfig.get_config()
    root = app_config.root_dir
    weights_file = root / mconfig.path
    config_file = root / mconfig.config
    diffusers_path = app_config.converted_ckpts_dir / weights_file.stem
    image_size = mconfig.get('width') or mconfig.get('height') or 512
        
    # return cached version if it exists
    if diffusers_path.exists():
        return diffusers_path

    # this avoids circular import error
    from .convert_ckpt_to_diffusers import convert_ldm_vae_to_diffusers
    if weights_file.suffix == '.safetensors':
        checkpoint = safetensors.torch.load_file(weights_file)
    else:
        checkpoint = torch.load(weights_file, map_location="cpu")

    # sometimes weights are hidden under "state_dict", and sometimes not
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    config = OmegaConf.load(config_file)

    vae_model = convert_ldm_vae_to_diffusers(
        checkpoint = checkpoint,
        vae_config = config,
        image_size = image_size
    )
    vae_model.save_pretrained(
        diffusers_path,
        safe_serialization=is_safetensors_available()
    )
    return diffusers_path
