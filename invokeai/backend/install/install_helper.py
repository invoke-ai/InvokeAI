"""
Utility (backend) functions used by model_install.py
"""
from pathlib import Path
from typing import Dict, List, Optional

import omegaconf
from huggingface_hub import HfFolder
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from tqdm import tqdm

import invokeai.configs as configs
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.model_manager import BaseModelType, ModelInstall, ModelInstallJob, ModelType
from invokeai.backend.model_manager.install import ModelSourceMetadata

# name of the starter models file
INITIAL_MODELS = "INITIAL_MODELS.yaml"
ACCESS_TOKEN = HfFolder.get_token()


class UnifiedModelInfo(BaseModel):
    name: Optional[str] = None
    base_model: Optional[BaseModelType] = None
    model_type: Optional[ModelType] = None
    source: Optional[str] = None
    description: Optional[str] = None
    recommended: bool = False
    installed: bool = False
    default: bool = False


@dataclass
class InstallSelections:
    install_models: List[UnifiedModelInfo] = Field(default_factory=list)
    remove_models: List[str] = Field(default_factory=list)


class TqdmProgress(object):
    _bars: Dict[int, tqdm]  # the tqdm object
    _last: Dict[int, int]  # last bytes downloaded

    def __init__(self):
        self._bars = dict()
        self._last = dict()

    def job_update(self, job: ModelInstallJob):
        job_id = job.id
        if job.status == "running":
            if job_id not in self._bars:
                dest = Path(job.destination).name
                self._bars[job_id] = tqdm(
                    desc=dest,
                    initial=0,
                    total=job.total_bytes,
                    unit="iB",
                    unit_scale=True,
                )
                self._last[job_id] = 0
            self._bars[job_id].update(job.bytes - self._last[job_id])
            self._last[job_id] = job.bytes


class InstallHelper(object):
    """Capture information stored jointly in INITIAL_MODELS.yaml and the installed models db."""

    all_models: Dict[str, UnifiedModelInfo] = dict()
    _installer: ModelInstall
    _config: InvokeAIAppConfig
    _installed_models: List[str] = []
    _starter_models: List[str] = []
    _default_model: Optional[str] = None
    _initial_models: omegaconf.DictConfig

    def __init__(self, config: InvokeAIAppConfig):
        self._config = config
        self._installer = ModelInstall(config=config, event_handlers=[TqdmProgress().job_update])
        self._initial_models = omegaconf.OmegaConf.load(Path(configs.__path__[0]) / INITIAL_MODELS)
        self._initialize_model_lists()

    @property
    def installer(self) -> ModelInstall:
        return self._installer

    def _initialize_model_lists(self):
        """
        Initialize our model slots.

        Set up the following:
        installed_models  -- list of installed model keys
        starter_models -- list of starter model keys from INITIAL_MODELS
        all_models -- dict of key => UnifiedModelInfo
        default_model -- key to default model
        """
        # previously-installed models
        for model in self._installer.store.all_models():
            info = UnifiedModelInfo.parse_obj(model.dict())
            info.installed = True
            key = f"{model.base_model.value}/{model.model_type.value}/{model.name}"
            self.all_models[key] = info
            self._installed_models.append(key)

        for key in self._initial_models.keys():
            if key in self.all_models:
                # we want to preserve the description
                description = self.all_models[key].description or self._initial_models[key].get("description")
                self.all_models[key].description = description
            else:
                base_model, model_type, model_name = key.split("/")
                info = UnifiedModelInfo(
                    name=model_name,
                    model_type=model_type,
                    base_model=base_model,
                    source=self._initial_models[key].source,
                    description=self._initial_models[key].get("description"),
                    recommended=self._initial_models[key].get("recommended", False),
                    default=self._initial_models[key].get("default", False),
                )
                self.all_models[key] = info
            if not self.default_model:
                self._default_model = key
            elif self._initial_models[key].get("default", False):
                self._default_model = key
            self._starter_models.append(key)

        # previously-installed models
        for model in self._installer.store.all_models():
            info = UnifiedModelInfo.parse_obj(model.dict())
            info.installed = True
            key = f"{model.base_model.value}/{model.model_type.value}/{model.name}"
            self.all_models[key] = info
            self._installed_models.append(key)

    def recommended_models(self) -> List[UnifiedModelInfo]:
        return [self._to_model(x) for x in self._starter_models if self._to_model(x).recommended]

    def installed_models(self) -> List[UnifiedModelInfo]:
        return [self._to_model(x) for x in self._installed_models]

    def starter_models(self) -> List[UnifiedModelInfo]:
        return [self._to_model(x) for x in self._starter_models]

    def default_model(self) -> UnifiedModelInfo:
        return self._to_model(self._default_model)

    def _to_model(self, key: str) -> UnifiedModelInfo:
        return self.all_models[key]

    def add_or_delete(self, selections: InstallSelections):
        installer = self._installer
        for model in selections.install_models:
            metadata = ModelSourceMetadata(description=model.description, name=model.name)
            installer.install(
                model.source,
                variant="fp16" if self._config.precision == "float16" else None,
                access_token=ACCESS_TOKEN,  # this is a global,
                metadata=metadata,
            )

        for model in selections.remove_models:
            parts = model.split("/")
            if len(parts) == 1:
                base_model, model_type, model_name = (None, None, model)
            else:
                base_model, model_type, model_name = parts
            matches = installer.store.search_by_name(
                base_model=base_model, model_type=model_type, model_name=model_name
            )
            if len(matches) > 1:
                print(f"{model} is ambiguous. Please use model_type:model_name (e.g. main:my_model) to disambiguate.")
            elif not matches:
                print(f"{model}: unknown model")
            else:
                for m in matches:
                    print(f"Deleting {m.model_type}:{m.name}")
                    installer.conditionally_delete(m.key)

        installer.wait_for_installs()
