"""Utility (backend) functions used by model_install.py"""
import re
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional

import omegaconf
from huggingface_hub import HfFolder
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from pydantic.networks import AnyHttpUrl
from requests import HTTPError
from tqdm import tqdm

import invokeai.configs as configs
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.download import DownloadQueueService
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.model_install import (
    HFModelSource,
    LocalModelSource,
    ModelInstallService,
    ModelInstallServiceBase,
    ModelSource,
    URLModelSource,
)
from invokeai.app.services.model_records import ModelRecordServiceBase, ModelRecordServiceSQL
from invokeai.app.services.shared.sqlite.sqlite_util import init_db
from invokeai.backend.model_manager import (
    BaseModelType,
    InvalidModelConfigException,
    ModelRepoVariant,
    ModelType,
)
from invokeai.backend.model_manager.metadata import UnknownMetadataException
from invokeai.backend.util.logging import InvokeAILogger

# name of the starter models file
INITIAL_MODELS = "INITIAL_MODELS.yaml"


def initialize_record_store(app_config: InvokeAIAppConfig) -> ModelRecordServiceBase:
    """Return an initialized ModelConfigRecordServiceBase object."""
    logger = InvokeAILogger.get_logger(config=app_config)
    image_files = DiskImageFileStorage(f"{app_config.output_path}/images")
    db = init_db(config=app_config, logger=logger, image_files=image_files)
    obj: ModelRecordServiceBase = ModelRecordServiceSQL(db)
    return obj


def initialize_installer(
    app_config: InvokeAIAppConfig, event_bus: Optional[EventServiceBase] = None
) -> ModelInstallServiceBase:
    """Return an initialized ModelInstallService object."""
    record_store = initialize_record_store(app_config)
    metadata_store = record_store.metadata_store
    download_queue = DownloadQueueService()
    installer = ModelInstallService(
        app_config=app_config,
        record_store=record_store,
        metadata_store=metadata_store,
        download_queue=download_queue,
        event_bus=event_bus,
    )
    download_queue.start()
    installer.start()
    return installer


class UnifiedModelInfo(BaseModel):
    """Catchall class for information in INITIAL_MODELS2.yaml."""

    name: Optional[str] = None
    base: Optional[BaseModelType] = None
    type: Optional[ModelType] = None
    source: Optional[str] = None
    subfolder: Optional[str] = None
    description: Optional[str] = None
    recommended: bool = False
    installed: bool = False
    default: bool = False
    requires: List[str] = Field(default_factory=list)


@dataclass
class InstallSelections:
    """Lists of models to install and remove."""

    install_models: List[UnifiedModelInfo] = Field(default_factory=list)
    remove_models: List[str] = Field(default_factory=list)


class TqdmEventService(EventServiceBase):
    """An event service to track downloads."""

    def __init__(self) -> None:
        """Create a new TqdmEventService object."""
        super().__init__()
        self._bars: Dict[str, tqdm] = {}
        self._last: Dict[str, int] = {}
        self._logger = InvokeAILogger.get_logger(__name__)

    def dispatch(self, event_name: str, payload: Any) -> None:
        """Dispatch an event by appending it to self.events."""
        data = payload["data"]
        source = data["source"]
        if payload["event"] == "model_install_downloading":
            dest = data["local_path"]
            total_bytes = data["total_bytes"]
            bytes = data["bytes"]
            if dest not in self._bars:
                self._bars[dest] = tqdm(desc=Path(dest).name, initial=0, total=total_bytes, unit="iB", unit_scale=True)
                self._last[dest] = 0
            self._bars[dest].update(bytes - self._last[dest])
            self._last[dest] = bytes
        elif payload["event"] == "model_install_completed":
            self._logger.info(f"{source}: installed successfully.")
        elif payload["event"] == "model_install_error":
            self._logger.warning(f"{source}: installation failed with error {data['error']}")
        elif payload["event"] == "model_install_cancelled":
            self._logger.warning(f"{source}: installation cancelled")


class InstallHelper(object):
    """Capture information stored jointly in INITIAL_MODELS.yaml and the installed models db."""

    def __init__(self, app_config: InvokeAIAppConfig, logger: Logger):
        """Create new InstallHelper object."""
        self._app_config = app_config
        self.all_models: Dict[str, UnifiedModelInfo] = {}

        omega = omegaconf.OmegaConf.load(Path(configs.__path__[0]) / INITIAL_MODELS)
        assert isinstance(omega, omegaconf.dictconfig.DictConfig)

        self._installer = initialize_installer(app_config, TqdmEventService())
        self._initial_models = omega
        self._installed_models: List[str] = []
        self._starter_models: List[str] = []
        self._default_model: Optional[str] = None
        self._logger = logger
        self._initialize_model_lists()

    @property
    def installer(self) -> ModelInstallServiceBase:
        """Return the installer object used internally."""
        return self._installer

    def _initialize_model_lists(self) -> None:
        """
        Initialize our model slots.

        Set up the following:
        installed_models  -- list of installed model keys
        starter_models -- list of starter model keys from INITIAL_MODELS
        all_models -- dict of key => UnifiedModelInfo
        default_model -- key to default model
        """
        # previously-installed models
        for model in self._installer.record_store.all_models():
            info = UnifiedModelInfo.parse_obj(model.dict())
            info.installed = True
            model_key = f"{model.base.value}/{model.type.value}/{model.name}"
            self.all_models[model_key] = info
            self._installed_models.append(model_key)

        for key in self._initial_models.keys():
            assert isinstance(key, str)
            if key in self.all_models:
                # we want to preserve the description
                description = self.all_models[key].description or self._initial_models[key].get("description")
                self.all_models[key].description = description
            else:
                base_model, model_type, model_name = key.split("/")
                info = UnifiedModelInfo(
                    name=model_name,
                    type=ModelType(model_type),
                    base=BaseModelType(base_model),
                    source=self._initial_models[key].source,
                    description=self._initial_models[key].get("description"),
                    recommended=self._initial_models[key].get("recommended", False),
                    default=self._initial_models[key].get("default", False),
                    subfolder=self._initial_models[key].get("subfolder"),
                    requires=list(self._initial_models[key].get("requires", [])),
                )
                self.all_models[key] = info
            if not self.default_model():
                self._default_model = key
            elif self._initial_models[key].get("default", False):
                self._default_model = key
            self._starter_models.append(key)

        # previously-installed models
        for model in self._installer.record_store.all_models():
            info = UnifiedModelInfo.parse_obj(model.dict())
            info.installed = True
            model_key = f"{model.base.value}/{model.type.value}/{model.name}"
            self.all_models[model_key] = info
            self._installed_models.append(model_key)

    def recommended_models(self) -> List[UnifiedModelInfo]:
        """List of the models recommended in INITIAL_MODELS.yaml."""
        return [self._to_model(x) for x in self._starter_models if self._to_model(x).recommended]

    def installed_models(self) -> List[UnifiedModelInfo]:
        """List of models already installed."""
        return [self._to_model(x) for x in self._installed_models]

    def starter_models(self) -> List[UnifiedModelInfo]:
        """List of starter models."""
        return [self._to_model(x) for x in self._starter_models]

    def default_model(self) -> Optional[UnifiedModelInfo]:
        """Return the default model."""
        return self._to_model(self._default_model) if self._default_model else None

    def _to_model(self, key: str) -> UnifiedModelInfo:
        return self.all_models[key]

    def _add_required_models(self, model_list: List[UnifiedModelInfo]) -> None:
        installed = {x.source for x in self.installed_models()}
        reverse_source = {x.source: x for x in self.all_models.values()}
        additional_models: List[UnifiedModelInfo] = []
        for model_info in model_list:
            for requirement in model_info.requires:
                if requirement not in installed and reverse_source.get(requirement):
                    additional_models.append(reverse_source[requirement])
        model_list.extend(additional_models)

    def _make_install_source(self, model_info: UnifiedModelInfo) -> ModelSource:
        assert model_info.source
        model_path_id_or_url = model_info.source.strip("\"' ")
        model_path = Path(model_path_id_or_url)

        if model_path.exists():  # local file on disk
            return LocalModelSource(path=model_path.absolute(), inplace=True)

        # parsing huggingface repo ids
        # we're going to do a little trick that allows for extended repo_ids of form "foo/bar:fp16"
        variants = "|".join([x.lower() for x in ModelRepoVariant.__members__])
        if match := re.match(f"^([^/]+/[^/]+?)(?::({variants}))?$", model_path_id_or_url):
            repo_id = match.group(1)
            repo_variant = ModelRepoVariant(match.group(2)) if match.group(2) else None
            subfolder = Path(model_info.subfolder) if model_info.subfolder else None
            return HFModelSource(
                repo_id=repo_id,
                access_token=HfFolder.get_token(),
                subfolder=subfolder,
                variant=repo_variant,
            )
        if re.match(r"^(http|https):", model_path_id_or_url):
            return URLModelSource(url=AnyHttpUrl(model_path_id_or_url))
        raise ValueError(f"Unsupported model source: {model_path_id_or_url}")

    def add_or_delete(self, selections: InstallSelections) -> None:
        """Add or delete selected models."""
        installer = self._installer
        self._add_required_models(selections.install_models)
        for model in selections.install_models:
            source = self._make_install_source(model)
            config = (
                {
                    "description": model.description,
                    "name": model.name,
                }
                if model.name
                else None
            )

            try:
                installer.import_model(
                    source=source,
                    config=config,
                )
            except (UnknownMetadataException, InvalidModelConfigException, HTTPError, OSError) as e:
                self._logger.warning(f"{source}: {e}")

        for model_to_remove in selections.remove_models:
            parts = model_to_remove.split("/")
            if len(parts) == 1:
                base_model, model_type, model_name = (None, None, model_to_remove)
            else:
                base_model, model_type, model_name = parts
            matches = installer.record_store.search_by_attr(
                base_model=BaseModelType(base_model) if base_model else None,
                model_type=ModelType(model_type) if model_type else None,
                model_name=model_name,
            )
            if len(matches) > 1:
                print(
                    f"{model_to_remove} is ambiguous. Please use model_base/model_type/model_name (e.g. sd-1/main/my_model) to disambiguate."
                )
            elif not matches:
                print(f"{model_to_remove}: unknown model")
            else:
                for m in matches:
                    print(f"Deleting {m.type}:{m.name}")
                    installer.delete(m.key)

        installer.wait_for_installs()
