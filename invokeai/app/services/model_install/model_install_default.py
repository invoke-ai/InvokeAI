from pathlib import Path
from typing import Dict, Optional, Union
from pydantic import BaseModel, Field
from queue import Queue
from pydantic.networks import AnyHttpUrl

from .model_install_base import InstallStatus, ModelInstallStatus, ModelInstallServiceBase
from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_records import ModelRecordServiceBase
from invokeai.app.services.events import EventServiceBase

class ModelInstallService(ModelInstallBase, BaseModel):
    """class for InvokeAI model installation."""

    config: InvokeAIAppConfig
    store: ModelRecordServiceBase
    event_bus: Optional[EventServiceBase] = Field(default=None)
    install_queue: Queue = Field(default_factory=Queue)

    def register_path(
            self,
            model_path: Union[Path, str],
            name: Optional[str] = None,
            description: Optional[str] = None,
            metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        raise NotImplementedError

    def install_path(
            self,
            model_path: Union[Path, str],
            name: Optional[str] = None,
            description: Optional[str] = None,
            metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        raise NotImplementedError

    def install_model(
            self,
            source: Union[str, Path, AnyHttpUrl],
            inplace: bool = True,
            name: Optional[str] = None,
            description: Optional[str] = None,
            variant: Optional[str] = None,
            subfolder: Optional[str] = None,
            metadata: Optional[Dict[str, str]] = None,
            access_token: Optional[str] = None,
    ) -> ModelInstallStatus:
        raise NotImplementedError

    def wait_for_installs(self) -> Dict[Union[str, Path, AnyHttpUrl], Optional[str]]:
        raise NotImplementedError

    def scan_directory(self, scan_dir: Path, install: bool = False) -> List[str]:
        raise NotImplementedError

    def sync_to_config(self):
        raise NotImplementedError

    def hash(self, model_path: Union[Path, str]) -> str:
        raise NotImplementedError

    # The following are internal methods
    def _create_name(self, model_path: Union[Path, str]) -> str:
        model_path = Path(model_path)
        if model_path.suffix in {'.safetensors', '.bin', '.pt', '.ckpt'}:
            return model_path.stem
        else:
            return model_path.name

    def _create_description(self, model_path: Union[Path, str]) -> str:
        info: ModelProbeInfo = ModelProbe.probe(Path(model_path))
        name: str = name or self._create_name(model_path)
        return f"a {info.model_type} model based on {info.base_type}"

    def _create_id(self, model_path: Union[Path, str], name: Optional[str] = None) -> str:
        name: str = name or self._create_name(model_path)
        raise NotImplementedError
