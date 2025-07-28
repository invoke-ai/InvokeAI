# invokeai/app/services/model_manager/model_manager_dummy.py

from typing import Any, Optional

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.download.download_base import DownloadQueueServiceBase
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_install.model_install_base import ModelInstallServiceBase
from invokeai.app.services.model_load.model_load_base import ModelLoadServiceBase
from invokeai.app.services.model_manager.model_manager_base import ModelManagerServiceBase
from invokeai.app.services.model_records.model_records_base import ModelRecordServiceBase
from typing_extensions import Self


class DummyModelRecordService(ModelRecordServiceBase):
    def add_model(self, config: Any) -> Any:
        pass

    def del_model(self, key: str) -> None:
        pass

    def update_model(self, key: str, changes: Any) -> Any:
        pass

    def get_model(self, key: str) -> Any:
        return None

    def get_model_by_hash(self, hash: str) -> Any:
        return None

    def list_models(self, page: int = 0, per_page: int = 10, order_by: Any = None) -> Any:
        return None

    def search_by_attr(self, *args, **kwargs) -> list[Any]:
        return []

    def exists(self, key: str) -> bool:
        return False

    def search_by_path(self, path: Any) -> list[Any]:
        return []

    def search_by_hash(self, hash: str) -> list[Any]:
        return []


class DummyModelInstallService(ModelInstallServiceBase):
    def __init__(self, app_config, record_store, download_queue, event_bus=None):
        pass

    def start(self, invoker: Any = None) -> None:
        pass

    def stop(self, invoker: Any = None) -> None:
        pass

    @property
    def app_config(self) -> Any:
        return None

    @property
    def record_store(self) -> Any:
        return None

    @property
    def event_bus(self) -> Any:
        return None

    def register_path(self, model_path: Any, config: Any = None) -> str:
        return ""

    def unregister(self, key: str) -> None:
        pass

    def delete(self, key: str) -> None:
        pass

    def unconditionally_delete(self, key: str) -> None:
        pass

    def install_path(self, model_path: Any, config: Any = None) -> str:
        return ""

    def heuristic_import(self, *args, **kwargs) -> Any:
        return None

    def import_model(self, source: Any, config: Any = None) -> Any:
        return None

    def get_job_by_source(self, source: Any) -> list[Any]:
        return []

    def list_jobs(self) -> list[Any]:
        return []

    def get_job_by_id(self, id: int) -> Any:
        return None

    def prune_jobs(self) -> None:
        pass

    def cancel_job(self, job: Any) -> None:
        pass

    def wait_for_job(self, job: Any, timeout: int = 0) -> Any:
        return None

    def wait_for_installs(self, timeout: int = 0) -> list[Any]:
        return []

    def sync_model_path(self, key: str) -> Any:
        return None

    def download_and_cache_model(self, source: Any) -> Any:
        return None


class DummyModelLoadService(ModelLoadServiceBase):
    def load_model(self, model_config: Any, submodel_type: Any = None) -> Any:
        return None

    def load_model_from_path(self, model_path: Any, loader: Any = None) -> Any:
        return None

    @property
    def ram_cache(self) -> Any:
        class DummyCache:
            @property
            def stats(self) -> Any:
                class DummyStats:
                    def __getattr__(self, name):
                        return 0

                return DummyStats()

            def make_room(self, size: int) -> None:
                pass

        return DummyCache()


class DummyDownloadQueue(DownloadQueueServiceBase):
    def start(self, *args, **kwargs) -> None: pass
    def stop(self, *args, **kwargs) -> None: pass
    def download(self, source: Any, dest: Any, priority: int = 10, access_token: Optional[str] = None, on_start: Any = None, on_progress: Any = None, on_complete: Any = None, on_cancelled: Any = None, on_error: Any = None) -> Any: pass
    def multifile_download(self, parts: Any, dest: Any, access_token: Optional[str] = None, submit_job: bool = True, on_start: Any = None, on_progress: Any = None, on_complete: Any = None, on_cancelled: Any = None, on_error: Any = None) -> Any: pass
    def submit_multifile_download(self, job: Any) -> None: pass
    def submit_download_job(self, job: Any, on_start: Any = None, on_progress: Any = None, on_complete: Any = None, on_cancelled: Any = None, on_error: Any = None) -> None: pass
    def list_jobs(self) -> list[Any]: return []
    def id_to_job(self, id: int) -> Any: pass
    def cancel_all_jobs(self) -> None: pass
    def prune_jobs(self) -> None: pass
    def cancel_job(self, job: Any) -> None: pass
    def join(self) -> None: pass
    def wait_for_job(self, job: Any, timeout: int = 0) -> Any: pass

class DummyModelManagerService(ModelManagerServiceBase):
    def __init__(self):
        class DummyAppConfig(InvokeAIAppConfig):
            def __init__(self):
                pass

        self._store = DummyModelRecordService()
        self._install = DummyModelInstallService(DummyAppConfig(), self._store, DummyDownloadQueue())
        self._load = DummyModelLoadService()

    @classmethod
    def build_model_manager(
        cls,
        app_config: InvokeAIAppConfig,
        model_record_service: ModelRecordServiceBase,
        download_queue: DownloadQueueServiceBase,
        events: EventServiceBase,
        execution_device: Any,
    ) -> Self:
        return cls()

    @property
    def store(self) -> ModelRecordServiceBase:
        return self._store

    @property
    def install(self) -> ModelInstallServiceBase:
        return self._install

    @property
    def load(self) -> ModelLoadServiceBase:
        return self._load

    def start(self, invoker: Invoker) -> None:
        pass

    def stop(self, invoker: Invoker) -> None:
        pass
