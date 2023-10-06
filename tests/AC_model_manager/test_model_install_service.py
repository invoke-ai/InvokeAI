import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from invokeai.app.services.config.invokeai_config import InvokeAIAppConfig
from invokeai.app.services.events import EventServiceBase
from invokeai.app.services.model_install_service import ModelInstallService
from invokeai.app.services.model_loader_service import ModelLoadService
from invokeai.app.services.model_record_service import ModelRecordServiceBase
from invokeai.backend.model_manager import BaseModelType, ModelType

# This is a very little embedding model that we can use to test installation
TEST_MODEL = "test_embedding.safetensors"


class DummyEvent(BaseModel):
    """Dummy Event to use with Dummy Event service."""

    event_name: str
    status: str


class DummyEventService(EventServiceBase):
    """Dummy event service for testing."""

    events: list

    def __init__(self):
        super().__init__()
        self.events = list()

    def dispatch(self, event_name: str, payload: Any) -> None:
        """Dispatch an event by appending it to self.events."""
        self.events.append(DummyEvent(event_name=event_name, status=payload["job"].status))


def test_install(datadir: Path):
    """Test installation of an itty-bitty embedding."""
    # create a temporary root directory for install to target
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "models").mkdir()
        (tmp_path / "configs").mkdir()
        config = InvokeAIAppConfig(
            root=tmp_path,
            model_config_db=tmp_path / "configs" / "models.yaml",
            models_dir=tmp_path / "models",
        )

        event_bus = DummyEventService()
        mm_store = ModelRecordServiceBase.get_impl(config)
        mm_load = ModelLoadService(config, mm_store)
        mm_install = ModelInstallService(config=config, store=mm_store, event_bus=event_bus)

        source = datadir / TEST_MODEL
        mm_install.install_model(source=source)
        id_map = mm_install.wait_for_installs()
        print(id_map)
        assert source in id_map, "model did not install; id_map empty"
        assert id_map[source] is not None, "model did not install: source field empty"

        # test the events
        assert len(event_bus.events) > 0, "no events received"
        assert len(event_bus.events) == 3

        event_names = set([x.event_name for x in event_bus.events])
        assert "model_event" in event_names
        event_payloads = set([x.status for x in event_bus.events])
        assert "enqueued" in event_payloads
        assert "running" in event_payloads
        assert "completed" in event_payloads

        key = id_map[source]
        model = mm_store.get_model(key)  # may raise an exception here
        assert Path(config.models_path / model.path).exists(), "generated path incorrect"
        assert model.base_model == BaseModelType.StableDiffusion1, "probe of model base type didn't work"
        assert model.model_type == ModelType.TextualInversion, "probe of model type didn't work"

        model_info = mm_load.get_model(key)
        assert model_info, "model did not load"
        with model_info as model:
            assert model is not None, "model context not working"
