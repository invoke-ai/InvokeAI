"""
Test model loading
"""

from pathlib import Path

from invokeai.app.services.model_install import ModelInstallServiceBase
from invokeai.app.services.model_load import ModelLoadServiceBase
from invokeai.backend.textual_inversion import TextualInversionModelRaw
from tests.backend.model_manager.model_manager_fixtures import *  # noqa F403

def test_loading(mm2_installer: ModelInstallServiceBase, mm2_loader: ModelLoadServiceBase, embedding_file: Path):
    store = mm2_installer.record_store
    matches = store.search_by_attr(model_name="test_embedding")
    assert len(matches) == 0
    key = mm2_installer.register_path(embedding_file)
    loaded_model = mm2_loader.load_model_by_config(store.get_model(key))
    assert loaded_model is not None
    assert loaded_model.config.key == key
    with loaded_model as model:
        assert isinstance(model, TextualInversionModelRaw)
