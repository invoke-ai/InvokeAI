"""
Test model loading
"""

from pathlib import Path

from invokeai.app.services.model_install import ModelInstallServiceBase
from invokeai.backend.embeddings.textual_inversion import TextualInversionModelRaw
from invokeai.backend.model_manager.load import AnyModelLoader
from tests.backend.model_manager_2.model_manager_2_fixtures import *  # noqa F403


def test_loading(mm2_installer: ModelInstallServiceBase, mm2_loader: AnyModelLoader, embedding_file: Path):
    store = mm2_installer.record_store
    matches = store.search_by_attr(model_name="test_embedding")
    assert len(matches) == 0
    key = mm2_installer.register_path(embedding_file)
    loaded_model = mm2_loader.load_model(store.get_model(key))
    assert loaded_model is not None
    assert loaded_model.config.key == key
    with loaded_model as model:
        assert isinstance(model, TextualInversionModelRaw)
