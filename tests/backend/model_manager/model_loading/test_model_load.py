"""
Test model loading
"""

from pathlib import Path

from invokeai.app.services.model_manager import ModelManagerServiceBase
from invokeai.backend.textual_inversion import TextualInversionModelRaw
from tests.backend.model_manager.model_manager_fixtures import *  # noqa F403


def test_loading(mm2_model_manager: ModelManagerServiceBase, embedding_file: Path):
    store = mm2_model_manager.store
    matches = store.search_by_attr(model_name="test_embedding")
    assert len(matches) == 0
    key = mm2_model_manager.install.register_path(embedding_file)
    loaded_model = mm2_model_manager.load.load_model(store.get_model(key))
    assert loaded_model is not None
    assert loaded_model.config.key == key
    with loaded_model as model:
        assert isinstance(model, TextualInversionModelRaw)

    config = mm2_model_manager.store.get_model(key)
    loaded_model_2 = mm2_model_manager.load.load_model(config)

    assert loaded_model.config.key == loaded_model_2.config.key
