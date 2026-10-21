"""The FLUX/SD3 T5 GGUF loader (``T5EncoderGGUFModel``).

Its llama.cpp key mapping, config inference and feed-forward patch are shared with the Wan UMT5
loader and tested in ``tests/backend/model_manager/util/test_llamacpp_t5.py``.
"""

from unittest.mock import MagicMock

import pytest

from invokeai.backend.model_manager.configs.t5_encoder import T5Encoder_GGUF_Config
from invokeai.backend.model_manager.load.model_loaders.flux import T5EncoderGGUFModel
from invokeai.backend.model_manager.taxonomy import SubModelType
from tests.backend.model_manager.load.umt5_gguf_fixture import write_tiny_umt5_gguf


def test_a_umt5_gguf_recorded_as_a_flux_t5_encoder_is_refused_with_the_way_out(tmp_path) -> None:
    """Before identification knew UMT5, city96's UMT5 GGUFs installed as FLUX/SD3 T5 encoders. Loaded
    as T5 v1.1, such a file runs every block on block 0's relative attention bias and encodes garbage
    without an error, so the old record has to be refused, pointing at the Model Manager action that
    turns it into a Wan T5 encoder."""
    path = tmp_path / "umt5-xxl-encoder-Q8_0.gguf"
    write_tiny_umt5_gguf(path)
    loader = object.__new__(T5EncoderGGUFModel)
    loader._ram_cache = MagicMock()
    config = T5Encoder_GGUF_Config.model_construct(path=str(path), name="umt5-xxl-encoder-Q8_0")

    with pytest.raises(ValueError, match="UMT5.*Re-identify model"):
        loader._load_model(config, SubModelType.TextEncoder2)
