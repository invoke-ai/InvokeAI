import logging

import torch

from invokeai.backend.image_util.triposplat.triposplat_model import TripoSplatModel
from invokeai.backend.model_manager.load.model_util import calc_model_size_by_data


class _FakePipe:
    """Stands in for TripoSplatPipeline: just the five submodule attributes the wrapper reads."""

    def __init__(self) -> None:
        module = torch.nn.Linear(4, 4)
        self.dinov3 = module
        self.vae_encoder = module
        self.rmbg = module
        self.flow_model = module
        self.decoder = module


def test_calc_model_size_by_data_dispatches_to_triposplat_calc_size():
    # TripoSplatModel is not an nn.Module — if it is missing from calc_model_size_by_data's isinstance
    # dispatch it silently sizes as 0 and the model cache cannot make room for the multi-GB pipeline.
    model = TripoSplatModel(_FakePipe())
    size = calc_model_size_by_data(logging.getLogger(__name__), model)
    assert size == model.calc_size()
    assert size > 0
