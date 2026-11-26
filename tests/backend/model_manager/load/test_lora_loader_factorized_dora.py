from logging import Logger
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

import invokeai.backend.model_manager.load.model_loaders.lora as lora_loader_module
from invokeai.backend.model_manager.load.model_loaders.lora import LoRALoader
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat
from invokeai.backend.patches.layers.dora_layer import DoRALayer, FactorizedDoRALayer
from invokeai.backend.patches.layers.loha_layer import LoHALayer
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer


def test_loader_keeps_the_magnitudes_of_weight_decomposed_lokr_and_loha_layers(tmp_path, monkeypatch) -> None:
    state_dict = {
        "lokr.lokr_w1": torch.randn(2, 2),
        "lokr.lokr_w2": torch.randn(3, 3),
        "lokr.dora_scale": torch.ones(6, 1),
        "loha.hada_w1_a": torch.randn(6, 2),
        "loha.hada_w1_b": torch.randn(2, 6),
        "loha.hada_w2_a": torch.randn(6, 2),
        "loha.hada_w2_b": torch.randn(2, 6),
        "loha.dora_magnitude": torch.ones(6, 1),
        "dora.lora_down.weight": torch.randn(2, 6),
        "dora.lora_up.weight": torch.randn(6, 2),
        "dora.dora_scale": torch.ones(6, 1),
    }
    monkeypatch.setattr(lora_loader_module, "load_file", lambda *_args, **_kwargs: state_dict)
    loader = LoRALoader.__new__(LoRALoader)
    loader._logger = MagicMock(spec=Logger)
    loader._model_base = BaseModelType.StableDiffusion1
    loader._torch_dtype = torch.float32
    config = SimpleNamespace(
        path=tmp_path / "factorized-dora.safetensors",
        name="factorized-dora",
        base=BaseModelType.StableDiffusion1,
        format=ModelFormat.LyCORIS,
    )

    model = loader._load_model(config)  # type: ignore[arg-type]

    lokr, loha = model.layers["lokr"], model.layers["loha"]
    assert isinstance(lokr, FactorizedDoRALayer) and isinstance(lokr.layer, LoKRLayer)
    assert isinstance(loha, FactorizedDoRALayer) and isinstance(loha.layer, LoHALayer)
    assert isinstance(model.layers["dora"], DoRALayer)
    assert lokr.magnitude_is_out_dim and loha.magnitude_is_out_dim
    assert torch.equal(lokr.dora_scale, state_dict["lokr.dora_scale"])
    assert torch.equal(loha.dora_scale, state_dict["loha.dora_magnitude"])
