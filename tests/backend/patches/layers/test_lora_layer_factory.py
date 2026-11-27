import pytest
import torch

from invokeai.backend.patches.layers.loha_layer import LoHALayer
from invokeai.backend.patches.layers.lokr_layer import LoKRLayer
from invokeai.backend.patches.layers.utils import any_lora_layer_from_state_dict


@pytest.mark.parametrize("magnitude_key", ["dora_scale", "dora_magnitude"])
def test_lokr_with_dora_magnitude_falls_back_to_plain_lokr(magnitude_key: str) -> None:
    values = {
        "lokr_w1": torch.randn(2, 2),
        "lokr_w2": torch.randn(3, 3),
        magnitude_key: torch.ones(6, 1),
        "alpha": torch.tensor(2.0),
    }

    layer = any_lora_layer_from_state_dict(values)

    assert isinstance(layer, LoKRLayer)
    torch.testing.assert_close(layer.get_weight(torch.empty(6, 6)), torch.kron(values["lokr_w1"], values["lokr_w2"]))


@pytest.mark.parametrize("magnitude_key", ["dora_scale", "dora_magnitude"])
def test_loha_with_dora_magnitude_falls_back_to_plain_loha(magnitude_key: str) -> None:
    values = {
        "hada_w1_a": torch.randn(6, 2),
        "hada_w1_b": torch.randn(2, 6),
        "hada_w2_a": torch.randn(6, 2),
        "hada_w2_b": torch.randn(2, 6),
        magnitude_key: torch.ones(6, 1),
        "alpha": torch.tensor(2.0),
    }

    layer = any_lora_layer_from_state_dict(values)

    assert isinstance(layer, LoHALayer)
    expected = (values["hada_w1_a"] @ values["hada_w1_b"]) * (values["hada_w2_a"] @ values["hada_w2_b"])
    torch.testing.assert_close(layer.get_weight(torch.empty(6, 6)), expected)
