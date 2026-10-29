import torch

from invokeai.backend.patches.layers.loha_layer import LoHALayer


def test_conv_factors_are_flattened_before_multiplication() -> None:
    torch.manual_seed(0)
    w1_a = torch.randn(8, 2, 1, 1)
    w1_b = torch.randn(2, 4, 3, 3)
    w2_a = torch.randn(8, 2, 1, 1)
    w2_b = torch.randn(2, 4, 3, 3)
    layer = LoHALayer(
        w1_a=w1_a,
        w1_b=w1_b,
        w2_a=w2_a,
        w2_b=w2_b,
        t1=None,
        t2=None,
        alpha=2.0,
        bias=None,
    )

    parameters = layer.get_parameters({"weight": torch.zeros(8, 4, 3, 3)}, weight=1.0)

    expected = ((w1_a.reshape(8, 2) @ w1_b.reshape(2, 36)) * (w2_a.reshape(8, 2) @ w2_b.reshape(2, 36))).reshape(
        8, 4, 3, 3
    )
    torch.testing.assert_close(parameters["weight"], expected)
