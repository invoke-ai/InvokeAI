import torch

from invokeai.backend.model_cache_v2.torch_autocast_context import TorchAutocastContext


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def test_torch_autocast_context():
    model = DummyModule()

    with TorchAutocastContext(to_device=torch.device("cuda")):
        x = torch.randn(10, 10, device="cuda")
        y = model(x)
        print(y.shape)
