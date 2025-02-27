import pytest
import torch


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 32)
        self.linear2 = torch.nn.Linear(32, 64)
        self.register_buffer("buffer1", torch.ones(64))
        # Non-persistent buffers are not included in the state dict. We need to make sure that this case is handled
        # correctly by the partial loading code.
        self.register_buffer("buffer2", torch.ones(64), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.linear2(x)
        x = x + self.buffer1
        x = x + self.buffer2
        return x


parameterize_mps_and_cuda = pytest.mark.parametrize(
    ("device"),
    [
        pytest.param(
            "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available.")
        ),
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")),
    ],
)

parameterize_keep_ram_copy = pytest.mark.parametrize("keep_ram_copy", [True, False])
