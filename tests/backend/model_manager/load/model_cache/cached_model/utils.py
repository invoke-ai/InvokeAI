import os

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


class TiedWeightsModule(torch.nn.Module):
    """A language model's shape: the output projection shares its weight with the embedding.

    Every Qwen3 text encoder InvokeAI loads is tied this way, and so are many transformers models. The state dict
    lists that one tensor under both names, which is what the cache has to account for and move as one.
    """

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(128, 64)
        self.head = torch.nn.Linear(64, 128, bias=False)
        self.head.weight = self.embed.weight
        self.linear = torch.nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.linear(self.embed(x)))


class TiedRequiredParameterModule(torch.nn.Module):
    """A tie that spans an autocast-wrapped module and a parameter the cache must keep on the compute device.

    Root parameters sit in no wrapped module, so the cache counts them among the weights a partially-loaded model
    cannot run without. Tying one to a wrapped module's weight is what makes "keep the required weights" and "move a
    tied group as a unit" pull in opposite directions.
    """

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 128, bias=False)
        self.required = torch.nn.Parameter(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.required.sum()


is_github_ci = os.getenv("GITHUB_ACTIONS") == "true"

parameterize_mps_and_cuda = pytest.mark.parametrize(
    ("device"),
    [
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                is_github_ci or not torch.backends.mps.is_available(),
                reason="MPS is very flaky in CI" if is_github_ci else "MPS is not available.",
            ),
        ),
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")),
    ],
)

parameterize_keep_ram_copy = pytest.mark.parametrize("keep_ram_copy", [True, False])
