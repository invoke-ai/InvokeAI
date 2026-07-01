import torch

# This model definition is based on:
# https://github.com/black-forest-labs/flux/blob/716724eb276d94397be99710a0a54d352664e23b/src/flux/modules/image_embedders.py#L66


class FluxReduxModel(torch.nn.Module):
    def __init__(self, redux_dim: int = 1152, txt_in_features: int = 4096) -> None:
        super().__init__()

        self.redux_dim = redux_dim

        self.redux_up = torch.nn.Linear(redux_dim, txt_in_features * 3)
        self.redux_down = torch.nn.Linear(txt_in_features * 3, txt_in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.redux_down(torch.nn.functional.silu(self.redux_up(x)))
