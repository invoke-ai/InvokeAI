import torch


class LoRALinearSidecarLayer(torch.nn.Module):
    """An implementation of a linear LoRA layer based on the paper 'LoRA: Low-Rank Adaptation of Large Language Models'.
    (https://arxiv.org/pdf/2106.09685.pdf)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        include_mid: bool,
        rank: int,
        alpha: float,
        weight: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize a LoRALinearLayer.
        Args:
            in_features (int): Inputs to this layer will be expected to have shape (..., in_features).
            out_features (int): This layer will produce outputs with shape (..., out_features).
            rank (int, optional): The internal rank of the layer. See the paper for details.
            alpha (float, optional): A scaling factor that enables tuning the rank without having to adjust the learning
                rate. The recommendation from the paper is to set alpha equal to the first rank that you try and then do
                not tune it further. See the paper for more details.
            device (torch.device, optional): Device where weights will be initialized.
            dtype (torch.dtype, optional): Weight dtype.
        Raises:
            ValueError: If the rank is greater than either in_features or out_features.
        """
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less than or equal to {min(in_features, out_features)}")

        self._down = torch.nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self._up = torch.nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self._mid = None
        if include_mid:
            self._mid = torch.nn.Linear(rank, rank, bias=False, device=device, dtype=dtype)

        # Register alpha as a buffer so that it is not trained, but still gets saved to the state_dict.
        self.register_buffer("alpha", torch.tensor(alpha, device=device, dtype=dtype))

        self._weight = weight
        self._rank = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._down(x)
        if self._mid is not None:
            x = self._mid(x)
        x = self._up(x)

        x *= self._weight * self.alpha / self._rank
        return x
