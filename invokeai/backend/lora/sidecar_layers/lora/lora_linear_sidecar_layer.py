import torch

from invokeai.backend.lora.layers.lora_layer import LoRALayer


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

    @classmethod
    def from_layers(cls, orig_layer: torch.nn.Module, lora_layer: LoRALayer, weight: float):
        # Initialize the LoRA layer.
        with torch.device("meta"):
            model = cls.from_orig_layer(
                orig_layer,
                include_mid=lora_layer.mid is not None,
                rank=lora_layer.rank,
                # TODO(ryand): Is this the right default in case of missing alpha?
                alpha=lora_layer.alpha if lora_layer.alpha is not None else lora_layer.rank,
                weight=weight,
            )

        # TODO(ryand): Are there cases where we need to reshape the weight matrices to match the conv layers?

        # Inject weight into the LoRA layer.
        model._up.weight.data = lora_layer.up
        model._down.weight.data = lora_layer.down
        if lora_layer.mid is not None:
            assert model._mid is not None
            model._mid.weight.data = lora_layer.mid

        return model

    @classmethod
    def from_orig_layer(
        cls,
        layer: torch.nn.Module,
        include_mid: bool,
        rank: int,
        alpha: float,
        weight: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        if not isinstance(layer, torch.nn.Linear):
            raise TypeError(f"'{__class__.__name__}' cannot be initialized from a layer of type '{type(layer)}'.")

        return cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            include_mid=include_mid,
            rank=rank,
            alpha=alpha,
            weight=weight,
            device=layer.weight.device if device is None else device,
            dtype=layer.weight.dtype if dtype is None else dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._down(x)
        if self._mid is not None:
            x = self._mid(x)
        x = self._up(x)

        x *= self._weight * self.alpha / self._rank
        return x
