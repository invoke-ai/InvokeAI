import torch
from torch import nn
def replace_linear_with_lora(
    module: nn.Module,
    max_rank: int,
    scale: float = 1.0,
) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            new_lora = LinearLora(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                rank=max_rank,
                scale=scale,
                dtype=getattr(child, "compute_dtype") if hasattr(child, "compute_dtype") else child.weight.dtype,
                device=child.weight.device,
            )
            new_lora.weight = child.weight
            new_lora.bias = child.bias if child.bias is not None else None
            setattr(module, name, new_lora)
        else:
            replace_linear_with_lora(
                module=child,
                max_rank=max_rank,
                scale=scale,
            )

class LinearLora(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
        lora_bias: bool = True,
        scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias is not None,
            device=device,
            dtype=dtype,
            *args,
            **kwargs,
        )
        assert isinstance(scale, float), "scale must be a float"
        self.scale = scale
        self.rank = rank
        self.lora_bias = lora_bias
        self.dtype = dtype
        self.device = device
        if rank > (new_rank := min(self.out_features, self.in_features)):
            self.rank = new_rank
        self.lora_A = nn.Linear(
            in_features=in_features,
            out_features=self.rank,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.lora_B = nn.Linear(
            in_features=self.rank,
            out_features=out_features,
            bias=self.lora_bias,
            dtype=dtype,
            device=device,
        )
    def set_scale(self, scale: float) -> None:
        assert isinstance(scale, float), "scalar value must be a float"
        self.scale = scale
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base_out = super().forward(input)
        _lora_out_B = self.lora_B(self.lora_A(input))
        lora_update = _lora_out_B * self.scale
        return base_out + lora_update