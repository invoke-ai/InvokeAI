import typing

import torch

from invokeai.backend.lora.layers.lora_layer import LoRALayer


class LoRAConvSidecarLayer(torch.nn.Module):
    """An implementation of a conv LoRA layer based on the paper 'LoRA: Low-Rank Adaptation of Large Language Models'.
    (https://arxiv.org/pdf/2106.09685.pdf)
    """

    @property
    def conv_module(self) -> type[torch.nn.Conv1d | torch.nn.Conv2d | torch.nn.Conv3d]:
        """The conv module to be set by child classes. One of torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d."""
        raise NotImplementedError(
            "LoRAConvLayer cannot be used directly. Use LoRAConv1dLayer, LoRAConv2dLayer, or LoRAConv3dLayer instead."
        )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        include_mid: bool,
        rank: int,
        alpha: float,
        weight: float,
        kernel_size: typing.Union[int, tuple[int]] = 1,
        stride: typing.Union[int, tuple[int]] = 1,
        padding: typing.Union[str, int, tuple[int]] = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize a LoRAConvLayer.
        Args:
            in_channels (int): The number of channels expected on inputs to this layer.
            out_channels (int): The number of channels on outputs from this layer.
            kernel_size: The kernel_size of the conv layer that this LoRA layer is mirroring. See torch.nn.Conv* docs.
            stride: The stride of the conv layer that this LoRA layer is mirroring. See torch.nn.Conv* docs.
            padding: The padding of the conv layer that this LoRA layer is mirroring. See torch.nn.Conv* docs.
            rank (int, optional): The internal rank of the layer. See the paper for details.
            alpha (float, optional): A scaling factor that enables tuning the rank without having to adjust the learning
                rate. The recommendation from the paper is to set alpha equal to the first rank that you try and then do
                not tune it further. See the paper for more details.
            device (torch.device, optional): Device where weights will be initialized.
            dtype (torch.dtype, optional): Weight dtype.
        Raises:
            ValueError: If the rank is greater than either in_channels or out_channels.
        """
        super().__init__()

        if rank > min(in_channels, out_channels):
            raise ValueError(f"LoRA rank {rank} must be less than or equal to {min(in_channels, out_channels)}")

        self._down = self.conv_module(
            in_channels,
            rank,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self._up = self.conv_module(rank, out_channels, kernel_size=1, stride=1, bias=False, device=device, dtype=dtype)
        self._mid = None
        if include_mid:
            self._mid = self.conv_module(rank, rank, kernel_size=1, stride=1, bias=False, device=device, dtype=dtype)

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
        if not isinstance(layer, cls.conv_module):
            raise TypeError(f"'{__class__.__name__}' cannot be initialized from a layer of type '{type(layer)}'.")

        return cls(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            include_mid=include_mid,
            weight=weight,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            rank=rank,
            alpha=alpha,
            device=layer.weight.device if device is None else device,
            dtype=layer.weight.dtype if dtype is None else dtype,
        )

    def forward(self, x: torch.Tensor):
        x = self._down(x)
        if self._mid is not None:
            x = self._mid(x)
        x = self._up(x)

        x *= self._weight * self.alpha / self._rank
        return x


class LoRAConv1dSidecarLayer(LoRAConvSidecarLayer):
    @property
    def conv_module(self):
        return torch.nn.Conv1d


class LoRAConv2dSidecarLayer(LoRAConvSidecarLayer):
    @property
    def conv_module(self):
        return torch.nn.Conv2d


class LoRAConv3dSidecarLayer(LoRAConvSidecarLayer):
    @property
    def conv_module(self):
        return torch.nn.Conv3d
