from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

BlockRunner = Callable[[nn.Module, torch.Tensor], torch.Tensor]
"""Runs one control block on the control image stream and returns the stream it produced. Whatever else the block
takes and updates (its own text stream, modulation, RoPE) the family closes over."""


class FunControlBranch(nn.Module, ABC):
    """``control_img_in`` and ``control_blocks``, with the base layers their hints land on.

    ``control_layers[k]`` is the index of the base block after whose output ``control_blocks[k]``'s hint is added.
    Block 0 carries ``before_proj`` and every block ``after_proj``, both ``Linear(hidden, hidden)``; the family's block
    class adds them to the base block it subclasses, and keeps that block's ``forward``.

    The abstract methods are the family seams the hooks go through: which base blocks to hook, how to read the base
    image stream from block 0's arguments, how to call a control block with them, and where in a block's output the hint
    goes.
    """

    def __init__(
        self, control_img_in: nn.Linear, control_blocks: Sequence[nn.Module], control_layers: Sequence[int]
    ) -> None:
        super().__init__()
        layers = tuple(control_layers)
        if len(control_blocks) != len(layers) or not layers:
            raise ValueError(f"{len(control_blocks)} control blocks for base layers {layers}: one block per layer.")
        # The hooks drop the hints once the last listed layer has taken its share, which needs the layers in order.
        if layers[0] < 0 or any(later <= earlier for earlier, later in zip(layers, layers[1:], strict=False)):
            raise ValueError(f"Control layers must be non-negative and strictly increasing, got {layers}.")
        self.control_img_in = control_img_in
        self.control_blocks = nn.ModuleList(control_blocks)
        self.control_layers = layers

    @property
    def hidden_size(self) -> int:
        return self.control_img_in.out_features

    @property
    def control_in_dim(self) -> int:
        return self.control_img_in.in_features

    @property
    def dtype(self) -> torch.dtype:
        return self.control_img_in.weight.dtype

    def compute_hints(
        self, control_context: torch.Tensor, x: torch.Tensor, run_block: BlockRunner
    ) -> list[torch.Tensor]:
        """One hint per control block, from the control context and the base image stream ``x`` as it enters base
        block 0. The chain runs in the branch's dtype; ``add_hint`` casts back."""
        c = self.control_img_in(control_context.to(self.dtype))
        hints: list[torch.Tensor] = []
        for index, block in enumerate(self.control_blocks):
            if index == 0:
                c = block.before_proj(c) + x.to(c.dtype)
            c = run_block(block, c)
            hints.append(block.after_proj(c))
        return hints

    @abstractmethod
    def base_blocks(self, transformer: nn.Module) -> Sequence[nn.Module]:
        """The base transformer's blocks, indexed as ``control_layers`` are."""

    @abstractmethod
    def base_hidden_size(self, transformer: nn.Module) -> int:
        """The width of the base image stream the hints are added to."""

    @abstractmethod
    def image_stream(self, block_inputs: Mapping[str, Any]) -> torch.Tensor:
        """The base image stream among base block 0's bound forward arguments."""

    @abstractmethod
    def block_runner(self, block_inputs: Mapping[str, Any]) -> BlockRunner:
        """A runner that calls a control block the way the base called its block 0, threading the branch's own copy of
        whatever else the block updates from call to call."""

    @abstractmethod
    def add_hint(self, output: Any, hint: torch.Tensor, scale: float) -> Any:
        """A base block's ``output`` with ``hint * scale`` added to its image stream and nothing else touched."""
