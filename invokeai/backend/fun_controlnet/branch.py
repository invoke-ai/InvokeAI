from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

BlockRunner = Callable[[nn.Module, torch.Tensor], torch.Tensor]
"""Runs one control block on the control image stream and returns the stream it produced. Whatever else the block
takes and updates (its own text stream, modulation, RoPE) the family closes over."""

StageRunner = Callable[[Mapping[str, Any], list[torch.Tensor]], tuple[list[torch.Tensor], torch.Tensor | None]]
"""Runs a stage's chain from its entry block's bound forward arguments and ``carry``, which holds the control stream
the previous stage handed on (empty for the first stage). The stage pops the stream it continues from, so nothing else
keeps it alive while the chain runs. Returns one hint per hint block and the control stream for the next stage."""


@dataclass(frozen=True)
class FunControlStage:
    """One place in the base forward where a control chain runs.

    ``entry`` is the base block whose pre-hook runs ``run`` on that block's bound arguments; the stage's hint ``k`` is
    added to the output of ``hint_blocks[k]``. A family with one chain has one stage; a family whose chain is split
    across the base's own stages (a refiner ahead of the main blocks) has several, which one forward reaches in order,
    each handing its control stream to the next.
    """

    entry: nn.Module
    hint_blocks: tuple[nn.Module, ...]
    run: StageRunner


def run_vace_chain(
    make_input: Callable[[], torch.Tensor],
    x: torch.Tensor,
    blocks: Sequence[nn.Module],
    run_block: BlockRunner,
    *,
    num_hints: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """The VACE recipe every Fun control chain follows: block 0 adds ``before_proj(c)`` to the base stream ``x``, every
    block runs on the control stream ``c``, and the first ``num_hints`` blocks' ``after_proj`` outputs are the hints.

    The chain builds its input ``c`` with ``make_input`` and holds the only reference to it, so each stream is freed as
    soon as the next one replaces it. An input the caller had built and kept would stay alive, an image-length hidden
    state, through every block. Returns the hints and the control stream after the last block. The chain runs in
    ``c``'s dtype; ``x`` is cast to it.
    """
    c = make_input()
    hints: list[torch.Tensor] = []
    for index, block in enumerate(blocks):
        if index == 0:
            c = block.before_proj(c) + x.to(c.dtype)
        c = run_block(block, c)
        if index < num_hints:
            hints.append(block.after_proj(c))
    return hints, c


def hint_blocks(base_blocks: Sequence[nn.Module], layers: Sequence[int]) -> tuple[nn.Module, ...]:
    """The base blocks at ``layers``, refused when the transformer is too shallow for them."""
    layers = tuple(layers)
    if len(base_blocks) <= layers[-1]:
        raise ValueError(
            f"The control branch adds hints after base layers {layers}; this transformer has {len(base_blocks)} blocks."
        )
    return tuple(base_blocks[layer] for layer in layers)


class FunControl(nn.Module, ABC):
    """A Fun control model as the hooks see it: its width and context features, the stages it runs inside a base
    transformer's forward, and how a hint goes onto a base block's output."""

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """The width of the hints."""

    @property
    @abstractmethod
    def control_in_dim(self) -> int:
        """The features per token of the control context."""

    @abstractmethod
    def base_hidden_size(self, transformer: nn.Module) -> int:
        """The width of the base image stream the hints are added to."""

    @abstractmethod
    def stages(self, transformer: nn.Module, control_context: torch.Tensor) -> Sequence[FunControlStage]:
        """The stages to hook into ``transformer``, in the order one forward reaches their entry blocks. Refuses a
        transformer whose blocks the hints cannot land on."""

    @abstractmethod
    def add_hint(self, output: Any, hint: torch.Tensor, scale: float) -> Any:
        """A base block's ``output`` with ``hint * scale`` added to its image stream and nothing else touched."""


class FunControlBranch(FunControl):
    """``control_img_in`` and ``control_blocks``, with the base layers their hints land on: one chain, entered at base
    block 0.

    ``control_layers[k]`` is the index of the base block after whose output ``control_blocks[k]``'s hint is added.
    Block 0 carries ``before_proj`` and every block ``after_proj``, both ``Linear(hidden, hidden)``; the family's block
    class adds them to the base block it subclasses, and keeps that block's ``forward``.

    The abstract methods are the family seams: which base blocks to hook, how to read the base image stream from block
    0's arguments, how to call a control block with them, and where in a block's output the hint goes.
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
        hints, _ = run_vace_chain(
            lambda: self.control_img_in(control_context.to(self.dtype)),
            x,
            self.control_blocks,
            run_block,
            num_hints=len(self.control_blocks),
        )
        return hints

    def stages(self, transformer: nn.Module, control_context: torch.Tensor) -> Sequence[FunControlStage]:
        blocks = self.base_blocks(transformer)
        targets = hint_blocks(blocks, self.control_layers)

        def run(inputs: Mapping[str, Any], _carry: list[torch.Tensor]) -> tuple[list[torch.Tensor], None]:
            return self.compute_hints(control_context, self.image_stream(inputs), self.block_runner(inputs)), None

        return (FunControlStage(entry=blocks[0], hint_blocks=targets, run=run),)

    @abstractmethod
    def base_blocks(self, transformer: nn.Module) -> Sequence[nn.Module]:
        """The base transformer's blocks, indexed as ``control_layers`` are."""

    @abstractmethod
    def image_stream(self, block_inputs: Mapping[str, Any]) -> torch.Tensor:
        """The base image stream among base block 0's bound forward arguments."""

    @abstractmethod
    def block_runner(self, block_inputs: Mapping[str, Any]) -> BlockRunner:
        """A runner that calls a control block the way the base called its block 0, threading the branch's own copy of
        whatever else the block updates from call to call."""
