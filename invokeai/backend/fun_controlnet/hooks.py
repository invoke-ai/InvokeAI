import inspect
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from invokeai.backend.fun_controlnet.branch import FunControlBranch


def should_apply(step_index: int, total_steps: int, begin_step_percent: float, end_step_percent: float) -> bool:
    """Whether the control applies at a step: ``begin <= step_index / total_steps <= end``, both ends included, as the
    Z-Image control extension counts."""
    if total_steps <= 0:
        return True
    return begin_step_percent <= step_index / total_steps <= end_step_percent


class FunControlHooks:
    """The hooks a branch is installed with. ``active`` gates the forwards that follow without re-installing.

    Base block 0's pre-hook computes the hints from that forward's real arguments; the post-hook of each listed base
    block adds its hint and the last one drops them, so no hint of one forward reaches the next and nothing stays alive
    through the layers that come after the last listed one.
    """

    def __init__(
        self, branch: FunControlBranch, blocks: Sequence[nn.Module], control_context: torch.Tensor, scale: float
    ) -> None:
        self.active = True
        self._branch = branch
        self._control_context = control_context
        self._scale = scale
        self._hints: list[torch.Tensor] | None = None
        # Bound once: the base calls its blocks by keyword, a checkpointing caller by position, and the family reads
        # the arguments by name either way.
        self._signature = inspect.signature(blocks[0].forward)
        last_layer = branch.control_layers[-1]
        self._handles: list[RemovableHandle] = [
            blocks[0].register_forward_pre_hook(self._compute_hints, with_kwargs=True)
        ]
        for position, layer in enumerate(branch.control_layers):
            self._handles.append(
                blocks[layer].register_forward_hook(self._hint_adder(position, clear=layer == last_layer))
            )

    def _compute_hints(self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        # Block 0 runs first in every forward: whatever a previous forward left behind is stale here.
        self._hints = None
        if not self.active:
            return None
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = bound.arguments
        self._hints = self._branch.compute_hints(
            self._control_context, self._branch.image_stream(inputs), self._branch.block_runner(inputs)
        )
        return None

    def _hint_adder(self, position: int, clear: bool) -> Callable[[nn.Module, tuple[Any, ...], Any], Any]:
        def add_hint(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
            if self._hints is None:
                return None
            hint = self._hints[position]
            if clear:
                self._hints = None
            return self._branch.add_hint(output, hint, self._scale)

        return add_hint

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._hints = None


@contextmanager
def install_fun_control(
    transformer: nn.Module, branch: FunControlBranch, control_context: torch.Tensor, *, scale: float
) -> Iterator[FunControlHooks]:
    """Run ``transformer`` with ``branch``'s hints for the duration of the context.

    The hints are recomputed on every forward (each denoising step, each guidance branch) from the control context,
    which is the same for both guidance branches. The hooks come off on exit however the body ends: the blocks belong to
    the model cache and outlive this call.
    """
    blocks = branch.base_blocks(transformer)
    base_hidden_size = branch.base_hidden_size(transformer)
    if base_hidden_size != branch.hidden_size:
        raise ValueError(
            f"The control branch is {branch.hidden_size} wide and the transformer {base_hidden_size}: "
            "this is not a control model for this transformer."
        )
    if len(blocks) <= branch.control_layers[-1]:
        raise ValueError(
            f"The control branch adds hints after base layers {branch.control_layers}; "
            f"this transformer has {len(blocks)} blocks."
        )
    if control_context.dim() != 3 or control_context.shape[-1] != branch.control_in_dim:
        raise ValueError(
            f"The control context must be (batch, tokens, {branch.control_in_dim}), got {tuple(control_context.shape)}."
        )
    hooks = FunControlHooks(branch, blocks, control_context, scale)
    try:
        yield hooks
    finally:
        hooks.remove()
