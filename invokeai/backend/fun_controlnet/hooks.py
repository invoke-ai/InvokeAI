import inspect
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from invokeai.backend.fun_controlnet.branch import FunControl, FunControlStage

HintAdder = Callable[[Any, torch.Tensor, float], Any]
"""``FunControl.add_hint``: a base block's output with a scaled hint added to its image stream."""


def should_apply(step_index: int, total_steps: int, begin_step_percent: float, end_step_percent: float) -> bool:
    """Whether the control applies at a step: ``begin <= step_index / total_steps <= end``, both ends included."""
    if total_steps <= 0:
        return True
    return begin_step_percent <= step_index / total_steps <= end_step_percent


class FunControlHooks:
    """The hooks a control model is installed with. ``active`` gates the forwards that follow without re-installing.

    Each stage's entry block runs that stage's chain in a pre-hook, on the block's real arguments and the control stream
    the stage before it carried; the post-hook of each of the stage's hint blocks adds its hint, and the last one drops
    them. The first stage's entry opens every forward and drops whatever an earlier one left behind, so no hint or
    control stream of one forward reaches the next, and none stays alive past the block that consumes it.
    """

    def __init__(self, stages: Sequence[FunControlStage], add_hint: HintAdder, scale: float) -> None:
        self.active = True
        self._stages = tuple(stages)
        self._add_hint = add_hint
        self._scale = scale
        self._hints: list[list[torch.Tensor] | None] = [None] * len(self._stages)
        # What the stage that ran last hands to the one after it, which pops it: held here, and only here, until then.
        self._carry: list[torch.Tensor] = []
        self._handles: list[RemovableHandle] = []
        for index, stage in enumerate(self._stages):
            # Bound once: the base calls its blocks by keyword or by position, and the family reads the arguments by
            # name either way.
            signature = inspect.signature(stage.entry.forward)
            self._handles.append(
                stage.entry.register_forward_pre_hook(self._stage_runner(index, signature), with_kwargs=True)
            )
            last = len(stage.hint_blocks) - 1
            for position, block in enumerate(stage.hint_blocks):
                self._handles.append(
                    block.register_forward_hook(self._hint_adder(index, position, clear=position == last))
                )

    def _stage_runner(
        self, index: int, signature: inspect.Signature
    ) -> Callable[[nn.Module, tuple[Any, ...], dict[str, Any]], None]:
        def run_stage(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if index == 0:
                # The first stage's entry runs first in every forward: whatever a previous forward left is stale here.
                self._hints = [None] * len(self._stages)
                self._carry.clear()
            if not self.active:
                return None
            if index > 0 and not self._carry:
                raise RuntimeError(
                    f"Control stage {index} was entered without the control stream of stage {index - 1}: the forward "
                    "did not run the block that starts the control."
                )
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            hints, handed_on = self._stages[index].run(bound.arguments, self._carry)
            self._hints[index] = hints
            self._carry.clear()
            # The last stage has no one to hand its stream to; holding it would keep it alive into the next forward.
            if handed_on is not None and index + 1 < len(self._stages):
                self._carry.append(handed_on)
            return None

        return run_stage

    def _hint_adder(self, stage: int, position: int, clear: bool) -> Callable[[nn.Module, tuple[Any, ...], Any], Any]:
        def add_hint(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
            hints = self._hints[stage]
            if hints is None:
                return None
            hint = hints[position]
            if clear:
                self._hints[stage] = None
            return self._add_hint(output, hint, self._scale)

        return add_hint

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._hints = [None] * len(self._stages)
        self._carry.clear()


@contextmanager
def install_fun_control(
    transformer: nn.Module, control: FunControl, control_context: torch.Tensor, *, scale: float
) -> Iterator[FunControlHooks]:
    """Run ``transformer`` with ``control``'s hints for the duration of the context.

    The hints are recomputed on every forward (each denoising step, each guidance branch) from the control context,
    which is the same for both guidance branches. The hooks come off on exit however the body ends: the blocks belong to
    the model cache and outlive this call.
    """
    base_hidden_size = control.base_hidden_size(transformer)
    if base_hidden_size != control.hidden_size:
        raise ValueError(
            f"The control branch is {control.hidden_size} wide and the transformer {base_hidden_size}: "
            "this is not a control model for this transformer."
        )
    if control_context.dim() != 3 or control_context.shape[-1] != control.control_in_dim:
        raise ValueError(
            f"The control context must be (batch, tokens, {control.control_in_dim}), got {tuple(control_context.shape)}."
        )
    hooks = FunControlHooks(control.stages(transformer, control_context), control.add_hint, scale)
    try:
        yield hooks
    finally:
        hooks.remove()
