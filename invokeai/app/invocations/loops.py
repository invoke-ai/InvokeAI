import copy
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, Field

from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import Input, InputField, OutputField, OutputScope, UIType
from invokeai.app.services.shared.invocation_context import InvocationContext

T = TypeVar("T")

LOOP_LINKAGE_FIELD = "loop_linkage"


def _copy_value(value: T) -> T:
    if isinstance(value, BaseModel):
        return value.model_copy(deep=True)
    return copy.deepcopy(value)


class LoopState(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)


@invocation_output("loop_state_output")
class LoopStateOutput(BaseInvocationOutput):
    state: LoopState = OutputField(description="The loop state")


@invocation_output("loop_state_value_output")
class LoopStateValueOutput(BaseInvocationOutput):
    value: Optional[Any] = OutputField(
        default=None,
        description="The value read from the loop state, or the configured default when the key is missing",
        ui_type=UIType.Any,
    )


@invocation("state_empty", title="Empty Loop State", tags=["loop", "state"], category="workflow", version="1.0.0")
class StateEmptyInvocation(BaseInvocation):
    """Creates an empty loop state."""

    def invoke(self, context: InvocationContext) -> LoopStateOutput:
        return LoopStateOutput(state=LoopState())


@invocation("state_get", title="Get Loop State Value", tags=["loop", "state"], category="workflow", version="1.0.3")
class StateGetInvocation(BaseInvocation):
    """Reads a value from loop state."""

    state: LoopState = InputField(description="The loop state to read")
    key: str = InputField(default="", description="The state key to read")
    default: Any = InputField(
        default=None,
        description="The value to return when the key is missing",
        ui_type=UIType.Any,
    )

    def invoke(self, context: InvocationContext) -> LoopStateValueOutput:
        return LoopStateValueOutput(value=_copy_value(self.state.values.get(self.key, self.default)))


@invocation("state_set", title="Set Loop State Value", tags=["loop", "state"], category="workflow", version="1.0.1")
class StateSetInvocation(BaseInvocation):
    """Returns loop state with one value set."""

    state: Optional[LoopState] = InputField(default=None, description="The loop state to update")
    key: str = InputField(default="", description="The state key to set")
    value: Any = InputField(
        default=None,
        description="The value to set. Connect an output to this input.",
        ui_type=UIType.Any,
    )

    def invoke(self, context: InvocationContext) -> LoopStateOutput:
        values = _copy_value((self.state or LoopState()).values)
        values[self.key] = _copy_value(self.value)
        return LoopStateOutput(state=LoopState(values=values))


@invocation(
    "state_merge", title="Merge Loop State Values", tags=["loop", "state"], category="workflow", version="1.0.1"
)
class StateMergeInvocation(BaseInvocation):
    """Returns loop state with multiple values merged."""

    state: Optional[LoopState] = InputField(default=None, description="The loop state to update")
    values: dict[str, Any] = InputField(
        default_factory=dict,
        description="The values to merge into the loop state. Connect an output to this input.",
        ui_type=UIType.Any,
    )

    def invoke(self, context: InvocationContext) -> LoopStateOutput:
        values = _copy_value((self.state or LoopState()).values)
        values.update(_copy_value(self.values))
        return LoopStateOutput(state=LoopState(values=values))


@invocation_output("for_output")
class ForInvocationOutput(BaseInvocationOutput):
    loop_linkage: Any = OutputField(
        description="The loop linkage to the matching ForReturn",
        title="Loop Linkage",
        ui_type=UIType.Any,
    )
    item: Optional[Any] = OutputField(
        default=None,
        description="The item for the current loop iteration, or None when the collection is empty",
        title="Collection Item",
        ui_type=UIType._CollectionItem,
        output_scope=OutputScope.Iteration,
    )
    index: int = OutputField(
        description="The index for the current loop iteration",
        title="Index",
        output_scope=OutputScope.Iteration,
    )
    total: int = OutputField(
        description="The total number of items in the loop collection",
        title="Total",
        output_scope=OutputScope.Iteration,
    )
    state: LoopState = OutputField(
        description="The state for the current loop iteration",
        title="State",
        output_scope=OutputScope.Iteration,
    )
    output_collection: list[Any] = OutputField(
        description="The collected loop body outputs",
        title="Output Collection",
        ui_type=UIType._Collection,
        output_scope=OutputScope.Final,
    )
    final_state: LoopState = OutputField(
        description="The final loop state",
        title="Final State",
        output_scope=OutputScope.Final,
    )


@invocation("for", version="1.3.1")
class ForInvocation(BaseInvocation):
    execution_effects_enabled = True

    collection: list[Any] = InputField(
        description="The list of items to iterate over",
        default=[],
        ui_type=UIType._Collection,
    )
    state: Optional[LoopState] = InputField(
        default=None,
        description="Optional initial loop state",
    )
    index: int = InputField(
        description="The internal iteration index for a prepared For execution node",
        default=-1,
        input=Input.Direct,
        ui_hidden=True,
    )

    def invoke(self, context: InvocationContext) -> ForInvocationOutput:
        if self.index < 0:
            raise NotImplementedError("For loop nodes must be executed as part of a workflow graph")

        state = self.state or LoopState()
        output = ForInvocationOutput(
            loop_linkage=LOOP_LINKAGE_FIELD,
            item=self.collection[self.index],
            index=self.index,
            total=len(self.collection),
            state=state,
            output_collection=[],
            final_state=state,
        )
        execution = getattr(context, "execution", None)
        if execution is not None and hasattr(execution, "start_continuation"):
            execution.start_continuation(
                "for",
                payload={
                    "index": self.index,
                    "total": len(self.collection),
                    "state": state.model_dump(mode="json"),
                },
            )
        return output


@invocation_output("for_return_output")
class ForReturnInvocationOutput(BaseInvocationOutput):
    output: Optional[Any] = OutputField(
        default=None,
        description="The output item to append to the loop output collection",
        title="Output",
        ui_type=UIType._CollectionItem,
        ui_hidden=True,
    )
    state: Optional[LoopState] = OutputField(
        default=None,
        description="The state to pass to the next loop iteration",
        title="State",
        ui_hidden=True,
    )


@invocation("for_return", version="1.3.2")
class ForReturnInvocation(BaseInvocation):
    execution_effects_enabled = True

    loop_linkage: Optional[Any] = InputField(
        default=None,
        description="The loop linkage from the matching For",
        input=Input.Connection,
        ui_type=UIType.Any,
    )
    output: Optional[Any] = InputField(
        default=None,
        description="The output item to append to the loop output collection",
        ui_type=UIType._CollectionItem,
    )
    state: Optional[LoopState] = InputField(
        default=None,
        description="The state to pass to the next loop iteration",
    )
    continue_condition: Optional[bool] = InputField(
        default=True,
        description="Whether to schedule the next loop iteration; false finalizes the loop",
    )

    def invoke(self, context: InvocationContext) -> ForReturnInvocationOutput:
        output = ForReturnInvocationOutput(output=self.output, state=self.state)
        execution = getattr(context, "execution", None)
        if execution is not None and hasattr(execution, "complete_continuation"):
            execution.complete_continuation(
                "for",
                payload={
                    "output": self.output,
                    "state": self.state.model_dump(mode="json") if self.state is not None else None,
                    "continue_condition": self.continue_condition,
                },
            )
        return output
