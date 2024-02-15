from typing import List, Optional, Union

from invokeai.app.invocations.baseinvocation import (
    BaseInvocationOutput,
    Input,
    InputField,
    InvocationContext,
    OutputField,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.controlnet_image_processors import ControlField, ControlNetInvocation
from invokeai.app.invocations.ip_adapter import IPAdapterField, IPAdapterInvocation
from invokeai.app.invocations.t2i_adapter import T2IAdapterField, T2IAdapterInvocation
from invokeai.app.shared.fields import FieldDescriptions


def append_list(new_item, items, item_cls):
    """Add an item to an exiting item or list of items then output as a list of items."""

    result = []
    if items is None or (isinstance(items, list) and len(items) == 0):
        pass
    elif isinstance(items, item_cls):
        result.append(items)
    elif isinstance(items, list) and all(isinstance(i, item_cls) for i in items):
        result.extend(items)
    else:
        raise ValueError(f"Invalid adapter list format: {items}")

    result.append(new_item)
    return result


@invocation_output("control_list_output")
class ControlListOutput(BaseInvocationOutput):
    # Outputs
    control_list: list[ControlField] = OutputField(description=FieldDescriptions.control)


@invocation(
    "controlnet-linked",
    title="ControlNet-Linked",
    tags=["controlnet"],
    category="controlnet",
    version="1.1.0",
)
class ControlNetLinkedInvocation(ControlNetInvocation):
    """Collects ControlNet info to pass to other nodes."""

    control_list: Optional[Union[ControlField, list[ControlField]]] = InputField(
        default=None,
        title="ControlNet-List",
        input=Input.Connection,
        ui_order=0,
    )

    def invoke(self, context: InvocationContext) -> ControlListOutput:
        # Call parent
        output = super().invoke(context).control
        # Append the control output to the input list
        control_list = append_list(output, self.control_list, ControlField)
        return ControlListOutput(control_list=control_list)


@invocation_output("ip_adapter_list_output")
class IPAdapterListOutput(BaseInvocationOutput):
    # Outputs
    ip_adapter_list: list[IPAdapterField] = OutputField(
        description=FieldDescriptions.ip_adapter, title="IP-Adapter-List"
    )


@invocation(
    "ip_adapter_linked",
    title="IP-Adapter-Linked",
    tags=["ip_adapter", "control"],
    category="ip_adapter",
    version="1.1.0",
)
class IPAdapterLinkedInvocation(IPAdapterInvocation):
    """Collects IP-Adapter info to pass to other nodes."""

    ip_adapter_list: Optional[Union[IPAdapterField, list[IPAdapterField]]] = InputField(
        description=FieldDescriptions.ip_adapter,
        title="IP-Adapter-List",
        default=None,
        input=Input.Connection,
        ui_order=0,
    )

    def invoke(self, context: InvocationContext) -> IPAdapterListOutput:
        # Call parent
        output = super().invoke(context).ip_adapter
        # Append the control output to the input list
        result = append_list(output, self.ip_adapter_list, IPAdapterField)
        return IPAdapterListOutput(ip_adapter_list=result)


@invocation_output("ip_adapters_output")
class T2IAdapterListOutput(BaseInvocationOutput):
    # Outputs
    t2i_adapter_list: list[T2IAdapterField] = OutputField(
        description=FieldDescriptions.t2i_adapter, title="T2I Adapter-List"
    )


@invocation(
    "t2i_adapter_linked",
    title="T2I-Adapter-Linked",
    tags=["t2i_adapter", "control"],
    category="t2i_adapter",
    version="1.0.0",
)
class T2IAdapterLinkedInvocation(T2IAdapterInvocation):
    """Collects T2I-Adapter info to pass to other nodes."""

    t2i_adapter_list: Optional[Union[T2IAdapterField, list[T2IAdapterField]]] = InputField(
        description=FieldDescriptions.ip_adapter,
        title="T2I-Adapter",
        default=None,
        input=Input.Connection,
        ui_order=0,
    )

    def invoke(self, context: InvocationContext) -> T2IAdapterListOutput:
        # Call parent
        output = super().invoke(context).t2i_adapter
        # Append the control output to the input list
        t2i_adapter_list = append_list(output, self.t2i_adapter_list, T2IAdapterField)
        return T2IAdapterListOutput(t2i_adapter_list=t2i_adapter_list)
