from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    FieldDescriptions,
    InputField,
    InvocationContext,
    OutputField,
    UIType,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.model import LoRAModelField
from invokeai.app.util.model_exclude_null import BaseModelExcludeNull

from ...version import __version__


class LoRAMetadataField(BaseModelExcludeNull):
    """LoRA metadata for an image generated in InvokeAI."""

    lora: LoRAModelField = Field(description="The LoRA model")
    weight: float = Field(description="The weight of the LoRA model")


class ImageMetadata(BaseModelExcludeNull):
    """An image's generation metadata"""

    metadata: Optional[dict] = Field(default=None, description="The metadata associated with the image")
    workflow: Optional[dict] = Field(default=None, description="The workflow associated with the image")


class MetadataItem(BaseModel):
    label: str = Field(description=FieldDescriptions.metadata_item_label)
    value: Any = Field(description=FieldDescriptions.metadata_item_value)


@invocation_output("metadata_item_output")
class MetadataItemOutput(BaseInvocationOutput):
    """Metadata Item Output"""

    item: MetadataItem = OutputField(description="Metadata Item")


@invocation("metadata_item", title="Metadata Item", tags=["metadata"], category="metadata", version="1.0.0")
class MetadataItemInvocation(BaseInvocation):
    """Used to create an arbitrary metadata item. Provide "label" and make a connection to "value" to store that data as the value."""

    label: str = InputField(description=FieldDescriptions.metadata_item_label)
    value: Any = InputField(description=FieldDescriptions.metadata_item_value, ui_type=UIType.Any)

    def invoke(self, context: InvocationContext) -> MetadataItemOutput:
        return MetadataItemOutput(item=MetadataItem(label=self.label, value=self.value))


class MetadataDict(BaseModel):
    """Accepts a single MetadataItem or collection of MetadataItems (use a Collect node)."""

    data: dict[str, Any] = Field(description="Metadata dict")


@invocation_output("metadata_dict")
class MetadataDictOutput(BaseInvocationOutput):
    metadata_dict: MetadataDict = OutputField(description="Metadata Dict")


@invocation("metadata", title="Metadata", tags=["metadata"], category="metadata", version="1.0.0")
class MetadataInvocation(BaseInvocation):
    """Takes a MetadataItem or collection of MetadataItems and outputs a MetadataDict."""

    items: Union[list[MetadataItem], MetadataItem] = InputField(description=FieldDescriptions.metadata_item_polymorphic)

    def invoke(self, context: InvocationContext) -> MetadataDictOutput:
        if isinstance(self.items, MetadataItem):
            # single metadata item
            data = {self.items.label: self.items.value}
        else:
            # collection of metadata items
            data = {item.label: item.value for item in self.items}

        data.update({"app_version": __version__})
        return MetadataDictOutput(metadata_dict=(MetadataDict(data=data)))


@invocation("merge_metadata_dict", title="Metadata Merge", tags=["metadata"], category="metadata", version="1.0.0")
class MergeMetadataDictInvocation(BaseInvocation):
    """Merged a collection of MetadataDict into a single MetadataDict."""

    collection: list[MetadataDict] = InputField(description=FieldDescriptions.metadata_dict_collection)

    def invoke(self, context: InvocationContext) -> MetadataDictOutput:
        data = {}
        for item in self.collection:
            data.update(item.data)

        return MetadataDictOutput(metadata_dict=MetadataDict(data=data))


class WithMetadata(BaseModel):
    metadata: Optional[MetadataDict] = InputField(default=None, description=FieldDescriptions.metadata)
