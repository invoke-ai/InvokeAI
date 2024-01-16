# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI team

from __future__ import annotations

import inspect
import re
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from inspect import signature
from types import UnionType
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterable, Literal, Optional, Type, TypeVar, Union, cast

import semver
from pydantic import BaseModel, ConfigDict, Field, RootModel, TypeAdapter, create_model
from pydantic.fields import FieldInfo, _Unset
from pydantic_core import PydanticUndefined

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowWithoutID
from invokeai.app.shared.fields import FieldDescriptions
from invokeai.app.util.metaenum import MetaEnum
from invokeai.app.util.misc import uuid_string
from invokeai.backend.util.logging import InvokeAILogger

if TYPE_CHECKING:
    from ..services.invocation_services import InvocationServices

logger = InvokeAILogger.get_logger()

CUSTOM_NODE_PACK_SUFFIX = "__invokeai-custom-node"


class InvalidVersionError(ValueError):
    pass


class InvalidFieldError(TypeError):
    pass


class Classification(str, Enum, metaclass=MetaEnum):
    """
    The classification of an Invocation.
    - `Stable`: The invocation, including its inputs/outputs and internal logic, is stable. You may build workflows with it, having confidence that they will not break because of a change in this invocation.
    - `Beta`: The invocation is not yet stable, but is planned to be stable in the future. Workflows built around this invocation may break, but we are committed to supporting this invocation long-term.
    - `Prototype`: The invocation is not yet stable and may be removed from the application at any time. Workflows built around this invocation may break, and we are *not* committed to supporting this invocation.
    """

    Stable = "stable"
    Beta = "beta"
    Prototype = "prototype"


class Input(str, Enum, metaclass=MetaEnum):
    """
    The type of input a field accepts.
    - `Input.Direct`: The field must have its value provided directly, when the invocation and field \
      are instantiated.
    - `Input.Connection`: The field must have its value provided by a connection.
    - `Input.Any`: The field may have its value provided either directly or by a connection.
    """

    Connection = "connection"
    Direct = "direct"
    Any = "any"


class FieldKind(str, Enum, metaclass=MetaEnum):
    """
    The kind of field.
    - `Input`: An input field on a node.
    - `Output`: An output field on a node.
    - `Internal`: A field which is treated as an input, but cannot be used in node definitions. Metadata is
    one example. It is provided to nodes via the WithMetadata class, and we want to reserve the field name
    "metadata" for this on all nodes. `FieldKind` is used to short-circuit the field name validation logic,
    allowing "metadata" for that field.
    - `NodeAttribute`: The field is a node attribute. These are fields which are not inputs or outputs,
    but which are used to store information about the node. For example, the `id` and `type` fields are node
    attributes.

    The presence of this in `json_schema_extra["field_kind"]` is used when initializing node schemas on app
    startup, and when generating the OpenAPI schema for the workflow editor.
    """

    Input = "input"
    Output = "output"
    Internal = "internal"
    NodeAttribute = "node_attribute"


class UIType(str, Enum, metaclass=MetaEnum):
    """
    Type hints for the UI for situations in which the field type is not enough to infer the correct UI type.

    - Model Fields
    The most common node-author-facing use will be for model fields. Internally, there is no difference
    between SD-1, SD-2 and SDXL model fields - they all use the class `MainModelField`. To ensure the
    base-model-specific UI is rendered, use e.g. `ui_type=UIType.SDXLMainModelField` to indicate that
    the field is an SDXL main model field.

    - Any Field
    We cannot infer the usage of `typing.Any` via schema parsing, so you *must* use `ui_type=UIType.Any` to
    indicate that the field accepts any type. Use with caution. This cannot be used on outputs.

    - Scheduler Field
    Special handling in the UI is needed for this field, which otherwise would be parsed as a plain enum field.

    - Internal Fields
    Similar to the Any Field, the `collect` and `iterate` nodes make use of `typing.Any`. To facilitate
    handling these types in the client, we use `UIType._Collection` and `UIType._CollectionItem`. These
    should not be used by node authors.

    - DEPRECATED Fields
    These types are deprecated and should not be used by node authors. A warning will be logged if one is
    used, and the type will be ignored. They are included here for backwards compatibility.
    """

    # region Model Field Types
    SDXLMainModel = "SDXLMainModelField"
    SDXLRefinerModel = "SDXLRefinerModelField"
    ONNXModel = "ONNXModelField"
    VaeModel = "VAEModelField"
    LoRAModel = "LoRAModelField"
    ControlNetModel = "ControlNetModelField"
    IPAdapterModel = "IPAdapterModelField"
    # endregion

    # region Misc Field Types
    Scheduler = "SchedulerField"
    Any = "AnyField"
    # endregion

    # region Internal Field Types
    _Collection = "CollectionField"
    _CollectionItem = "CollectionItemField"
    # endregion

    # region DEPRECATED
    Boolean = "DEPRECATED_Boolean"
    Color = "DEPRECATED_Color"
    Conditioning = "DEPRECATED_Conditioning"
    Control = "DEPRECATED_Control"
    Float = "DEPRECATED_Float"
    Image = "DEPRECATED_Image"
    Integer = "DEPRECATED_Integer"
    Latents = "DEPRECATED_Latents"
    String = "DEPRECATED_String"
    BooleanCollection = "DEPRECATED_BooleanCollection"
    ColorCollection = "DEPRECATED_ColorCollection"
    ConditioningCollection = "DEPRECATED_ConditioningCollection"
    ControlCollection = "DEPRECATED_ControlCollection"
    FloatCollection = "DEPRECATED_FloatCollection"
    ImageCollection = "DEPRECATED_ImageCollection"
    IntegerCollection = "DEPRECATED_IntegerCollection"
    LatentsCollection = "DEPRECATED_LatentsCollection"
    StringCollection = "DEPRECATED_StringCollection"
    BooleanPolymorphic = "DEPRECATED_BooleanPolymorphic"
    ColorPolymorphic = "DEPRECATED_ColorPolymorphic"
    ConditioningPolymorphic = "DEPRECATED_ConditioningPolymorphic"
    ControlPolymorphic = "DEPRECATED_ControlPolymorphic"
    FloatPolymorphic = "DEPRECATED_FloatPolymorphic"
    ImagePolymorphic = "DEPRECATED_ImagePolymorphic"
    IntegerPolymorphic = "DEPRECATED_IntegerPolymorphic"
    LatentsPolymorphic = "DEPRECATED_LatentsPolymorphic"
    StringPolymorphic = "DEPRECATED_StringPolymorphic"
    MainModel = "DEPRECATED_MainModel"
    UNet = "DEPRECATED_UNet"
    Vae = "DEPRECATED_Vae"
    CLIP = "DEPRECATED_CLIP"
    Collection = "DEPRECATED_Collection"
    CollectionItem = "DEPRECATED_CollectionItem"
    Enum = "DEPRECATED_Enum"
    WorkflowField = "DEPRECATED_WorkflowField"
    IsIntermediate = "DEPRECATED_IsIntermediate"
    BoardField = "DEPRECATED_BoardField"
    MetadataItem = "DEPRECATED_MetadataItem"
    MetadataItemCollection = "DEPRECATED_MetadataItemCollection"
    MetadataItemPolymorphic = "DEPRECATED_MetadataItemPolymorphic"
    MetadataDict = "DEPRECATED_MetadataDict"
    # endregion


class UIComponent(str, Enum, metaclass=MetaEnum):
    """
    The type of UI component to use for a field, used to override the default components, which are
    inferred from the field type.
    """

    None_ = "none"
    Textarea = "textarea"
    Slider = "slider"


class InputFieldJSONSchemaExtra(BaseModel):
    """
    Extra attributes to be added to input fields and their OpenAPI schema. Used during graph execution,
    and by the workflow editor during schema parsing and UI rendering.
    """

    input: Input
    orig_required: bool
    field_kind: FieldKind
    default: Optional[Any] = None
    orig_default: Optional[Any] = None
    ui_hidden: bool = False
    ui_type: Optional[UIType] = None
    ui_component: Optional[UIComponent] = None
    ui_order: Optional[int] = None
    ui_choice_labels: Optional[dict[str, str]] = None

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_serialization_defaults_required=True,
    )


class OutputFieldJSONSchemaExtra(BaseModel):
    """
    Extra attributes to be added to input fields and their OpenAPI schema. Used by the workflow editor
    during schema parsing and UI rendering.
    """

    field_kind: FieldKind
    ui_hidden: bool
    ui_type: Optional[UIType]
    ui_order: Optional[int]

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_serialization_defaults_required=True,
    )


def InputField(
    # copied from pydantic's Field
    # TODO: Can we support default_factory?
    default: Any = _Unset,
    default_factory: Callable[[], Any] | None = _Unset,
    title: str | None = _Unset,
    description: str | None = _Unset,
    pattern: str | None = _Unset,
    strict: bool | None = _Unset,
    gt: float | None = _Unset,
    ge: float | None = _Unset,
    lt: float | None = _Unset,
    le: float | None = _Unset,
    multiple_of: float | None = _Unset,
    allow_inf_nan: bool | None = _Unset,
    max_digits: int | None = _Unset,
    decimal_places: int | None = _Unset,
    min_length: int | None = _Unset,
    max_length: int | None = _Unset,
    # custom
    input: Input = Input.Any,
    ui_type: Optional[UIType] = None,
    ui_component: Optional[UIComponent] = None,
    ui_hidden: bool = False,
    ui_order: Optional[int] = None,
    ui_choice_labels: Optional[dict[str, str]] = None,
) -> Any:
    """
    Creates an input field for an invocation.

    This is a wrapper for Pydantic's [Field](https://docs.pydantic.dev/latest/api/fields/#pydantic.fields.Field) \
    that adds a few extra parameters to support graph execution and the node editor UI.

    :param Input input: [Input.Any] The kind of input this field requires. \
      `Input.Direct` means a value must be provided on instantiation. \
      `Input.Connection` means the value must be provided by a connection. \
      `Input.Any` means either will do.

    :param UIType ui_type: [None] Optionally provides an extra type hint for the UI. \
      In some situations, the field's type is not enough to infer the correct UI type. \
      For example, model selection fields should render a dropdown UI component to select a model. \
      Internally, there is no difference between SD-1, SD-2 and SDXL model fields, they all use \
      `MainModelField`. So to ensure the base-model-specific UI is rendered, you can use \
      `UIType.SDXLMainModelField` to indicate that the field is an SDXL main model field.

    :param UIComponent ui_component: [None] Optionally specifies a specific component to use in the UI. \
      The UI will always render a suitable component, but sometimes you want something different than the default. \
      For example, a `string` field will default to a single-line input, but you may want a multi-line textarea instead. \
      For this case, you could provide `UIComponent.Textarea`.

    :param bool ui_hidden: [False] Specifies whether or not this field should be hidden in the UI.

    :param int ui_order: [None] Specifies the order in which this field should be rendered in the UI.

    :param dict[str, str] ui_choice_labels: [None] Specifies the labels to use for the choices in an enum field.
    """

    json_schema_extra_ = InputFieldJSONSchemaExtra(
        input=input,
        ui_type=ui_type,
        ui_component=ui_component,
        ui_hidden=ui_hidden,
        ui_order=ui_order,
        ui_choice_labels=ui_choice_labels,
        field_kind=FieldKind.Input,
        orig_required=True,
    )

    """
    There is a conflict between the typing of invocation definitions and the typing of an invocation's
    `invoke()` function.

    On instantiation of a node, the invocation definition is used to create the python class. At this time,
    any number of fields may be optional, because they may be provided by connections.

    On calling of `invoke()`, however, those fields may be required.

    For example, consider an ResizeImageInvocation with an `image: ImageField` field.

    `image` is required during the call to `invoke()`, but when the python class is instantiated,
    the field may not be present. This is fine, because that image field will be provided by a
    connection from an ancestor node, which outputs an image.

    This means we want to type the `image` field as optional for the node class definition, but required
    for the `invoke()` function.

    If we use `typing.Optional` in the node class definition, the field will be typed as optional in the
    `invoke()` method, and we'll have to do a lot of runtime checks to ensure the field is present - or
    any static type analysis tools will complain.

    To get around this, in node class definitions, we type all fields correctly for the `invoke()` function,
    but secretly make them optional in `InputField()`. We also store the original required bool and/or default
    value. When we call `invoke()`, we use this stored information to do an additional check on the class.
    """

    if default_factory is not _Unset and default_factory is not None:
        default = default_factory()
        logger.warn('"default_factory" is not supported, calling it now to set "default"')

    # These are the args we may wish pass to the pydantic `Field()` function
    field_args = {
        "default": default,
        "title": title,
        "description": description,
        "pattern": pattern,
        "strict": strict,
        "gt": gt,
        "ge": ge,
        "lt": lt,
        "le": le,
        "multiple_of": multiple_of,
        "allow_inf_nan": allow_inf_nan,
        "max_digits": max_digits,
        "decimal_places": decimal_places,
        "min_length": min_length,
        "max_length": max_length,
    }

    # We only want to pass the args that were provided, otherwise the `Field()`` function won't work as expected
    provided_args = {k: v for (k, v) in field_args.items() if v is not PydanticUndefined}

    # Because we are manually making fields optional, we need to store the original required bool for reference later
    json_schema_extra_.orig_required = default is PydanticUndefined

    # Make Input.Any and Input.Connection fields optional, providing None as a default if the field doesn't already have one
    if input is Input.Any or input is Input.Connection:
        default_ = None if default is PydanticUndefined else default
        provided_args.update({"default": default_})
        if default is not PydanticUndefined:
            # Before invoking, we'll check for the original default value and set it on the field if the field has no value
            json_schema_extra_.default = default
            json_schema_extra_.orig_default = default
    elif default is not PydanticUndefined:
        default_ = default
        provided_args.update({"default": default_})
        json_schema_extra_.orig_default = default_

    return Field(
        **provided_args,
        json_schema_extra=json_schema_extra_.model_dump(exclude_none=True),
    )


def OutputField(
    # copied from pydantic's Field
    default: Any = _Unset,
    title: str | None = _Unset,
    description: str | None = _Unset,
    pattern: str | None = _Unset,
    strict: bool | None = _Unset,
    gt: float | None = _Unset,
    ge: float | None = _Unset,
    lt: float | None = _Unset,
    le: float | None = _Unset,
    multiple_of: float | None = _Unset,
    allow_inf_nan: bool | None = _Unset,
    max_digits: int | None = _Unset,
    decimal_places: int | None = _Unset,
    min_length: int | None = _Unset,
    max_length: int | None = _Unset,
    # custom
    ui_type: Optional[UIType] = None,
    ui_hidden: bool = False,
    ui_order: Optional[int] = None,
) -> Any:
    """
    Creates an output field for an invocation output.

    This is a wrapper for Pydantic's [Field](https://docs.pydantic.dev/1.10/usage/schema/#field-customization) \
    that adds a few extra parameters to support graph execution and the node editor UI.

    :param UIType ui_type: [None] Optionally provides an extra type hint for the UI. \
      In some situations, the field's type is not enough to infer the correct UI type. \
      For example, model selection fields should render a dropdown UI component to select a model. \
      Internally, there is no difference between SD-1, SD-2 and SDXL model fields, they all use \
      `MainModelField`. So to ensure the base-model-specific UI is rendered, you can use \
      `UIType.SDXLMainModelField` to indicate that the field is an SDXL main model field.

    :param bool ui_hidden: [False] Specifies whether or not this field should be hidden in the UI. \

    :param int ui_order: [None] Specifies the order in which this field should be rendered in the UI. \
    """
    return Field(
        default=default,
        title=title,
        description=description,
        pattern=pattern,
        strict=strict,
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        multiple_of=multiple_of,
        allow_inf_nan=allow_inf_nan,
        max_digits=max_digits,
        decimal_places=decimal_places,
        min_length=min_length,
        max_length=max_length,
        json_schema_extra=OutputFieldJSONSchemaExtra(
            ui_type=ui_type,
            ui_hidden=ui_hidden,
            ui_order=ui_order,
            field_kind=FieldKind.Output,
        ).model_dump(exclude_none=True),
    )


class UIConfigBase(BaseModel):
    """
    Provides additional node configuration to the UI.
    This is used internally by the @invocation decorator logic. Do not use this directly.
    """

    tags: Optional[list[str]] = Field(default_factory=None, description="The node's tags")
    title: Optional[str] = Field(default=None, description="The node's display name")
    category: Optional[str] = Field(default=None, description="The node's category")
    version: str = Field(
        description='The node\'s version. Should be a valid semver string e.g. "1.0.0" or "3.8.13".',
    )
    node_pack: Optional[str] = Field(default=None, description="Whether or not this is a custom node")
    classification: Classification = Field(default=Classification.Stable, description="The node's classification")

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_serialization_defaults_required=True,
    )


class InvocationContext:
    """Initialized and provided to on execution of invocations."""

    services: InvocationServices
    graph_execution_state_id: str
    queue_id: str
    queue_item_id: int
    queue_batch_id: str
    workflow: Optional[WorkflowWithoutID]

    def __init__(
        self,
        services: InvocationServices,
        queue_id: str,
        queue_item_id: int,
        queue_batch_id: str,
        graph_execution_state_id: str,
        workflow: Optional[WorkflowWithoutID],
    ):
        self.services = services
        self.graph_execution_state_id = graph_execution_state_id
        self.queue_id = queue_id
        self.queue_item_id = queue_item_id
        self.queue_batch_id = queue_batch_id
        self.workflow = workflow


class BaseInvocationOutput(BaseModel):
    """
    Base class for all invocation outputs.

    All invocation outputs must use the `@invocation_output` decorator to provide their unique type.
    """

    _output_classes: ClassVar[set[BaseInvocationOutput]] = set()

    @classmethod
    def register_output(cls, output: BaseInvocationOutput) -> None:
        """Registers an invocation output."""
        cls._output_classes.add(output)

    @classmethod
    def get_outputs(cls) -> Iterable[BaseInvocationOutput]:
        """Gets all invocation outputs."""
        return cls._output_classes

    @classmethod
    def get_outputs_union(cls) -> UnionType:
        """Gets a union of all invocation outputs."""
        outputs_union = Union[tuple(cls._output_classes)]  # type: ignore [valid-type]
        return outputs_union  # type: ignore [return-value]

    @classmethod
    def get_output_types(cls) -> Iterable[str]:
        """Gets all invocation output types."""
        return (i.get_type() for i in BaseInvocationOutput.get_outputs())

    @staticmethod
    def json_schema_extra(schema: dict[str, Any], model_class: Type[BaseModel]) -> None:
        """Adds various UI-facing attributes to the invocation output's OpenAPI schema."""
        # Because we use a pydantic Literal field with default value for the invocation type,
        # it will be typed as optional in the OpenAPI schema. Make it required manually.
        if "required" not in schema or not isinstance(schema["required"], list):
            schema["required"] = []
        schema["required"].extend(["type"])

    @classmethod
    def get_type(cls) -> str:
        """Gets the invocation output's type, as provided by the `@invocation_output` decorator."""
        return cls.model_fields["type"].default

    model_config = ConfigDict(
        protected_namespaces=(),
        validate_assignment=True,
        json_schema_serialization_defaults_required=True,
        json_schema_extra=json_schema_extra,
    )


class RequiredConnectionException(Exception):
    """Raised when an field which requires a connection did not receive a value."""

    def __init__(self, node_id: str, field_name: str):
        super().__init__(f"Node {node_id} missing connections for field {field_name}")


class MissingInputException(Exception):
    """Raised when an field which requires some input, but did not receive a value."""

    def __init__(self, node_id: str, field_name: str):
        super().__init__(f"Node {node_id} missing value or connection for field {field_name}")


class BaseInvocation(ABC, BaseModel):
    """
    All invocations must use the `@invocation` decorator to provide their unique type.
    """

    _invocation_classes: ClassVar[set[BaseInvocation]] = set()

    @classmethod
    def get_type(cls) -> str:
        """Gets the invocation's type, as provided by the `@invocation` decorator."""
        return cls.model_fields["type"].default

    @classmethod
    def register_invocation(cls, invocation: BaseInvocation) -> None:
        """Registers an invocation."""
        cls._invocation_classes.add(invocation)

    @classmethod
    def get_invocations_union(cls) -> UnionType:
        """Gets a union of all invocation types."""
        invocations_union = Union[tuple(cls._invocation_classes)]  # type: ignore [valid-type]
        return invocations_union  # type: ignore [return-value]

    @classmethod
    def get_invocations(cls) -> Iterable[BaseInvocation]:
        """Gets all invocations, respecting the allowlist and denylist."""
        app_config = InvokeAIAppConfig.get_config()
        allowed_invocations: set[BaseInvocation] = set()
        for sc in cls._invocation_classes:
            invocation_type = sc.get_type()
            is_in_allowlist = (
                invocation_type in app_config.allow_nodes if isinstance(app_config.allow_nodes, list) else True
            )
            is_in_denylist = (
                invocation_type in app_config.deny_nodes if isinstance(app_config.deny_nodes, list) else False
            )
            if is_in_allowlist and not is_in_denylist:
                allowed_invocations.add(sc)
        return allowed_invocations

    @classmethod
    def get_invocations_map(cls) -> dict[str, BaseInvocation]:
        """Gets a map of all invocation types to their invocation classes."""
        return {i.get_type(): i for i in BaseInvocation.get_invocations()}

    @classmethod
    def get_invocation_types(cls) -> Iterable[str]:
        """Gets all invocation types."""
        return (i.get_type() for i in BaseInvocation.get_invocations())

    @classmethod
    def get_output_annotation(cls) -> BaseInvocationOutput:
        """Gets the invocation's output annotation (i.e. the return annotation of its `invoke()` method)."""
        return signature(cls.invoke).return_annotation

    @staticmethod
    def json_schema_extra(schema: dict[str, Any], model_class: Type[BaseModel], *args, **kwargs) -> None:
        """Adds various UI-facing attributes to the invocation's OpenAPI schema."""
        uiconfig = cast(UIConfigBase | None, getattr(model_class, "UIConfig", None))
        if uiconfig is not None:
            if uiconfig.title is not None:
                schema["title"] = uiconfig.title
            if uiconfig.tags is not None:
                schema["tags"] = uiconfig.tags
            if uiconfig.category is not None:
                schema["category"] = uiconfig.category
            if uiconfig.node_pack is not None:
                schema["node_pack"] = uiconfig.node_pack
            schema["classification"] = uiconfig.classification
            schema["version"] = uiconfig.version
        if "required" not in schema or not isinstance(schema["required"], list):
            schema["required"] = []
        schema["required"].extend(["type", "id"])

    @abstractmethod
    def invoke(self, context: InvocationContext) -> BaseInvocationOutput:
        """Invoke with provided context and return outputs."""
        pass

    def invoke_internal(self, context: InvocationContext) -> BaseInvocationOutput:
        """
        Internal invoke method, calls `invoke()` after some prep.
        Handles optional fields that are required to call `invoke()` and invocation cache.
        """
        for field_name, field in self.model_fields.items():
            if not field.json_schema_extra or callable(field.json_schema_extra):
                # something has gone terribly awry, we should always have this and it should be a dict
                continue

            # Here we handle the case where the field is optional in the pydantic class, but required
            # in the `invoke()` method.

            orig_default = field.json_schema_extra.get("orig_default", PydanticUndefined)
            orig_required = field.json_schema_extra.get("orig_required", True)
            input_ = field.json_schema_extra.get("input", None)
            if orig_default is not PydanticUndefined and not hasattr(self, field_name):
                setattr(self, field_name, orig_default)
            if orig_required and orig_default is PydanticUndefined and getattr(self, field_name) is None:
                if input_ == Input.Connection:
                    raise RequiredConnectionException(self.model_fields["type"].default, field_name)
                elif input_ == Input.Any:
                    raise MissingInputException(self.model_fields["type"].default, field_name)

        # skip node cache codepath if it's disabled
        if context.services.configuration.node_cache_size == 0:
            return self.invoke(context)

        output: BaseInvocationOutput
        if self.use_cache:
            key = context.services.invocation_cache.create_key(self)
            cached_value = context.services.invocation_cache.get(key)
            if cached_value is None:
                context.services.logger.debug(f'Invocation cache miss for type "{self.get_type()}": {self.id}')
                output = self.invoke(context)
                context.services.invocation_cache.save(key, output)
                return output
            else:
                context.services.logger.debug(f'Invocation cache hit for type "{self.get_type()}": {self.id}')
                return cached_value
        else:
            context.services.logger.debug(f'Skipping invocation cache for "{self.get_type()}": {self.id}')
            return self.invoke(context)

    id: str = Field(
        default_factory=uuid_string,
        description="The id of this instance of an invocation. Must be unique among all instances of invocations.",
        json_schema_extra={"field_kind": FieldKind.NodeAttribute},
    )
    is_intermediate: bool = Field(
        default=False,
        description="Whether or not this is an intermediate invocation.",
        json_schema_extra={"ui_type": "IsIntermediate", "field_kind": FieldKind.NodeAttribute},
    )
    use_cache: bool = Field(
        default=True,
        description="Whether or not to use the cache",
        json_schema_extra={"field_kind": FieldKind.NodeAttribute},
    )

    UIConfig: ClassVar[Type[UIConfigBase]]

    model_config = ConfigDict(
        protected_namespaces=(),
        validate_assignment=True,
        json_schema_extra=json_schema_extra,
        json_schema_serialization_defaults_required=True,
        coerce_numbers_to_str=True,
    )


TBaseInvocation = TypeVar("TBaseInvocation", bound=BaseInvocation)


RESERVED_NODE_ATTRIBUTE_FIELD_NAMES = {
    "id",
    "is_intermediate",
    "use_cache",
    "type",
    "workflow",
}

RESERVED_INPUT_FIELD_NAMES = {
    "metadata",
}

RESERVED_OUTPUT_FIELD_NAMES = {"type"}


class _Model(BaseModel):
    pass


with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    # Get all pydantic model attrs, methods, etc
    RESERVED_PYDANTIC_FIELD_NAMES = {m[0] for m in inspect.getmembers(_Model())}


def validate_fields(model_fields: dict[str, FieldInfo], model_type: str) -> None:
    """
    Validates the fields of an invocation or invocation output:
    - Must not override any pydantic reserved fields
    - Must have a type annotation
    - Must have a json_schema_extra dict
    - Must have field_kind in json_schema_extra
    - Field name must not be reserved, according to its field_kind
    """
    for name, field in model_fields.items():
        if name in RESERVED_PYDANTIC_FIELD_NAMES:
            raise InvalidFieldError(f'Invalid field name "{name}" on "{model_type}" (reserved by pydantic)')

        if not field.annotation:
            raise InvalidFieldError(f'Invalid field type "{name}" on "{model_type}" (missing annotation)')

        if not isinstance(field.json_schema_extra, dict):
            raise InvalidFieldError(
                f'Invalid field definition for "{name}" on "{model_type}" (missing json_schema_extra dict)'
            )

        field_kind = field.json_schema_extra.get("field_kind", None)

        # must have a field_kind
        if not isinstance(field_kind, FieldKind):
            raise InvalidFieldError(
                f'Invalid field definition for "{name}" on "{model_type}" (maybe it\'s not an InputField or OutputField?)'
            )

        if field_kind is FieldKind.Input and (
            name in RESERVED_NODE_ATTRIBUTE_FIELD_NAMES or name in RESERVED_INPUT_FIELD_NAMES
        ):
            raise InvalidFieldError(f'Invalid field name "{name}" on "{model_type}" (reserved input field name)')

        if field_kind is FieldKind.Output and name in RESERVED_OUTPUT_FIELD_NAMES:
            raise InvalidFieldError(f'Invalid field name "{name}" on "{model_type}" (reserved output field name)')

        if (field_kind is FieldKind.Internal) and name not in RESERVED_INPUT_FIELD_NAMES:
            raise InvalidFieldError(
                f'Invalid field name "{name}" on "{model_type}" (internal field without reserved name)'
            )

        # node attribute fields *must* be in the reserved list
        if (
            field_kind is FieldKind.NodeAttribute
            and name not in RESERVED_NODE_ATTRIBUTE_FIELD_NAMES
            and name not in RESERVED_OUTPUT_FIELD_NAMES
        ):
            raise InvalidFieldError(
                f'Invalid field name "{name}" on "{model_type}" (node attribute field without reserved name)'
            )

        ui_type = field.json_schema_extra.get("ui_type", None)
        if isinstance(ui_type, str) and ui_type.startswith("DEPRECATED_"):
            logger.warn(f"\"UIType.{ui_type.split('_')[-1]}\" is deprecated, ignoring")
            field.json_schema_extra.pop("ui_type")
    return None


def invocation(
    invocation_type: str,
    title: Optional[str] = None,
    tags: Optional[list[str]] = None,
    category: Optional[str] = None,
    version: Optional[str] = None,
    use_cache: Optional[bool] = True,
    classification: Classification = Classification.Stable,
) -> Callable[[Type[TBaseInvocation]], Type[TBaseInvocation]]:
    """
    Registers an invocation.

    :param str invocation_type: The type of the invocation. Must be unique among all invocations.
    :param Optional[str] title: Adds a title to the invocation. Use if the auto-generated title isn't quite right. Defaults to None.
    :param Optional[list[str]] tags: Adds tags to the invocation. Invocations may be searched for by their tags. Defaults to None.
    :param Optional[str] category: Adds a category to the invocation. Used to group the invocations in the UI. Defaults to None.
    :param Optional[str] version: Adds a version to the invocation. Must be a valid semver string. Defaults to None.
    :param Optional[bool] use_cache: Whether or not to use the invocation cache. Defaults to True. The user may override this in the workflow editor.
    :param Classification classification: The classification of the invocation. Defaults to FeatureClassification.Stable. Use Beta or Prototype if the invocation is unstable.
    """

    def wrapper(cls: Type[TBaseInvocation]) -> Type[TBaseInvocation]:
        # Validate invocation types on creation of invocation classes
        # TODO: ensure unique?
        if re.compile(r"^\S+$").match(invocation_type) is None:
            raise ValueError(f'"invocation_type" must consist of non-whitespace characters, got "{invocation_type}"')

        if invocation_type in BaseInvocation.get_invocation_types():
            raise ValueError(f'Invocation type "{invocation_type}" already exists')

        validate_fields(cls.model_fields, invocation_type)

        # Add OpenAPI schema extras
        uiconfig_name = cls.__qualname__ + ".UIConfig"
        if not hasattr(cls, "UIConfig") or cls.UIConfig.__qualname__ != uiconfig_name:
            cls.UIConfig = type(uiconfig_name, (UIConfigBase,), {})
        cls.UIConfig.title = title
        cls.UIConfig.tags = tags
        cls.UIConfig.category = category
        cls.UIConfig.classification = classification

        # Grab the node pack's name from the module name, if it's a custom node
        is_custom_node = cls.__module__.rsplit(".", 1)[0] == "invokeai.app.invocations"
        if is_custom_node:
            cls.UIConfig.node_pack = cls.__module__.split(".")[0]
        else:
            cls.UIConfig.node_pack = None

        if version is not None:
            try:
                semver.Version.parse(version)
            except ValueError as e:
                raise InvalidVersionError(f'Invalid version string for node "{invocation_type}": "{version}"') from e
            cls.UIConfig.version = version
        else:
            logger.warn(f'No version specified for node "{invocation_type}", using "1.0.0"')
            cls.UIConfig.version = "1.0.0"

        if use_cache is not None:
            cls.model_fields["use_cache"].default = use_cache

        # Add the invocation type to the model.

        # You'd be tempted to just add the type field and rebuild the model, like this:
        # cls.model_fields.update(type=FieldInfo.from_annotated_attribute(Literal[invocation_type], invocation_type))
        # cls.model_rebuild() or cls.model_rebuild(force=True)

        # Unfortunately, because the `GraphInvocation` uses a forward ref in its `graph` field's annotation, this does
        # not work. Instead, we have to create a new class with the type field and patch the original class with it.

        invocation_type_annotation = Literal[invocation_type]  # type: ignore
        invocation_type_field = Field(
            title="type", default=invocation_type, json_schema_extra={"field_kind": FieldKind.NodeAttribute}
        )

        docstring = cls.__doc__
        cls = create_model(
            cls.__qualname__,
            __base__=cls,
            __module__=cls.__module__,
            type=(invocation_type_annotation, invocation_type_field),
        )
        cls.__doc__ = docstring

        # TODO: how to type this correctly? it's typed as ModelMetaclass, a private class in pydantic
        BaseInvocation.register_invocation(cls)  # type: ignore

        return cls

    return wrapper


TBaseInvocationOutput = TypeVar("TBaseInvocationOutput", bound=BaseInvocationOutput)


def invocation_output(
    output_type: str,
) -> Callable[[Type[TBaseInvocationOutput]], Type[TBaseInvocationOutput]]:
    """
    Adds metadata to an invocation output.

    :param str output_type: The type of the invocation output. Must be unique among all invocation outputs.
    """

    def wrapper(cls: Type[TBaseInvocationOutput]) -> Type[TBaseInvocationOutput]:
        # Validate output types on creation of invocation output classes
        # TODO: ensure unique?
        if re.compile(r"^\S+$").match(output_type) is None:
            raise ValueError(f'"output_type" must consist of non-whitespace characters, got "{output_type}"')

        if output_type in BaseInvocationOutput.get_output_types():
            raise ValueError(f'Invocation type "{output_type}" already exists')

        validate_fields(cls.model_fields, output_type)

        # Add the output type to the model.

        output_type_annotation = Literal[output_type]  # type: ignore
        output_type_field = Field(
            title="type", default=output_type, json_schema_extra={"field_kind": FieldKind.NodeAttribute}
        )

        docstring = cls.__doc__
        cls = create_model(
            cls.__qualname__,
            __base__=cls,
            __module__=cls.__module__,
            type=(output_type_annotation, output_type_field),
        )
        cls.__doc__ = docstring

        BaseInvocationOutput.register_output(cls)  # type: ignore # TODO: how to type this correctly?

        return cls

    return wrapper


class MetadataField(RootModel):
    """
    Pydantic model for metadata with custom root of type dict[str, Any].
    Metadata is stored without a strict schema.
    """

    root: dict[str, Any] = Field(description="The metadata")


MetadataFieldValidator = TypeAdapter(MetadataField)


class WithMetadata(BaseModel):
    metadata: Optional[MetadataField] = Field(
        default=None,
        description=FieldDescriptions.metadata,
        json_schema_extra=InputFieldJSONSchemaExtra(
            field_kind=FieldKind.Internal,
            input=Input.Connection,
            orig_required=False,
        ).model_dump(exclude_none=True),
    )


class WithWorkflow:
    workflow = None

    def __init_subclass__(cls) -> None:
        logger.warn(
            f"{cls.__module__.split('.')[0]}.{cls.__name__}: WithWorkflow is deprecated. Use `context.workflow` to access the workflow."
        )
        super().__init_subclass__()
