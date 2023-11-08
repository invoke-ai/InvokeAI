# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from enum import Enum
from inspect import signature
from types import UnionType
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterable, Literal, Optional, Type, TypeVar, Union

import semver
from pydantic import BaseModel, ConfigDict, Field, RootModel, TypeAdapter, create_model
from pydantic.fields import FieldInfo, _Unset
from pydantic_core import PydanticUndefined

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.shared.fields import FieldDescriptions
from invokeai.app.util.misc import uuid_string

if TYPE_CHECKING:
    from ..services.invocation_services import InvocationServices


class InvalidVersionError(ValueError):
    pass


class InvalidFieldError(TypeError):
    pass


class Input(str, Enum):
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


class UIType(str, Enum):
    """
    Type hints for the UI.
    If a field should be provided a data type that does not exactly match the python type of the field, \
    use this to provide the type that should be used instead. See the node development docs for detail \
    on adding a new field type, which involves client-side changes.
    """

    # region Primitives
    Boolean = "boolean"
    Color = "ColorField"
    Conditioning = "ConditioningField"
    Control = "ControlField"
    Float = "float"
    Image = "ImageField"
    Integer = "integer"
    Latents = "LatentsField"
    String = "string"
    # endregion

    # region Collection Primitives
    BooleanCollection = "BooleanCollection"
    ColorCollection = "ColorCollection"
    ConditioningCollection = "ConditioningCollection"
    ControlCollection = "ControlCollection"
    FloatCollection = "FloatCollection"
    ImageCollection = "ImageCollection"
    IntegerCollection = "IntegerCollection"
    LatentsCollection = "LatentsCollection"
    StringCollection = "StringCollection"
    # endregion

    # region Polymorphic Primitives
    BooleanPolymorphic = "BooleanPolymorphic"
    ColorPolymorphic = "ColorPolymorphic"
    ConditioningPolymorphic = "ConditioningPolymorphic"
    ControlPolymorphic = "ControlPolymorphic"
    FloatPolymorphic = "FloatPolymorphic"
    ImagePolymorphic = "ImagePolymorphic"
    IntegerPolymorphic = "IntegerPolymorphic"
    LatentsPolymorphic = "LatentsPolymorphic"
    StringPolymorphic = "StringPolymorphic"
    # endregion

    # region Models
    MainModel = "MainModelField"
    SDXLMainModel = "SDXLMainModelField"
    SDXLRefinerModel = "SDXLRefinerModelField"
    ONNXModel = "ONNXModelField"
    VaeModel = "VaeModelField"
    LoRAModel = "LoRAModelField"
    ControlNetModel = "ControlNetModelField"
    IPAdapterModel = "IPAdapterModelField"
    UNet = "UNetField"
    Vae = "VaeField"
    CLIP = "ClipField"
    # endregion

    # region Iterate/Collect
    Collection = "Collection"
    CollectionItem = "CollectionItem"
    # endregion

    # region Misc
    Enum = "enum"
    Scheduler = "Scheduler"
    WorkflowField = "WorkflowField"
    IsIntermediate = "IsIntermediate"
    BoardField = "BoardField"
    Any = "Any"
    MetadataItem = "MetadataItem"
    MetadataItemCollection = "MetadataItemCollection"
    MetadataItemPolymorphic = "MetadataItemPolymorphic"
    MetadataDict = "MetadataDict"
    # endregion


class UIComponent(str, Enum):
    """
    The type of UI component to use for a field, used to override the default components, which are \
    inferred from the field type.
    """

    None_ = "none"
    Textarea = "textarea"
    Slider = "slider"


class _InputField(BaseModel):
    """
    *DO NOT USE*
    This helper class is used to tell the client about our custom field attributes via OpenAPI
    schema generation, and Typescript type generation from that schema. It serves no functional
    purpose in the backend.
    """

    input: Input
    ui_hidden: bool
    ui_type: Optional[UIType]
    ui_component: Optional[UIComponent]
    ui_order: Optional[int]
    ui_choice_labels: Optional[dict[str, str]]
    item_default: Optional[Any]

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_serialization_defaults_required=True,
    )


class _OutputField(BaseModel):
    """
    *DO NOT USE*
    This helper class is used to tell the client about our custom field attributes via OpenAPI
    schema generation, and Typescript type generation from that schema. It serves no functional
    purpose in the backend.
    """

    ui_hidden: bool
    ui_type: Optional[UIType]
    ui_order: Optional[int]

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_serialization_defaults_required=True,
    )


def get_type(klass: BaseModel) -> str:
    """Helper function to get an invocation or invocation output's type. This is the default value of the `type` field."""
    return klass.model_fields["type"].default


def InputField(
    # copied from pydantic's Field
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
    item_default: Optional[Any] = None,
) -> Any:
    """
    Creates an input field for an invocation.

    This is a wrapper for Pydantic's [Field](https://docs.pydantic.dev/1.10/usage/schema/#field-customization) \
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

    : param bool ui_hidden: [False] Specifies whether or not this field should be hidden in the UI.

    : param int ui_order: [None] Specifies the order in which this field should be rendered in the UI. \

    : param bool item_default: [None] Specifies the default item value, if this is a collection input. \
      Ignored for non-collection fields.
    """

    json_schema_extra_: dict[str, Any] = dict(
        input=input,
        ui_type=ui_type,
        ui_component=ui_component,
        ui_hidden=ui_hidden,
        ui_order=ui_order,
        item_default=item_default,
        ui_choice_labels=ui_choice_labels,
        _field_kind="input",
    )

    field_args = dict(
        default=default,
        default_factory=default_factory,
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
    )

    """
    Invocation definitions have their fields typed correctly for their `invoke()` functions.
    This typing is often more specific than the actual invocation definition requires, because
    fields may have values provided only by connections.

    For example, consider an ResizeImageInvocation with an `image: ImageField` field.

    `image` is required during the call to `invoke()`, but when the python class is instantiated,
    the field may not be present. This is fine, because that image field will be provided by a
    an ancestor node that outputs the image.

    So we'd like to type that `image` field as `Optional[ImageField]`. If we do that, however, then
    we need to handle a lot of extra logic in the `invoke()` function to check if the field has a
    value or not. This is very tedious.

    Ideally, the invocation definition would be able to specify that the field is required during
    invocation, but optional during instantiation. So the field would be typed as `image: ImageField`,
    but when calling the `invoke()` function, we raise an error if the field is not present.

    To do this, we need to do a bit of fanagling to make the pydantic field optional, and then do
    extra validation when calling `invoke()`.

    There is some additional logic here to cleaning create the pydantic field via the wrapper.
    """

    # Filter out field args not provided
    provided_args = {k: v for (k, v) in field_args.items() if v is not PydanticUndefined}

    if (default is not PydanticUndefined) and (default_factory is not PydanticUndefined):
        raise ValueError("Cannot specify both default and default_factory")

    # because we are manually making fields optional, we need to store the original required bool for reference later
    if default is PydanticUndefined and default_factory is PydanticUndefined:
        json_schema_extra_.update(dict(orig_required=True))
    else:
        json_schema_extra_.update(dict(orig_required=False))

    # make Input.Any and Input.Connection fields optional, providing None as a default if the field doesn't already have one
    if (input is Input.Any or input is Input.Connection) and default_factory is PydanticUndefined:
        default_ = None if default is PydanticUndefined else default
        provided_args.update(dict(default=default_))
        if default is not PydanticUndefined:
            # before invoking, we'll grab the original default value and set it on the field if the field wasn't provided a value
            json_schema_extra_.update(dict(default=default))
            json_schema_extra_.update(dict(orig_default=default))
    elif default is not PydanticUndefined and default_factory is PydanticUndefined:
        default_ = default
        provided_args.update(dict(default=default_))
        json_schema_extra_.update(dict(orig_default=default_))
    elif default_factory is not PydanticUndefined:
        provided_args.update(dict(default_factory=default_factory))
        # TODO: cannot serialize default_factory...
        # json_schema_extra_.update(dict(orig_default_factory=default_factory))

    return Field(
        **provided_args,
        json_schema_extra=json_schema_extra_,
    )


def OutputField(
    # copied from pydantic's Field
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

    : param bool ui_hidden: [False] Specifies whether or not this field should be hidden in the UI. \

    : param int ui_order: [None] Specifies the order in which this field should be rendered in the UI. \
    """
    return Field(
        default=default,
        default_factory=default_factory,
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
        json_schema_extra=dict(
            ui_type=ui_type,
            ui_hidden=ui_hidden,
            ui_order=ui_order,
            _field_kind="output",
        ),
    )


class UIConfigBase(BaseModel):
    """
    Provides additional node configuration to the UI.
    This is used internally by the @invocation decorator logic. Do not use this directly.
    """

    tags: Optional[list[str]] = Field(default_factory=None, description="The node's tags")
    title: Optional[str] = Field(default=None, description="The node's display name")
    category: Optional[str] = Field(default=None, description="The node's category")
    version: Optional[str] = Field(
        default=None,
        description='The node\'s version. Should be a valid semver string e.g. "1.0.0" or "3.8.13".',
    )

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

    def __init__(
        self,
        services: InvocationServices,
        queue_id: str,
        queue_item_id: int,
        queue_batch_id: str,
        graph_execution_state_id: str,
    ):
        self.services = services
        self.graph_execution_state_id = graph_execution_state_id
        self.queue_id = queue_id
        self.queue_item_id = queue_item_id
        self.queue_batch_id = queue_batch_id


class BaseInvocationOutput(BaseModel):
    """
    Base class for all invocation outputs.

    All invocation outputs must use the `@invocation_output` decorator to provide their unique type.
    """

    _output_classes: ClassVar[set[BaseInvocationOutput]] = set()

    @classmethod
    def register_output(cls, output: BaseInvocationOutput) -> None:
        cls._output_classes.add(output)

    @classmethod
    def get_outputs(cls) -> Iterable[BaseInvocationOutput]:
        return cls._output_classes

    @classmethod
    def get_outputs_union(cls) -> UnionType:
        outputs_union = Union[tuple(cls._output_classes)]  # type: ignore [valid-type]
        return outputs_union  # type: ignore [return-value]

    @classmethod
    def get_output_types(cls) -> Iterable[str]:
        return map(lambda i: get_type(i), BaseInvocationOutput.get_outputs())

    @staticmethod
    def json_schema_extra(schema: dict[str, Any], model_class: Type[BaseModel]) -> None:
        # Because we use a pydantic Literal field with default value for the invocation type,
        # it will be typed as optional in the OpenAPI schema. Make it required manually.
        if "required" not in schema or not isinstance(schema["required"], list):
            schema["required"] = list()
        schema["required"].extend(["type"])

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
    def register_invocation(cls, invocation: BaseInvocation) -> None:
        cls._invocation_classes.add(invocation)

    @classmethod
    def get_invocations_union(cls) -> UnionType:
        invocations_union = Union[tuple(cls._invocation_classes)]  # type: ignore [valid-type]
        return invocations_union  # type: ignore [return-value]

    @classmethod
    def get_invocations(cls) -> Iterable[BaseInvocation]:
        app_config = InvokeAIAppConfig.get_config()
        allowed_invocations: set[BaseInvocation] = set()
        for sc in cls._invocation_classes:
            invocation_type = get_type(sc)
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
        # Get the type strings out of the literals and into a dictionary
        return dict(
            map(
                lambda i: (get_type(i), i),
                BaseInvocation.get_invocations(),
            )
        )

    @classmethod
    def get_invocation_types(cls) -> Iterable[str]:
        return map(lambda i: get_type(i), BaseInvocation.get_invocations())

    @classmethod
    def get_output_type(cls) -> BaseInvocationOutput:
        return signature(cls.invoke).return_annotation

    @staticmethod
    def json_schema_extra(schema: dict[str, Any], model_class: Type[BaseModel]) -> None:
        # Add the various UI-facing attributes to the schema. These are used to build the invocation templates.
        uiconfig = getattr(model_class, "UIConfig", None)
        if uiconfig and hasattr(uiconfig, "title"):
            schema["title"] = uiconfig.title
        if uiconfig and hasattr(uiconfig, "tags"):
            schema["tags"] = uiconfig.tags
        if uiconfig and hasattr(uiconfig, "category"):
            schema["category"] = uiconfig.category
        if uiconfig and hasattr(uiconfig, "version"):
            schema["version"] = uiconfig.version
        if "required" not in schema or not isinstance(schema["required"], list):
            schema["required"] = list()
        schema["required"].extend(["type", "id"])

    @abstractmethod
    def invoke(self, context: InvocationContext) -> BaseInvocationOutput:
        """Invoke with provided context and return outputs."""
        pass

    def invoke_internal(self, context: InvocationContext) -> BaseInvocationOutput:
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

    def get_type(self) -> str:
        return self.model_fields["type"].default

    id: str = Field(
        default_factory=uuid_string,
        description="The id of this instance of an invocation. Must be unique among all instances of invocations.",
        json_schema_extra=dict(_field_kind="internal"),
    )
    is_intermediate: bool = Field(
        default=False,
        description="Whether or not this is an intermediate invocation.",
        json_schema_extra=dict(ui_type=UIType.IsIntermediate, _field_kind="internal"),
    )
    use_cache: bool = Field(
        default=True, description="Whether or not to use the cache", json_schema_extra=dict(_field_kind="internal")
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


RESERVED_INPUT_FIELD_NAMES = {
    "id",
    "is_intermediate",
    "use_cache",
    "type",
    "workflow",
    "metadata",
}

RESERVED_OUTPUT_FIELD_NAMES = {"type"}


class _Model(BaseModel):
    pass


# Get all pydantic model attrs, methods, etc
RESERVED_PYDANTIC_FIELD_NAMES = set(map(lambda m: m[0], inspect.getmembers(_Model())))


def validate_fields(model_fields: dict[str, FieldInfo], model_type: str) -> None:
    """
    Validates the fields of an invocation or invocation output:
    - must not override any pydantic reserved fields
    - must be created via `InputField`, `OutputField`, or be an internal field defined in this file
    """
    for name, field in model_fields.items():
        if name in RESERVED_PYDANTIC_FIELD_NAMES:
            raise InvalidFieldError(f'Invalid field name "{name}" on "{model_type}" (reserved by pydantic)')

        field_kind = (
            # _field_kind is defined via InputField(), OutputField() or by one of the internal fields defined in this file
            field.json_schema_extra.get("_field_kind", None)
            if field.json_schema_extra
            else None
        )

        # must have a field_kind
        if field_kind is None or field_kind not in {"input", "output", "internal"}:
            raise InvalidFieldError(
                f'Invalid field definition for "{name}" on "{model_type}" (maybe it\'s not an InputField or OutputField?)'
            )

        if field_kind == "input" and name in RESERVED_INPUT_FIELD_NAMES:
            raise InvalidFieldError(f'Invalid field name "{name}" on "{model_type}" (reserved input field name)')

        if field_kind == "output" and name in RESERVED_OUTPUT_FIELD_NAMES:
            raise InvalidFieldError(f'Invalid field name "{name}" on "{model_type}" (reserved output field name)')

        # internal fields *must* be in the reserved list
        if (
            field_kind == "internal"
            and name not in RESERVED_INPUT_FIELD_NAMES
            and name not in RESERVED_OUTPUT_FIELD_NAMES
        ):
            raise InvalidFieldError(
                f'Invalid field name "{name}" on "{model_type}" (internal field without reserved name)'
            )

    return None


def invocation(
    invocation_type: str,
    title: Optional[str] = None,
    tags: Optional[list[str]] = None,
    category: Optional[str] = None,
    version: Optional[str] = None,
    use_cache: Optional[bool] = True,
) -> Callable[[Type[TBaseInvocation]], Type[TBaseInvocation]]:
    """
    Registers an invocation.

    :param str invocation_type: The type of the invocation. Must be unique among all invocations.
    :param Optional[str] title: Adds a title to the invocation. Use if the auto-generated title isn't quite right. Defaults to None.
    :param Optional[list[str]] tags: Adds tags to the invocation. Invocations may be searched for by their tags. Defaults to None.
    :param Optional[str] category: Adds a category to the invocation. Used to group the invocations in the UI. Defaults to None.
    :param Optional[str] version: Adds a version to the invocation. Must be a valid semver string. Defaults to None.
    :param Optional[bool] use_cache: Whether or not to use the invocation cache. Defaults to True. The user may override this in the workflow editor.
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
        uiconf_name = cls.__qualname__ + ".UIConfig"
        if not hasattr(cls, "UIConfig") or cls.UIConfig.__qualname__ != uiconf_name:
            cls.UIConfig = type(uiconf_name, (UIConfigBase,), dict())
        if title is not None:
            cls.UIConfig.title = title
        if tags is not None:
            cls.UIConfig.tags = tags
        if category is not None:
            cls.UIConfig.category = category
        if version is not None:
            try:
                semver.Version.parse(version)
            except ValueError as e:
                raise InvalidVersionError(f'Invalid version string for node "{invocation_type}": "{version}"') from e
            cls.UIConfig.version = version
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
            title="type", default=invocation_type, json_schema_extra=dict(_field_kind="internal")
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
        output_type_field = Field(title="type", default=output_type, json_schema_extra=dict(_field_kind="internal"))

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


class WorkflowField(RootModel):
    """
    Pydantic model for workflows with custom root of type dict[str, Any].
    Workflows are stored without a strict schema.
    """

    root: dict[str, Any] = Field(description="The workflow")


WorkflowFieldValidator = TypeAdapter(WorkflowField)


class WithWorkflow(BaseModel):
    workflow: Optional[WorkflowField] = Field(
        default=None, description=FieldDescriptions.workflow, json_schema_extra=dict(_field_kind="internal")
    )


class MetadataField(RootModel):
    """
    Pydantic model for metadata with custom root of type dict[str, Any].
    Metadata is stored without a strict schema.
    """

    root: dict[str, Any] = Field(description="The metadata")


MetadataFieldValidator = TypeAdapter(MetadataField)


class WithMetadata(BaseModel):
    metadata: Optional[MetadataField] = Field(
        default=None, description=FieldDescriptions.metadata, json_schema_extra=dict(_field_kind="internal")
    )
