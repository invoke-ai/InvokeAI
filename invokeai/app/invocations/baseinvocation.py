# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654) and the InvokeAI team

from __future__ import annotations

import inspect
import re
import sys
import types
import typing
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    ClassVar,
    Iterable,
    Literal,
    Optional,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import semver
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from invokeai.app.invocations.fields import (
    FieldKind,
    Input,
)
from invokeai.app.services.config.config_default import get_config
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.metaenum import MetaEnum
from invokeai.app.util.misc import uuid_string
from invokeai.backend.util.logging import InvokeAILogger

if TYPE_CHECKING:
    from invokeai.app.services.invocation_services import InvocationServices

logger = InvokeAILogger.get_logger()


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
    - `Deprecated`: The invocation is deprecated and may be removed in a future version.
    - `Internal`: The invocation is not intended for use by end-users. It may be changed or removed at any time, but is exposed for users to play with.
    - `Special`: The invocation is a special case and does not fit into any of the other classifications.
    """

    Stable = "stable"
    Beta = "beta"
    Prototype = "prototype"
    Deprecated = "deprecated"
    Internal = "internal"
    Special = "special"


class Bottleneck(str, Enum, metaclass=MetaEnum):
    """
    The bottleneck of an invocation.
    - `Network`: The invocation's execution is network-bound.
    - `GPU`: The invocation's execution is GPU-bound.
    """

    Network = "network"
    GPU = "gpu"


class UIConfigBase(BaseModel):
    """
    Provides additional node configuration to the UI.
    This is used internally by the @invocation decorator logic. Do not use this directly.
    """

    tags: Optional[list[str]] = Field(default=None, description="The node's tags")
    title: Optional[str] = Field(default=None, description="The node's display name")
    category: Optional[str] = Field(default=None, description="The node's category")
    version: str = Field(
        description='The node\'s version. Should be a valid semver string e.g. "1.0.0" or "3.8.13".',
    )
    node_pack: str = Field(description="The node pack that this node belongs to, will be 'invokeai' for built-in nodes")
    classification: Classification = Field(default=Classification.Stable, description="The node's classification")

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_serialization_defaults_required=True,
    )


class OriginalModelField(TypedDict):
    annotation: Any
    field_info: FieldInfo


class BaseInvocationOutput(BaseModel):
    """
    Base class for all invocation outputs.

    All invocation outputs must use the `@invocation_output` decorator to provide their unique type.
    """

    output_meta: Optional[dict[str, JsonValue]] = Field(
        default=None,
        description="Optional dictionary of metadata for the invocation output, unrelated to the invocation's actual output value. This is not exposed as an output field.",
        json_schema_extra={"field_kind": FieldKind.NodeAttribute},
    )

    @staticmethod
    def json_schema_extra(schema: dict[str, Any], model_class: Type[BaseInvocationOutput]) -> None:
        """Adds various UI-facing attributes to the invocation output's OpenAPI schema."""
        # Because we use a pydantic Literal field with default value for the invocation type,
        # it will be typed as optional in the OpenAPI schema. Make it required manually.
        if "required" not in schema or not isinstance(schema["required"], list):
            schema["required"] = []
        schema["class"] = "output"
        schema["required"].extend(["type"])

    @classmethod
    def get_type(cls) -> str:
        """Gets the invocation output's type, as provided by the `@invocation_output` decorator."""
        return cls.model_fields["type"].default

    _original_model_fields: ClassVar[dict[str, OriginalModelField]] = {}
    """The original model fields, before any modifications were made by the @invocation_output decorator."""

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

    @classmethod
    def get_type(cls) -> str:
        """Gets the invocation's type, as provided by the `@invocation` decorator."""
        return cls.model_fields["type"].default

    @classmethod
    def get_output_annotation(cls) -> Type[BaseInvocationOutput]:
        """Gets the invocation's output annotation (i.e. the return annotation of its `invoke()` method)."""
        return signature(cls.invoke).return_annotation

    @staticmethod
    def json_schema_extra(schema: dict[str, Any], model_class: Type[BaseInvocation]) -> None:
        """Adds various UI-facing attributes to the invocation's OpenAPI schema."""
        if title := model_class.UIConfig.title:
            schema["title"] = title
        if tags := model_class.UIConfig.tags:
            schema["tags"] = tags
        if category := model_class.UIConfig.category:
            schema["category"] = category
        if node_pack := model_class.UIConfig.node_pack:
            schema["node_pack"] = node_pack
        schema["classification"] = model_class.UIConfig.classification
        schema["version"] = model_class.UIConfig.version
        if "required" not in schema or not isinstance(schema["required"], list):
            schema["required"] = []
        schema["class"] = "invocation"
        schema["required"].extend(["type", "id"])

    @abstractmethod
    def invoke(self, context: InvocationContext) -> BaseInvocationOutput:
        """Invoke with provided context and return outputs."""
        pass

    def invoke_internal(self, context: InvocationContext, services: "InvocationServices") -> BaseInvocationOutput:
        """
        Internal invoke method, calls `invoke()` after some prep.
        Handles optional fields that are required to call `invoke()` and invocation cache.
        """
        for field_name, field in type(self).model_fields.items():
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
                    raise RequiredConnectionException(type(self).model_fields["type"].default, field_name)
                elif input_ == Input.Any:
                    raise MissingInputException(type(self).model_fields["type"].default, field_name)

        # skip node cache codepath if it's disabled
        if services.configuration.node_cache_size == 0:
            return self.invoke(context)

        output: BaseInvocationOutput
        if self.use_cache:
            key = services.invocation_cache.create_key(self)
            cached_value = services.invocation_cache.get(key)
            if cached_value is None:
                services.logger.debug(f'Invocation cache miss for type "{self.get_type()}": {self.id}')
                output = self.invoke(context)
                services.invocation_cache.save(key, output)
                return output
            else:
                services.logger.debug(f'Invocation cache hit for type "{self.get_type()}": {self.id}')
                return cached_value
        else:
            services.logger.debug(f'Skipping invocation cache for "{self.get_type()}": {self.id}')
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

    bottleneck: ClassVar[Bottleneck]

    UIConfig: ClassVar[UIConfigBase]

    model_config = ConfigDict(
        protected_namespaces=(),
        validate_assignment=True,
        json_schema_extra=json_schema_extra,
        json_schema_serialization_defaults_required=False,
        coerce_numbers_to_str=True,
    )

    _original_model_fields: ClassVar[dict[str, OriginalModelField]] = {}
    """The original model fields, before any modifications were made by the @invocation decorator."""


TBaseInvocation = TypeVar("TBaseInvocation", bound=BaseInvocation)


class InvocationRegistry:
    _invocation_classes: ClassVar[set[type[BaseInvocation]]] = set()
    _output_classes: ClassVar[set[type[BaseInvocationOutput]]] = set()

    @classmethod
    def register_invocation(cls, invocation: type[BaseInvocation]) -> None:
        """Registers an invocation."""

        invocation_type = invocation.get_type()
        node_pack = invocation.UIConfig.node_pack

        # Log a warning when an existing invocation is being clobbered by the one we are registering
        clobbered_invocation = InvocationRegistry.get_invocation_for_type(invocation_type)
        if clobbered_invocation is not None:
            # This should always be true - we just checked if the invocation type was in the set
            clobbered_node_pack = clobbered_invocation.UIConfig.node_pack

            if clobbered_node_pack == "invokeai":
                # The invocation being clobbered is a core invocation
                logger.warning(f'Overriding core node "{invocation_type}" with node from "{node_pack}"')
            else:
                # The invocation being clobbered is a custom invocation
                logger.warning(
                    f'Overriding node "{invocation_type}" from "{node_pack}" with node from "{clobbered_node_pack}"'
                )
            cls._invocation_classes.remove(clobbered_invocation)

        cls._invocation_classes.add(invocation)
        cls.invalidate_invocation_typeadapter()

    @classmethod
    @lru_cache(maxsize=1)
    def get_invocation_typeadapter(cls) -> TypeAdapter[Any]:
        """Gets a pydantic TypeAdapter for the union of all invocation types.

        This is used to parse serialized invocations into the correct invocation class.

        This method is cached to avoid rebuilding the TypeAdapter on every access. If the invocation allowlist or
        denylist is changed, the cache should be cleared to ensure the TypeAdapter is updated and validation respects
        the updated allowlist and denylist.

        @see https://docs.pydantic.dev/latest/concepts/type_adapter/
        """
        return TypeAdapter(Annotated[Union[tuple(cls.get_invocation_classes())], Field(discriminator="type")])

    @classmethod
    def invalidate_invocation_typeadapter(cls) -> None:
        """Invalidates the cached invocation type adapter."""
        cls.get_invocation_typeadapter.cache_clear()

    @classmethod
    def get_invocation_classes(cls) -> Iterable[type[BaseInvocation]]:
        """Gets all invocations, respecting the allowlist and denylist."""
        app_config = get_config()
        allowed_invocations: set[type[BaseInvocation]] = set()
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
    def get_invocations_map(cls) -> dict[str, type[BaseInvocation]]:
        """Gets a map of all invocation types to their invocation classes."""
        return {i.get_type(): i for i in cls.get_invocation_classes()}

    @classmethod
    def get_invocation_types(cls) -> Iterable[str]:
        """Gets all invocation types."""
        return (i.get_type() for i in cls.get_invocation_classes())

    @classmethod
    def get_invocation_for_type(cls, invocation_type: str) -> type[BaseInvocation] | None:
        """Gets the invocation class for a given invocation type."""
        return cls.get_invocations_map().get(invocation_type)

    @classmethod
    def register_output(cls, output: "type[TBaseInvocationOutput]") -> None:
        """Registers an invocation output."""
        output_type = output.get_type()

        # Log a warning when an existing invocation is being clobbered by the one we are registering
        clobbered_output = InvocationRegistry.get_output_for_type(output_type)
        if clobbered_output is not None:
            # TODO(psyche): We do not record the node pack of the output, so we cannot log it here
            logger.warning(f'Overriding invocation output "{output_type}"')
            cls._output_classes.remove(clobbered_output)

        cls._output_classes.add(output)
        cls.invalidate_output_typeadapter()

    @classmethod
    def get_output_classes(cls) -> Iterable[type[BaseInvocationOutput]]:
        """Gets all invocation outputs."""
        return cls._output_classes

    @classmethod
    def get_outputs_map(cls) -> dict[str, type[BaseInvocationOutput]]:
        """Gets a map of all output types to their output classes."""
        return {i.get_type(): i for i in cls.get_output_classes()}

    @classmethod
    @lru_cache(maxsize=1)
    def get_output_typeadapter(cls) -> TypeAdapter[Any]:
        """Gets a pydantic TypeAdapter for the union of all invocation output types.

        This is used to parse serialized invocation outputs into the correct invocation output class.

        This method is cached to avoid rebuilding the TypeAdapter on every access. If the invocation allowlist or
        denylist is changed, the cache should be cleared to ensure the TypeAdapter is updated and validation respects
        the updated allowlist and denylist.

        @see https://docs.pydantic.dev/latest/concepts/type_adapter/
        """
        return TypeAdapter(Annotated[Union[tuple(cls._output_classes)], Field(discriminator="type")])

    @classmethod
    def invalidate_output_typeadapter(cls) -> None:
        """Invalidates the cached invocation output type adapter."""
        cls.get_output_typeadapter.cache_clear()

    @classmethod
    def get_output_types(cls) -> Iterable[str]:
        """Gets all invocation output types."""
        return (i.get_type() for i in cls.get_output_classes())

    @classmethod
    def get_output_for_type(cls, output_type: str) -> type[BaseInvocationOutput] | None:
        """Gets the output class for a given output type."""
        return cls.get_outputs_map().get(output_type)


RESERVED_NODE_ATTRIBUTE_FIELD_NAMES = {
    "id",
    "is_intermediate",
    "use_cache",
    "type",
    "workflow",
    "bottleneck",
}

RESERVED_INPUT_FIELD_NAMES = {"metadata", "board"}

RESERVED_OUTPUT_FIELD_NAMES = {"type", "output_meta"}


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
            logger.warning(f'"UIType.{ui_type.split("_")[-1]}" is deprecated, ignoring')
            field.json_schema_extra.pop("ui_type")
    return None


class NoDefaultSentinel:
    pass


def validate_field_default(
    cls_name: str, field_name: str, invocation_type: str, annotation: Any, field_info: FieldInfo
) -> None:
    """Validates the default value of a field against its pydantic field definition."""

    assert isinstance(field_info.json_schema_extra, dict), "json_schema_extra is not a dict"

    # By the time we are doing this, we've already done some pydantic magic by overriding the original default value.
    # We store the original default value in the json_schema_extra dict, so we can validate it here.
    orig_default = field_info.json_schema_extra.get("orig_default", NoDefaultSentinel)

    if orig_default is NoDefaultSentinel:
        return

    # To validate the default value, we can create a temporary pydantic model with the field we are validating as its
    # only field. Then validate the default value against this temporary model.
    TempDefaultValidator = cast(BaseModel, create_model(cls_name, **{field_name: (annotation, field_info)}))

    try:
        TempDefaultValidator.model_validate({field_name: orig_default})
    except Exception as e:
        raise InvalidFieldError(
            f'Default value for field "{field_name}" on invocation "{invocation_type}" is invalid, {e}'
        ) from e


def is_optional(annotation: Any) -> bool:
    """
    Checks if the given annotation is optional (i.e. Optional[X], Union[X, None] or X | None).
    """
    origin = typing.get_origin(annotation)
    # PEP 604 unions (int|None) have origin types.UnionType
    is_union = origin is typing.Union or origin is types.UnionType
    if not is_union:
        return False
    return any(arg is type(None) for arg in typing.get_args(annotation))


def invocation(
    invocation_type: str,
    title: Optional[str] = None,
    tags: Optional[list[str]] = None,
    category: Optional[str] = None,
    version: Optional[str] = None,
    use_cache: Optional[bool] = True,
    classification: Classification = Classification.Stable,
    bottleneck: Bottleneck = Bottleneck.GPU,
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
    :param Bottleneck bottleneck: The bottleneck of the invocation. Defaults to Bottleneck.GPU. Use Network if the invocation is network-bound.
    """

    def wrapper(cls: Type[TBaseInvocation]) -> Type[TBaseInvocation]:
        # Validate invocation types on creation of invocation classes
        # TODO: ensure unique?
        if re.compile(r"^\S+$").match(invocation_type) is None:
            raise ValueError(f'"invocation_type" must consist of non-whitespace characters, got "{invocation_type}"')

        # The node pack is the module name - will be "invokeai" for built-in nodes
        node_pack = cls.__module__.split(".")[0]

        validate_fields(cls.model_fields, invocation_type)

        fields: dict[str, tuple[Any, FieldInfo]] = {}

        original_model_fields: dict[str, OriginalModelField] = {}

        for field_name, field_info in cls.model_fields.items():
            annotation = field_info.annotation
            assert annotation is not None, f"{field_name} on invocation {invocation_type} has no type annotation."
            assert isinstance(field_info.json_schema_extra, dict), (
                f"{field_name} on invocation {invocation_type} has a non-dict json_schema_extra, did you forget to use InputField?"
            )

            original_model_fields[field_name] = OriginalModelField(annotation=annotation, field_info=field_info)

            validate_field_default(cls.__name__, field_name, invocation_type, annotation, field_info)

            if field_info.default is None and not is_optional(annotation):
                annotation = annotation | None

            fields[field_name] = (annotation, field_info)

        # Add OpenAPI schema extras
        uiconfig: dict[str, Any] = {}
        uiconfig["title"] = title
        uiconfig["tags"] = tags
        uiconfig["category"] = category
        uiconfig["classification"] = classification
        uiconfig["node_pack"] = node_pack

        if version is not None:
            try:
                semver.Version.parse(version)
            except ValueError as e:
                raise InvalidVersionError(f'Invalid version string for node "{invocation_type}": "{version}"') from e
            uiconfig["version"] = version
        else:
            logger.warning(f'No version specified for node "{invocation_type}", using "1.0.0"')
            uiconfig["version"] = "1.0.0"

        cls.UIConfig = UIConfigBase(**uiconfig)

        if use_cache is not None:
            cls.model_fields["use_cache"].default = use_cache

        cls.bottleneck = bottleneck

        # Add the invocation type to the model.

        # You'd be tempted to just add the type field and rebuild the model, like this:
        # cls.model_fields.update(type=FieldInfo.from_annotated_attribute(Literal[invocation_type], invocation_type))
        # cls.model_rebuild() or cls.model_rebuild(force=True)

        # Unfortunately, because the `GraphInvocation` uses a forward ref in its `graph` field's annotation, this does
        # not work. Instead, we have to create a new class with the type field and patch the original class with it.

        invocation_type_annotation = Literal[invocation_type]

        # Field() returns an instance of FieldInfo, but thanks to a pydantic implementation detail, it is _typed_ as Any.
        # This cast makes the type annotation match the class's true type.
        invocation_type_field_info = cast(
            FieldInfo,
            Field(title="type", default=invocation_type, json_schema_extra={"field_kind": FieldKind.NodeAttribute}),
        )

        fields["type"] = (invocation_type_annotation, invocation_type_field_info)

        # Invocation outputs must be registered using the @invocation_output decorator, but it is possible that the
        # output is registered _after_ this invocation is registered. It depends on module import ordering.
        #
        # We can only confirm the output for an invocation is registered after all modules are imported. There's
        # only really one good time to do that - during application startup, in `run_app.py`, after loading all
        # custom nodes.
        #
        # We can still do some basic validation here - ensure the invoke method is defined and returns an instance
        # of BaseInvocationOutput.

        # Validate the `invoke()` method is implemented
        if "invoke" in cls.__abstractmethods__:
            raise ValueError(f'Invocation "{invocation_type}" must implement the "invoke" method')

        # And validate that `invoke()` returns a subclass of `BaseInvocationOutput
        invoke_return_annotation = signature(cls.invoke).return_annotation

        try:
            # TODO(psyche): If `invoke()` is not defined, `return_annotation` ends up as the string "BaseInvocationOutput"
            # instead of the class `BaseInvocationOutput`. This may be a pydantic bug: https://github.com/pydantic/pydantic/issues/7978
            if isinstance(invoke_return_annotation, str):
                invoke_return_annotation = getattr(sys.modules[cls.__module__], invoke_return_annotation)

            assert invoke_return_annotation is not BaseInvocationOutput
            assert issubclass(invoke_return_annotation, BaseInvocationOutput)
        except Exception:
            raise ValueError(
                f'Invocation "{invocation_type}" must have a return annotation of a subclass of BaseInvocationOutput (got "{invoke_return_annotation}")'
            )

        docstring = cls.__doc__
        new_class = create_model(cls.__qualname__, __base__=cls, __module__=cls.__module__, **fields)  # type: ignore
        new_class.__doc__ = docstring
        new_class._original_model_fields = original_model_fields

        InvocationRegistry.register_invocation(new_class)

        return new_class

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

        validate_fields(cls.model_fields, output_type)

        fields: dict[str, tuple[Any, FieldInfo]] = {}

        for field_name, field_info in cls.model_fields.items():
            annotation = field_info.annotation
            assert annotation is not None, f"{field_name} on invocation output {output_type} has no type annotation."
            assert isinstance(field_info.json_schema_extra, dict), (
                f"{field_name} on invocation output {output_type} has a non-dict json_schema_extra, did you forget to use InputField?"
            )

            cls._original_model_fields[field_name] = OriginalModelField(annotation=annotation, field_info=field_info)

            if field_info.default is not PydanticUndefined and is_optional(annotation):
                annotation = annotation | None
            fields[field_name] = (annotation, field_info)

        # Add the output type to the model.
        output_type_annotation = Literal[output_type]

        # Field() returns an instance of FieldInfo, but thanks to a pydantic implementation detail, it is _typed_ as Any.
        # This cast makes the type annotation match the class's true type.
        output_type_field_info = cast(
            FieldInfo,
            Field(title="type", default=output_type, json_schema_extra={"field_kind": FieldKind.NodeAttribute}),
        )

        fields["type"] = (output_type_annotation, output_type_field_info)

        docstring = cls.__doc__
        new_class = create_model(cls.__qualname__, __base__=cls, __module__=cls.__module__, **fields)
        new_class.__doc__ = docstring

        InvocationRegistry.register_output(new_class)

        return new_class

    return wrapper
