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
from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from invokeai.app.invocations.fields import (
    FieldKind,
    Input,
)
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.shared.invocation_context import InvocationContext
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

    def invoke_internal(self, context: InvocationContext, services: "InvocationServices") -> BaseInvocationOutput:
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

RESERVED_INPUT_FIELD_NAMES = {"metadata", "board"}

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
