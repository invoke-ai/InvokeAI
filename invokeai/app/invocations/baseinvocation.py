# Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from enum import Enum
from inspect import signature
import re
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Callable,
    ClassVar,
    Literal,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_type_hints,
)

from pydantic import BaseModel, Field, validator
from pydantic.fields import Undefined, ModelField
from pydantic.typing import NoArgAnyCallable
import semver

if TYPE_CHECKING:
    from ..services.invocation_services import InvocationServices


class InvalidVersionError(ValueError):
    pass


class FieldDescriptions:
    denoising_start = "When to start denoising, expressed a percentage of total steps"
    denoising_end = "When to stop denoising, expressed a percentage of total steps"
    cfg_scale = "Classifier-Free Guidance scale"
    scheduler = "Scheduler to use during inference"
    positive_cond = "Positive conditioning tensor"
    negative_cond = "Negative conditioning tensor"
    noise = "Noise tensor"
    clip = "CLIP (tokenizer, text encoder, LoRAs) and skipped layer count"
    unet = "UNet (scheduler, LoRAs)"
    vae = "VAE"
    cond = "Conditioning tensor"
    controlnet_model = "ControlNet model to load"
    vae_model = "VAE model to load"
    lora_model = "LoRA model to load"
    main_model = "Main model (UNet, VAE, CLIP) to load"
    sdxl_main_model = "SDXL Main model (UNet, VAE, CLIP1, CLIP2) to load"
    sdxl_refiner_model = "SDXL Refiner Main Modde (UNet, VAE, CLIP2) to load"
    onnx_main_model = "ONNX Main model (UNet, VAE, CLIP) to load"
    lora_weight = "The weight at which the LoRA is applied to each model"
    compel_prompt = "Prompt to be parsed by Compel to create a conditioning tensor"
    raw_prompt = "Raw prompt text (no parsing)"
    sdxl_aesthetic = "The aesthetic score to apply to the conditioning tensor"
    skipped_layers = "Number of layers to skip in text encoder"
    seed = "Seed for random number generation"
    steps = "Number of steps to run"
    width = "Width of output (px)"
    height = "Height of output (px)"
    control = "ControlNet(s) to apply"
    denoised_latents = "Denoised latents tensor"
    latents = "Latents tensor"
    strength = "Strength of denoising (proportional to steps)"
    core_metadata = "Optional core metadata to be written to image"
    interp_mode = "Interpolation mode"
    torch_antialias = "Whether or not to apply antialiasing (bilinear or bicubic only)"
    fp32 = "Whether or not to use full float32 precision"
    precision = "Precision to use"
    tiled = "Processing using overlapping tiles (reduce memory consumption)"
    detect_res = "Pixel resolution for detection"
    image_res = "Pixel resolution for output image"
    safe_mode = "Whether or not to use safe mode"
    scribble_mode = "Whether or not to use scribble mode"
    scale_factor = "The factor by which to scale"
    blend_alpha = (
        "Blending factor. 0.0 = use input A only, 1.0 = use input B only, 0.5 = 50% mix of input A and input B."
    )
    num_1 = "The first number"
    num_2 = "The second number"
    mask = "The mask to use for the operation"


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
    MetadataField = "MetadataField"
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
    item_default: Optional[Any]


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


def InputField(
    *args: Any,
    default: Any = Undefined,
    default_factory: Optional[NoArgAnyCallable] = None,
    alias: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    exclude: Optional[Union[AbstractSet[Union[int, str]], Mapping[Union[int, str], Any], Any]] = None,
    include: Optional[Union[AbstractSet[Union[int, str]], Mapping[Union[int, str], Any], Any]] = None,
    const: Optional[bool] = None,
    gt: Optional[float] = None,
    ge: Optional[float] = None,
    lt: Optional[float] = None,
    le: Optional[float] = None,
    multiple_of: Optional[float] = None,
    allow_inf_nan: Optional[bool] = None,
    max_digits: Optional[int] = None,
    decimal_places: Optional[int] = None,
    min_items: Optional[int] = None,
    max_items: Optional[int] = None,
    unique_items: Optional[bool] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allow_mutation: bool = True,
    regex: Optional[str] = None,
    discriminator: Optional[str] = None,
    repr: bool = True,
    input: Input = Input.Any,
    ui_type: Optional[UIType] = None,
    ui_component: Optional[UIComponent] = None,
    ui_hidden: bool = False,
    ui_order: Optional[int] = None,
    item_default: Optional[Any] = None,
    **kwargs: Any,
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
      Ignored for non-collection fields..
    """
    return Field(
        *args,
        default=default,
        default_factory=default_factory,
        alias=alias,
        title=title,
        description=description,
        exclude=exclude,
        include=include,
        const=const,
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        multiple_of=multiple_of,
        allow_inf_nan=allow_inf_nan,
        max_digits=max_digits,
        decimal_places=decimal_places,
        min_items=min_items,
        max_items=max_items,
        unique_items=unique_items,
        min_length=min_length,
        max_length=max_length,
        allow_mutation=allow_mutation,
        regex=regex,
        discriminator=discriminator,
        repr=repr,
        input=input,
        ui_type=ui_type,
        ui_component=ui_component,
        ui_hidden=ui_hidden,
        ui_order=ui_order,
        item_default=item_default,
        **kwargs,
    )


def OutputField(
    *args: Any,
    default: Any = Undefined,
    default_factory: Optional[NoArgAnyCallable] = None,
    alias: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    exclude: Optional[Union[AbstractSet[Union[int, str]], Mapping[Union[int, str], Any], Any]] = None,
    include: Optional[Union[AbstractSet[Union[int, str]], Mapping[Union[int, str], Any], Any]] = None,
    const: Optional[bool] = None,
    gt: Optional[float] = None,
    ge: Optional[float] = None,
    lt: Optional[float] = None,
    le: Optional[float] = None,
    multiple_of: Optional[float] = None,
    allow_inf_nan: Optional[bool] = None,
    max_digits: Optional[int] = None,
    decimal_places: Optional[int] = None,
    min_items: Optional[int] = None,
    max_items: Optional[int] = None,
    unique_items: Optional[bool] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allow_mutation: bool = True,
    regex: Optional[str] = None,
    discriminator: Optional[str] = None,
    repr: bool = True,
    ui_type: Optional[UIType] = None,
    ui_hidden: bool = False,
    ui_order: Optional[int] = None,
    **kwargs: Any,
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
        *args,
        default=default,
        default_factory=default_factory,
        alias=alias,
        title=title,
        description=description,
        exclude=exclude,
        include=include,
        const=const,
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        multiple_of=multiple_of,
        allow_inf_nan=allow_inf_nan,
        max_digits=max_digits,
        decimal_places=decimal_places,
        min_items=min_items,
        max_items=max_items,
        unique_items=unique_items,
        min_length=min_length,
        max_length=max_length,
        allow_mutation=allow_mutation,
        regex=regex,
        discriminator=discriminator,
        repr=repr,
        ui_type=ui_type,
        ui_hidden=ui_hidden,
        ui_order=ui_order,
        **kwargs,
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
        default=None, description='The node\'s version. Should be a valid semver string e.g. "1.0.0" or "3.8.13".'
    )


class InvocationContext:
    services: InvocationServices
    graph_execution_state_id: str

    def __init__(self, services: InvocationServices, graph_execution_state_id: str):
        self.services = services
        self.graph_execution_state_id = graph_execution_state_id


class BaseInvocationOutput(BaseModel):
    """
    Base class for all invocation outputs.

    All invocation outputs must use the `@invocation_output` decorator to provide their unique type.
    """

    @classmethod
    def get_all_subclasses_tuple(cls):
        subclasses = []
        toprocess = [cls]
        while len(toprocess) > 0:
            next = toprocess.pop(0)
            next_subclasses = next.__subclasses__()
            subclasses.extend(next_subclasses)
            toprocess.extend(next_subclasses)
        return tuple(subclasses)

    class Config:
        @staticmethod
        def schema_extra(schema: dict[str, Any], model_class: Type[BaseModel]) -> None:
            if "required" not in schema or not isinstance(schema["required"], list):
                schema["required"] = list()
            schema["required"].extend(["type"])


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
    A node to process inputs and produce outputs.
    May use dependency injection in __init__ to receive providers.

    All invocations must use the `@invocation` decorator to provide their unique type.
    """

    @classmethod
    def get_all_subclasses(cls):
        subclasses = []
        toprocess = [cls]
        while len(toprocess) > 0:
            next = toprocess.pop(0)
            next_subclasses = next.__subclasses__()
            subclasses.extend(next_subclasses)
            toprocess.extend(next_subclasses)
        return subclasses

    @classmethod
    def get_invocations(cls):
        return tuple(BaseInvocation.get_all_subclasses())

    @classmethod
    def get_invocations_map(cls):
        # Get the type strings out of the literals and into a dictionary
        return dict(
            map(
                lambda t: (get_args(get_type_hints(t)["type"])[0], t),
                BaseInvocation.get_all_subclasses(),
            )
        )

    @classmethod
    def get_output_type(cls):
        return signature(cls.invoke).return_annotation

    class Config:
        @staticmethod
        def schema_extra(schema: dict[str, Any], model_class: Type[BaseModel]) -> None:
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

    def __init__(self, **data):
        # nodes may have required fields, that can accept input from connections
        # on instantiation of the model, we need to exclude these from validation
        restore = dict()
        try:
            field_names = list(self.__fields__.keys())
            for field_name in field_names:
                # if the field is required and may get its value from a connection, exclude it from validation
                field = self.__fields__[field_name]
                _input = field.field_info.extra.get("input", None)
                if _input in [Input.Connection, Input.Any] and field.required:
                    if field_name not in data:
                        restore[field_name] = self.__fields__.pop(field_name)
            # instantiate the node, which will validate the data
            super().__init__(**data)
        finally:
            # restore the removed fields
            for field_name, field in restore.items():
                self.__fields__[field_name] = field

    def invoke_internal(self, context: InvocationContext) -> BaseInvocationOutput:
        for field_name, field in self.__fields__.items():
            _input = field.field_info.extra.get("input", None)
            if field.required and not hasattr(self, field_name):
                if _input == Input.Connection:
                    raise RequiredConnectionException(self.__fields__["type"].default, field_name)
                elif _input == Input.Any:
                    raise MissingInputException(self.__fields__["type"].default, field_name)
        return self.invoke(context)

    id: str = Field(
        description="The id of this instance of an invocation. Must be unique among all instances of invocations."
    )
    is_intermediate: bool = InputField(
        default=False, description="Whether or not this is an intermediate invocation.", ui_type=UIType.IsIntermediate
    )
    workflow: Optional[str] = InputField(
        default=None,
        description="The workflow to save with the image",
        ui_type=UIType.WorkflowField,
    )

    @validator("workflow", pre=True)
    def validate_workflow_is_json(cls, v):
        if v is None:
            return None
        try:
            json.loads(v)
        except json.decoder.JSONDecodeError:
            raise ValueError("Workflow must be valid JSON")
        return v

    UIConfig: ClassVar[Type[UIConfigBase]]


GenericBaseInvocation = TypeVar("GenericBaseInvocation", bound=BaseInvocation)


def invocation(
    invocation_type: str,
    title: Optional[str] = None,
    tags: Optional[list[str]] = None,
    category: Optional[str] = None,
    version: Optional[str] = None,
) -> Callable[[Type[GenericBaseInvocation]], Type[GenericBaseInvocation]]:
    """
    Adds metadata to an invocation.

    :param str invocation_type: The type of the invocation. Must be unique among all invocations.
    :param Optional[str] title: Adds a title to the invocation. Use if the auto-generated title isn't quite right. Defaults to None.
    :param Optional[list[str]] tags: Adds tags to the invocation. Invocations may be searched for by their tags. Defaults to None.
    :param Optional[str] category: Adds a category to the invocation. Used to group the invocations in the UI. Defaults to None.
    """

    def wrapper(cls: Type[GenericBaseInvocation]) -> Type[GenericBaseInvocation]:
        # Validate invocation types on creation of invocation classes
        # TODO: ensure unique?
        if re.compile(r"^\S+$").match(invocation_type) is None:
            raise ValueError(f'"invocation_type" must consist of non-whitespace characters, got "{invocation_type}"')

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

        # Add the invocation type to the pydantic model of the invocation
        invocation_type_annotation = Literal[invocation_type]  # type: ignore
        invocation_type_field = ModelField.infer(
            name="type",
            value=invocation_type,
            annotation=invocation_type_annotation,
            class_validators=None,
            config=cls.__config__,
        )
        cls.__fields__.update({"type": invocation_type_field})
        # to support 3.9, 3.10 and 3.11, as described in https://docs.python.org/3/howto/annotations.html
        if annotations := cls.__dict__.get("__annotations__", None):
            annotations.update({"type": invocation_type_annotation})
        return cls

    return wrapper


GenericBaseInvocationOutput = TypeVar("GenericBaseInvocationOutput", bound=BaseInvocationOutput)


def invocation_output(
    output_type: str,
) -> Callable[[Type[GenericBaseInvocationOutput]], Type[GenericBaseInvocationOutput]]:
    """
    Adds metadata to an invocation output.

    :param str output_type: The type of the invocation output. Must be unique among all invocation outputs.
    """

    def wrapper(cls: Type[GenericBaseInvocationOutput]) -> Type[GenericBaseInvocationOutput]:
        # Validate output types on creation of invocation output classes
        # TODO: ensure unique?
        if re.compile(r"^\S+$").match(output_type) is None:
            raise ValueError(f'"output_type" must consist of non-whitespace characters, got "{output_type}"')

        # Add the output type to the pydantic model of the invocation output
        output_type_annotation = Literal[output_type]  # type: ignore
        output_type_field = ModelField.infer(
            name="type",
            value=output_type,
            annotation=output_type_annotation,
            class_validators=None,
            config=cls.__config__,
        )
        cls.__fields__.update({"type": output_type_field})

        # to support 3.9, 3.10 and 3.11, as described in https://docs.python.org/3/howto/annotations.html
        if annotations := cls.__dict__.get("__annotations__", None):
            annotations.update({"type": output_type_annotation})

        return cls

    return wrapper
