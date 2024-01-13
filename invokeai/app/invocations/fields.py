from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field, RootModel, TypeAdapter
from pydantic.fields import _Unset
from pydantic_core import PydanticUndefined

from invokeai.app.util.metaenum import MetaEnum
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger()


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


class UIComponent(str, Enum, metaclass=MetaEnum):
    """
    The type of UI component to use for a field, used to override the default components, which are
    inferred from the field type.
    """

    None_ = "none"
    Textarea = "textarea"
    Slider = "slider"


class FieldDescriptions:
    denoising_start = "When to start denoising, expressed a percentage of total steps"
    denoising_end = "When to stop denoising, expressed a percentage of total steps"
    cfg_scale = "Classifier-Free Guidance scale"
    cfg_rescale_multiplier = "Rescale multiplier for CFG guidance, used for models trained with zero-terminal SNR"
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
    ip_adapter = "IP-Adapter to apply"
    t2i_adapter = "T2I-Adapter(s) to apply"
    denoised_latents = "Denoised latents tensor"
    latents = "Latents tensor"
    strength = "Strength of denoising (proportional to steps)"
    metadata = "Optional metadata to be saved with the image"
    metadata_collection = "Collection of Metadata"
    metadata_item_polymorphic = "A single metadata item or collection of metadata items"
    metadata_item_label = "Label for this metadata item"
    metadata_item_value = "The value for this metadata item (may be any type)"
    workflow = "Optional workflow to be saved with the image"
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
    board = "The board to save the image to"
    image = "The image to process"
    tile_size = "Tile size"
    inclusive_low = "The inclusive low value"
    exclusive_high = "The exclusive high value"
    decimal_places = "The number of decimal places to round to"
    freeu_s1 = 'Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to mitigate the "oversmoothing effect" in the enhanced denoising process.'
    freeu_s2 = 'Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to mitigate the "oversmoothing effect" in the enhanced denoising process.'
    freeu_b1 = "Scaling factor for stage 1 to amplify the contributions of backbone features."
    freeu_b2 = "Scaling factor for stage 2 to amplify the contributions of backbone features."


class MetadataField(RootModel):
    """
    Pydantic model for metadata with custom root of type dict[str, Any].
    Metadata is stored without a strict schema.
    """

    root: dict[str, Any] = Field(description="The metadata")


MetadataFieldValidator = TypeAdapter(MetadataField)


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
    # endregion
