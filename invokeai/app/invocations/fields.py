from enum import Enum
from typing import Any, Callable, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, RootModel, TypeAdapter, model_validator
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
    MainModel = "MainModelField"
    FluxMainModel = "FluxMainModelField"
    SD3MainModel = "SD3MainModelField"
    SDXLMainModel = "SDXLMainModelField"
    SDXLRefinerModel = "SDXLRefinerModelField"
    ONNXModel = "ONNXModelField"
    VAEModel = "VAEModelField"
    FluxVAEModel = "FluxVAEModelField"
    LoRAModel = "LoRAModelField"
    ControlNetModel = "ControlNetModelField"
    IPAdapterModel = "IPAdapterModelField"
    T2IAdapterModel = "T2IAdapterModelField"
    T5EncoderModel = "T5EncoderModelField"
    CLIPEmbedModel = "CLIPEmbedModelField"
    CLIPLEmbedModel = "CLIPLEmbedModelField"
    CLIPGEmbedModel = "CLIPGEmbedModelField"
    SpandrelImageToImageModel = "SpandrelImageToImageModelField"
    ControlLoRAModel = "ControlLoRAModelField"
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
    t5_encoder = "T5 tokenizer and text encoder"
    clip_embed_model = "CLIP Embed loader"
    clip_g_model = "CLIP-G Embed loader"
    unet = "UNet (scheduler, LoRAs)"
    transformer = "Transformer"
    mmditx = "MMDiTX"
    vae = "VAE"
    cond = "Conditioning tensor"
    controlnet_model = "ControlNet model to load"
    vae_model = "VAE model to load"
    lora_model = "LoRA model to load"
    control_lora_model = "Control LoRA model to load"
    main_model = "Main model (UNet, VAE, CLIP) to load"
    flux_model = "Flux model (Transformer) to load"
    sd3_model = "SD3 model (MMDiTX) to load"
    sdxl_main_model = "SDXL Main model (UNet, VAE, CLIP1, CLIP2) to load"
    sdxl_refiner_model = "SDXL Refiner Main Modde (UNet, VAE, CLIP2) to load"
    onnx_main_model = "ONNX Main model (UNet, VAE, CLIP) to load"
    spandrel_image_to_image_model = "Image-to-Image model"
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
    vae_tile_size = "The tile size for VAE tiling in pixels (image space). If set to 0, the default tile size for the model will be used. Larger tile sizes generally produce better results at the cost of higher memory usage."
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
    denoise_mask = "A mask of the region to apply the denoising process to. Values of 0.0 represent the regions to be fully denoised, and 1.0 represent the regions to be preserved."
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
    instantx_control_mode = "The control mode for InstantX ControlNet union models. Ignored for other ControlNet models. The standard mapping is: canny (0), tile (1), depth (2), blur (3), pose (4), gray (5), low quality (6). Negative values will be treated as 'None'."


class ImageField(BaseModel):
    """An image primitive field"""

    image_name: str = Field(description="The name of the image")


class BoardField(BaseModel):
    """A board primitive field"""

    board_id: str = Field(description="The id of the board")


class DenoiseMaskField(BaseModel):
    """An inpaint mask field"""

    mask_name: str = Field(description="The name of the mask image")
    masked_latents_name: Optional[str] = Field(default=None, description="The name of the masked image latents")
    gradient: bool = Field(default=False, description="Used for gradient inpainting")


class TensorField(BaseModel):
    """A tensor primitive field."""

    tensor_name: str = Field(description="The name of a tensor.")


class LatentsField(BaseModel):
    """A latents tensor primitive field"""

    latents_name: str = Field(description="The name of the latents")
    seed: Optional[int] = Field(default=None, description="Seed used to generate this latents")


class ColorField(BaseModel):
    """A color primitive field"""

    r: int = Field(ge=0, le=255, description="The red component")
    g: int = Field(ge=0, le=255, description="The green component")
    b: int = Field(ge=0, le=255, description="The blue component")
    a: int = Field(ge=0, le=255, description="The alpha component")

    def tuple(self) -> Tuple[int, int, int, int]:
        return (self.r, self.g, self.b, self.a)


class FluxConditioningField(BaseModel):
    """A conditioning tensor primitive value"""

    conditioning_name: str = Field(description="The name of conditioning tensor")
    mask: Optional[TensorField] = Field(
        default=None,
        description="The mask associated with this conditioning tensor. Excluded regions should be set to False, "
        "included regions should be set to True.",
    )


class SD3ConditioningField(BaseModel):
    """A conditioning tensor primitive value"""

    conditioning_name: str = Field(description="The name of conditioning tensor")


class ConditioningField(BaseModel):
    """A conditioning tensor primitive value"""

    conditioning_name: str = Field(description="The name of conditioning tensor")
    mask: Optional[TensorField] = Field(
        default=None,
        description="The mask associated with this conditioning tensor. Excluded regions should be set to False, "
        "included regions should be set to True.",
    )


class BoundingBoxField(BaseModel):
    """A bounding box primitive value."""

    x_min: int = Field(ge=0, description="The minimum x-coordinate of the bounding box (inclusive).")
    x_max: int = Field(ge=0, description="The maximum x-coordinate of the bounding box (exclusive).")
    y_min: int = Field(ge=0, description="The minimum y-coordinate of the bounding box (inclusive).")
    y_max: int = Field(ge=0, description="The maximum y-coordinate of the bounding box (exclusive).")

    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="The score associated with the bounding box. In the range [0, 1]. This value is typically set "
        "when the bounding box was produced by a detector and has an associated confidence score.",
    )

    @model_validator(mode="after")
    def check_coords(self):
        if self.x_min > self.x_max:
            raise ValueError(f"x_min ({self.x_min}) is greater than x_max ({self.x_max}).")
        if self.y_min > self.y_max:
            raise ValueError(f"y_min ({self.y_min}) is greater than y_max ({self.y_max}).")
        return self

    def tuple(self) -> Tuple[int, int, int, int]:
        """
        Returns the bounding box as a tuple suitable for use with PIL's `Image.crop()` method.
        This method returns a tuple of the form (left, upper, right, lower) == (x_min, y_min, x_max, y_max).
        """
        return (self.x_min, self.y_min, self.x_max, self.y_max)


class MetadataField(RootModel[dict[str, Any]]):
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
    """
    Inherit from this class if your node needs a metadata input field.
    """

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


class WithBoard(BaseModel):
    """
    Inherit from this class if your node needs a board input field.
    """

    board: Optional[BoardField] = Field(
        default=None,
        description=FieldDescriptions.board,
        json_schema_extra=InputFieldJSONSchemaExtra(
            field_kind=FieldKind.Internal,
            input=Input.Direct,
            orig_required=False,
        ).model_dump(exclude_none=True),
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
