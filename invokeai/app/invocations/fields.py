from enum import Enum
from typing import Any, Callable, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, RootModel, TypeAdapter
from pydantic.fields import _Unset
from pydantic_core import PydanticUndefined

from invokeai.app.util.metaenum import MetaEnum
from invokeai.backend.image_util.segment_anything.shared import BoundingBox
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ClipVariantType,
    ModelFormat,
    ModelType,
    ModelVariantType,
)
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

    # region Misc Field Types
    Scheduler = "SchedulerField"
    Any = "AnyField"
    # endregion

    # region Internal Field Types
    _Collection = "CollectionField"
    _CollectionItem = "CollectionItemField"
    _IsIntermediate = "IsIntermediate"
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
    BoardField = "DEPRECATED_BoardField"
    MetadataItem = "DEPRECATED_MetadataItem"
    MetadataItemCollection = "DEPRECATED_MetadataItemCollection"
    MetadataItemPolymorphic = "DEPRECATED_MetadataItemPolymorphic"
    MetadataDict = "DEPRECATED_MetadataDict"

    # Deprecated Model Field Types - use ui_model_[base|type|variant|format] instead
    MainModel = "DEPRECATED_MainModelField"
    CogView4MainModel = "DEPRECATED_CogView4MainModelField"
    FluxMainModel = "DEPRECATED_FluxMainModelField"
    SD3MainModel = "DEPRECATED_SD3MainModelField"
    SDXLMainModel = "DEPRECATED_SDXLMainModelField"
    SDXLRefinerModel = "DEPRECATED_SDXLRefinerModelField"
    ONNXModel = "DEPRECATED_ONNXModelField"
    VAEModel = "DEPRECATED_VAEModelField"
    FluxVAEModel = "DEPRECATED_FluxVAEModelField"
    LoRAModel = "DEPRECATED_LoRAModelField"
    ControlNetModel = "DEPRECATED_ControlNetModelField"
    IPAdapterModel = "DEPRECATED_IPAdapterModelField"
    T2IAdapterModel = "DEPRECATED_T2IAdapterModelField"
    T5EncoderModel = "DEPRECATED_T5EncoderModelField"
    CLIPEmbedModel = "DEPRECATED_CLIPEmbedModelField"
    CLIPLEmbedModel = "DEPRECATED_CLIPLEmbedModelField"
    CLIPGEmbedModel = "DEPRECATED_CLIPGEmbedModelField"
    SpandrelImageToImageModel = "DEPRECATED_SpandrelImageToImageModelField"
    ControlLoRAModel = "DEPRECATED_ControlLoRAModelField"
    SigLipModel = "DEPRECATED_SigLipModelField"
    FluxReduxModel = "DEPRECATED_FluxReduxModelField"
    LlavaOnevisionModel = "DEPRECATED_LLaVAModelField"
    Imagen3Model = "DEPRECATED_Imagen3ModelField"
    Imagen4Model = "DEPRECATED_Imagen4ModelField"
    ChatGPT4oModel = "DEPRECATED_ChatGPT4oModelField"
    Gemini2_5Model = "DEPRECATED_Gemini2_5ModelField"
    FluxKontextModel = "DEPRECATED_FluxKontextModelField"
    Veo3Model = "DEPRECATED_Veo3ModelField"
    RunwayModel = "DEPRECATED_RunwayModelField"
    # endregion


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
    glm_encoder = "GLM (THUDM) tokenizer and text encoder"
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
    cogview4_model = "CogView4 model (Transformer) to load"
    sdxl_main_model = "SDXL Main model (UNet, VAE, CLIP1, CLIP2) to load"
    sdxl_refiner_model = "SDXL Refiner Main Modde (UNet, VAE, CLIP2) to load"
    onnx_main_model = "ONNX Main model (UNet, VAE, CLIP) to load"
    spandrel_image_to_image_model = "Image-to-Image model"
    vllm_model = "VLLM model"
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
    flux_redux_conditioning = "FLUX Redux conditioning tensor"
    vllm_model = "The VLLM model to use"
    flux_fill_conditioning = "FLUX Fill conditioning tensor"
    flux_kontext_conditioning = "FLUX Kontext conditioning (reference image)"


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


class FluxReduxConditioningField(BaseModel):
    """A FLUX Redux conditioning tensor primitive value"""

    conditioning: TensorField = Field(description="The Redux image conditioning tensor.")
    mask: Optional[TensorField] = Field(
        default=None,
        description="The mask associated with this conditioning tensor. Excluded regions should be set to False, "
        "included regions should be set to True.",
    )


class FluxFillConditioningField(BaseModel):
    """A FLUX Fill conditioning field."""

    image: ImageField = Field(description="The FLUX Fill reference image.")
    mask: TensorField = Field(description="The FLUX Fill inpaint mask.")


class FluxKontextConditioningField(BaseModel):
    """A conditioning field for FLUX Kontext (reference image)."""

    image: ImageField = Field(description="The Kontext reference image.")


class SD3ConditioningField(BaseModel):
    """A conditioning tensor primitive value"""

    conditioning_name: str = Field(description="The name of conditioning tensor")


class CogView4ConditioningField(BaseModel):
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


class BoundingBoxField(BoundingBox):
    """A bounding box primitive value."""

    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="The score associated with the bounding box. In the range [0, 1]. This value is typically set "
        "when the bounding box was produced by a detector and has an associated confidence score.",
    )


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
    field_kind: FieldKind
    orig_required: bool = True
    default: Optional[Any] = None
    orig_default: Optional[Any] = None
    ui_hidden: bool = False
    ui_type: Optional[UIType] = None
    ui_component: Optional[UIComponent] = None
    ui_order: Optional[int] = None
    ui_choice_labels: Optional[dict[str, str]] = None
    ui_model_base: Optional[list[BaseModelType]] = None
    ui_model_type: Optional[list[ModelType]] = None
    ui_model_variant: Optional[list[ClipVariantType | ModelVariantType]] = None
    ui_model_format: Optional[list[ModelFormat]] = None

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_serialization_defaults_required=True,
        use_enum_values=True,
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
        logger.warning(
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
    ui_hidden: bool = False
    ui_order: Optional[int] = None
    ui_type: Optional[UIType] = None

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_serialization_defaults_required=True,
        use_enum_values=True,
    )


def migrate_model_ui_type(ui_type: UIType | str, json_schema_extra: dict[str, Any]) -> bool:
    """Migrate deprecated model-specifier ui_type values to new-style ui_model_[base|type|variant|format] in json_schema_extra."""
    if not isinstance(ui_type, UIType):
        ui_type = UIType(ui_type)

    ui_model_type: list[ModelType] | None = None
    ui_model_base: list[BaseModelType] | None = None
    ui_model_format: list[ModelFormat] | None = None
    ui_model_variant: list[ClipVariantType | ModelVariantType] | None = None

    match ui_type:
        case UIType.MainModel:
            ui_model_base = [BaseModelType.StableDiffusion1, BaseModelType.StableDiffusion2]
            ui_model_type = [ModelType.Main]
        case UIType.CogView4MainModel:
            ui_model_base = [BaseModelType.CogView4]
            ui_model_type = [ModelType.Main]
        case UIType.FluxMainModel:
            ui_model_base = [BaseModelType.Flux]
            ui_model_type = [ModelType.Main]
        case UIType.SD3MainModel:
            ui_model_base = [BaseModelType.StableDiffusion3]
            ui_model_type = [ModelType.Main]
        case UIType.SDXLMainModel:
            ui_model_base = [BaseModelType.StableDiffusionXL]
            ui_model_type = [ModelType.Main]
        case UIType.SDXLRefinerModel:
            ui_model_base = [BaseModelType.StableDiffusionXLRefiner]
            ui_model_type = [ModelType.Main]
        case UIType.VAEModel:
            ui_model_type = [ModelType.VAE]
        case UIType.FluxVAEModel:
            ui_model_base = [BaseModelType.Flux]
            ui_model_type = [ModelType.VAE]
        case UIType.LoRAModel:
            ui_model_type = [ModelType.LoRA]
        case UIType.ControlNetModel:
            ui_model_type = [ModelType.ControlNet]
        case UIType.IPAdapterModel:
            ui_model_type = [ModelType.IPAdapter]
        case UIType.T2IAdapterModel:
            ui_model_type = [ModelType.T2IAdapter]
        case UIType.T5EncoderModel:
            ui_model_type = [ModelType.T5Encoder]
        case UIType.CLIPEmbedModel:
            ui_model_type = [ModelType.CLIPEmbed]
        case UIType.CLIPLEmbedModel:
            ui_model_type = [ModelType.CLIPEmbed]
            ui_model_variant = [ClipVariantType.L]
        case UIType.CLIPGEmbedModel:
            ui_model_type = [ModelType.CLIPEmbed]
            ui_model_variant = [ClipVariantType.G]
        case UIType.SpandrelImageToImageModel:
            ui_model_type = [ModelType.SpandrelImageToImage]
        case UIType.ControlLoRAModel:
            ui_model_type = [ModelType.ControlLoRa]
        case UIType.SigLipModel:
            ui_model_type = [ModelType.SigLIP]
        case UIType.FluxReduxModel:
            ui_model_type = [ModelType.FluxRedux]
        case UIType.LlavaOnevisionModel:
            ui_model_type = [ModelType.LlavaOnevision]
        case _:
            pass

    did_migrate = False

    if ui_model_type is not None:
        json_schema_extra["ui_model_type"] = [m.value for m in ui_model_type]
        did_migrate = True
    if ui_model_base is not None:
        json_schema_extra["ui_model_base"] = [m.value for m in ui_model_base]
        did_migrate = True
    if ui_model_format is not None:
        json_schema_extra["ui_model_format"] = [m.value for m in ui_model_format]
        did_migrate = True
    if ui_model_variant is not None:
        json_schema_extra["ui_model_variant"] = [m.value for m in ui_model_variant]
        did_migrate = True

    return did_migrate


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
    ui_hidden: Optional[bool] = None,
    ui_order: Optional[int] = None,
    ui_choice_labels: Optional[dict[str, str]] = None,
    ui_model_base: Optional[BaseModelType | list[BaseModelType]] = None,
    ui_model_type: Optional[ModelType | list[ModelType]] = None,
    ui_model_variant: Optional[ClipVariantType | ModelVariantType | list[ClipVariantType | ModelVariantType]] = None,
    ui_model_format: Optional[ModelFormat | list[ModelFormat]] = None,
) -> Any:
    """
    Creates an input field for an invocation.

    This is a wrapper for Pydantic's [Field](https://docs.pydantic.dev/latest/api/fields/#pydantic.fields.Field)
    that adds a few extra parameters to support graph execution and the node editor UI.

    If the field is a `ModelIdentifierField`, use the `ui_model_[base|type|variant|format]` args to filter the model list
    in the Workflow Editor. Otherwise, use `ui_type` to provide extra type hints for the UI.

    Don't use both `ui_type` and `ui_model_[base|type|variant|format]` - if both are provided, a warning will be
    logged and `ui_type` will be ignored.

    Args:
        input: The kind of input this field requires.
        - `Input.Direct` means a value must be provided on instantiation.
        - `Input.Connection` means the value must be provided by a connection.
        - `Input.Any` means either will do.

        ui_type: Optionally provides an extra type hint for the UI. In some situations, the field's type is not enough
        to infer the correct UI type. For example, Scheduler fields are enums, but we want to render a special scheduler
        dropdown in the UI. Use `UIType.Scheduler` to indicate this.

        ui_component: Optionally specifies a specific component to use in the UI. The UI will always render a suitable
        component, but sometimes you want something different than the default. For example, a `string` field will
        default to a single-line input, but you may want a multi-line textarea instead. In this case, you could use
        `UIComponent.Textarea`.

        ui_hidden: Specifies whether or not this field should be hidden in the UI.

        ui_order: Specifies the order in which this field should be rendered in the UI. If omitted, the field will be
        rendered after all fields with an explicit order, in the order they are defined in the Invocation class.

        ui_model_base: Specifies the base model architectures to filter the model list by in the Workflow Editor. For
        example, `ui_model_base=BaseModelType.StableDiffusionXL` will show only SDXL architecture models. This arg is
        only valid if this Input field is annotated as a `ModelIdentifierField`.

        ui_model_type: Specifies the model type(s) to filter the model list by in the Workflow Editor. For example,
        `ui_model_type=ModelType.VAE` will show only VAE models. This arg is only valid if this Input field is
        annotated as a `ModelIdentifierField`.

        ui_model_variant: Specifies the model variant(s) to filter the model list by in the Workflow Editor. For example,
        `ui_model_variant=ModelVariantType.Inpainting` will show only inpainting models. This arg is only valid if this
        Input field is annotated as a `ModelIdentifierField`.

        ui_model_format: Specifies the model format(s) to filter the model list by in the Workflow Editor. For example,
        `ui_model_format=ModelFormat.Diffusers` will show only models in the diffusers format. This arg is only valid
        if this Input field is annotated as a `ModelIdentifierField`.

        ui_choice_labels: Specifies the labels to use for the choices in an enum field. If omitted, the enum values
        will be used. This arg is only valid if the field is annotated with as a `Literal`. For example,
        `Literal["choice1", "choice2", "choice3"]` with `ui_choice_labels={"choice1": "Choice 1", "choice2": "Choice 2",
        "choice3": "Choice 3"}` will render a dropdown with the labels "Choice 1", "Choice 2" and "Choice 3".
    """

    json_schema_extra_ = InputFieldJSONSchemaExtra(
        input=input,
        field_kind=FieldKind.Input,
    )

    if ui_component is not None:
        json_schema_extra_.ui_component = ui_component
    if ui_hidden is not None:
        json_schema_extra_.ui_hidden = ui_hidden
    if ui_order is not None:
        json_schema_extra_.ui_order = ui_order
    if ui_choice_labels is not None:
        json_schema_extra_.ui_choice_labels = ui_choice_labels
    if ui_model_base is not None:
        if isinstance(ui_model_base, list):
            json_schema_extra_.ui_model_base = ui_model_base
        else:
            json_schema_extra_.ui_model_base = [ui_model_base]
    if ui_model_type is not None:
        if isinstance(ui_model_type, list):
            json_schema_extra_.ui_model_type = ui_model_type
        else:
            json_schema_extra_.ui_model_type = [ui_model_type]
    if ui_model_variant is not None:
        if isinstance(ui_model_variant, list):
            json_schema_extra_.ui_model_variant = ui_model_variant
        else:
            json_schema_extra_.ui_model_variant = [ui_model_variant]
    if ui_model_format is not None:
        if isinstance(ui_model_format, list):
            json_schema_extra_.ui_model_format = ui_model_format
        else:
            json_schema_extra_.ui_model_format = [ui_model_format]
    if ui_type is not None:
        json_schema_extra_.ui_type = ui_type

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
        logger.warning('"default_factory" is not supported, calling it now to set "default"')

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
        json_schema_extra=json_schema_extra_.model_dump(exclude_unset=True),
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

    This is a wrapper for Pydantic's [Field](https://docs.pydantic.dev/1.10/usage/schema/#field-customization)
    that adds a few extra parameters to support graph execution and the node editor UI.

    Args:
        ui_type: Optionally provides an extra type hint for the UI. In some situations, the field's type is not enough
        to infer the correct UI type. For example, Scheduler fields are enums, but we want to render a special scheduler
        dropdown in the UI. Use `UIType.Scheduler` to indicate this.

        ui_hidden: Specifies whether or not this field should be hidden in the UI.

        ui_order: Specifies the order in which this field should be rendered in the UI. If omitted, the field will be
        rendered after all fields with an explicit order, in the order they are defined in the Invocation class.
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
            ui_hidden=ui_hidden,
            ui_order=ui_order,
            ui_type=ui_type,
            field_kind=FieldKind.Output,
        ).model_dump(exclude_none=True),
    )
