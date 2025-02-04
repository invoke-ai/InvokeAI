import copy
from typing import List, Optional

from pydantic import BaseModel, Field

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, Input, InputField, OutputField, UIType
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.shared.models import FreeUConfig
from invokeai.backend.model_manager.config import (
    AnyModelConfig,
    BaseModelType,
    ModelType,
    SubModelType,
)


class ModelIdentifierField(BaseModel):
    key: str = Field(description="The model's unique key")
    hash: str = Field(description="The model's BLAKE3 hash")
    name: str = Field(description="The model's name")
    base: BaseModelType = Field(description="The model's base model type")
    type: ModelType = Field(description="The model's type")
    submodel_type: Optional[SubModelType] = Field(
        description="The submodel to load, if this is a main model", default=None
    )

    @classmethod
    def from_config(
        cls, config: "AnyModelConfig", submodel_type: Optional[SubModelType] = None
    ) -> "ModelIdentifierField":
        return cls(
            key=config.key,
            hash=config.hash,
            name=config.name,
            base=config.base,
            type=config.type,
            submodel_type=submodel_type,
        )


class LoRAField(BaseModel):
    lora: ModelIdentifierField = Field(description="Info to load lora model")
    weight: float = Field(description="Weight to apply to lora model")


class UNetField(BaseModel):
    unet: ModelIdentifierField = Field(description="Info to load unet submodel")
    scheduler: ModelIdentifierField = Field(description="Info to load scheduler submodel")
    loras: List[LoRAField] = Field(description="LoRAs to apply on model loading")
    seamless_axes: List[str] = Field(default_factory=list, description='Axes("x" and "y") to which apply seamless')
    freeu_config: Optional[FreeUConfig] = Field(default=None, description="FreeU configuration")


class CLIPField(BaseModel):
    tokenizer: ModelIdentifierField = Field(description="Info to load tokenizer submodel")
    text_encoder: ModelIdentifierField = Field(description="Info to load text_encoder submodel")
    skipped_layers: int = Field(description="Number of skipped layers in text_encoder")
    loras: List[LoRAField] = Field(description="LoRAs to apply on model loading")


class T5EncoderField(BaseModel):
    tokenizer: ModelIdentifierField = Field(description="Info to load tokenizer submodel")
    text_encoder: ModelIdentifierField = Field(description="Info to load text_encoder submodel")
    loras: List[LoRAField] = Field(description="LoRAs to apply on model loading")


class VAEField(BaseModel):
    vae: ModelIdentifierField = Field(description="Info to load vae submodel")
    seamless_axes: List[str] = Field(default_factory=list, description='Axes("x" and "y") to which apply seamless')


class ControlLoRAField(LoRAField):
    img: ImageField = Field(description="Image to use in structural conditioning")


class TransformerField(BaseModel):
    transformer: ModelIdentifierField = Field(description="Info to load Transformer submodel")
    loras: List[LoRAField] = Field(description="LoRAs to apply on model loading")


@invocation_output("unet_output")
class UNetOutput(BaseInvocationOutput):
    """Base class for invocations that output a UNet field."""

    unet: UNetField = OutputField(description=FieldDescriptions.unet, title="UNet")


@invocation_output("vae_output")
class VAEOutput(BaseInvocationOutput):
    """Base class for invocations that output a VAE field"""

    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation_output("clip_output")
class CLIPOutput(BaseInvocationOutput):
    """Base class for invocations that output a CLIP field"""

    clip: CLIPField = OutputField(description=FieldDescriptions.clip, title="CLIP")


@invocation_output("model_loader_output")
class ModelLoaderOutput(UNetOutput, CLIPOutput, VAEOutput):
    """Model loader output"""

    pass


@invocation_output("model_identifier_output")
class ModelIdentifierOutput(BaseInvocationOutput):
    """Model identifier output"""

    model: ModelIdentifierField = OutputField(description="Model identifier", title="Model")


@invocation(
    "model_identifier",
    title="Model identifier",
    tags=["model"],
    category="model",
    version="1.0.0",
    classification=Classification.Prototype,
)
class ModelIdentifierInvocation(BaseInvocation):
    """Selects any model, outputting it its identifier. Be careful with this one! The identifier will be accepted as
    input for any model, even if the model types don't match. If you connect this to a mismatched input, you'll get an
    error."""

    model: ModelIdentifierField = InputField(description="The model to select", title="Model")

    def invoke(self, context: InvocationContext) -> ModelIdentifierOutput:
        if not context.models.exists(self.model.key):
            raise Exception(f"Unknown model {self.model.key}")

        return ModelIdentifierOutput(model=self.model)


@invocation(
    "main_model_loader",
    title="Main Model",
    tags=["model"],
    category="model",
    version="1.0.3",
)
class MainModelLoaderInvocation(BaseInvocation):
    """Loads a main model, outputting its submodels."""

    model: ModelIdentifierField = InputField(description=FieldDescriptions.main_model, ui_type=UIType.MainModel)
    # TODO: precision?

    def invoke(self, context: InvocationContext) -> ModelLoaderOutput:
        # TODO: not found exceptions
        if not context.models.exists(self.model.key):
            raise Exception(f"Unknown model {self.model.key}")

        unet = self.model.model_copy(update={"submodel_type": SubModelType.UNet})
        scheduler = self.model.model_copy(update={"submodel_type": SubModelType.Scheduler})
        tokenizer = self.model.model_copy(update={"submodel_type": SubModelType.Tokenizer})
        text_encoder = self.model.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        vae = self.model.model_copy(update={"submodel_type": SubModelType.VAE})

        return ModelLoaderOutput(
            unet=UNetField(unet=unet, scheduler=scheduler, loras=[]),
            clip=CLIPField(tokenizer=tokenizer, text_encoder=text_encoder, loras=[], skipped_layers=0),
            vae=VAEField(vae=vae),
        )


@invocation_output("lora_loader_output")
class LoRALoaderOutput(BaseInvocationOutput):
    """Model loader output"""

    unet: Optional[UNetField] = OutputField(default=None, description=FieldDescriptions.unet, title="UNet")
    clip: Optional[CLIPField] = OutputField(default=None, description=FieldDescriptions.clip, title="CLIP")


@invocation("lora_loader", title="LoRA", tags=["model"], category="model", version="1.0.3")
class LoRALoaderInvocation(BaseInvocation):
    """Apply selected lora to unet and text_encoder."""

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.lora_model, title="LoRA", ui_type=UIType.LoRAModel
    )
    weight: float = InputField(default=0.75, description=FieldDescriptions.lora_weight)
    unet: Optional[UNetField] = InputField(
        default=None,
        description=FieldDescriptions.unet,
        input=Input.Connection,
        title="UNet",
    )
    clip: Optional[CLIPField] = InputField(
        default=None,
        description=FieldDescriptions.clip,
        input=Input.Connection,
        title="CLIP",
    )

    def invoke(self, context: InvocationContext) -> LoRALoaderOutput:
        lora_key = self.lora.key

        if not context.models.exists(lora_key):
            raise Exception(f"Unknown lora: {lora_key}!")

        if self.unet is not None and any(lora.lora.key == lora_key for lora in self.unet.loras):
            raise Exception(f'LoRA "{lora_key}" already applied to unet')

        if self.clip is not None and any(lora.lora.key == lora_key for lora in self.clip.loras):
            raise Exception(f'LoRA "{lora_key}" already applied to clip')

        output = LoRALoaderOutput()

        if self.unet is not None:
            output.unet = self.unet.model_copy(deep=True)
            output.unet.loras.append(
                LoRAField(
                    lora=self.lora,
                    weight=self.weight,
                )
            )

        if self.clip is not None:
            output.clip = self.clip.model_copy(deep=True)
            output.clip.loras.append(
                LoRAField(
                    lora=self.lora,
                    weight=self.weight,
                )
            )

        return output


@invocation_output("lora_selector_output")
class LoRASelectorOutput(BaseInvocationOutput):
    """Model loader output"""

    lora: LoRAField = OutputField(description="LoRA model and weight", title="LoRA")


@invocation("lora_selector", title="LoRA Selector", tags=["model"], category="model", version="1.0.1")
class LoRASelectorInvocation(BaseInvocation):
    """Selects a LoRA model and weight."""

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.lora_model, title="LoRA", ui_type=UIType.LoRAModel
    )
    weight: float = InputField(default=0.75, description=FieldDescriptions.lora_weight)

    def invoke(self, context: InvocationContext) -> LoRASelectorOutput:
        return LoRASelectorOutput(lora=LoRAField(lora=self.lora, weight=self.weight))


@invocation("lora_collection_loader", title="LoRA Collection Loader", tags=["model"], category="model", version="1.1.0")
class LoRACollectionLoader(BaseInvocation):
    """Applies a collection of LoRAs to the provided UNet and CLIP models."""

    loras: Optional[LoRAField | list[LoRAField]] = InputField(
        default=None, description="LoRA models and weights. May be a single LoRA or collection.", title="LoRAs"
    )
    unet: Optional[UNetField] = InputField(
        default=None,
        description=FieldDescriptions.unet,
        input=Input.Connection,
        title="UNet",
    )
    clip: Optional[CLIPField] = InputField(
        default=None,
        description=FieldDescriptions.clip,
        input=Input.Connection,
        title="CLIP",
    )

    def invoke(self, context: InvocationContext) -> LoRALoaderOutput:
        output = LoRALoaderOutput()
        loras = self.loras if isinstance(self.loras, list) else [self.loras]
        added_loras: list[str] = []

        if self.unet is not None:
            output.unet = self.unet.model_copy(deep=True)
        if self.clip is not None:
            output.clip = self.clip.model_copy(deep=True)

        for lora in loras:
            if lora is None:
                continue
            if lora.lora.key in added_loras:
                continue

            if not context.models.exists(lora.lora.key):
                raise Exception(f"Unknown lora: {lora.lora.key}!")

            assert lora.lora.base in (BaseModelType.StableDiffusion1, BaseModelType.StableDiffusion2)

            added_loras.append(lora.lora.key)

            if self.unet is not None and output.unet is not None:
                output.unet.loras.append(lora)

            if self.clip is not None and output.clip is not None:
                output.clip.loras.append(lora)

        return output


@invocation_output("sdxl_lora_loader_output")
class SDXLLoRALoaderOutput(BaseInvocationOutput):
    """SDXL LoRA Loader Output"""

    unet: Optional[UNetField] = OutputField(default=None, description=FieldDescriptions.unet, title="UNet")
    clip: Optional[CLIPField] = OutputField(default=None, description=FieldDescriptions.clip, title="CLIP 1")
    clip2: Optional[CLIPField] = OutputField(default=None, description=FieldDescriptions.clip, title="CLIP 2")


@invocation(
    "sdxl_lora_loader",
    title="SDXL LoRA",
    tags=["lora", "model"],
    category="model",
    version="1.0.3",
)
class SDXLLoRALoaderInvocation(BaseInvocation):
    """Apply selected lora to unet and text_encoder."""

    lora: ModelIdentifierField = InputField(
        description=FieldDescriptions.lora_model, title="LoRA", ui_type=UIType.LoRAModel
    )
    weight: float = InputField(default=0.75, description=FieldDescriptions.lora_weight)
    unet: Optional[UNetField] = InputField(
        default=None,
        description=FieldDescriptions.unet,
        input=Input.Connection,
        title="UNet",
    )
    clip: Optional[CLIPField] = InputField(
        default=None,
        description=FieldDescriptions.clip,
        input=Input.Connection,
        title="CLIP 1",
    )
    clip2: Optional[CLIPField] = InputField(
        default=None,
        description=FieldDescriptions.clip,
        input=Input.Connection,
        title="CLIP 2",
    )

    def invoke(self, context: InvocationContext) -> SDXLLoRALoaderOutput:
        lora_key = self.lora.key

        if not context.models.exists(lora_key):
            raise Exception(f"Unknown lora: {lora_key}!")

        if self.unet is not None and any(lora.lora.key == lora_key for lora in self.unet.loras):
            raise Exception(f'LoRA "{lora_key}" already applied to unet')

        if self.clip is not None and any(lora.lora.key == lora_key for lora in self.clip.loras):
            raise Exception(f'LoRA "{lora_key}" already applied to clip')

        if self.clip2 is not None and any(lora.lora.key == lora_key for lora in self.clip2.loras):
            raise Exception(f'LoRA "{lora_key}" already applied to clip2')

        output = SDXLLoRALoaderOutput()

        if self.unet is not None:
            output.unet = self.unet.model_copy(deep=True)
            output.unet.loras.append(
                LoRAField(
                    lora=self.lora,
                    weight=self.weight,
                )
            )

        if self.clip is not None:
            output.clip = self.clip.model_copy(deep=True)
            output.clip.loras.append(
                LoRAField(
                    lora=self.lora,
                    weight=self.weight,
                )
            )

        if self.clip2 is not None:
            output.clip2 = self.clip2.model_copy(deep=True)
            output.clip2.loras.append(
                LoRAField(
                    lora=self.lora,
                    weight=self.weight,
                )
            )

        return output


@invocation(
    "sdxl_lora_collection_loader",
    title="SDXL LoRA Collection Loader",
    tags=["model"],
    category="model",
    version="1.1.0",
)
class SDXLLoRACollectionLoader(BaseInvocation):
    """Applies a collection of SDXL LoRAs to the provided UNet and CLIP models."""

    loras: Optional[LoRAField | list[LoRAField]] = InputField(
        default=None, description="LoRA models and weights. May be a single LoRA or collection.", title="LoRAs"
    )
    unet: Optional[UNetField] = InputField(
        default=None,
        description=FieldDescriptions.unet,
        input=Input.Connection,
        title="UNet",
    )
    clip: Optional[CLIPField] = InputField(
        default=None,
        description=FieldDescriptions.clip,
        input=Input.Connection,
        title="CLIP",
    )
    clip2: Optional[CLIPField] = InputField(
        default=None,
        description=FieldDescriptions.clip,
        input=Input.Connection,
        title="CLIP 2",
    )

    def invoke(self, context: InvocationContext) -> SDXLLoRALoaderOutput:
        output = SDXLLoRALoaderOutput()
        loras = self.loras if isinstance(self.loras, list) else [self.loras]
        added_loras: list[str] = []

        if self.unet is not None:
            output.unet = self.unet.model_copy(deep=True)

        if self.clip is not None:
            output.clip = self.clip.model_copy(deep=True)

        if self.clip2 is not None:
            output.clip2 = self.clip2.model_copy(deep=True)

        for lora in loras:
            if lora is None:
                continue
            if lora.lora.key in added_loras:
                continue

            if not context.models.exists(lora.lora.key):
                raise Exception(f"Unknown lora: {lora.lora.key}!")

            assert lora.lora.base is BaseModelType.StableDiffusionXL

            added_loras.append(lora.lora.key)

            if self.unet is not None and output.unet is not None:
                output.unet.loras.append(lora)

            if self.clip is not None and output.clip is not None:
                output.clip.loras.append(lora)

            if self.clip2 is not None and output.clip2 is not None:
                output.clip2.loras.append(lora)

        return output


@invocation("vae_loader", title="VAE", tags=["vae", "model"], category="model", version="1.0.3")
class VAELoaderInvocation(BaseInvocation):
    """Loads a VAE model, outputting a VaeLoaderOutput"""

    vae_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.vae_model, title="VAE", ui_type=UIType.VAEModel
    )

    def invoke(self, context: InvocationContext) -> VAEOutput:
        key = self.vae_model.key

        if not context.models.exists(key):
            raise Exception(f"Unknown vae: {key}!")

        return VAEOutput(vae=VAEField(vae=self.vae_model))


@invocation_output("seamless_output")
class SeamlessModeOutput(BaseInvocationOutput):
    """Modified Seamless Model output"""

    unet: Optional[UNetField] = OutputField(default=None, description=FieldDescriptions.unet, title="UNet")
    vae: Optional[VAEField] = OutputField(default=None, description=FieldDescriptions.vae, title="VAE")


@invocation(
    "seamless",
    title="Seamless",
    tags=["seamless", "model"],
    category="model",
    version="1.0.1",
)
class SeamlessModeInvocation(BaseInvocation):
    """Applies the seamless transformation to the Model UNet and VAE."""

    unet: Optional[UNetField] = InputField(
        default=None,
        description=FieldDescriptions.unet,
        input=Input.Connection,
        title="UNet",
    )
    vae: Optional[VAEField] = InputField(
        default=None,
        description=FieldDescriptions.vae_model,
        input=Input.Connection,
        title="VAE",
    )
    seamless_y: bool = InputField(default=True, input=Input.Any, description="Specify whether Y axis is seamless")
    seamless_x: bool = InputField(default=True, input=Input.Any, description="Specify whether X axis is seamless")

    def invoke(self, context: InvocationContext) -> SeamlessModeOutput:
        # Conditionally append 'x' and 'y' based on seamless_x and seamless_y
        unet = copy.deepcopy(self.unet)
        vae = copy.deepcopy(self.vae)

        seamless_axes_list = []

        if self.seamless_x:
            seamless_axes_list.append("x")
        if self.seamless_y:
            seamless_axes_list.append("y")

        if unet is not None:
            unet.seamless_axes = seamless_axes_list
        if vae is not None:
            vae.seamless_axes = seamless_axes_list

        return SeamlessModeOutput(unet=unet, vae=vae)


@invocation("freeu", title="FreeU", tags=["freeu"], category="unet", version="1.0.1")
class FreeUInvocation(BaseInvocation):
    """
    Applies FreeU to the UNet. Suggested values (b1/b2/s1/s2):

    SD1.5: 1.2/1.4/0.9/0.2,
    SD2: 1.1/1.2/0.9/0.2,
    SDXL: 1.1/1.2/0.6/0.4,
    """

    unet: UNetField = InputField(description=FieldDescriptions.unet, input=Input.Connection, title="UNet")
    b1: float = InputField(default=1.2, ge=-1, le=3, description=FieldDescriptions.freeu_b1)
    b2: float = InputField(default=1.4, ge=-1, le=3, description=FieldDescriptions.freeu_b2)
    s1: float = InputField(default=0.9, ge=-1, le=3, description=FieldDescriptions.freeu_s1)
    s2: float = InputField(default=0.2, ge=-1, le=3, description=FieldDescriptions.freeu_s2)

    def invoke(self, context: InvocationContext) -> UNetOutput:
        self.unet.freeu_config = FreeUConfig(s1=self.s1, s2=self.s2, b1=self.b1, b2=self.b2)
        return UNetOutput(unet=self.unet)
