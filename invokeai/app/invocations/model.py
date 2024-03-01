import copy
from typing import List, Optional

from pydantic import BaseModel, Field

from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.shared.models import FreeUConfig

from ...backend.model_manager import SubModelType
from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)


class ModelInfo(BaseModel):
    key: str = Field(description="Key of model as returned by ModelRecordServiceBase.get_model()")
    submodel_type: Optional[SubModelType] = Field(default=None, description="Info to load submodel")


class LoraInfo(ModelInfo):
    weight: float = Field(description="Lora's weight which to use when apply to model")


class UNetField(BaseModel):
    unet: ModelInfo = Field(description="Info to load unet submodel")
    scheduler: ModelInfo = Field(description="Info to load scheduler submodel")
    loras: List[LoraInfo] = Field(description="Loras to apply on model loading")
    seamless_axes: List[str] = Field(default_factory=list, description='Axes("x" and "y") to which apply seamless')
    freeu_config: Optional[FreeUConfig] = Field(default=None, description="FreeU configuration")


class ClipField(BaseModel):
    tokenizer: ModelInfo = Field(description="Info to load tokenizer submodel")
    text_encoder: ModelInfo = Field(description="Info to load text_encoder submodel")
    skipped_layers: int = Field(description="Number of skipped layers in text_encoder")
    loras: List[LoraInfo] = Field(description="Loras to apply on model loading")


class VaeField(BaseModel):
    # TODO: better naming?
    vae: ModelInfo = Field(description="Info to load vae submodel")
    seamless_axes: List[str] = Field(default_factory=list, description='Axes("x" and "y") to which apply seamless')


@invocation_output("unet_output")
class UNetOutput(BaseInvocationOutput):
    """Base class for invocations that output a UNet field."""

    unet: UNetField = OutputField(description=FieldDescriptions.unet, title="UNet")


@invocation_output("vae_output")
class VAEOutput(BaseInvocationOutput):
    """Base class for invocations that output a VAE field"""

    vae: VaeField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation_output("clip_output")
class CLIPOutput(BaseInvocationOutput):
    """Base class for invocations that output a CLIP field"""

    clip: ClipField = OutputField(description=FieldDescriptions.clip, title="CLIP")


@invocation_output("model_loader_output")
class ModelLoaderOutput(UNetOutput, CLIPOutput, VAEOutput):
    """Model loader output"""

    pass


class MainModelField(BaseModel):
    """Main model field"""

    key: str = Field(description="Model key")


class LoRAModelField(BaseModel):
    """LoRA model field"""

    key: str = Field(description="LoRA model key")


@invocation(
    "main_model_loader",
    title="Main Model",
    tags=["model"],
    category="model",
    version="1.0.1",
)
class MainModelLoaderInvocation(BaseInvocation):
    """Loads a main model, outputting its submodels."""

    model: MainModelField = InputField(description=FieldDescriptions.main_model, input=Input.Direct)
    # TODO: precision?

    def invoke(self, context: InvocationContext) -> ModelLoaderOutput:
        key = self.model.key

        # TODO: not found exceptions
        if not context.models.exists(key):
            raise Exception(f"Unknown model {key}")

        return ModelLoaderOutput(
            unet=UNetField(
                unet=ModelInfo(
                    key=key,
                    submodel_type=SubModelType.UNet,
                ),
                scheduler=ModelInfo(
                    key=key,
                    submodel_type=SubModelType.Scheduler,
                ),
                loras=[],
            ),
            clip=ClipField(
                tokenizer=ModelInfo(
                    key=key,
                    submodel_type=SubModelType.Tokenizer,
                ),
                text_encoder=ModelInfo(
                    key=key,
                    submodel_type=SubModelType.TextEncoder,
                ),
                loras=[],
                skipped_layers=0,
            ),
            vae=VaeField(
                vae=ModelInfo(
                    key=key,
                    submodel_type=SubModelType.Vae,
                ),
            ),
        )


@invocation_output("lora_loader_output")
class LoraLoaderOutput(BaseInvocationOutput):
    """Model loader output"""

    unet: Optional[UNetField] = OutputField(default=None, description=FieldDescriptions.unet, title="UNet")
    clip: Optional[ClipField] = OutputField(default=None, description=FieldDescriptions.clip, title="CLIP")


@invocation("lora_loader", title="LoRA", tags=["model"], category="model", version="1.0.1")
class LoraLoaderInvocation(BaseInvocation):
    """Apply selected lora to unet and text_encoder."""

    lora: LoRAModelField = InputField(description=FieldDescriptions.lora_model, input=Input.Direct, title="LoRA")
    weight: float = InputField(default=0.75, description=FieldDescriptions.lora_weight)
    unet: Optional[UNetField] = InputField(
        default=None,
        description=FieldDescriptions.unet,
        input=Input.Connection,
        title="UNet",
    )
    clip: Optional[ClipField] = InputField(
        default=None,
        description=FieldDescriptions.clip,
        input=Input.Connection,
        title="CLIP",
    )

    def invoke(self, context: InvocationContext) -> LoraLoaderOutput:
        if self.lora is None:
            raise Exception("No LoRA provided")

        lora_key = self.lora.key

        if not context.models.exists(lora_key):
            raise Exception(f"Unkown lora: {lora_key}!")

        if self.unet is not None and any(lora.key == lora_key for lora in self.unet.loras):
            raise Exception(f'Lora "{lora_key}" already applied to unet')

        if self.clip is not None and any(lora.key == lora_key for lora in self.clip.loras):
            raise Exception(f'Lora "{lora_key}" already applied to clip')

        output = LoraLoaderOutput()

        if self.unet is not None:
            output.unet = copy.deepcopy(self.unet)
            output.unet.loras.append(
                LoraInfo(
                    key=lora_key,
                    submodel_type=None,
                    weight=self.weight,
                )
            )

        if self.clip is not None:
            output.clip = copy.deepcopy(self.clip)
            output.clip.loras.append(
                LoraInfo(
                    key=lora_key,
                    submodel_type=None,
                    weight=self.weight,
                )
            )

        return output


@invocation_output("sdxl_lora_loader_output")
class SDXLLoraLoaderOutput(BaseInvocationOutput):
    """SDXL LoRA Loader Output"""

    unet: Optional[UNetField] = OutputField(default=None, description=FieldDescriptions.unet, title="UNet")
    clip: Optional[ClipField] = OutputField(default=None, description=FieldDescriptions.clip, title="CLIP 1")
    clip2: Optional[ClipField] = OutputField(default=None, description=FieldDescriptions.clip, title="CLIP 2")


@invocation(
    "sdxl_lora_loader",
    title="SDXL LoRA",
    tags=["lora", "model"],
    category="model",
    version="1.0.1",
)
class SDXLLoraLoaderInvocation(BaseInvocation):
    """Apply selected lora to unet and text_encoder."""

    lora: LoRAModelField = InputField(description=FieldDescriptions.lora_model, input=Input.Direct, title="LoRA")
    weight: float = InputField(default=0.75, description=FieldDescriptions.lora_weight)
    unet: Optional[UNetField] = InputField(
        default=None,
        description=FieldDescriptions.unet,
        input=Input.Connection,
        title="UNet",
    )
    clip: Optional[ClipField] = InputField(
        default=None,
        description=FieldDescriptions.clip,
        input=Input.Connection,
        title="CLIP 1",
    )
    clip2: Optional[ClipField] = InputField(
        default=None,
        description=FieldDescriptions.clip,
        input=Input.Connection,
        title="CLIP 2",
    )

    def invoke(self, context: InvocationContext) -> SDXLLoraLoaderOutput:
        if self.lora is None:
            raise Exception("No LoRA provided")

        lora_key = self.lora.key

        if not context.models.exists(lora_key):
            raise Exception(f"Unknown lora: {lora_key}!")

        if self.unet is not None and any(lora.key == lora_key for lora in self.unet.loras):
            raise Exception(f'Lora "{lora_key}" already applied to unet')

        if self.clip is not None and any(lora.key == lora_key for lora in self.clip.loras):
            raise Exception(f'Lora "{lora_key}" already applied to clip')

        if self.clip2 is not None and any(lora.key == lora_key for lora in self.clip2.loras):
            raise Exception(f'Lora "{lora_key}" already applied to clip2')

        output = SDXLLoraLoaderOutput()

        if self.unet is not None:
            output.unet = copy.deepcopy(self.unet)
            output.unet.loras.append(
                LoraInfo(
                    key=lora_key,
                    submodel_type=None,
                    weight=self.weight,
                )
            )

        if self.clip is not None:
            output.clip = copy.deepcopy(self.clip)
            output.clip.loras.append(
                LoraInfo(
                    key=lora_key,
                    submodel_type=None,
                    weight=self.weight,
                )
            )

        if self.clip2 is not None:
            output.clip2 = copy.deepcopy(self.clip2)
            output.clip2.loras.append(
                LoraInfo(
                    key=lora_key,
                    submodel_type=None,
                    weight=self.weight,
                )
            )

        return output


class VAEModelField(BaseModel):
    """Vae model field"""

    key: str = Field(description="Model's key")


@invocation("vae_loader", title="VAE", tags=["vae", "model"], category="model", version="1.0.1")
class VaeLoaderInvocation(BaseInvocation):
    """Loads a VAE model, outputting a VaeLoaderOutput"""

    vae_model: VAEModelField = InputField(
        description=FieldDescriptions.vae_model,
        input=Input.Direct,
        title="VAE",
    )

    def invoke(self, context: InvocationContext) -> VAEOutput:
        key = self.vae_model.key

        if not context.models.exists(key):
            raise Exception(f"Unkown vae: {key}!")

        return VAEOutput(vae=VaeField(vae=ModelInfo(key=key)))


@invocation_output("seamless_output")
class SeamlessModeOutput(BaseInvocationOutput):
    """Modified Seamless Model output"""

    unet: Optional[UNetField] = OutputField(default=None, description=FieldDescriptions.unet, title="UNet")
    vae: Optional[VaeField] = OutputField(default=None, description=FieldDescriptions.vae, title="VAE")


@invocation(
    "seamless",
    title="Seamless",
    tags=["seamless", "model"],
    category="model",
    version="1.0.0",
)
class SeamlessModeInvocation(BaseInvocation):
    """Applies the seamless transformation to the Model UNet and VAE."""

    unet: Optional[UNetField] = InputField(
        default=None,
        description=FieldDescriptions.unet,
        input=Input.Connection,
        title="UNet",
    )
    vae: Optional[VaeField] = InputField(
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


@invocation("freeu", title="FreeU", tags=["freeu"], category="unet", version="1.0.0")
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
