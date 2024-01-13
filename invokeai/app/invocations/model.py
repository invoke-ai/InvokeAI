import copy
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from invokeai.app.invocations.fields import FieldDescriptions, Input, InputField, OutputField
from invokeai.app.shared.models import FreeUConfig

from ...backend.model_management import BaseModelType, ModelType, SubModelType
from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    invocation,
    invocation_output,
)


class ModelInfo(BaseModel):
    model_name: str = Field(description="Info to load submodel")
    base_model: BaseModelType = Field(description="Base model")
    model_type: ModelType = Field(description="Info to load submodel")
    submodel: Optional[SubModelType] = Field(default=None, description="Info to load submodel")

    model_config = ConfigDict(protected_namespaces=())


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
    """Base class for invocations that output a UNet field"""

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

    model_name: str = Field(description="Name of the model")
    base_model: BaseModelType = Field(description="Base model")
    model_type: ModelType = Field(description="Model Type")

    model_config = ConfigDict(protected_namespaces=())


class LoRAModelField(BaseModel):
    """LoRA model field"""

    model_name: str = Field(description="Name of the LoRA model")
    base_model: BaseModelType = Field(description="Base model")

    model_config = ConfigDict(protected_namespaces=())


@invocation(
    "main_model_loader",
    title="Main Model",
    tags=["model"],
    category="model",
    version="1.0.0",
)
class MainModelLoaderInvocation(BaseInvocation):
    """Loads a main model, outputting its submodels."""

    model: MainModelField = InputField(description=FieldDescriptions.main_model, input=Input.Direct)
    # TODO: precision?

    def invoke(self, context: InvocationContext) -> ModelLoaderOutput:
        base_model = self.model.base_model
        model_name = self.model.model_name
        model_type = ModelType.Main

        # TODO: not found exceptions
        if not context.services.model_manager.model_exists(
            model_name=model_name,
            base_model=base_model,
            model_type=model_type,
        ):
            raise Exception(f"Unknown {base_model} {model_type} model: {model_name}")

        """
        if not context.services.model_manager.model_exists(
            model_name=self.model_name,
            model_type=SDModelType.Diffusers,
            submodel=SDModelType.Tokenizer,
        ):
            raise Exception(
                f"Failed to find tokenizer submodel in {self.model_name}! Check if model corrupted"
            )

        if not context.services.model_manager.model_exists(
            model_name=self.model_name,
            model_type=SDModelType.Diffusers,
            submodel=SDModelType.TextEncoder,
        ):
            raise Exception(
                f"Failed to find text_encoder submodel in {self.model_name}! Check if model corrupted"
            )

        if not context.services.model_manager.model_exists(
            model_name=self.model_name,
            model_type=SDModelType.Diffusers,
            submodel=SDModelType.UNet,
        ):
            raise Exception(
                f"Failed to find unet submodel from {self.model_name}! Check if model corrupted"
            )
        """

        return ModelLoaderOutput(
            unet=UNetField(
                unet=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.UNet,
                ),
                scheduler=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Scheduler,
                ),
                loras=[],
            ),
            clip=ClipField(
                tokenizer=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Tokenizer,
                ),
                text_encoder=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.TextEncoder,
                ),
                loras=[],
                skipped_layers=0,
            ),
            vae=VaeField(
                vae=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                    submodel=SubModelType.Vae,
                ),
            ),
        )


@invocation_output("lora_loader_output")
class LoraLoaderOutput(BaseInvocationOutput):
    """Model loader output"""

    unet: Optional[UNetField] = OutputField(default=None, description=FieldDescriptions.unet, title="UNet")
    clip: Optional[ClipField] = OutputField(default=None, description=FieldDescriptions.clip, title="CLIP")


@invocation("lora_loader", title="LoRA", tags=["model"], category="model", version="1.0.0")
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

        base_model = self.lora.base_model
        lora_name = self.lora.model_name

        if not context.services.model_manager.model_exists(
            base_model=base_model,
            model_name=lora_name,
            model_type=ModelType.Lora,
        ):
            raise Exception(f"Unkown lora name: {lora_name}!")

        if self.unet is not None and any(lora.model_name == lora_name for lora in self.unet.loras):
            raise Exception(f'Lora "{lora_name}" already applied to unet')

        if self.clip is not None and any(lora.model_name == lora_name for lora in self.clip.loras):
            raise Exception(f'Lora "{lora_name}" already applied to clip')

        output = LoraLoaderOutput()

        if self.unet is not None:
            output.unet = copy.deepcopy(self.unet)
            output.unet.loras.append(
                LoraInfo(
                    base_model=base_model,
                    model_name=lora_name,
                    model_type=ModelType.Lora,
                    submodel=None,
                    weight=self.weight,
                )
            )

        if self.clip is not None:
            output.clip = copy.deepcopy(self.clip)
            output.clip.loras.append(
                LoraInfo(
                    base_model=base_model,
                    model_name=lora_name,
                    model_type=ModelType.Lora,
                    submodel=None,
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
    version="1.0.0",
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

        base_model = self.lora.base_model
        lora_name = self.lora.model_name

        if not context.services.model_manager.model_exists(
            base_model=base_model,
            model_name=lora_name,
            model_type=ModelType.Lora,
        ):
            raise Exception(f"Unknown lora name: {lora_name}!")

        if self.unet is not None and any(lora.model_name == lora_name for lora in self.unet.loras):
            raise Exception(f'Lora "{lora_name}" already applied to unet')

        if self.clip is not None and any(lora.model_name == lora_name for lora in self.clip.loras):
            raise Exception(f'Lora "{lora_name}" already applied to clip')

        if self.clip2 is not None and any(lora.model_name == lora_name for lora in self.clip2.loras):
            raise Exception(f'Lora "{lora_name}" already applied to clip2')

        output = SDXLLoraLoaderOutput()

        if self.unet is not None:
            output.unet = copy.deepcopy(self.unet)
            output.unet.loras.append(
                LoraInfo(
                    base_model=base_model,
                    model_name=lora_name,
                    model_type=ModelType.Lora,
                    submodel=None,
                    weight=self.weight,
                )
            )

        if self.clip is not None:
            output.clip = copy.deepcopy(self.clip)
            output.clip.loras.append(
                LoraInfo(
                    base_model=base_model,
                    model_name=lora_name,
                    model_type=ModelType.Lora,
                    submodel=None,
                    weight=self.weight,
                )
            )

        if self.clip2 is not None:
            output.clip2 = copy.deepcopy(self.clip2)
            output.clip2.loras.append(
                LoraInfo(
                    base_model=base_model,
                    model_name=lora_name,
                    model_type=ModelType.Lora,
                    submodel=None,
                    weight=self.weight,
                )
            )

        return output


class VAEModelField(BaseModel):
    """Vae model field"""

    model_name: str = Field(description="Name of the model")
    base_model: BaseModelType = Field(description="Base model")

    model_config = ConfigDict(protected_namespaces=())


@invocation("vae_loader", title="VAE", tags=["vae", "model"], category="model", version="1.0.0")
class VaeLoaderInvocation(BaseInvocation):
    """Loads a VAE model, outputting a VaeLoaderOutput"""

    vae_model: VAEModelField = InputField(
        description=FieldDescriptions.vae_model,
        input=Input.Direct,
        title="VAE",
    )

    def invoke(self, context: InvocationContext) -> VAEOutput:
        base_model = self.vae_model.base_model
        model_name = self.vae_model.model_name
        model_type = ModelType.Vae

        if not context.services.model_manager.model_exists(
            base_model=base_model,
            model_name=model_name,
            model_type=model_type,
        ):
            raise Exception(f"Unkown vae name: {model_name}!")
        return VAEOutput(
            vae=VaeField(
                vae=ModelInfo(
                    model_name=model_name,
                    base_model=base_model,
                    model_type=model_type,
                )
            )
        )


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
