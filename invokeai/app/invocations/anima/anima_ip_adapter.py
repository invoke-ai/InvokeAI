"""Anima IP-Adapter invocation: encodes a reference image for the SigLIP2 Anima IP-Adapter."""

import re

import torch
from PIL import Image
from pydantic import BaseModel, Field, model_validator
from transformers import SiglipVisionModel
from typing_extensions import Self

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    Classification,
    invocation,
    invocation_output,
)
from invokeai.app.invocations.fields import FieldDescriptions, ImageField, InputField, OutputField
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.invocations.util import validate_begin_end_step
from invokeai.app.services.model_records.model_records_base import ModelRecordChanges
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.anima.ip_adapter import encode_image_tokens, siglip_fp32_working_bytes, siglip_pixel_values
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.starter_models.common import siglip2_base_512
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType

# The SigLIP2 checkpoint is ~1.5GB.
_ENCODER_INSTALL_TIMEOUT_SECONDS = 600


def _hf_repo_id(source: str) -> str | None:
    """The Hugging Face repo id a whole-repo model source names (`owner/repo`, its `:variant` form or its
    huggingface.co URL), lower-cased; None for a sub-folder of a repo or any other source."""
    source = re.sub(r"^https?://huggingface\.co/", "", source.strip(), flags=re.IGNORECASE).rstrip("/")
    if "::" in source or not re.fullmatch(r"[^/:]+/[^/:]+(:[^:]*)?", source):
        return None
    return source.split(":")[0].lower()


class AnimaIPAdapterField(BaseModel):
    """A reference image encoded for an Anima IP-Adapter."""

    tokens_name: str = Field(description="The name of the SigLIP2 image tokens tensor (strength not applied)")
    image_name: str = Field(description="The name of the reference image the tokens encode")
    ip_adapter_model: ModelIdentifierField = Field(description="The Anima IP-Adapter model")
    strength: float = Field(default=1.0, ge=0.0, le=2.0, description="Scales the image tokens; 0 turns it off")
    begin_step_percent: float = Field(
        default=0.0, ge=0.0, le=1.0, description="When the IP-Adapter is first applied (% of total steps)"
    )
    end_step_percent: float = Field(
        default=1.0, ge=0.0, le=1.0, description="When the IP-Adapter is last applied (% of total steps)"
    )


@invocation_output("anima_ip_adapter_output")
class AnimaIPAdapterOutput(BaseInvocationOutput):
    ip_adapter: AnimaIPAdapterField = OutputField(description=FieldDescriptions.ip_adapter, title="IP-Adapter")


@invocation(
    "anima_ip_adapter",
    title="IP-Adapter - Anima",
    tags=["ip_adapter", "anima", "reference", "image"],
    category="conditioning",
    version="1.0.0",
    classification=Classification.Prototype,
    idle_gpu_offloadable=True,
)
class AnimaIPAdapterInvocation(BaseInvocation):
    """Encodes a reference image with SigLIP2 for an Anima IP-Adapter, which carries its appearance into the image.

    The SigLIP2 encoder (google/siglip2-base-patch16-512) is installed on first use.
    """

    image: ImageField = InputField(description="The reference image.")
    ip_adapter_model: ModelIdentifierField = InputField(
        description=FieldDescriptions.ip_adapter,
        title="IP-Adapter Model",
        ui_model_base=BaseModelType.Anima,
        ui_model_type=ModelType.IPAdapter,
    )
    strength: float = InputField(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Scales the image tokens before the adapter projects them. 0 turns the adapter off.",
    )
    begin_step_percent: float = InputField(
        default=0.0, ge=0.0, le=1.0, description="When the IP-Adapter is first applied (% of total steps)"
    )
    end_step_percent: float = InputField(
        default=1.0, ge=0.0, le=1.0, description="When the IP-Adapter is last applied (% of total steps)"
    )

    @model_validator(mode="after")
    def validate_begin_end_step_percent(self) -> Self:
        validate_begin_end_step(self.begin_step_percent, self.end_step_percent)
        return self

    def invoke(self, context: InvocationContext) -> AnimaIPAdapterOutput:
        if self.strength == 0:
            # Denoise drops a strength-0 image before loading its tokens, so nothing is encoded (or installed) for it.
            tokens = torch.zeros(0)
        else:
            image = context.images.get_pil(self.image.image_name, "RGB")
            # Stored on CPU: this node may run on a borrowed idle GPU, and the denoise node runs on the session's.
            tokens = self._encode(context, image).detach().to("cpu")
        return AnimaIPAdapterOutput(
            ip_adapter=AnimaIPAdapterField(
                tokens_name=context.tensors.save(tokens),
                image_name=self.image.image_name,
                ip_adapter_model=self.ip_adapter_model,
                strength=self.strength,
                begin_step_percent=self.begin_step_percent,
                end_step_percent=self.end_step_percent,
            )
        )

    @torch.no_grad()
    def _encode(self, context: InvocationContext, image: Image.Image) -> torch.Tensor:
        encoder_info = context.models.load(self._get_siglip_model(context).key)
        with encoder_info.model_on_device(working_mem_bytes=siglip_fp32_working_bytes(encoder_info.model)) as (
            _,
            encoder,
        ):
            if not isinstance(encoder, SiglipVisionModel):
                raise TypeError(f"Expected a SigLIP vision model, got {type(encoder).__name__}.")
            return encode_image_tokens(encoder, siglip_pixel_values(image), encoder_info.compute_device)

    @staticmethod
    def _get_siglip_model(context: InvocationContext) -> AnyModelConfig:
        repo_id = _hf_repo_id(siglip2_base_512.source)

        def installed() -> list[AnyModelConfig]:
            return [
                m
                for m in context.models.search_by_attrs(base=BaseModelType.Any, type=ModelType.SigLIP)
                if _hf_repo_id(m.source) == repo_id or m.name == siglip2_base_512.name
            ]

        models = installed()
        if not models and context._services.configuration.multiuser:
            user = context._services.users.get(context._data.queue_item.user_id)
            if not (user and user.is_admin):
                raise RuntimeError(
                    f"The SigLIP2 encoder ({siglip2_base_512.source}) required by the Anima IP-Adapter is not "
                    "installed. Ask an administrator to install it from the Model Manager and try again."
                )
        if not models:
            context.logger.warning(
                f"The SigLIP2 encoder required by the Anima IP-Adapter ({siglip2_base_512.source}) is not installed. "
                "Downloading and installing it now; this may take a while."
            )
            install = context._services.model_manager.install
            job = install.heuristic_import(
                siglip2_base_512.source,
                config=ModelRecordChanges(name=siglip2_base_512.name, type=ModelType.SigLIP),
            )
            install.wait_for_job(job, timeout=_ENCODER_INSTALL_TIMEOUT_SECONDS)
            models = installed()
            if not models:
                raise RuntimeError(
                    f"Could not install the SigLIP2 encoder ({siglip2_base_512.source}) required by the Anima "
                    "IP-Adapter. Install it from the Model Manager and try again."
                )
        return models[0]
