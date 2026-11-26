from types import SimpleNamespace

import pytest

from invokeai.app.invocations.fields import ImageField
from invokeai.app.invocations.flux.flux_control_lora_loader import FluxControlLoRALoaderInvocation
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


def _invoke(stored_base: BaseModelType):
    lora = ModelIdentifierField(
        key="control", hash="control", name="control", base=BaseModelType.Flux, type=ModelType.ControlLoRa
    )
    invocation = FluxControlLoRALoaderInvocation(id="loader", lora=lora, image=ImageField(image_name="depth.png"))
    record = SimpleNamespace(name="Depth", base=stored_base, type=ModelType.ControlLoRa)
    context = SimpleNamespace(models=SimpleNamespace(exists=lambda _key: True, get_config=lambda _identifier: record))
    return invocation.invoke(context)


def test_a_flux_control_lora_passes_and_a_krea2_one_is_refused() -> None:
    assert _invoke(BaseModelType.Flux).control_lora.lora.key == "control"

    with pytest.raises(ValueError, match="'Depth' is a krea-2 control_lora model, not a FLUX Control LoRA"):
        _invoke(BaseModelType.Krea2)
