from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from diffusers import UNet2DConditionModel

from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase
from invokeai.backend.util.fp8 import get_model_compute_dtype

if TYPE_CHECKING:
    from invokeai.app.invocations.model import ModelIdentifierField
    from invokeai.app.services.shared.invocation_context import InvocationContext
    from invokeai.backend.util.original_weights_storage import OriginalWeightsStorage


class LoRAExt(ExtensionBase):
    def __init__(
        self,
        node_context: InvocationContext,
        model_id: ModelIdentifierField,
        weight: float,
    ):
        super().__init__()
        self._node_context = node_context
        self._model_id = model_id
        self._weight = weight

    @contextmanager
    def patch_unet(self, unet: UNet2DConditionModel, original_weights: OriginalWeightsStorage):
        lora_info = self._node_context.models.load(self._model_id)
        assert isinstance(lora_info.model, ModelPatchRaw)
        # Pin the LoRA's cache record for the whole patched scope, mirroring the plural
        # apply_smart_model_patches. Without the pin, dropping the LoadedModel handle would release
        # the record's post-admission grace and a peer cache could evict it (un-counting RAM that
        # is still referenced here). The pin must span the yield, not just the application: despite
        # force_direct_patching=True, fp8-storage modules are routed to sidecar patching (direct
        # patching is impossible on float8 weights — see apply_smart_model_patch), which stores a
        # live reference to this cached patch's layers inside the UNet's modules for the whole
        # denoise.
        with lora_info.model_in_ram() as lora_model:
            LayerPatcher.apply_smart_model_patch(
                model=unet,
                prefix="lora_unet_",
                patch=lora_model,
                patch_weight=self._weight,
                original_weights=original_weights,
                original_modules={},
                dtype=get_model_compute_dtype(unet),
                force_direct_patching=True,
                force_sidecar_patching=False,
            )
            del lora_model

            yield
