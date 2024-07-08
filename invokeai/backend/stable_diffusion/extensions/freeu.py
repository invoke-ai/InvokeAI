from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

from diffusers import UNet2DConditionModel

from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase

if TYPE_CHECKING:
    from invokeai.app.shared.models import FreeUConfig


class FreeUExt(ExtensionBase):
    def __init__(
        self,
        freeu_config: Optional[FreeUConfig],
        priority: int,
    ):
        super().__init__(priority=priority)
        self.freeu_config = freeu_config

    @contextmanager
    def patch_unet(self, state_dict: dict, unet: UNet2DConditionModel):
        did_apply_freeu = False
        try:
            assert hasattr(unet, "enable_freeu")  # mypy doesn't pick up this attribute?
            if self.freeu_config is not None:
                unet.enable_freeu(
                    b1=self.freeu_config.b1,
                    b2=self.freeu_config.b2,
                    s1=self.freeu_config.s1,
                    s2=self.freeu_config.s2,
                )
                did_apply_freeu = True

            yield

        finally:
            assert hasattr(unet, "disable_freeu")  # mypy doesn't pick up this attribute?
            if did_apply_freeu:
                unet.disable_freeu()
