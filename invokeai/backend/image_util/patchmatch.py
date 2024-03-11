"""
This module defines a singleton object, "patchmatch" that
wraps the actual patchmatch object. It respects the global
"try_patchmatch" attribute, so that patchmatch loading can
be suppressed or deferred
"""

import numpy as np

import invokeai.backend.util.logging as logger
from invokeai.app.services.config.config_default import get_config

config = get_config()


class PatchMatch:
    """
    Thin class wrapper around the patchmatch function.
    """

    patch_match = None
    tried_load: bool = False

    def __init__(self):
        super().__init__()

    @classmethod
    def _load_patch_match(self):
        if self.tried_load:
            return
        if config.patchmatch:
            from patchmatch import patch_match as pm

            if pm.patchmatch_available:
                logger.info("Patchmatch initialized")
            else:
                logger.info("Patchmatch not loaded (nonfatal)")
            self.patch_match = pm
        else:
            logger.info("Patchmatch loading disabled")
        self.tried_load = True

    @classmethod
    def patchmatch_available(self) -> bool:
        self._load_patch_match()
        return self.patch_match and self.patch_match.patchmatch_available

    @classmethod
    def inpaint(self, *args, **kwargs) -> np.ndarray:
        if self.patchmatch_available():
            return self.patch_match.inpaint(*args, **kwargs)
