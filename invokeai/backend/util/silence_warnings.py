"""Context class to silence transformers and diffusers warnings."""
import warnings
from typing import Any

from diffusers import logging as diffusers_logging
from transformers import logging as transformers_logging


class SilenceWarnings(object):
    """Use in context to temporarily turn off warnings from transformers & diffusers modules.

    with SilenceWarnings():
        # do something
    """

    def __init__(self) -> None:
        self.transformers_verbosity = transformers_logging.get_verbosity()
        self.diffusers_verbosity = diffusers_logging.get_verbosity()

    def __enter__(self) -> None:
        transformers_logging.set_verbosity_error()
        diffusers_logging.set_verbosity_error()
        warnings.simplefilter("ignore")

    def __exit__(self, *args: Any) -> None:
        transformers_logging.set_verbosity(self.transformers_verbosity)
        diffusers_logging.set_verbosity(self.diffusers_verbosity)
        warnings.simplefilter("default")
