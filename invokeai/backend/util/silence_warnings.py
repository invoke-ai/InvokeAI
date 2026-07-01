import warnings
from contextlib import ContextDecorator

from diffusers.utils import logging as diffusers_logging
from transformers import logging as transformers_logging


# Inherit from ContextDecorator to allow using SilenceWarnings as both a context manager and a decorator.
class SilenceWarnings(ContextDecorator):
    """A context manager that disables warnings from transformers & diffusers modules while active.

    As context manager:
    ```
    with SilenceWarnings():
        # do something
    ```

    As decorator:
    ```
    @SilenceWarnings()
    def some_function():
        # do something
    ```
    """

    def __enter__(self) -> None:
        self._transformers_verbosity = transformers_logging.get_verbosity()
        self._diffusers_verbosity = diffusers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        diffusers_logging.set_verbosity_error()
        warnings.simplefilter("ignore")

    def __exit__(self, *args) -> None:
        transformers_logging.set_verbosity(self._transformers_verbosity)
        diffusers_logging.set_verbosity(self._diffusers_verbosity)
        warnings.simplefilter("default")
