import logging
from typing import Any, MutableMapping


# Issue with type hints related to LoggerAdapter: https://github.com/python/typeshed/issues/7855
class PrefixedLoggerAdapter(logging.LoggerAdapter):  # type: ignore
    def __init__(self, logger: logging.Logger, prefix: str):
        super().__init__(logger, {})
        self.prefix = prefix

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> tuple[str, MutableMapping[str, Any]]:
        return f"[{self.prefix}] {msg}", kwargs
