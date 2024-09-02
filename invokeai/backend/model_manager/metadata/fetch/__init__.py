"""
Initialization file for invokeai.backend.model_manager.metadata.fetch

Usage:
from invokeai.backend.model_manager.metadata.fetch import (
    HuggingFaceMetadataFetch,
)

data = HuggingFaceMetadataFetch().from_id("<repo_id>")
assert isinstance(data, HuggingFaceMetadata)
"""

from invokeai.backend.model_manager.metadata.fetch.fetch_base import ModelMetadataFetchBase
from invokeai.backend.model_manager.metadata.fetch.huggingface import HuggingFaceMetadataFetch

__all__ = ["ModelMetadataFetchBase", "HuggingFaceMetadataFetch"]
