"""
Initialization file for invokeai.backend.model_manager.metadata.fetch

Usage:
from invokeai.backend.model_manager.metadata.fetch import (
    CivitaiMetadataFetch,
    HuggingFaceMetadataFetch,
)
from invokeai.backend.model_manager.metadata import CivitaiMetadata

data = CivitaiMetadataFetch().from_url("https://civitai.com/models/206883/split")
assert isinstance(data, CivitaiMetadata)
if data.allow_commercial_use:
   print("Commercial use of this model is allowed")
"""

from .fetch_base import ModelMetadataFetchBase
from .huggingface import HuggingFaceMetadataFetch

__all__ = ["ModelMetadataFetchBase", "HuggingFaceMetadataFetch"]
