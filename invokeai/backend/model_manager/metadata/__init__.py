"""
Initialization file for invokeai.backend.model_manager.metadata

Usage:

from invokeai.backend.model_manager.metadata import(
   AnyModelRepoMetadata,
   CommercialUsage,
   LicenseRestrictions,
   HuggingFaceMetadata,
)

from invokeai.backend.model_manager.metadata.fetch import CivitaiMetadataFetch

data = CivitaiMetadataFetch().from_url("https://civitai.com/models/206883/split")
assert isinstance(data, CivitaiMetadata)
if data.allow_commercial_use:
   print("Commercial use of this model is allowed")
"""

from .fetch import HuggingFaceMetadataFetch, ModelMetadataFetchBase
from .metadata_base import (
    AnyModelRepoMetadata,
    AnyModelRepoMetadataValidator,
    BaseMetadata,
    HuggingFaceMetadata,
    ModelMetadataWithFiles,
    RemoteModelFile,
    UnknownMetadataException,
)

__all__ = [
    "AnyModelRepoMetadata",
    "AnyModelRepoMetadataValidator",
    "HuggingFaceMetadata",
    "HuggingFaceMetadataFetch",
    "ModelMetadataFetchBase",
    "BaseMetadata",
    "ModelMetadataWithFiles",
    "RemoteModelFile",
    "UnknownMetadataException",
]
