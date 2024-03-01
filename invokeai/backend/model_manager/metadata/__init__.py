"""
Initialization file for invokeai.backend.model_manager.metadata

Usage:

from invokeai.backend.model_manager.metadata import(
   AnyModelRepoMetadata,
   CommercialUsage,
   LicenseRestrictions,
   HuggingFaceMetadata,
   CivitaiMetadata,
)

from invokeai.backend.model_manager.metadata.fetch import CivitaiMetadataFetch

data = CivitaiMetadataFetch().from_url("https://civitai.com/models/206883/split")
assert isinstance(data, CivitaiMetadata)
if data.allow_commercial_use:
   print("Commercial use of this model is allowed")
"""

from .fetch import CivitaiMetadataFetch, HuggingFaceMetadataFetch, ModelMetadataFetchBase
from .metadata_base import (
    AnyModelRepoMetadata,
    AnyModelRepoMetadataValidator,
    BaseMetadata,
    CivitaiMetadata,
    HuggingFaceMetadata,
    ModelMetadataWithFiles,
    RemoteModelFile,
    UnknownMetadataException,
)

__all__ = [
    "AnyModelRepoMetadata",
    "AnyModelRepoMetadataValidator",
    "CivitaiMetadata",
    "CivitaiMetadataFetch",
    "HuggingFaceMetadata",
    "HuggingFaceMetadataFetch",
    "ModelMetadataFetchBase",
    "BaseMetadata",
    "ModelMetadataWithFiles",
    "RemoteModelFile",
    "UnknownMetadataException",
]
