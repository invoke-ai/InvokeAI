"""
Initialization file for invokeai.backend.model_manager.metadata

Usage:

from invokeai.backend.model_manager.metadata import(
   AnyModelRepoMetadata,
   CommercialUsage,
   LicenseRestrictions,
   HuggingFaceMetadata,
)

from invokeai.backend.model_manager.metadata.fetch import HuggingFaceMetadataFetch

data = HuggingFaceMetadataFetch().from_id("<REPO_ID>")
assert isinstance(data, HuggingFaceMetadata)
"""

from invokeai.backend.model_manager.metadata.fetch import (
    CivitaiMetadataFetch,
    HuggingFaceMetadataFetch,
    ModelMetadataFetchBase,
)
from invokeai.backend.model_manager.metadata.metadata_base import (
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
