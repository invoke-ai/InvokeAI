"""
Initialization file for invokeai.backend.util
"""
from .attention import auto_detect_slice_size  # noqa: F401
from .devices import (  # noqa: F401
    CPU_DEVICE,
    CUDA_DEVICE,
    MPS_DEVICE,
    choose_precision,
    choose_torch_device,
    normalize_device,
    torch_dtype,
)
from .logging import InvokeAILogger
from .util import ( # TO DO: Clean this up; remove the unused symbols
    GIG,
    Chdir,
    ask_user,  # noqa
    directory_size,
    download_with_resume,
    instantiate_from_config, # noqa
    url_attachment_name,  # noqa
    )

__all__ = ["GIG", "directory_size","Chdir", "download_with_resume", "InvokeAILogger", "choose_precision", "choose_torch_device"]
