"""
Initialization file for invokeai.backend.util
"""
from .devices import (  # noqa: F401
    CPU_DEVICE,
    CUDA_DEVICE,
    MPS_DEVICE,
    choose_precision,
    choose_torch_device,
    normalize_device,
    torch_dtype,
)
from .util import (  # noqa: F401
    ask_user,
    download_with_resume,
    instantiate_from_config,
    url_attachment_name,
    Chdir,
)
from .attention import auto_detect_slice_size  # noqa: F401
