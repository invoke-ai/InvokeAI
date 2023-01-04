import sys
import os

if sys.platform == 'darwin':
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

__app_id__ = "invoke-ai/InvokeAI"
__app_name__ = "InvokeAI"
__version__ = "2.2.5"
