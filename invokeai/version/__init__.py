'''
initialization file for invokeai
'''
import invokeai
from .invokeai_version import __version__
__app_id__= 'invoke-ai/InvokeAI'
__app_name__= 'InvokeAI'

# copy these attributes into the invokeai namespace
setattr(invokeai,'__version__',__version__)
setattr(invokeai,'__app_id__',__app_id__)
setattr(invokeai,'__app_name__',__app_name__)

def _ignore_xformers_triton_message_on_windows():
    import logging
    logging.getLogger("xformers").addFilter(
        lambda record: 'A matching Triton is not available' not in record.getMessage())

# In order to be effective, this needs to happen before anything could possibly import xformers.
_ignore_xformers_triton_message_on_windows()
