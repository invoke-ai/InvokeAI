"""
Initialization file for invokeai.backend.image_util methods.
"""
from .patchmatch import PatchMatch  # noqa: F401
from .pngwriter import PngWriter, PromptFormatter, retrieve_metadata, write_metadata  # noqa: F401
from .seamless import configure_model_padding  # noqa: F401
from .txt2mask import Txt2Mask  # noqa: F401
from .util import InitImageResizer, make_grid  # noqa: F401


def debug_image(debug_image, debug_text, debug_show=True, debug_result=False, debug_status=False):
    from PIL import ImageDraw

    if not debug_status:
        return

    image_copy = debug_image.copy().convert("RGBA")
    ImageDraw.Draw(image_copy).text((5, 5), debug_text, (255, 0, 0))

    if debug_show:
        image_copy.show()

    if debug_result:
        return image_copy
