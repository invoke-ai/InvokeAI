from typing import Literal

LATENT_SCALE_FACTOR = 8
"""
HACK: Many nodes are currently hard-coded to use a fixed latent scale factor of 8. This is fragile, and will need to
be addressed if future models use a different latent scale factor. Also, note that there may be places where the scale
factor is hard-coded to a literal '8' rather than using this constant.
The ratio of image:latent dimensions is LATENT_SCALE_FACTOR:1, or 8:1.
"""

IMAGE_MODES = Literal["L", "RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV", "I", "F"]
"""A literal type for PIL image modes supported by Invoke"""
