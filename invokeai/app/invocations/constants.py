from typing import Literal

from invokeai.backend.stable_diffusion.schedulers import SCHEDULER_MAP

LATENT_SCALE_FACTOR = 8
"""
HACK: Many nodes are currently hard-coded to use a fixed latent scale factor of 8. This is fragile, and will need to
be addressed if future models use a different latent scale factor. Also, note that there may be places where the scale
factor is hard-coded to a literal '8' rather than using this constant.
The ratio of image:latent dimensions is LATENT_SCALE_FACTOR:1, or 8:1.
"""

SCHEDULER_NAME_VALUES = Literal[tuple(SCHEDULER_MAP.keys())]
"""A literal type representing the valid scheduler names."""
