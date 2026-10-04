"""What an architecture can generate, and what its modes are called in image metadata.

Two separate facts that happen to travel together. The first is a capability the UI needs — an
architecture that cannot inpaint must not offer the tool. The second is a naming convention: every
generated image records a mode string like `flux_inpaint` in its metadata, and those strings are
persisted in user galleries and workflow files. They cannot be changed, only declared.

The slug is not the base value. It is `z_image` where the enum says `z-image`, `krea2` where the enum
says `krea-2`, `ideogram4`, `sd3`, `ernie_image` — and SD 1.x and 2.x use no prefix at all, emitting
a bare `txt2img`. Deriving it from `BaseModelType` would be wrong in seven of fourteen cases, so it
is declared, and a test reconstructs `GENERATION_MODES` from the declarations to prove the set is
complete and unchanged.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal

from invokeai.backend.architectures.facet import Facet
from invokeai.backend.architectures.registry import generative_bases, get

GenerationModeKind = Literal[
    "txt2img",
    "img2img",
    "inpaint",
    "outpaint",
    "t2v",
    "i2v",
    "lf2v",
    "flf2v",
    "interpolate",
    "extend_video",
    "ref2v",
    "a2v",
    "v2a",
]
"""The kinds of generation a mode string names.

`t2v`/`i2v` produce video, as do the conditioning variants: `lf2v` (last frame to video), `flf2v`
(first and last frame to video), `interpolate` (between two given images), `extend_video`
(continue an existing clip), `ref2v` (video from a set of image and video references) and `a2v`
(picture for an existing soundtrack). `v2a` is the one video mode that produces *sound* rather than
picture: it takes a clip and generates the soundtrack for it. The rest produce images."""


@dataclass(frozen=True)
class ModalityFacet(Facet):
    """What this architecture can produce."""

    REQUIRED: ClassVar[bool] = True

    modes: frozenset[GenerationModeKind]
    """Empty is meaningful: the SDXL refiner generates nothing on its own."""

    metadata_slug: str | None = None
    """The prefix its mode strings carry in image metadata. `None` means unprefixed.

    Persisted in every image a user has ever generated. Changing one does not migrate anything — it
    orphans the old value.
    """

    def metadata_modes(self) -> frozenset[str]:
        """The mode strings this architecture writes into image metadata."""
        prefix = f"{self.metadata_slug}_" if self.metadata_slug else ""
        return frozenset(f"{prefix}{mode}" for mode in self.modes)


def generation_modes() -> frozenset[str]:
    """Every mode string any architecture can write.

    Compared against `GENERATION_MODES` in a test rather than used to define it: that literal is a
    type, and it is what pydantic validates metadata against.
    """
    return frozenset(
        mode
        for base in generative_bases()
        for facet in [get(base, ModalityFacet)]
        if facet is not None
        for mode in facet.metadata_modes()
    )
