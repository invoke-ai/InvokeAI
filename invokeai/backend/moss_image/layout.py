"""MOSS-Image layout prompts built from canvas regions.

MOSS-Image reads either plain text or a compact JSON layout: a ``summary`` with the global ``caption``, a flat list
of ``items`` (each a ``label``, a ``caption``, a ``bbox`` and, for text, the literal ``text``), and the box
convention the model was trained on. The box conversion and the serialization are ported from OpenMOSS' prompt
enhancer, https://github.com/OpenMOSS/MOSS-Image at commit c1355e6b1986dfe8fba3c0455783a05899212342,
``skills/moss-image-prompt-enhancer/scripts/compile_prompt.py`` (``centered_box`` and the end of
``compile_prompt``; vocabulary in ``references/layout.md``), Apache-2.0 (see ``NOTICE.md``). The reference compiles a
whole draft; InvokeAI builds only the items, from Regional Guidance layers, and either wraps them around the user's
prompt or appends them to a layout the user already wrote.

Differences from the reference:

- A region's ``kind`` is the item ``label``. Only ``text`` regions carry ``text``: the reference also accepts it on
  objects, but InvokeAI's region panel offers the visible-text field for text regions only, so a value on another
  kind would be a stale one that reaches the model unseen.
- Items have no ``tags`` or ``palette``, and their keys are always in the order ``label, caption, bbox, text``. The
  reference compiler keeps whatever order a draft has, and the sources disagree: ``USERGUIDE.md`` and
  ``references/layout.md`` put ``text`` before ``bbox``, while the gallery's layout prompts in the same repository
  commit (``assets/images/011/parameters.json`` and ``014``) use ``label, caption, bbox, text``, which is what this
  module emits (``002`` puts ``bbox`` first).
"""

import json
import math
from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# The compiled layout's box convention: pixel offsets from the canvas centre, ordered [y1, x1, y2, x2].
_COORD_SPACE = "abs_centralized"
_BBOX_FORMAT = "yxyx"

# Four pixel edges [x1, y1, x2, y2] from the top-left of the image. Strict floats take JSON ints and floats but,
# like the reference's `type(v) in (int, float)` check, reject booleans and numeric strings.
MossImageLayoutBbox = Annotated[
    list[Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]],
    Field(min_length=4, max_length=4),
]


class MossImageLayoutRegion(BaseModel):
    """One MOSS-Image layout element: what a canvas region depicts, and where."""

    kind: Literal["object", "character", "text"] = Field(
        description="What the region depicts. Becomes the layout item's `label`."
    )
    caption: str = Field(description="The region's visible appearance. Must not be blank.")
    text: str | None = Field(
        default=None,
        description="The exact visible characters of a `text` region (required and non-blank there). Must be null "
        "for other kinds.",
    )
    bbox: MossImageLayoutBbox = Field(
        description="Pixel box [x1, y1, x2, y2] from the image's top-left, with x1 < x2 and y1 < y2. Checked against "
        "the image size when the prompt is built."
    )

    @field_validator("caption")
    @classmethod
    def _caption_is_not_blank(cls, caption: str) -> str:
        if not caption.strip():
            raise ValueError("caption must not be blank")
        return caption

    @field_validator("bbox")
    @classmethod
    def _bbox_has_area(cls, bbox: list[float]) -> list[float]:
        x1, y1, x2, y2 = bbox
        if not (x1 < x2 and y1 < y2):
            raise ValueError(f"bbox must be [x1, y1, x2, y2] with x1 < x2 and y1 < y2, got {bbox}")
        return bbox

    @model_validator(mode="after")
    def _text_matches_kind(self) -> "MossImageLayoutRegion":
        if self.kind == "text":
            if self.text is None or not self.text.strip():
                raise ValueError("a text region needs the exact text it shows")
        elif self.text is not None:
            raise ValueError(f"only text regions carry text; this {self.kind} region has text={self.text!r}")
        return self


def build_moss_image_layout_prompt(
    prompt: str, regions: Sequence[MossImageLayoutRegion], width: int, height: int
) -> str:
    """The prompt MOSS-Image encodes for ``prompt`` with ``regions`` placed on a ``width`` x ``height`` image.

    - No regions: ``prompt`` unchanged, whether plain text or a JSON layout.
    - The prompt is a compiled layout (a JSON object with ``summary`` and ``items``): the regions are appended to its
      ``items``; every other key keeps its value and position. A layout in another box convention is refused rather
      than mixed with these boxes.
    - Anything else, including JSON that is not such a layout and text that is not valid JSON, is the caption of a new
      layout: ``{"summary":{"caption":prompt},"items":[...],"bbox_coord_space":...,"bbox_format":...}``.

    Raises ValueError for a blank caption, a layout in another convention, or a region whose box leaves the image or
    has no area at whole-pixel precision.
    """
    if not regions:
        return prompt

    items = [_layout_item(index, region, width, height) for index, region in enumerate(regions)]
    layout = _parse_compiled_layout(prompt)
    if layout is None:
        if not prompt.strip():
            raise ValueError("MOSS-Image layout regions need a prompt: it becomes the layout's summary caption.")
        layout = {
            "summary": {"caption": prompt},
            "items": items,
            "bbox_coord_space": _COORD_SPACE,
            "bbox_format": _BBOX_FORMAT,
        }
    else:
        layout["items"].extend(items)
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _layout_item(index: int, region: MossImageLayoutRegion, width: int, height: int) -> dict[str, Any]:
    name = f"Layout region {index + 1} ({region.caption!r})"
    x1, y1, x2, y2 = region.bbox
    # x1 and y1 are >= 0 and below x2 and y2 (MossImageLayoutRegion), so these two bounds keep the box on the image.
    if x2 > width or y2 > height:
        raise ValueError(f"{name} has bbox {region.bbox}, which leaves the {width}x{height} image.")
    # Enclose the box in whole pixels, then centre it (reference `centered_box`).
    bbox = [
        math.floor(round(y1, 9)) - height // 2,
        math.floor(round(x1, 9)) - width // 2,
        math.ceil(round(y2, 9)) - height // 2,
        math.ceil(round(x2, 9)) - width // 2,
    ]
    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise ValueError(f"{name} has bbox {region.bbox}, which has no area at whole-pixel precision.")
    item: dict[str, Any] = {"label": region.kind, "caption": region.caption, "bbox": bbox}
    if region.text is not None:
        item["text"] = region.text
    return item


def _parse_compiled_layout(prompt: str) -> dict[str, Any] | None:
    """The prompt as a compiled layout, or None when it is caption text."""
    try:
        parsed = json.loads(prompt, parse_constant=_reject_constant)
    except ValueError:
        return None
    if not (isinstance(parsed, dict) and "summary" in parsed and "items" in parsed):
        return None
    if (
        not isinstance(parsed["items"], list)
        or parsed.get("bbox_coord_space") != _COORD_SPACE
        or parsed.get("bbox_format") != _BBOX_FORMAT
    ):
        raise ValueError(
            "The prompt is a MOSS-Image layout, but regions can only be added to a compiled one: a list of items with "
            f'"bbox_coord_space":"{_COORD_SPACE}" and "bbox_format":"{_BBOX_FORMAT}".'
        )
    return parsed


def _reject_constant(value: str) -> None:
    # NaN and Infinity are not JSON, though Python's parser reads them.
    raise ValueError(f"Invalid JSON number: {value}")
