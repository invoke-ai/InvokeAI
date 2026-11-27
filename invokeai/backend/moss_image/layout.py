"""MOSS-Image layout prompts built from canvas regions.

MOSS-Image reads either plain text or a compact JSON layout: a ``summary`` with the global ``caption``, a flat list
of ``items`` (each a ``label``, a ``caption``, a ``bbox`` and, for text, the literal ``text``), and the box
convention the model was trained on. The box conversion and the serialization are ported from OpenMOSS' prompt
enhancer, https://github.com/OpenMOSS/MOSS-Image at commit c1355e6b1986dfe8fba3c0455783a05899212342,
``skills/moss-image-prompt-enhancer/scripts/compile_prompt.py`` (``centered_box`` and the end of
``compile_prompt``; vocabulary in ``references/layout.md``), Apache-2.0 (see ``NOTICE.md``). The reference compiles a
whole draft; InvokeAI builds items from Regional Guidance layers, and either wraps them around the user's prompt or
appends them to a layout the user already wrote. It also compiles a whole draft, as the reference does, when the prompt
is one: that is what Expand Prompt's "MOSS-Image Prompt Enhancer" system prompt asks the LLM to write for layouts.

Differences from the reference:

- A region's ``kind`` is the item ``label``. Only ``text`` regions carry ``text``: the reference also accepts it on
  objects, but InvokeAI's region panel offers the visible-text field for text regions only, so a value on another
  kind would be a stale one that reaches the model unseen.
- Region items have no ``tags`` or ``palette``, and their keys are always in the order ``label, caption, bbox, text``. The
  reference compiler keeps whatever order a draft has, and the sources disagree: ``USERGUIDE.md`` and
  ``references/layout.md`` put ``text`` before ``bbox``, while the gallery's layout prompts in the same repository
  commit (``assets/images/011/parameters.json`` and ``014``) use ``label, caption, bbox, text``, which is what this
  module emits (``002`` puts ``bbox`` first). A compiled draft keeps its own key order, as in the reference.
- A draft is compiled at the generation size, not its own: boxes are scaled per axis from the draft's ``width`` and
  ``height``, which only set the coordinate frame (so they need not be multiples of 16). Expand Prompt never changes
  the image size.
- Drafts are read leniently, because they come from an LLM and the user cannot easily repair its output. The
  reference rejects these; this module repairs them:

  - Unknown fields (at the root, in ``summary`` or on items, such as ``id`` or ``notes``) are dropped. The reference
    rejects them so that nothing a user wrote is lost unseen; here the alternative is a failed generation, and the
    model would not read those fields anyway.
  - An unknown item ``label`` becomes ``text`` when the item has non-blank ``text``, otherwise ``object``.
  - A duplicate JSON key keeps its last value, as Python's parser does; the reference rejects duplicates.
  - A box that leaves the canvas is clamped to it. An item whose box has no area, before or after clamping or at
    whole-pixel precision after scaling, is dropped; only a draft left without items is an error.

  Everything else is checked as in the reference, with an error naming the item: a missing or blank
  ``summary.caption`` (it is the global description, and inventing one from the items would describe a different
  image), a ``text`` item without text, a blank caption, malformed ``meta``, ``palette``, ``tags`` or ``canvas``, and
  boxes that are not four finite numbers.
"""

import json
import math
import re
from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# The compiled layout's box convention: pixel offsets from the canvas centre, ordered [y1, x1, y2, x2].
_COORD_SPACE = "abs_centralized"
_BBOX_FORMAT = "yxyx"

# The draft's box convention: pixel edges [x1, y1, x2, y2] from the top-left corner.
_DRAFT_COORD_SPACE = "abs"
_DRAFT_BBOX_FORMAT = "xyxy"
_LABELS = ("object", "character", "text")
_CANVASES = ("single", "multi_panel", "collage")
_SUMMARY_FIELDS = ("caption", "meta", "canvas", "palette")
_ITEM_FIELDS = ("label", "caption", "bbox", "text", "tags", "palette")
# LLMs like to wrap JSON in a Markdown code fence, optionally naming the language.
_CODE_FENCE = re.compile(r"\A```[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)\r?\n?```\Z", re.DOTALL)

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

    - The prompt is a layout draft (a JSON object with ``"bbox_coord_space":"abs"`` and ``"bbox_format":"xyxy"``,
      optionally in a Markdown code fence): it is compiled at ``width`` x ``height`` (see the module docstring), and
      the regions follow its items.
    - No regions: any other prompt unchanged, whether plain text or a compiled JSON layout.
    - The prompt is a compiled layout (a JSON object with ``summary`` and ``items``): the regions are appended to its
      ``items``; every other key keeps its value and position. A layout in another box convention is refused rather
      than mixed with these boxes.
    - Anything else, including JSON that is not such a layout and text that is not valid JSON, is the caption of a new
      layout: ``{"summary":{"caption":prompt},"items":[...],"bbox_coord_space":...,"bbox_format":...}``.

    Raises ValueError for an invalid draft, a blank caption, a layout in another convention, or a region whose box
    leaves the image or has no area at whole-pixel precision.
    """
    draft = _parse_layout_draft(prompt)
    if draft is not None:
        layout = _compile_layout_draft(draft, width, height)
    elif not regions:
        return prompt
    else:
        layout = _parse_compiled_layout(prompt)

    items = [_layout_item(index, region, width, height) for index, region in enumerate(regions)]
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
    bbox = _centered_box(x1, y1, x2, y2, width, height)
    if bbox is None:
        raise ValueError(f"{name} has bbox {region.bbox}, which has no area at whole-pixel precision.")
    item: dict[str, Any] = {"label": region.kind, "caption": region.caption, "bbox": bbox}
    if region.text is not None:
        item["text"] = region.text
    return item


def _centered_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> list[int] | None:
    """The box enclosed in whole pixels, then centred as [y1, x1, y2, x2] (reference ``centered_box``); None when
    nothing is left."""
    bbox = [
        math.floor(round(y1, 9)) - height // 2,
        math.floor(round(x1, 9)) - width // 2,
        math.ceil(round(y2, 9)) - height // 2,
        math.ceil(round(x2, 9)) - width // 2,
    ]
    return bbox if bbox[0] < bbox[2] and bbox[1] < bbox[3] else None


def _parse_layout_draft(prompt: str) -> dict[str, Any] | None:
    """The prompt as an absolute-pixel layout draft, or None when it is something else."""
    source = prompt.strip()
    fenced = _CODE_FENCE.match(source)
    if fenced is not None:
        source = fenced.group(1)
    try:
        parsed = json.loads(source, parse_constant=_reject_constant)
    except ValueError:
        return None
    if (
        isinstance(parsed, dict)
        and parsed.get("bbox_coord_space") == _DRAFT_COORD_SPACE
        and parsed.get("bbox_format") == _DRAFT_BBOX_FORMAT
    ):
        return parsed
    return None


def _compile_layout_draft(draft: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """The draft as a compiled layout at ``width`` x ``height``, leniently validated (see the module docstring)."""
    draft_width = draft.get("width")
    draft_height = draft.get("height")
    if not (_is_number(draft_width) and _is_number(draft_height) and draft_width > 0 and draft_height > 0):
        raise ValueError("Layout draft: width and height must be positive numbers.")

    summary = draft.get("summary")
    if not isinstance(summary, dict) or not _is_text(summary.get("caption")):
        raise ValueError("Layout draft: the summary has no caption.")
    summary = {key: value for key, value in summary.items() if key in _SUMMARY_FIELDS}
    if "canvas" in summary and summary["canvas"] not in _CANVASES:
        raise ValueError("Layout draft: the summary canvas must be single, multi_panel or collage.")
    for key in ("meta", "palette"):
        if key in summary and not _is_text_list(summary[key]):
            raise ValueError(f"Layout draft: the summary {key} must be a list of non-blank strings.")

    items = draft.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Layout draft: it has no items.")
    frame = (draft_width, draft_height, width, height)
    compiled = [
        compiled_item
        for index, item in enumerate(items)
        if (compiled_item := _compile_draft_item(f"Layout draft: item {index + 1}", item, *frame)) is not None
    ]
    if not compiled:
        raise ValueError("Layout draft: no item has a box on the canvas.")
    return {"summary": summary, "items": compiled, "bbox_coord_space": _COORD_SPACE, "bbox_format": _BBOX_FORMAT}


def _compile_draft_item(
    name: str, item: Any, draft_width: float, draft_height: float, width: int, height: int
) -> dict[str, Any] | None:
    """One compiled item in the draft's key order, or None when its box has no area on the canvas."""
    if not isinstance(item, dict):
        raise ValueError(f"{name} is not an object.")
    item = {key: value for key, value in item.items() if key in _ITEM_FIELDS}
    if not _is_text(item.get("caption")):
        raise ValueError(f"{name} has no caption.")
    if "text" in item and not _is_text(item["text"]):
        raise ValueError(f"{name} has blank text.")
    if item.get("label") not in _LABELS:
        item["label"] = "text" if "text" in item else "object"
    if item["label"] == "text" and "text" not in item:
        raise ValueError(f"{name} has no text.")
    for key in ("tags", "palette"):
        if key in item and not _is_text_list(item[key]):
            raise ValueError(f"{name}: its {key} must be a list of non-blank strings.")

    box = item.get("bbox")
    if not (isinstance(box, list) and len(box) == 4 and all(_is_number(value) for value in box)):
        raise ValueError(f"{name} needs a box of four numbers [x1, y1, x2, y2].")
    x1, y1, x2, y2 = max(box[0], 0), max(box[1], 0), min(box[2], draft_width), min(box[3], draft_height)
    if not (x1 < x2 and y1 < y2):
        return None
    # Per axis, so the draft's frame maps onto the whole image even when its shape differs.
    scale_x, scale_y = width / draft_width, height / draft_height
    bbox = _centered_box(x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y, width, height)
    if bbox is None:
        return None
    item["bbox"] = bbox
    return item


def _is_number(value: Any) -> bool:
    # Like the reference's `type(v) in (int, float)` check: booleans and numeric strings are not numbers.
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_text_list(value: Any) -> bool:
    return isinstance(value, list) and all(_is_text(entry) for entry in value)


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
