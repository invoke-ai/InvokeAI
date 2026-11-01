"""FLUX.3 layout prompts: boxes from canvas regions, in the box-table form the transformer reads.

FLUX.3 reads a layout as the caption, one space, and one JSON list, the box table, serialized with
``json.dumps(rows, ensure_ascii=False)``. An object row is ``{"id", "bbox", "desc"}``, the box ``[y_min, x_min, y_max,
x_max]`` in whole numbers on a 0-1000 grid stretched over the image, origin top-left. When FLUX.3 edits reference
images it reads edit rows ``{"id", "from", "src_bbox", "tgt_bbox", "desc"}``: ``"from": "target"`` with
``"src_bbox": null`` places new content at ``tgt_bbox`` in the output, ``"from": "ref_image_N"`` moves content out of a
reference. This is the format of the reference implementation, github.com/black-forest-labs/flux-3-image-klein at
commit 45e57c9fb06f9751cdced29a65c5d2171003be05: ``src/flux3/generate.py`` and ``src/flux3/bbox.py``, and
``scripts/app.py``, whose layout UI writes the boxes of an edit as target rows. Apache-2.0, see ``NOTICE.md``.

``_split_caption``, ``_valid_box``, ``_BOX_ID`` and ``_GRID`` port ``split``, ``valid_box``, ``ID`` and ``SCALE`` from
``bbox.py``, and ``_detect_box_table`` the table check of its ``detect``. They are ported rather than imported from
diffusers' copy (``modular_pipelines/flux3/prompt_upsampler.py``): that module loads transformers' Qwen3.5 modeling,
and every node module is imported at server start.

Differences from the reference, which places boxes given as 0-1 fractions (``to_canvas``):

- Regions arrive on the 0-1000 grid already: the canvas normalizes each mask's bounds against the generation frame.
  A region's box must be one ``_valid_box`` accepts, so a table this module writes is detected again.
- A prompt that already ends with a box table gets the regions appended to that table, numbered after its highest
  ``obj_N``. The reference, given boxes, appends a second table after the whole prompt.
- The caption is stripped on both sides; the reference strips only its end.
"""

import json
import re
from collections.abc import Sequence
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator

# The grid's extent, the reference's `SCALE`, and a row id, its `ID`.
_GRID = 1000
_BOX_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_OBJECT_ID = re.compile(r"obj_([0-9]+)")
_BOX_KEYS = ("bbox", "src_bbox", "tgt_bbox")

# Four whole numbers [y_min, x_min, y_max, x_max] on the 0-1000 grid. Strict ints refuse booleans, as the reference's
# `isinstance` checks do, and floats: the reference rounds every box it places onto the grid (`to_canvas`).
Flux3LayoutBbox = Annotated[
    list[Annotated[int, Field(ge=0, le=_GRID, strict=True)]],
    Field(min_length=4, max_length=4),
]


class Flux3LayoutRegion(BaseModel):
    """One FLUX.3 box: what a canvas region depicts, and where."""

    desc: str = Field(description="What the region depicts. Becomes the row's `desc`, stripped; must not be blank.")
    bbox: Flux3LayoutBbox = Field(
        description="Box [y_min, x_min, y_max, x_max] in whole numbers from 0 to 1000 across the image, origin "
        "top-left, with y_min < y_max and x_min < x_max."
    )

    @field_validator("desc")
    @classmethod
    def _desc_is_stripped_and_not_blank(cls, desc: str) -> str:
        # The reference's layout UI strips the description it writes into a row.
        desc = desc.strip()
        if not desc:
            raise ValueError("desc must not be blank")
        return desc

    @field_validator("bbox")
    @classmethod
    def _bbox_is_a_grid_box(cls, bbox: list[int]) -> list[int]:
        # The type keeps each coordinate on the grid; the box also needs area. [0, 0, 1, 1] has area but reads as 0-1
        # fractions to the reference parser, so the table would no longer be detected as one.
        if _valid_box(bbox) is None:
            raise ValueError(
                "bbox must be [y_min, x_min, y_max, x_max] with y_min < y_max, x_min < x_max and a coordinate above 1 "
                f"(smaller ones read as 0-1 fractions), got {bbox}"
            )
        return bbox


def build_flux3_layout_prompt(prompt: str, regions: Sequence[Flux3LayoutRegion], *, edit_targets: bool = False) -> str:
    """The prompt FLUX.3 encodes for ``prompt`` with ``regions`` placed as boxes.

    - No regions: ``prompt`` unchanged, box table or not.
    - The prompt ends with a box table: its rows are kept as they are, edit rows included, and the regions follow them
      with ids numbered after the highest ``obj_N``, so no id repeats. A trailing list whose items are not all box rows
      (a caption ending in ``["SALE"]``, say) is not a table.
    - Anything else is the caption, stripped, followed by a table of the regions.

    Each region becomes an object row ``{"id", "bbox", "desc"}``, or with ``edit_targets`` an edit row that places new
    content in the output, ``{"id", "from": "target", "src_bbox": null, "tgt_bbox", "desc"}``: the rows FLUX.3 reads
    when it edits reference images.

    Raises ValueError when the caption is blank: the boxes place parts of the scene it describes.
    """
    if not regions:
        return prompt

    found = _detect_box_table(prompt)
    caption, rows = found if found is not None else (prompt, [])
    caption = caption.strip()
    if not caption:
        raise ValueError(
            "FLUX.3 layout regions need a prompt: their boxes place parts of the scene it describes, so it cannot be "
            "blank or only a box table."
        )
    next_index = 1 + max((int(match[1]) for row in rows if (match := _OBJECT_ID.fullmatch(row["id"]))), default=0)
    for offset, region in enumerate(regions):
        row_id = f"obj_{next_index + offset}"
        if edit_targets:
            rows.append(
                {"id": row_id, "from": "target", "src_bbox": None, "tgt_bbox": region.bbox, "desc": region.desc}
            )
        else:
            rows.append({"id": row_id, "bbox": region.bbox, "desc": region.desc})
    return f"{caption} {json.dumps(rows, ensure_ascii=False)}"


def _detect_box_table(prompt: str) -> tuple[str, list[dict[str, Any]]] | None:
    """``(caption, rows)`` when the prompt ends with a box table on the 0-1000 grid, else None: every row has an id and
    a valid box under ``bbox``, ``src_bbox`` or ``tgt_bbox``. The rows are returned as parsed."""
    caption, rows = _split_caption(prompt)
    if not rows or not all(_is_box_row(row) for row in rows):
        return None
    return caption, rows


def _is_box_row(row: object) -> bool:
    return (
        isinstance(row, dict)
        and isinstance(row.get("id"), str)
        and _BOX_ID.fullmatch(row["id"]) is not None
        and any(_valid_box(row.get(key)) is not None for key in _BOX_KEYS)
    )


def _split_caption(text: str) -> tuple[str, list[Any] | None]:
    """``(caption, rows)``: the rows are the JSON list that closes the text, parsed from the last ``[`` the rest
    parses from. Not a regex: descriptions may hold brackets and escaped quotes."""
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "[":
            try:
                rows = json.loads(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(rows, list):
                return text[:i].rstrip(), rows
    return text.strip(), None


def _valid_box(box: object) -> list[Any] | None:
    """The box as a list if it is a well-formed [y_min, x_min, y_max, x_max] on the 0-1000 grid, else None."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or abs(v) == float("inf") for v in box):
        return None
    y1, x1, y2, x2 = box
    # max <= 1 means 0-1 coordinates, not the 0-1000 grid
    if y1 < 0 or x1 < 0 or y2 <= y1 or x2 <= x1 or y2 > _GRID or x2 > _GRID or max(box) <= 1:
        return None
    return list(box)
