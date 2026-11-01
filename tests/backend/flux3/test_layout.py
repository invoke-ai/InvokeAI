"""FLUX.3 layout prompts built from regions (`invokeai/backend/flux3/layout.py`).

Every expected prompt is written out by hand: the model reads the exact string, so key order, separators and the one
space between caption and table are part of the result. Where the reference implementation
(github.com/black-forest-labs/flux-3-image-klein, Apache-2.0) builds the same prompt, diffusers' port of its
`to_canvas` (`boxes_to_canvas`) is the second oracle.
"""

import json

import pytest
from diffusers.modular_pipelines.flux3.prompt_upsampler import boxes_to_canvas
from pydantic import ValidationError

from invokeai.backend.flux3.layout import Flux3LayoutRegion, build_flux3_layout_prompt


def _region(desc: str, bbox: list) -> Flux3LayoutRegion:
    return Flux3LayoutRegion.model_validate({"desc": desc, "bbox": bbox})


BALLOON = _region("a red hot-air balloon", [80, 100, 450, 420])
LIGHTHOUSE = _region("a lighthouse on a cliff", [300, 600, 950, 900])
# The reference README's boxes, in the 0-1 coordinates its generate.py takes; BALLOON and LIGHTHOUSE on the grid.
README_BOXES = [
    {"id": "obj_1", "desc": "a red hot-air balloon", "bbox": [0.08, 0.10, 0.45, 0.42]},
    {"id": "obj_2", "desc": "a lighthouse on a cliff", "bbox": [0.30, 0.60, 0.95, 0.90]},
]


def test_regions_follow_the_stripped_caption_as_one_table_of_object_rows_in_region_order() -> None:
    prompt = build_flux3_layout_prompt("  A misty coast at dawn.\n", [BALLOON, LIGHTHOUSE])

    assert prompt == (
        'A misty coast at dawn. [{"id": "obj_1", "bbox": [80, 100, 450, 420], "desc": "a red hot-air balloon"}, '
        '{"id": "obj_2", "bbox": [300, 600, 950, 900], "desc": "a lighthouse on a cliff"}]'
    )
    # The reference's generate.py builds the same prompt from the README's boxes.
    assert prompt == f"A misty coast at dawn. {json.dumps(boxes_to_canvas(README_BOXES), ensure_ascii=False)}"


def test_edit_targets_become_rows_that_place_new_content_in_the_output() -> None:
    prompt = build_flux3_layout_prompt("Add a balloon and a lighthouse.", [BALLOON, LIGHTHOUSE], edit_targets=True)

    assert prompt == (
        'Add a balloon and a lighthouse. [{"id": "obj_1", "from": "target", "src_bbox": null, '
        '"tgt_bbox": [80, 100, 450, 420], "desc": "a red hot-air balloon"}, '
        '{"id": "obj_2", "from": "target", "src_bbox": null, "tgt_bbox": [300, 600, 950, 900], '
        '"desc": "a lighthouse on a cliff"}]'
    )
    # The reference's layout UI turns the boxes of an edit into target rows before `to_canvas`.
    app_boxes = [
        {"id": box["id"], "desc": box["desc"], "from": "target", "src_bbox": None, "tgt_bbox": box["bbox"]}
        for box in README_BOXES
    ]
    assert prompt == f"Add a balloon and a lighthouse. {json.dumps(boxes_to_canvas(app_boxes), ensure_ascii=False)}"


@pytest.mark.parametrize(
    "prompt",
    [
        pytest.param("", id="empty"),
        pytest.param("  a beach at noon \n", id="plain-text-with-whitespace"),
        pytest.param('a kite [{"id": "obj_1", "bbox": [0, 0, 500, 500], "desc": "a kite"}]', id="box-table"),
    ],
)
@pytest.mark.parametrize("edit_targets", [False, True], ids=["object-rows", "edit-targets"])
def test_without_regions_the_prompt_passes_through_unchanged(prompt: str, edit_targets: bool) -> None:
    assert build_flux3_layout_prompt(prompt, [], edit_targets=edit_targets) == prompt


def test_regions_extend_the_table_a_prompt_ends_with_numbered_after_its_highest_object_id() -> None:
    # obj_10 is the highest id, though obj_2 sorts after it as text and comes last. The edit row and the desc with
    # brackets and escaped quotes come through as they were written.
    prompt = (
        "A café terrace. "
        '[{"id": "obj_10", "bbox": [0, 0, 400, 500], "desc": "a sign reading \\"[SALE]\\""}, '
        '{"id": "person_1", "bbox": [200, 500, 900, 700], "desc": "a waiter"}, '
        '{"id": "obj_2", "from": "target", "src_bbox": null, "tgt_bbox": [600, 0, 1000, 300], "desc": "a bicycle"}]'
    )

    result = build_flux3_layout_prompt(prompt, [BALLOON, LIGHTHOUSE])

    assert result == (
        "A café terrace. "
        '[{"id": "obj_10", "bbox": [0, 0, 400, 500], "desc": "a sign reading \\"[SALE]\\""}, '
        '{"id": "person_1", "bbox": [200, 500, 900, 700], "desc": "a waiter"}, '
        '{"id": "obj_2", "from": "target", "src_bbox": null, "tgt_bbox": [600, 0, 1000, 300], "desc": "a bicycle"}, '
        '{"id": "obj_11", "bbox": [80, 100, 450, 420], "desc": "a red hot-air balloon"}, '
        '{"id": "obj_12", "bbox": [300, 600, 950, 900], "desc": "a lighthouse on a cliff"}]'
    )


def test_edit_targets_extend_an_edit_table_numbered_after_the_object_id_of_any_row() -> None:
    # obj_4 moves a cup out of the reference; its id is the highest, held by a row with only src/tgt boxes.
    prompt = (
        "Move the cup to the right. "
        '[{"id": "obj_4", "from": "ref_image_0", "src_bbox": [100, 100, 300, 300], "tgt_bbox": [600, 600, 800, 800], '
        '"desc": "a white cup"}, '
        '{"id": "obj_2", "from": "target", "src_bbox": null, "tgt_bbox": [0, 0, 200, 200], "desc": "a lamp"}]'
    )

    result = build_flux3_layout_prompt(prompt, [BALLOON], edit_targets=True)

    assert result == (
        "Move the cup to the right. "
        '[{"id": "obj_4", "from": "ref_image_0", "src_bbox": [100, 100, 300, 300], "tgt_bbox": [600, 600, 800, 800], '
        '"desc": "a white cup"}, '
        '{"id": "obj_2", "from": "target", "src_bbox": null, "tgt_bbox": [0, 0, 200, 200], "desc": "a lamp"}, '
        '{"id": "obj_5", "from": "target", "src_bbox": null, "tgt_bbox": [80, 100, 450, 420], '
        '"desc": "a red hot-air balloon"}]'
    )


def test_a_table_without_object_ids_gets_regions_from_obj_1() -> None:
    prompt = 'Two dogs. [{"id": "dog_1", "bbox": [500, 0, 1000, 400], "desc": "a beagle"}]'

    assert build_flux3_layout_prompt(prompt, [BALLOON]) == (
        'Two dogs. [{"id": "dog_1", "bbox": [500, 0, 1000, 400], "desc": "a beagle"}, '
        '{"id": "obj_1", "bbox": [80, 100, 450, 420], "desc": "a red hot-air balloon"}]'
    )


@pytest.mark.parametrize("edit_targets", [False, True], ids=["object-rows", "edit-targets"])
@pytest.mark.parametrize(
    "bbox",
    [
        pytest.param([0, 0, 1000, 1000], id="whole-image"),
        pytest.param([999, 999, 1000, 1000], id="last-grid-unit"),
        pytest.param([0, 0, 1, 2], id="smallest-box-not-read-as-fractions"),
    ],
)
def test_a_table_written_here_is_extended_by_later_regions(bbox: list[int], edit_targets: bool) -> None:
    once = build_flux3_layout_prompt("a scene", [_region("a cup", bbox)], edit_targets=edit_targets)

    twice = build_flux3_layout_prompt(once, [_region("a saucer", [500, 500, 600, 600])], edit_targets=edit_targets)

    assert twice.startswith("a scene [")
    assert [row["id"] for row in json.loads(twice.removeprefix("a scene "))] == ["obj_1", "obj_2"]


@pytest.mark.parametrize(
    "prompt",
    [
        pytest.param("A shop window with a poster [SALE]", id="bracketed-word"),
        pytest.param('A shop window with a poster ["SALE"]', id="json-list-of-strings"),
        pytest.param("A grid of tiles [1, 2, 3, 4]", id="json-list-of-numbers"),
        pytest.param("A cat []", id="empty-list"),
        pytest.param(
            'A cat [{"id": "obj_1", "bbox": [0.1, 0.1, 0.5, 0.5], "desc": "a cat"}]', id="row-in-0-1-coordinates"
        ),
        pytest.param('A cat [{"id": "obj 1", "bbox": [0, 0, 500, 500], "desc": "a cat"}]', id="row-id-with-a-space"),
        pytest.param('A cat [{"bbox": [0, 0, 500, 500], "desc": "a cat"}]', id="row-without-id"),
    ],
)
def test_a_trailing_list_that_is_not_a_box_table_stays_in_the_caption(prompt: str) -> None:
    assert build_flux3_layout_prompt(prompt, [BALLOON]) == (
        f'{prompt} [{{"id": "obj_1", "bbox": [80, 100, 450, 420], "desc": "a red hot-air balloon"}}]'
    )


@pytest.mark.parametrize(
    "prompt",
    [
        pytest.param("", id="empty"),
        pytest.param(" \n\t", id="whitespace"),
        pytest.param('[{"id": "obj_1", "bbox": [0, 0, 500, 500], "desc": "a cat"}]', id="only-a-box-table"),
    ],
)
def test_regions_around_a_blank_caption_are_refused(prompt: str) -> None:
    with pytest.raises(ValueError, match="need a prompt"):
        build_flux3_layout_prompt(prompt, [BALLOON])


def test_descriptions_reach_the_table_stripped_and_unescaped() -> None:
    region = _region(' ein Schild „Zur Linde“ mit "Kaffee" 🍰\n', [0, 0, 1000, 1000])

    prompt = build_flux3_layout_prompt("Ein Straßencafé.", [region])

    assert prompt == (
        'Ein Straßencafé. [{"id": "obj_1", "bbox": [0, 0, 1000, 1000], '
        '"desc": "ein Schild „Zur Linde“ mit \\"Kaffee\\" 🍰"}]'
    )


@pytest.mark.parametrize(
    "region",
    [
        pytest.param({"desc": "a cup", "bbox": [100, 100, 100, 200]}, id="no-height"),
        pytest.param({"desc": "a cup", "bbox": [100, 100, 200, 100]}, id="no-width"),
        pytest.param({"desc": "a cup", "bbox": [200, 100, 100, 200]}, id="y-reversed"),
        pytest.param({"desc": "a cup", "bbox": [100, 200, 200, 100]}, id="x-reversed"),
        pytest.param({"desc": "a cup", "bbox": [0, 0, 1001, 500]}, id="beyond-the-grid"),
        pytest.param({"desc": "a cup", "bbox": [-1, 0, 500, 500]}, id="negative"),
        pytest.param({"desc": "a cup", "bbox": [0, 0, True, 500]}, id="bool"),
        pytest.param({"desc": "a cup", "bbox": [0, 0, 500.0, 500]}, id="float"),
        pytest.param({"desc": "a cup", "bbox": [0, 0, "500", 500]}, id="numeric-string"),
        pytest.param({"desc": "a cup", "bbox": [0, 0, 500]}, id="three-coordinates"),
        # Reads as 0-1 fractions to the reference parser, so the table would not be detected again.
        pytest.param({"desc": "a cup", "bbox": [0, 0, 1, 1]}, id="read-as-fractions"),
        pytest.param({"desc": " \n", "bbox": [0, 0, 500, 500]}, id="blank-desc"),
        # Ideogram 4's region shape: the description under another key would otherwise reach the model as nothing.
        pytest.param({"prompt": "a cup", "bbox": [0, 0, 500, 500]}, id="prompt-instead-of-desc"),
    ],
)
def test_a_region_outside_the_grid_contract_is_refused(region: dict) -> None:
    with pytest.raises(ValidationError):
        Flux3LayoutRegion.model_validate(region)
