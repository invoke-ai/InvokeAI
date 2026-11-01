"""MOSS-Image layout prompts built from regions (`invokeai/backend/moss_image/layout.py`).

Expectations marked "OpenMOSS" are ported from the prompt-enhancer skill of https://github.com/OpenMOSS/MOSS-Image at
commit c1355e6b1986dfe8fba3c0455783a05899212342 (`skills/moss-image-prompt-enhancer/scripts/test_compile_prompt.py`
and the example in `references/layout.md`), Apache-2.0. The rest follow from that reference's rules. Every expected
prompt is written out by hand: the model reads the exact string, so key order and separators are part of the result.
"""

import json
import math

import pytest

from invokeai.backend.moss_image.layout import MossImageLayoutRegion, build_moss_image_layout_prompt


def _region(kind: str, caption: str, bbox: list, text: str | None = None) -> MossImageLayoutRegion:
    return MossImageLayoutRegion.model_validate({"kind": kind, "caption": caption, "text": text, "bbox": bbox})


def _box(bbox: list, width: int, height: int) -> list[int]:
    prompt = build_moss_image_layout_prompt("a scene", [_region("object", "a cup", bbox)], width, height)
    return json.loads(prompt)["items"][0]["bbox"]


def test_the_reference_example_compiles_to_its_documented_boxes_in_gallery_key_order() -> None:
    # OpenMOSS references/layout.md: the 1536x1024 campaign image; "For the cup, conversion yields [-288,-592,352,32]".
    # Boxes and serialization follow the reference. The item key order does not: layout.md writes `text` before
    # `bbox` and the reference keeps a draft's order, while this module emits the order of the gallery's layout
    # prompts (assets/images/011 and 014), `label, caption, bbox, text`.
    regions = [
        _region("object", "An ivory ceramic cup.", [176, 224, 800, 864]),
        _region("text", "Large dark-green Chinese title.", [928, 272, 1408, 448], text="把日常，过成诗。"),
        _region("text", "Smaller supporting copy on two lines.", [928, 512, 1376, 656], text="一杯温暖\n一刻从容"),
    ]

    prompt = build_moss_image_layout_prompt("A warm studio campaign for a ceramic cup.", regions, 1536, 1024)

    assert prompt == (
        '{"summary":{"caption":"A warm studio campaign for a ceramic cup."},"items":['
        '{"label":"object","caption":"An ivory ceramic cup.","bbox":[-288,-592,352,32]},'
        '{"label":"text","caption":"Large dark-green Chinese title.","bbox":[-240,160,-64,640],"text":"把日常，过成诗。"},'
        '{"label":"text","caption":"Smaller supporting copy on two lines.","bbox":[0,160,144,608],'
        '"text":"一杯温暖\\n一刻从容"}'
        '],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}'
    )


def test_text_keeps_its_exact_characters_without_ascii_escapes() -> None:
    # OpenMOSS test_landscape_conversion_preserves_content_and_source: Chinese, quotes, a line break and an emoji
    # survive; the serialized prompt itself is one line.
    text = '把日常，过成诗。\n"MOSS" 🪴'
    regions = [_region("text", "Dark-green lettering.", [928, 272, 1408, 448], text=text)]

    prompt = build_moss_image_layout_prompt("A ceramic campaign.", regions, 1536, 1024)

    assert '"bbox":[-240,160,-64,640],"text":"把日常，过成诗。\\n\\"MOSS\\" 🪴"}' in prompt
    assert "\n" not in prompt
    assert json.loads(prompt)["items"][0]["text"] == text


def test_each_kind_becomes_the_item_label_in_region_order_under_the_verbatim_prompt() -> None:
    regions = [
        _region("object", "a cup", [0, 0, 32, 32]),
        _region("character", "a girl", [32, 0, 64, 64]),
        _region("text", "a sign", [0, 32, 32, 64], text="OPEN"),
    ]

    layout = json.loads(build_moss_image_layout_prompt("a shop front\n", regions, 64, 64))

    assert layout["summary"] == {"caption": "a shop front\n"}
    assert layout["items"] == [
        {"label": "object", "caption": "a cup", "bbox": [-32, -32, 0, 0]},
        {"label": "character", "caption": "a girl", "bbox": [-32, 0, 32, 32]},
        {"label": "text", "caption": "a sign", "bbox": [0, -32, 32, 0], "text": "OPEN"},
    ]


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        # OpenMOSS test_portrait_edges_and_fractional_enclosure, on a 768x1344 canvas.
        ([0, 0, 768, 1344], [-672, -384, 672, 384]),
        ([10.9, 20.1, 100.01, 200.99], [-652, -374, -471, -283]),
        ([1.00000000001, 2, 100, 200], [-670, -383, -472, -284]),
        # Rounding to 9 decimals comes before ceil, so float noise above an edge does not widen the box by a pixel.
        ([2, 1, 100.0000000001, 200.0000000001], [-671, -382, -472, -284]),
    ],
)
def test_boxes_enclose_their_pixels_and_centre_on_the_canvas(bbox: list, expected: list[int]) -> None:
    assert _box(bbox, 768, 1344) == expected


@pytest.mark.parametrize(
    "bbox",
    [
        # OpenMOSS test_invalid_geometry_is_rejected, on a 1536x1024 canvas.
        [-1, 0, 100, 100],
        [0, 0, 1537, 100],
        [0, 0, 100, 1025],
        [100, 0, 50, 100],
        [0, 0, 0, 100],
        [0, 0, 100],
        [True, 0, 100, 100],
        [0, 0, math.nan, 100],
        [0, 0, math.inf, 100],
        [0, 0, "100", 100],
        [0, 0, 0.00000000001, 100],
        [0, 0, 100, 100, 100],
    ],
)
def test_invalid_geometry_is_rejected(bbox: list) -> None:
    with pytest.raises(ValueError):
        _box(bbox, 1536, 1024)


def test_a_box_the_image_cannot_hold_names_its_region() -> None:
    cup = _region("object", "a cup", [0, 0, 100, 100])

    with pytest.raises(ValueError, match=r"Layout region 2 \('a kite'\) .* leaves the 1536x1024 image"):
        build_moss_image_layout_prompt("a beach", [cup, _region("object", "a kite", [0, 0, 1600, 100])], 1536, 1024)
    with pytest.raises(ValueError, match=r"Layout region 1 \('a speck'\) .* no area at whole-pixel precision"):
        build_moss_image_layout_prompt("a beach", [_region("object", "a speck", [0, 0, 1e-11, 100]), cup], 1536, 1024)


@pytest.mark.parametrize(
    "prompt",
    [
        "  a fox at dawn \n",
        '{\n  "summary": {"caption": "a fox"},\n  "items": []\n}',
        '{"scene_type":"photo","elements":[]}',
        "{not json",
        "",
    ],
)
def test_without_regions_the_prompt_passes_through_unchanged(prompt: str) -> None:
    assert build_moss_image_layout_prompt(prompt, [], 1024, 1024) == prompt


def test_regions_are_appended_to_a_compiled_layout_which_keeps_everything_else() -> None:
    # Key order as written (not the order this module emits), summary keys and item extras the regions do not use.
    existing = (
        '{"bbox_coord_space": "abs_centralized",\n'
        ' "summary": {"meta": ["poster"], "canvas": "single", "caption": "A tea poster."},\n'
        ' "items": [{"label": "panel", "caption": "The frame.", "tags": ["border"], "bbox": [-512, -512, 512, 512]}],\n'
        ' "bbox_format": "yxyx"}'
    )

    prompt = build_moss_image_layout_prompt(
        existing, [_region("text", "Brush lettering.", [412, 100, 612, 300], text="茶")], 1024, 1024
    )

    assert prompt == (
        '{"bbox_coord_space":"abs_centralized",'
        '"summary":{"meta":["poster"],"canvas":"single","caption":"A tea poster."},"items":['
        '{"label":"panel","caption":"The frame.","tags":["border"],"bbox":[-512,-512,512,512]},'
        '{"label":"text","caption":"Brush lettering.","bbox":[-412,-100,-212,100],"text":"茶"}'
        '],"bbox_format":"yxyx"}'
    )


@pytest.mark.parametrize(
    "prompt",
    [
        pytest.param(
            '{"summary":{"caption":"x"},"items":[],"bbox_coord_space":"normalized_1000","bbox_format":"yxyx"}',
            id="other-coord-space",
        ),
        pytest.param(
            '{"summary":{"caption":"x"},"items":[],"bbox_coord_space":"abs_centralized","bbox_format":"xyxy"}',
            id="other-bbox-format",
        ),
        pytest.param('{"summary":{"caption":"x"},"items":[]}', id="no-convention"),
        pytest.param(
            '{"summary":{"caption":"x"},"items":{},"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}',
            id="items-not-a-list",
        ),
    ],
)
def test_a_layout_that_is_not_compiled_is_refused(prompt: str) -> None:
    with pytest.raises(ValueError, match="compiled"):
        build_moss_image_layout_prompt(prompt, [_region("object", "a cup", [0, 0, 16, 16])], 1024, 1024)


@pytest.mark.parametrize(
    "prompt",
    [
        # The model's other JSON prompt schema (`elements`, as in several of its gallery prompts) has no `items`.
        '{"scene_type":"poster","gen_brief":"A tea poster.","elements":[],"bbox_coord_space":"abs_centralized"}',
        '["a", "list"]',
        '"a quoted string"',
        '{"summary": {"caption": "cut off"}, "items": [',
        '{"summary":{},"items":[],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx","seed":NaN}',
        # A layout needs both `summary` and `items`; compiled conventions alone do not make one.
        '{"items":[],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}',
    ],
)
def test_a_prompt_that_is_not_a_layout_becomes_the_caption(prompt: str) -> None:
    layout = json.loads(
        build_moss_image_layout_prompt(prompt, [_region("object", "a cup", [496, 496, 528, 528])], 1024, 1024)
    )

    assert layout == {
        "summary": {"caption": prompt},
        "items": [{"label": "object", "caption": "a cup", "bbox": [-16, -16, 16, 16]}],
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }


@pytest.mark.parametrize("prompt", ["", " \n"])
def test_regions_need_a_prompt_to_caption_the_layout(prompt: str) -> None:
    # OpenMOSS requires a non-blank summary.caption.
    with pytest.raises(ValueError, match="need a prompt"):
        build_moss_image_layout_prompt(prompt, [_region("object", "a cup", [0, 0, 16, 16])], 1024, 1024)


# Layout drafts: the absolute-pixel JSON that Expand Prompt's MOSS-Image Prompt Enhancer asks an LLM to write.

_DRAFT = (
    '{"width":1536,"height":1024,"bbox_coord_space":"abs","bbox_format":"xyxy",'
    '"summary":{"meta":["poster"],"canvas":"single","caption":"A ceramic campaign."},"items":['
    '{"label":"object","caption":"An ivory cup.","bbox":[176,224,800,864],"tags":["product"]},'
    '{"label":"text","caption":"Dark-green lettering.","text":"把日常，过成诗。\\n\\"MOSS\\" 🪴","bbox":[928,272,1408,448]},'
    '{"label":"character","caption":"A potter.","palette":["ochre"],"bbox":[10.9,20.1,100.01,200.99]}]}'
)
# What the OpenMOSS reference `compile_prompt` returns for _DRAFT, captured from it at the commit named above.
_DRAFT_COMPILED_BY_REFERENCE = (
    '{"summary":{"meta":["poster"],"canvas":"single","caption":"A ceramic campaign."},"items":['
    '{"label":"object","caption":"An ivory cup.","bbox":[-288,-592,352,32],"tags":["product"]},'
    '{"label":"text","caption":"Dark-green lettering.","text":"把日常，过成诗。\\n\\"MOSS\\" 🪴","bbox":[-240,160,-64,640]},'
    '{"label":"character","caption":"A potter.","palette":["ochre"],"bbox":[-492,-758,-311,-667]}'
    '],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}'
)


def _draft(**changes: object) -> dict:
    draft = json.loads(_DRAFT)
    draft.update(changes)
    return draft


def _items(*items: object) -> dict:
    return _draft(items=list(items))


def _compile(draft: dict, width: int = 1536, height: int = 1024) -> dict:
    return json.loads(build_moss_image_layout_prompt(json.dumps(draft, ensure_ascii=False), [], width, height))


def test_a_draft_at_its_own_size_compiles_exactly_as_the_reference_does() -> None:
    assert build_moss_image_layout_prompt(_DRAFT, [], 1536, 1024) == _DRAFT_COMPILED_BY_REFERENCE


@pytest.mark.parametrize(
    "prompt",
    [
        pytest.param(f"\n  {_DRAFT}\n", id="whitespace"),
        pytest.param(f"```json\n{_DRAFT}\n```", id="json-fence"),
        pytest.param(f"  ```\n{_DRAFT}\n```\n", id="bare-fence"),
        pytest.param(f"```json5\n{_DRAFT}\n```", id="json5-fence"),
        pytest.param(f"```ld-json \t\n{_DRAFT}\n```", id="ld-json-fence"),
        pytest.param(f"```json\r\n{_DRAFT}\r\n```", id="crlf-fence"),
    ],
)
def test_a_draft_wrapped_the_way_llms_reply_is_compiled(prompt: str) -> None:
    assert build_moss_image_layout_prompt(prompt, [], 1536, 1024) == _DRAFT_COMPILED_BY_REFERENCE


def test_a_draft_is_scaled_per_axis_to_the_generation_size() -> None:
    # 1536 -> 1024 wide (2/3), height unchanged; edges are enclosed in whole pixels after scaling.
    layout = _compile(_draft(), 1024, 1024)

    assert [item["bbox"] for item in layout["items"]] == [
        [-288, -395, 352, 22],
        [-240, 106, -64, 427],
        [-492, -505, -311, -445],
    ]


def test_a_draft_frame_need_not_be_a_multiple_of_16() -> None:
    # Small LLMs write sizes like 1000; they only set the frame the boxes are measured in.
    draft = _draft(
        width=1000, height=1000, items=[{"label": "object", "caption": "a cup", "bbox": [250, 500, 750, 1000]}]
    )
    assert _compile(draft, 1024, 1024)["items"][0]["bbox"] == [0, -256, 512, 256]


def test_regions_follow_the_compiled_draft_items() -> None:
    prompt = build_moss_image_layout_prompt(_DRAFT, [_region("object", "a vase", [0, 0, 16, 16])], 1536, 1024)

    items = json.loads(prompt)["items"]
    assert [item["caption"] for item in items] == ["An ivory cup.", "Dark-green lettering.", "A potter.", "a vase"]
    assert items[-1]["bbox"] == [-512, -768, -496, -752]


def test_unknown_fields_are_dropped_and_unknown_labels_become_objects_or_text() -> None:
    draft = _draft(id="draft-1", notes="ignore", width=1000, height=1000)
    draft["summary"]["mood"] = "calm"
    draft["items"] = [
        {"id": 1, "label": "product", "caption": "a cup", "bbox": [0, 0, 500, 500], "children": []},
        {"label": "headline", "caption": "a title", "text": "TEA", "bbox": [500, 0, 1000, 500]},
    ]

    assert _compile(draft, 1024, 1024) == {
        "summary": {"meta": ["poster"], "canvas": "single", "caption": "A ceramic campaign."},
        "items": [
            {"label": "object", "caption": "a cup", "bbox": [-512, -512, 0, 0]},
            {"label": "text", "caption": "a title", "text": "TEA", "bbox": [-512, 0, 0, 512]},
        ],
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }


def test_boxes_leaving_the_canvas_are_clamped_and_boxes_without_area_dropped() -> None:
    layout = _compile(
        _items(
            {"label": "object", "caption": "clamped", "bbox": [-20, -20, 1600, 512]},
            {"label": "object", "caption": "off canvas", "bbox": [1600, 0, 1700, 100]},
            {"label": "object", "caption": "inverted", "bbox": [100, 0, 50, 100]},
            {"label": "object", "caption": "a hairline", "bbox": [0, 0, 0.00000000001, 100]},
        )
    )

    assert layout["items"] == [{"label": "object", "caption": "clamped", "bbox": [-512, -768, 0, 768]}]


def test_a_draft_without_any_box_on_the_canvas_is_refused() -> None:
    with pytest.raises(ValueError, match="no item has a box on the canvas"):
        _compile(_items({"label": "object", "caption": "off canvas", "bbox": [1600, 0, 1700, 100]}))


@pytest.mark.parametrize(
    ("draft", "message"),
    [
        # OpenMOSS: text items require literal text.
        (_items({"label": "text", "caption": "a title", "bbox": [0, 0, 16, 16]}), "item 1 has no text"),
        (_items({"label": "text", "caption": "a title", "text": " ", "bbox": [0, 0, 16, 16]}), "item 1 has blank text"),
        (_items({"label": "object", "caption": "", "bbox": [0, 0, 16, 16]}), "item 1 has no caption"),
        (_items({"label": "object", "caption": "a cup", "bbox": [0, 0, 16]}), "item 1 needs a box"),
        # OpenMOSS: booleans and numeric strings are not coordinates.
        (_items({"label": "object", "caption": "a cup", "bbox": [True, 0, 16, 16]}), "item 1 needs a box"),
        (_items({"label": "object", "caption": "a cup", "bbox": [0, 0, "16", 16]}), "item 1 needs a box"),
        (_items({"label": "object", "caption": "a cup", "tags": "cup", "bbox": [0, 0, 16, 16]}), "item 1: its tags"),
        (_items("a cup"), "item 1 is not an object"),
        (_draft(items=[]), "it has no items"),
        (_draft(summary={"meta": ["poster"]}), "the summary has no caption"),
        (_draft(summary={"caption": "  "}), "the summary has no caption"),
        (_draft(summary={"caption": "x", "canvas": "grid"}), "the summary canvas"),
        (_draft(summary={"caption": "x", "meta": "poster"}), "the summary meta"),
        # OpenMOSS rejects these sizes too; a frame that is not a multiple of 16 is accepted here.
        (_draft(width=0), "width and height"),
        (_draft(width=True), "width and height"),
        (_draft(height="1024"), "width and height"),
    ],
)
def test_an_invalid_draft_is_refused_naming_what_is_wrong(draft: dict, message: str) -> None:
    with pytest.raises(ValueError, match=f"^Layout draft: {message}"):
        _compile(draft)


def test_a_draft_number_that_is_not_json_makes_the_prompt_a_caption() -> None:
    # OpenMOSS rejects NaN; such text is not JSON, so it passes through like any other caption.
    prompt = _DRAFT.replace("[176,224,800,864]", "[176,224,NaN,864]")
    assert build_moss_image_layout_prompt(prompt, [], 1536, 1024) == prompt


@pytest.mark.parametrize(
    "prompt",
    [
        # Other conventions are not drafts; without regions they pass through as before.
        '{"width":1024,"height":1024,"bbox_coord_space":"abs_centralized","bbox_format":"xyxy","summary":{},"items":[]}',
        '{"width":1024,"height":1024,"bbox_coord_space":"normalized_1000","bbox_format":"xyxy","items":[]}',
        # Prose around a fence is not unwrapped.
        f"Here is the layout:\n```json\n{_DRAFT}\n```",
    ],
)
def test_json_that_is_not_a_draft_is_left_alone(prompt: str) -> None:
    assert build_moss_image_layout_prompt(prompt, [], 1536, 1024) == prompt
