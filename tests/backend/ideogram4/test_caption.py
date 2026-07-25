"""Tests for the Ideogram 4 runtime caption assembly (Python port of buildIdeogram4Caption).

Kept behaviorally identical to the frontend assembler so that moving the work into the
ideogram4_caption_builder node (so dynamic prompts / batching vary the encoded caption) does not
change the encoded output for a given (prompt, regions, palette).
"""

import json

from invokeai.backend.ideogram4.caption import build_ideogram4_caption


def test_raw_json_passthrough_is_verbatim():
    raw = '{"high_level_description": "a cat"}'
    # Regions and palette are ignored for raw-JSON prompts; whitespace preserved verbatim.
    assert build_ideogram4_caption(raw, [("ignored", [0, 0, 10, 10])], ["#FFFFFF"]) == raw


def test_plain_text_without_regions_or_palette_returns_trimmed_prompt():
    assert build_ideogram4_caption("  a serene lake  ", [], []) == "a serene lake"


def test_region_with_bbox_builds_obj_element_in_key_order():
    result = build_ideogram4_caption("a scene", [("a red bird", [10, 20, 300, 400])], [])
    parsed = json.loads(result)
    assert parsed == {
        "high_level_description": "a scene",
        "compositional_deconstruction": {
            "background": "",
            "elements": [{"type": "obj", "bbox": [10, 20, 300, 400], "desc": "a red bird"}],
        },
    }
    # obj element key order must be type, bbox, desc (trained schema order).
    assert list(parsed["compositional_deconstruction"]["elements"][0].keys()) == ["type", "bbox", "desc"]


def test_region_without_bbox_omits_bbox_key():
    result = build_ideogram4_caption("a scene", [("no drawn content", None)], [])
    element = json.loads(result)["compositional_deconstruction"]["elements"][0]
    assert element == {"type": "obj", "desc": "no drawn content"}


def test_blank_region_prompt_is_dropped():
    result = build_ideogram4_caption("a scene", [("   ", [0, 0, 10, 10]), ("kept", None)], [])
    elements = json.loads(result)["compositional_deconstruction"]["elements"]
    assert len(elements) == 1
    assert elements[0]["desc"] == "kept"


def test_palette_is_normalized_and_invalid_entries_dropped():
    result = build_ideogram4_caption("a scene", [], ["#ff0000", "not-a-color", "#00FF00"])
    parsed = json.loads(result)
    assert parsed["style_description"]["color_palette"] == ["#FF0000", "#00FF00"]


def test_palette_only_still_builds_structured_caption():
    result = build_ideogram4_caption("a scene", [], ["#123ABC"])
    parsed = json.loads(result)
    # Strict key order: high_level_description, style_description, compositional_deconstruction.
    assert list(parsed.keys()) == ["high_level_description", "style_description", "compositional_deconstruction"]
    assert parsed["compositional_deconstruction"]["elements"] == []


def test_compact_separators_and_non_ascii_preserved():
    result = build_ideogram4_caption("café ☕", [("naïve", None)], [])
    # Compact separators (no spaces after , or :) and raw non-ASCII characters (not \\uXXXX escapes).
    assert ", " not in result
    assert '": ' not in result
    assert "café ☕" in result
    assert "naïve" in result
