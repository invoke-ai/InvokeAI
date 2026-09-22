"""Runtime assembly of Ideogram 4's structured JSON caption.

This is the Python port of the frontend ``buildIdeogram4Caption`` (buildIdeogram4Prompt.ts). It runs
at generation time (inside the ``ideogram4_caption_builder`` node) rather than at graph-build time, so
that dynamic-prompt expansions and prompt batching — which vary the *global* prompt — are reflected in
the encoded caption. The regions (description + bbox) and color palette are fixed per generation and
are passed in as node inputs.

Schema notes (must match the reference the model was trained on):
  - bbox is ``[y_min, x_min, y_max, x_max]``, normalized to 0–1000, origin top-left. The graph builder
    computes it from canvas coordinates; here it is passed through verbatim.
  - Key order matters: ``high_level_description``, (``style_description``), ``compositional_deconstruction``;
    ``obj`` elements use ``type``, ``bbox``, ``desc``. Python dicts preserve insertion order.
  - Serialized with compact separators and non-ASCII preserved (matches JS ``JSON.stringify`` and the
    reference's ``ensure_ascii=False``).
"""

import json
import re
from typing import Optional

_HEX_COLOR_RE = re.compile(r"#[0-9A-F]{6}")

# Centered, deliberately sub-full-frame bbox ([y_min, x_min, y_max, x_max], 0–1000) for the element we
# auto-synthesize when the user drew no regions. Empirically, Ideogram 4's built-in safety filter blocks
# "degenerate" captions — an empty ``elements`` list, or a single *full-frame* [0,0,1000,1000] element
# whose desc merely repeats ``high_level_description``. A partial bbox (or a distinct desc) avoids the
# block; since a no-region prompt only gives us one description, we make the default element partial.
_DEFAULT_SCENE_BBOX = (100, 100, 900, 900)


def build_ideogram4_caption(
    global_prompt: str,
    regions: list[tuple[str, Optional[list[int]]]],
    color_palette: list[str],
) -> str:
    """Assemble the Ideogram 4 prompt from a global prompt, regions, and an optional color palette.

    - Raw-JSON passthrough: if the trimmed global prompt already starts with ``{`` it is returned
      verbatim (the user controls the JSON; regions/palette are ignored).
    - Otherwise a structured JSON caption is always built — regions become ``obj`` elements and a
      palette becomes ``style_description.color_palette``. A prompt with neither still becomes a
      minimal caption (just ``high_level_description``), never bare plain text: Ideogram is trained on
      the JSON schema and its safety filter false-positives far more on plain text.

    ``regions`` is a list of ``(description, bbox)`` where bbox is ``[y_min, x_min, y_max, x_max]`` or
    None. Regions with a blank description are dropped.
    """
    trimmed = global_prompt.strip()

    # The user pasted a structured caption (or any JSON object) — use it verbatim.
    if trimmed.startswith("{"):
        return global_prompt

    elements: list[dict] = []
    for description, bbox in regions:
        if not description.strip():
            continue
        if bbox is not None:
            elements.append({"type": "obj", "bbox": bbox, "desc": description})
        else:
            elements.append({"type": "obj", "desc": description})

    # No usable regions: synthesize one default element describing the whole scene from the prompt, with
    # a partial (non-full-frame) bbox. An empty `elements` list — or a full-frame element repeating the
    # prompt — trips Ideogram's safety filter (verified empirically); a partial default box avoids it.
    if not elements:
        elements = [{"type": "obj", "bbox": list(_DEFAULT_SCENE_BBOX), "desc": trimmed}]

    # Normalize the palette to uppercase #RRGGBB (the schema's required hex form); drop invalid entries.
    # Mirror the JS order: uppercase first, then validate.
    palette = [c.upper() for c in color_palette if _HEX_COLOR_RE.fullmatch(c.upper())]

    # Always emit a structured caption (never bare plain text): Ideogram 4 is trained on the JSON schema
    # and its built-in safety filter false-positives far more often on plain-text prompts than on
    # structured JSON. A prompt with no regions/palette becomes a minimal caption carrying just the
    # high_level_description (with an empty compositional_deconstruction). Raw-JSON pastes still pass
    # through verbatim above.
    compositional_deconstruction = {"background": "", "elements": elements}
    if palette:
        caption = {
            "high_level_description": trimmed,
            "style_description": {"color_palette": palette},
            "compositional_deconstruction": compositional_deconstruction,
        }
    else:
        caption = {
            "high_level_description": trimmed,
            "compositional_deconstruction": compositional_deconstruction,
        }

    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
