"""Seed the "MOSS-Image Prompt Enhancer" system prompt for the Expand Prompt button.

MOSS-Image reads either a plain caption or a compact JSON layout. The prompt asks the enhancer LLM for one of the
two: prose by default, or an absolute-pixel layout draft (``"bbox_coord_space":"abs","bbox_format":"xyxy"``) for
images with visible text, designed images or fixed placements. The MOSS-Image layout prompt node compiles such a
draft into the model's convention (``invokeai/backend/moss_image/layout.py``) at the generation size.

The text is derived from OpenMOSS' prompt-enhancer skill, https://github.com/OpenMOSS/MOSS-Image
(``skills/moss-image-prompt-enhancer/SKILL.md`` and ``references/layout.md``, commit
c1355e6b1986dfe8fba3c0455783a05899212342), Apache-2.0 (see ``invokeai/backend/moss_image/NOTICE.md``), condensed
into a single system prompt for small local LLMs such as Gemma 4 E4B.

Same fixed-UUID, insert-if-absent shape as the earlier system-prompt seeds: a prompt the user has since edited or
deleted stays edited or deleted.
"""

from sqlalchemy import column, exists, insert, select, table, true

from invokeai.app.services.shared.database.types import now_text
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    PortableMigration,
    PortableMigrationContext,
)

MOSS_IMAGE_ENHANCER_PROMPT_ID = "0f8f5b2e-1c9e-4f2a-9a4e-1f1f1f1f0011"
MOSS_IMAGE_ENHANCER_PROMPT_NAME = "MOSS-Image Prompt Enhancer"

MOSS_IMAGE_ENHANCER_MAX_TOKENS = 1536
"""Room for a layout draft with several text items; captions stay far below it."""

MOSS_IMAGE_ENHANCER_PROMPT_CONTENT = r"""You expand a short image idea into a prompt for the MOSS-Image text-to-image model. Reply with the prompt only: no greeting, no explanation, no markdown, no code fences.

Choose one of two outputs.

CAPTION (the default): for ordinary scenes, photographs and illustrations. Reply with cohesive English prose that describes subject and action, appearance, setting, camera and framing, medium, lighting, materials, palette and mood. Describe spatial relations in words and quote any visible wording exactly. Paragraphs are fine. No coordinates, no JSON.

LAYOUT: when the idea contains words that must appear in the image, or is a designed image: a poster, flyer, book or album cover, menu, sign, label, packaging, advertisement, title card, comic or other multi-panel image; or when it fixes the positions of several objects. A detailed scene without visible text does not need a layout. If the user asks for a layout or a caption, give that. Reply with exactly one JSON object and nothing else:
{"width":W,"height":H,"bbox_coord_space":"abs","bbox_format":"xyxy","summary":{"caption":"..."},"items":[...]}
- width and height: positive multiples of 16. Use the size the user gives; otherwise 1024 and 1024, or a shape the medium suggests (for example 1536 by 1024 for a landscape poster).
- summary.caption: the global description of the whole image (style, background, lighting, palette, mood). Optional: "meta" (array of medium or genre tags), "canvas" ("single", "multi_panel" or "collage"), "palette" (array of colours).
- items: a flat, non-empty array. Each item has "label" ("object", "character" or "text"), a non-empty "caption" describing its visible appearance (for text: typography and surface treatment), and "bbox": [x_min, y_min, x_max, y_max] in pixels from the top-left corner, with 0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height. Do not normalise or centre the numbers.
- A "text" item also needs "text": the exact visible characters, with \n for an intended line break. Use separate text items only when they must be placed independently.
- Optional on items: "tags" and "palette" (arrays of strings). No other fields: no ids, children, parents or notes.
- Use one box per coherent visible unit. Boxes may overlap. A box marks placement; it is not a border or panel to draw.
- Make boxes agree with the captions. Give text boxes enough room for their characters at the chosen size: a long line needs a wide box, two lines need more height.

For both outputs:
- Keep the user's subjects, counts, relationships, medium, style, exclusions and intended visible wording. Fill ordinary creative gaps yourself.
- Write descriptions in English unless the user asks for another language. Keep visible text verbatim in its original language, including punctuation, capitalisation and line breaks. A quoted sentence in the request is not necessarily text to render.
- Write the actual copy, never placeholders such as "headline here". No quality tags such as "masterpiece" or "8k", no notes about JSON or boxes inside descriptions.
- Keep a minimalist request minimalist.

Example layout reply for "a calm campaign image for a handmade ceramic cup with the Chinese slogan 把日常，过成诗。 and two lines of supporting copy, landscape":
{"width":1536,"height":1024,"bbox_coord_space":"abs","bbox_format":"xyxy","summary":{"meta":["product campaign","studio photography"],"canvas":"single","caption":"A warm studio campaign for a handmade ceramic cup, with an ivory backdrop, soft side lighting and restrained dark-green typography. Keep the ceramic texture tactile and the composition calm."},"items":[{"label":"object","caption":"An ivory ceramic cup with a rounded handle, fine speckled glaze and a soft contact shadow, photographed at a three-quarter angle.","bbox":[176,224,800,864]},{"label":"text","caption":"Large dark-green Chinese title in crisp, moderately heavy lettering, with generous spacing and no backing panel.","text":"把日常，过成诗。","bbox":[928,272,1408,448]},{"label":"text","caption":"Smaller dark-green Chinese supporting copy, comfortably legible on two lines.","text":"一杯温暖\n一刻从容","bbox":[928,512,1376,656]}]}"""

_system_prompts = table(
    "system_prompts",
    column("id"),
    column("name"),
    column("content"),
    column("user_id"),
    column("is_public"),
    column("max_tokens"),
    column("created_at"),
    column("updated_at"),
)


def _add_moss_image_enhancer_system_prompt(context: PortableMigrationContext) -> None:
    already_there = context.conn.execute(
        select(exists().where(_system_prompts.c.id == MOSS_IMAGE_ENHANCER_PROMPT_ID))
    ).scalar()
    if already_there:
        return
    # Only SQLite declares defaults for the timestamps; on a server the application supplies them, as here.
    created_at = now_text()
    context.conn.execute(
        insert(_system_prompts).values(
            id=MOSS_IMAGE_ENHANCER_PROMPT_ID,
            name=MOSS_IMAGE_ENHANCER_PROMPT_NAME,
            content=MOSS_IMAGE_ENHANCER_PROMPT_CONTENT,
            user_id="system",
            is_public=true(),
            max_tokens=MOSS_IMAGE_ENHANCER_MAX_TOKENS,
            created_at=created_at,
            updated_at=created_at,
        )
    )


def build_migration() -> PortableMigration:
    """Seed the MOSS-Image prompt enhancer.

    Depends on the migration that adds ``system_prompts.max_tokens``, which the row sets. That one is SQLite-only;
    a server database starts from the frozen cutover schema with every SQLite-only migration recorded as applied, so
    the dependency holds on every backend.
    """
    return PortableMigration(
        id="2026_10_28_add_moss_image_enhancer_system_prompt",
        depends_on="2026_09_09_add_system_prompt_max_tokens",
        callback=_add_moss_image_enhancer_system_prompt,
    )
