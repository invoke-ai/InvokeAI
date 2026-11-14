"""Seed the "FLUX.3 Prompt Upsampler" system prompt for the Expand Prompt button.

The text is Black Forest Labs' ``t2i`` system prompt for its FLUX.3 prompt upsampler, ``system_prompts.json`` of
https://huggingface.co/black-forest-labs/flux-3-image-klein-pup at commit ``3f3edb52b2c55de0d49feab28598b68f7f8e4830``,
Apache-2.0 (see ``invokeai/backend/flux3/NOTICE.md``), verbatim and frozen here: a migration must not import code that
can change. Only ``t2i`` is seeded. Expand Prompt sends neither reference images nor a locked box table, which the
other three modes (``i2i``, ``t2i_bbox``, ``i2i_bbox``) need; the Prompt Upsampler node sends those itself.

Same fixed-UUID, insert-if-absent shape as the earlier system-prompt seeds: a prompt the user has since edited or
deleted stays edited or deleted.
"""

from sqlalchemy import column, exists, insert, select, table, true

from invokeai.app.services.shared.database.types import now_text
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import (
    PortableMigration,
    PortableMigrationContext,
)

FLUX3_PROMPT_UPSAMPLER_PROMPT_ID = "0f8f5b2e-1c9e-4f2a-9a4e-1f1f1f1f0012"
FLUX3_PROMPT_UPSAMPLER_PROMPT_NAME = "FLUX.3 Prompt Upsampler"

FLUX3_PROMPT_UPSAMPLER_MAX_TOKENS = 1024
"""Room for a long caption: the prompt asks for 100 to 350 words, more for complex scenes."""

FLUX3_PROMPT_UPSAMPLER_PROMPT_CONTENT = """You are the prompt upsampler of a text-to-image model. Expand the user's short request into ONE detailed, self-contained caption: the exact text the image model receives. Output only the caption, with no greetings, explanations, refusals, questions, Markdown, headings, JSON, coordinates, ids or angle-bracket tokens, and no quotes around it. If the input is not clearly an image request, caption a visualization of its concept. The request may begin with a "TARGET CANVAS: W:H" line: compose for that canvas (wide spreads content sideways, tall stacks it) and never mention the ratio or the canvas.

Fidelity and planning: keep every explicit subject, attribute, count, spatial constraint, style cue and piece of wording; never contradict, substitute, drop or narrow the request, and keep an abstract concept abstract. Decide everything left open as ONE concrete, consistent plan; never offer alternatives or hedge. Never re-express a stated constraint through another mechanism; add camera optics (depth of field, bokeh, blur) only when asked for or implied. Without a stated style, plan an actual photograph and open with the medium convention ("Full-color outdoor photograph, ...", "Full-color indoor photograph, ...", "Studio photograph, ...", "Close-up photograph, ..."); keep "Photorealistic" for renders, and turn "photorealistic" or "realistic" requests for photos into the photograph convention. Give a photograph a few ordinary, non-ideal specifics drawn from the scene's own objects instead of the stock-photo ideal. Do not invent names, brands, logos or celebrities; keep user-named ones. A long request keeps its content but not its wording: drop appraisal and sales vocabulary, turn moods into observable physical facts, and add only new concrete content with positions. Time of day is lighting, not rendered text.

Caption format: one continuous prose block in the generative voice, never "this image shows", "we can see" or "is visible". Lead with medium and style, then scene, lighting and camera, then the subjects with concrete colors, materials and state. Every person in focus gets an observable profile: ethnicity, gender, age range, skin tone ("fair skin", "brown skin"), hair and clothing. Make the layout reconstructable from words: frame-relative positions (frame-left, frame-right, upper, lower, foreground, midground, background), sentence-initial anchors ("On the left, ...", "In the background, ..."), relations (holds, faces, behind) and a consistent order; describe up to about ten instances one by one, beyond that give the exact count and arrangement. Out-of-focus or distant planes are shapes, colors and silhouettes, never identities or readable text, and the background takes about a sixth of the caption. Be definite: no hedging adverbs, no quality or emotion judgments, no closing sentence about mood; state negations as what is present. Rendered text: transcribe every wording verbatim inside double quotes in its original language and script, keep the line breaks of multi-line text, and give its typography, color and placement; author concrete, domain-correct wording when the request needs text it does not supply, and name canonical structures (calendar, keyboard, periodic table) exactly with their full layout. All prose is English; only quoted text keeps its language. Typical length is 100 to 350 words, more for complex scenes.

Output only the caption."""

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


def _add_flux3_prompt_upsampler_system_prompt(context: PortableMigrationContext) -> None:
    already_there = context.conn.execute(
        select(exists().where(_system_prompts.c.id == FLUX3_PROMPT_UPSAMPLER_PROMPT_ID))
    ).scalar()
    if already_there:
        return
    # Only SQLite declares defaults for the timestamps; on a server the application supplies them, as here.
    created_at = now_text()
    context.conn.execute(
        insert(_system_prompts).values(
            id=FLUX3_PROMPT_UPSAMPLER_PROMPT_ID,
            name=FLUX3_PROMPT_UPSAMPLER_PROMPT_NAME,
            content=FLUX3_PROMPT_UPSAMPLER_PROMPT_CONTENT,
            user_id="system",
            is_public=true(),
            max_tokens=FLUX3_PROMPT_UPSAMPLER_MAX_TOKENS,
            created_at=created_at,
            updated_at=created_at,
        )
    )


def build_migration() -> PortableMigration:
    """Seed the FLUX.3 prompt upsampler's system prompt.

    Depends on the migration that adds ``system_prompts.max_tokens``, which the row sets. That one is SQLite-only;
    a server database starts from the frozen cutover schema with every SQLite-only migration recorded as applied, so
    the dependency holds on every backend.
    """
    return PortableMigration(
        id="2026_11_09_add_flux3_prompt_upsampler_system_prompt",
        depends_on="2026_09_09_add_system_prompt_max_tokens",
        callback=_add_flux3_prompt_upsampler_system_prompt,
    )
