"""Seed the LTX-2.5 Text-to-Video and Image-to-Video system prompts for the Expand Prompt button.

LTX-2.5's own text encoder is not trained to enhance prompts; Lightricks pairs the model with
`google/gemma-4-E2B-it` and these two "capstyle_plus" system prompts, which rewrite a short request
into the single-paragraph audio-visual caption style the model was trained on. The texts are
verbatim from diffusers 0.40.0 (`pipelines/ltx2/utils.py`, `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT` and
`LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT`); they are copied rather than imported so that a later diffusers
release cannot change what this migration seeds.

The Image-to-Video prompt describes the video from its first frame, so Expand Prompt sends that
frame along when the model can read images.

Same fixed-UUID / ``INSERT OR IGNORE`` shape as the MiniMax H3 Ref2VA seed: a prompt the user has
since edited or deleted stays edited or deleted.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration

LTX2_5_T2V_PROMPT_ID = "0f8f5b2e-1c9e-4f2a-9a4e-1f1f1f1f0009"
LTX2_5_I2V_PROMPT_ID = "0f8f5b2e-1c9e-4f2a-9a4e-1f1f1f1f0010"

LTX2_5_MAX_TOKENS = 600
"""Upstream's own cap (`GEMMA4_PROMPT_ENHANCEMENT_CONFIG.max_new_tokens`). The prompts ask for
150-220 words; measured outputs with Gemma-4-E2B ran 118-257 tokens."""

LTX2_5_T2V_PROMPT_CONTENT = """You are given a user's short text-to-video request. Write a single, highly detailed audio-visual caption describing the video that best fulfills that request, in the EXACT style of the training captions used for this video model. The generated video is scored against the user's ORIGINAL request, so preserve every element the user stated; expand faithfully into the full caption style without contradicting or dropping anything they asked for.

Match this captioning style precisely:

1. Begin immediately with the action or visual detail. Do NOT use "The scene opens…", "We see…", "There is…".

2. Objective, observable description only. Do not infer emotions or intentions — describe what is visible and audible (e.g. not "he looks sad" but "his eyebrows angle downward and his lips are pressed together").

3. Full visual detail: environment (materials, textures, lighting, colors), character appearance (clothing, posture, facial details), and the spatial positioning of all elements. When a human appears, identify them specifically (gendered terms when clearly implied; differentiate multiple people consistently) and describe visible physical attributes — apparent gender presentation, skin tone, estimated age group, hair color/length/style, build, clothing and accessories. Do not infer ethnicity, nationality, religion, or culture.

4. Precise motion and cinematic description. For every shot you MUST include, woven naturally into the prose (never as tags or labels):
   - Shot type (exactly one: extreme wide shot / wide shot / medium shot / medium close-up / close-up / extreme close-up)
   - Camera motion (always stated; if none, explicitly say the camera remains static). Camera movement is expected and good — match the user if they specified it, otherwise choose the treatment that best presents the requested scene.
   - Camera viewpoint relative to subject (front-facing / back-facing / side view / over-the-shoulder / top-down / low-angle / high-angle).
   Express these as flowing prose: "a medium shot frames…, captured from a front-facing angle as the camera slowly pans…". Never as "medium shot, static camera —".

5. Complete soundscape, integrated naturally: any dialogue (quote it exactly, in the original language), tone of voice, background music (type, mood, volume changes), and environmental sounds (footsteps, wind, traffic, animals). If the request implies sound, describe it plausibly.

6. Strict chronological, real-time flow using transitions like "Initially…", "A moment later…", "Simultaneously…". Keep every stated action in motion.

7. One single continuous paragraph. No bullet points, no section headers, no labels like "Audio:" or "Visual:". Exhaustive and lossless — include background elements, subtle movements, lighting, secondary sounds — detailed enough to reconstruct the scene. Aim for a rich, complete paragraph (roughly 150–220 words).

If the user wrote in another language, produce the English caption of the same content. Output ONLY the caption text — no JSON, no preamble.

AESTHETIC QUALITY (in addition to the above, without breaking the objective caption style): render the described scene with strong visual production value — cinematic, film-grade color and contrast, beautiful natural lighting, crisp fine detail and texture, pleasing composition and depth. Weave these quality descriptors naturally into the same observable prose (e.g. "warm cinematic lighting", "richly saturated film-grade color", "crisp high-resolution detail") — describe how the exact requested scene LOOKS at its most visually striking, never adding new objects or actions. Keep everything else (framing triple, soundscape, chronological single paragraph, faithfulness) exactly as specified.
"""

LTX2_5_I2V_PROMPT_CONTENT = """You are given a REFERENCE IMAGE (the exact first frame of the video) and a user's short image-to-video request. Write a single, highly detailed audio-visual caption describing the video that BEGINS from this exact reference image and best fulfills that request, in the EXACT style of the training captions used for this video model. The generated video is scored against the user's ORIGINAL request, so preserve every element the user stated; expand faithfully into the full caption style without contradicting or dropping anything they asked for.

FIRST-FRAME / IMAGE GROUNDING (do this first): the opening of your caption must match the reference image exactly — same subject(s), identity, appearance, clothing, setting, lighting, and composition as shown. The video starts on this frame; describe it faithfully, then narrate chronologically as the user's requested action unfolds from it. Never contradict, replace, or invent things not consistent with the image. Single continuous take — no hard cuts.

Match this captioning style precisely:

1. Begin immediately with the action or visual detail. Do NOT use "The scene opens…", "We see…", "There is…".

2. Objective, observable description only. Do not infer emotions or intentions — describe what is visible and audible (e.g. not "he looks sad" but "his eyebrows angle downward and his lips are pressed together").

3. Full visual detail: environment (materials, textures, lighting, colors), character appearance (clothing, posture, facial details), and the spatial positioning of all elements — grounded in and consistent with the reference image. When a human appears, identify them specifically (gendered terms when clearly implied; differentiate multiple people consistently) and describe visible physical attributes — apparent gender presentation, skin tone, estimated age group, hair color/length/style, build, clothing and accessories. Do not infer ethnicity, nationality, religion, or culture.

4. Precise motion and cinematic description. For every shot you MUST include, woven naturally into the prose (never as tags or labels):
   - Shot type (exactly one: extreme wide shot / wide shot / medium shot / medium close-up / close-up / extreme close-up) — consistent with how the reference image is framed at the start.
   - Camera motion (always stated; if none, explicitly say the camera remains static). Camera movement is expected and good — match the user if they specified it, otherwise choose the treatment that best presents the requested scene starting from this frame.
   - Camera viewpoint relative to subject (front-facing / back-facing / side view / over-the-shoulder / top-down / low-angle / high-angle) — matching the reference image's viewpoint at the opening.
   Express these as flowing prose: "a medium shot frames…, captured from a front-facing angle as the camera slowly pans…". Never as "medium shot, static camera —".

5. Complete soundscape, integrated naturally: any dialogue (quote it exactly, in the original language), tone of voice, background music (type, mood, volume changes), and environmental sounds (footsteps, wind, traffic, animals). If the request implies sound, describe it plausibly.

6. Strict chronological, real-time flow using transitions like "Initially…", "A moment later…", "Simultaneously…". Keep the user's requested motion/action central and in motion throughout.

7. One single continuous paragraph. No bullet points, no section headers, no labels like "Audio:" or "Visual:". Exhaustive and lossless — include background elements, subtle movements, lighting, secondary sounds — detailed enough to reconstruct the scene. Aim for a rich, complete paragraph (roughly 150–220 words).

If the user wrote in another language, produce the English caption of the same content. Output ONLY the caption text — no JSON, no preamble.

AESTHETIC QUALITY (in addition to the above, without breaking the objective caption style or contradicting the reference image): render the described scene with strong visual production value — cinematic, film-grade color and contrast, beautiful natural lighting, crisp fine detail and texture, pleasing composition and depth. Weave these quality descriptors naturally into the same observable prose (e.g. "warm cinematic lighting", "richly saturated film-grade color", "crisp high-resolution detail") — describe how the exact requested scene, starting from this frame, LOOKS at its most visually striking, never adding new objects or actions and never contradicting the first frame. Keep everything else (first-frame grounding, framing triple, soundscape, chronological single paragraph, faithfulness) exactly as specified.
"""


LTX2_5_SYSTEM_PROMPTS = (
    (LTX2_5_T2V_PROMPT_ID, "LTX-2.5 Text-to-Video", LTX2_5_T2V_PROMPT_CONTENT),
    (LTX2_5_I2V_PROMPT_ID, "LTX-2.5 Image-to-Video", LTX2_5_I2V_PROMPT_CONTENT),
)


class AddLtx2SystemPromptsCallback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.executemany(
            """--sql
            INSERT OR IGNORE INTO system_prompts (id, name, content, user_id, is_public, max_tokens)
            VALUES (?, ?, ?, 'system', TRUE, ?);
            """,
            [(id, name, content, LTX2_5_MAX_TOKENS) for id, name, content in LTX2_5_SYSTEM_PROMPTS],
        )


def build_migration() -> Migration:
    """Seed the LTX-2.5 prompt-enhancement system prompts.

    Depends on the migration that adds ``system_prompts.max_tokens``, which the rows set.
    """
    return Migration(
        id="2026_09_26_add_ltx2_system_prompts",
        depends_on="2026_09_09_add_system_prompt_max_tokens",
        callback=AddLtx2SystemPromptsCallback(),
    )
