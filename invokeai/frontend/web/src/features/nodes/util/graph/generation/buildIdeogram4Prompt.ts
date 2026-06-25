import type { RootState } from 'app/store/store';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { selectIdeogram4ColorPalette, selectPositivePrompt } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { Rect } from 'features/controlLayers/store/types';

/**
 * Ideogram 4 is prompted with a structured JSON caption describing the scene as a list of regions,
 * each with a bounding box and a description. This module assembles that caption from InvokeAI's
 * Canvas Regional Guidance layers — the bbox numbers live inside the prompt string (Ideogram 4 does
 * not use spatial attention masks), so this is pure string assembly with no backend mask handling.
 *
 * See the reference prompting guide for the schema; the key points used here:
 *   - bbox is `[y_min, x_min, y_max, x_max]`, normalized to 0–1000, origin at top-left.
 *   - Key order matters (the model was trained on a consistent order). `obj` elements use
 *     `type`, `bbox`, `desc`. We rely on JS object insertion order being preserved by JSON.stringify.
 *   - The reference serializes with compact separators and `ensure_ascii=False`; JSON.stringify
 *     already produces compact `,`/`:` separators and preserves non-ASCII characters.
 */

/** Ideogram 4 normalizes spatial coordinates to a 0–1000 grid with the origin at the top-left. */
const IDEOGRAM4_COORD_MAX = 1000;

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

export type Ideogram4Bbox = [number, number, number, number];

/**
 * Converts a region's rect (canvas/layer coordinates — the same space as the generation bbox) into an
 * Ideogram 4 bounding box `[y_min, x_min, y_max, x_max]`, normalized to 0–1000 relative to the
 * generation bbox, clamped and rounded to integers.
 */
export const rectToIdeogram4Bbox = (regionRect: Rect, genBbox: Rect): Ideogram4Bbox => {
  const norm = (value: number, origin: number, extent: number): number =>
    extent <= 0 ? 0 : clamp(Math.round(((value - origin) / extent) * IDEOGRAM4_COORD_MAX), 0, IDEOGRAM4_COORD_MAX);
  const yMin = norm(regionRect.y, genBbox.y, genBbox.height);
  const xMin = norm(regionRect.x, genBbox.x, genBbox.width);
  const yMax = norm(regionRect.y + regionRect.height, genBbox.y, genBbox.height);
  const xMax = norm(regionRect.x + regionRect.width, genBbox.x, genBbox.width);
  return [yMin, xMin, yMax, xMax];
};

export type Ideogram4RegionInput = {
  /** The region's positive prompt — becomes the element's `desc`. */
  prompt: string;
  /** The region's normalized bbox, or null when the region has no drawn content. */
  bbox: Ideogram4Bbox | null;
};

/** An `obj`-type element. Key order matches the training schema: `type`, `bbox`, `desc`. */
type Ideogram4Element = { type: 'obj'; bbox: Ideogram4Bbox; desc: string } | { type: 'obj'; desc: string };

export type Ideogram4PromptResult = {
  /** The final prompt string to feed to the text encoder. */
  prompt: string;
  /**
   * Whether the prompt is a structured caption (assembled JSON or raw-JSON passthrough). When true,
   * the graph builder must NOT let the linear batch inject the raw positive prompt over it, otherwise
   * the assembled caption would be clobbered by the plain prompt text.
   */
  isStructured: boolean;
};

/**
 * Assembles an Ideogram 4 prompt from a global prompt, a set of regions, and an optional color palette.
 *
 * - Raw-JSON passthrough: if the global prompt is already a JSON object (trimmed, starts with `{`), it
 *   is used verbatim and treated as structured (the palette is ignored — the user controls the JSON).
 * - With regions and/or a color palette: a structured JSON caption is built — the global prompt becomes
 *   `high_level_description`, each region becomes an `obj` element, and the palette (if any) becomes
 *   `style_description.color_palette`.
 * - Otherwise: the plain global prompt is returned (the model accepts plain text). This keeps dynamic
 *   prompts and prompt batching working, since the caller can let the batch inject it directly.
 */
export const buildIdeogram4Caption = (
  globalPrompt: string,
  regions: Ideogram4RegionInput[],
  colorPalette: string[] = []
): Ideogram4PromptResult => {
  const trimmed = globalPrompt.trim();

  // The user pasted a structured caption (or any JSON object) — use it verbatim.
  if (trimmed.startsWith('{')) {
    return { prompt: globalPrompt, isStructured: true };
  }

  const elements: Ideogram4Element[] = regions
    .filter((region) => region.prompt.trim().length > 0)
    .map((region) =>
      region.bbox ? { type: 'obj', bbox: region.bbox, desc: region.prompt } : { type: 'obj', desc: region.prompt }
    );

  // Normalize the palette to uppercase #RRGGBB (the schema's required hex form); drop invalid entries.
  const palette = colorPalette.map((c) => c.toUpperCase()).filter((c) => /^#[0-9A-F]{6}$/.test(c));

  // Nothing structured to encode — fall back to the plain prompt (documented to work).
  if (elements.length === 0 && palette.length === 0) {
    return { prompt: trimmed, isStructured: false };
  }

  // Strict key order: high_level_description, (style_description), compositional_deconstruction.
  // style_description here carries only color_palette; the other style fields are left to raw-JSON use.
  const compositional_deconstruction = { background: '', elements };
  const caption =
    palette.length > 0
      ? { high_level_description: trimmed, style_description: { color_palette: palette }, compositional_deconstruction }
      : { high_level_description: trimmed, compositional_deconstruction };

  return { prompt: JSON.stringify(caption), isStructured: true };
};

/**
 * Reads the global prompt and each enabled Regional Guidance layer (prompt + normalized bbox) from
 * canvas state, then assembles the Ideogram 4 prompt. Regions with no drawn content contribute their
 * description without a bbox.
 */
export const buildIdeogram4Prompt = (state: RootState, manager: CanvasManager | null): Ideogram4PromptResult => {
  const globalPrompt = selectPositivePrompt(state);
  const colorPalette = selectIdeogram4ColorPalette(state);

  // No canvas manager (e.g. the Generate tab) → no regions to read.
  if (manager === null) {
    return buildIdeogram4Caption(globalPrompt, [], colorPalette);
  }

  const canvas = selectCanvasSlice(state);
  const genBbox = canvas.bbox.rect;

  const regions: Ideogram4RegionInput[] = [];
  for (const region of canvas.regionalGuidance.entities) {
    if (!region.isEnabled) {
      continue;
    }
    const prompt = region.positivePrompt;
    if (!prompt || prompt.trim().length === 0) {
      continue;
    }
    const adapter = manager.adapters.regionMasks.get(region.id);
    const bbox =
      adapter && adapter.renderer.hasObjects()
        ? rectToIdeogram4Bbox(adapter.transformer.getRelativeRect(), genBbox)
        : null;
    regions.push({ prompt, bbox });
  }

  return buildIdeogram4Caption(globalPrompt, regions, colorPalette);
};
