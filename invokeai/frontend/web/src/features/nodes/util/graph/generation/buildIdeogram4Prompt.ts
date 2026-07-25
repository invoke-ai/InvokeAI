import type { RootState } from 'app/store/store';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { selectIdeogram4ColorPalette, selectPositivePrompt } from 'features/controlLayers/store/paramsSlice';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import type { Rect } from 'features/controlLayers/store/types';

/**
 * Ideogram 4 is prompted with a structured JSON caption describing the scene as a list of regions,
 * each with a bounding box and a description. The bbox numbers live inside the prompt string (Ideogram 4
 * does not use spatial attention masks).
 *
 * This module reads the raw inputs (global prompt, per-region description + bbox, color palette) from
 * canvas state. The actual JSON assembly happens at generation time in the backend
 * `ideogram4_caption_builder` node (see backend `build_ideogram4_caption`) so that dynamic-prompt
 * expansions and prompt batching — which vary the global prompt — are folded into the encoded caption.
 * Only the bbox normalization stays here, since it needs the canvas manager / generation bbox.
 */

/** Ideogram 4 normalizes spatial coordinates to a 0–1000 grid with the origin at the top-left. */
const IDEOGRAM4_COORD_MAX = 1000;

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

type Ideogram4Bbox = [number, number, number, number];

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

export type Ideogram4PromptInputs = {
  /** The global positive prompt (batch-injectable; becomes `high_level_description` / raw text). */
  globalPrompt: string;
  /** Enabled Regional Guidance layers with a non-empty prompt (description + normalized bbox). */
  regions: Ideogram4RegionInput[];
  /** The raw color palette (normalized/validated by the backend caption builder). */
  colorPalette: string[];
};

/**
 * Reads the raw Ideogram 4 prompt inputs from state: the global prompt, each enabled Regional Guidance
 * layer (prompt + normalized bbox), and the color palette. The JSON caption is assembled later, at
 * generation time, by the backend `ideogram4_caption_builder` node — so the batch-injectable global
 * prompt is reflected in the encoded caption. Regions with no drawn content contribute a null bbox.
 */
export const collectIdeogram4PromptInputs = (
  state: RootState,
  manager: CanvasManager | null
): Ideogram4PromptInputs => {
  const globalPrompt = selectPositivePrompt(state);
  const colorPalette = selectIdeogram4ColorPalette(state);

  // No canvas manager (e.g. the Generate tab) → no regions to read.
  if (manager === null) {
    return { globalPrompt, regions: [], colorPalette };
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

  return { globalPrompt, regions, colorPalette };
};
