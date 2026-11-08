/**
 * Before/after pixel history uses layer-local rects, stable across cache growth and shifted origins. The
 * engine-provided {@link ImagePatchApply} owns writes and persistence; this entry accounts for both buffers' byte
 * lengths.
 */

import type { Rect } from '@workbench/canvas-engine/types';

import type { HistoryEntry } from './history';

import { NO_HELD_ASSET_REFS } from './history';

/**
 * Engine-owned pixel write at the layer-local rect, including invalidation/versioning and dirty persistence
 * marking. It first makes the layer's cache trustworthy (rasterizing an evicted or stale one) and rejects, leaving
 * the cache unchanged, when it cannot.
 */
export type ImagePatchApply = (layerId: string, rect: Rect, pixels: ImageData) => Promise<void>;

/** Options for {@link createImagePatchEntry}. */
export interface CreateImagePatchEntryOptions {
  layerId: string;
  /** The painted region in LAYER-LOCAL space; its w/h must match the ImageData. */
  rect: Rect;
  /** Cache pixels within `rect` before the stroke. */
  before: ImageData;
  /** Cache pixels within `rect` after the stroke. */
  after: ImageData;
  /** Entry label (e.g. "Brush stroke" / "Eraser stroke"). */
  label: string;
  /** The engine's pixel-write bridge. */
  apply: ImagePatchApply;
}

/** Throws if an ImageData's dimensions disagree with the patch rect. */
const assertDims = (rect: Rect, image: ImageData, which: 'before' | 'after'): void => {
  if (image.width !== rect.width || image.height !== rect.height) {
    throw new Error(
      `imagePatch: ${which} ImageData (${image.width}x${image.height}) does not match rect (${rect.width}x${rect.height})`
    );
  }
};

/** Creates a reversible paint-edit entry from a committed stroke's before/after pixels. */
export const createImagePatchEntry = (opts: CreateImagePatchEntryOptions): HistoryEntry => {
  const { after, apply, before, label, layerId, rect } = opts;
  assertDims(rect, before, 'before');
  assertDims(rect, after, 'after');

  // Snapshot the rect so a later external mutation of the caller's object can't
  // shift where undo/redo write.
  const patchRect: Rect = { height: rect.height, width: rect.width, x: rect.x, y: rect.y };
  const bytes = before.data.byteLength + after.data.byteLength;

  return {
    bytes,
    heldAssetRefs: NO_HELD_ASSET_REFS,
    label,
    redo: () => apply(layerId, patchRect, after),
    undo: () => apply(layerId, patchRect, before),
  };
};
