/**
 * A paint-edit history entry: a before/after pair of `ImageData` over a
 * LAYER-LOCAL rect, reversed by putting the pixels back into the layer's cache
 * surface. Layer-local coordinates are stable across cache growth/reallocation
 * (the cache's content rect origin can shift as strokes grow it), so an old
 * entry replays correctly no matter how the cache has been resized since.
 *
 * Painting (P2.1) commits a stroke as a {@link StrokeCommittedEvent} carrying the
 * `dirtyRect` plus `beforeImageData`/`afterImageData` (both sized to that rect).
 * This wraps that pair into a {@link HistoryEntry}: undo puts `before`, redo puts
 * `after`. The actual pixel write is delegated to an engine-provided
 * {@link ImagePatchApply} so this module stays free of the cache/backend/bitmap
 * store — it just owns the before/after bookkeeping and the byte accounting.
 *
 * Byte cost = both buffers' `byteLength`, which is what the history budget bounds.
 *
 * Zero React, zero DOM (ImageData is a plain data carrier here), zero import-time
 * side effects.
 */

import type { Rect } from '@workbench/canvas-engine/types';

import type { HistoryEntry } from './history';

/**
 * Writes `pixels` into `layerId`'s cache at `rect`'s origin and propagates the
 * edit (invalidate/version bump + mark the layer dirty for persistence). Provided
 * by the engine; the entry never touches the cache or backend directly.
 */
export type ImagePatchApply = (layerId: string, rect: Rect, pixels: ImageData) => void;

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
    label,
    redo: () => apply(layerId, patchRect, after),
    undo: () => apply(layerId, patchRect, before),
  };
};
