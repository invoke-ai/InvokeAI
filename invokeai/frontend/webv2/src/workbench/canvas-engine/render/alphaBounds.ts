/**
 * Alpha-derived content bounds for an RGBA pixel buffer.
 *
 * A layer's content rect is otherwise pure geometry — a paint source reports its
 * persisted bitmap's dimensions, and the raster cache only ever GROWS (the stroke
 * chunk-pads its extent, and the eraser grows it too). So a layer whose pixels
 * have all been erased still describes a full-size rectangle, and every consumer
 * of that rectangle — the move outline, the transform frame, fit-to-content —
 * frames a region holding nothing. These functions are the pixel-side answer:
 * where the visible pixels ACTUALLY are.
 *
 * ANY non-zero alpha counts as content, matching the selection state's mask test
 * (`selection/selectionState.ts`) rather than the mask outline's solidity
 * threshold. A higher threshold would silently discard faint pixels a user
 * deliberately painted, which is a worse failure than an outline that lingers
 * around near-invisible content.
 *
 * Zero React, zero import-time side effects.
 */

import type { Rect } from '@workbench/canvas-engine/types';

/**
 * An RGBA pixel buffer. Only the ALPHA channel is read, and only the first
 * `width * height * 4` bytes. `ImageData` satisfies this structurally, so callers
 * can pass a readback directly and tests can pass a plain literal — no canvas
 * required.
 */
export interface AlphaPixels {
  readonly data: Uint8ClampedArray;
  readonly height: number;
  readonly width: number;
}

/** The all-zero rect returned for a buffer with no visible pixels. */
const EMPTY_RECT: Rect = { height: 0, width: 0, x: 0, y: 0 };

/**
 * True when the buffer is too small to describe `width * height` RGBA pixels —
 * a partially-populated readback is never trusted to prove emptiness.
 */
const isUnreadable = (pixels: AlphaPixels): boolean =>
  pixels.width <= 0 || pixels.height <= 0 || pixels.data.length < pixels.width * pixels.height * 4;

/**
 * True when any pixel in `pixels` has non-zero alpha.
 *
 * Early-exits on the first hit, so this is strictly cheaper than
 * {@link alphaBounds} when only the yes/no answer is needed. An unreadable buffer
 * reports `false` — it describes no pixels, so it has no visible ones.
 */
export const hasVisiblePixels = (pixels: AlphaPixels): boolean => {
  if (isUnreadable(pixels)) {
    return false;
  }
  const { data } = pixels;
  const end = pixels.width * pixels.height * 4;
  for (let index = 3; index < end; index += 4) {
    if (data[index] !== 0) {
      return true;
    }
  }
  return false;
};

/**
 * The tight bounds of the non-transparent pixels in `pixels`, expressed in the
 * BUFFER's own pixel coordinates (origin at its top-left) — the caller translates
 * into layer-local space by its surface origin.
 *
 * Returns an EMPTY rect when every pixel has alpha 0, so `math/rect`'s `isEmpty`
 * is the emptiness test. The result is always inclusive of both edges: a single
 * opaque pixel yields a 1x1 rect at that pixel.
 *
 * Rows are scanned edge-inward, skipping the span already known to be inside the
 * running horizontal bounds — so a fully-covered buffer costs roughly two probes
 * per row rather than a full pass.
 */
export const alphaBounds = (pixels: AlphaPixels): Rect => {
  if (isUnreadable(pixels)) {
    return EMPTY_RECT;
  }
  const { data, height, width } = pixels;
  let minX = width;
  let minY = -1;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y += 1) {
    const rowStart = y * width * 4 + 3;
    let rowMinX = -1;
    for (let x = 0; x < width; x += 1) {
      if (data[rowStart + x * 4] !== 0) {
        rowMinX = x;
        break;
      }
    }
    if (rowMinX === -1) {
      continue;
    }
    let rowMaxX = rowMinX;
    // Everything between the running bounds is already accounted for, so only
    // scan inward from the right edge down to whichever is further right.
    for (let x = width - 1; x > Math.max(rowMinX, maxX); x -= 1) {
      if (data[rowStart + x * 4] !== 0) {
        rowMaxX = x;
        break;
      }
    }
    if (rowMinX < minX) {
      minX = rowMinX;
    }
    if (rowMaxX > maxX) {
      maxX = rowMaxX;
    }
    if (minY === -1) {
      minY = y;
    }
    maxY = y;
  }
  if (maxY === -1) {
    return EMPTY_RECT;
  }
  return { height: maxY - minY + 1, width: maxX - minX + 1, x: minX, y: minY };
};
