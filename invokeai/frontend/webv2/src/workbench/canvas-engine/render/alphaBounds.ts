/**
 * Alpha-derived content bounds for an RGBA pixel buffer — where a layer's visible
 * pixels actually are, as opposed to the geometric extent its cache grew to.
 *
 * ANY non-zero alpha counts as content (matching `selection/selectionState.ts`, not
 * the mask outline's solidity threshold): a higher threshold would silently discard
 * faint pixels the user painted.
 *
 * Zero React, zero import-time side effects.
 */

import type { Rect } from '@workbench/canvas-engine/types';

/**
 * An RGBA pixel buffer; only the alpha channel is read. `ImageData` satisfies this
 * structurally, so tests can pass a plain literal — no canvas required.
 */
export interface AlphaPixels {
  readonly data: Uint8ClampedArray;
  readonly height: number;
  readonly width: number;
}

const EMPTY_RECT: Rect = { height: 0, width: 0, x: 0, y: 0 };

/** Too small to describe `width * height` pixels — never trusted to prove emptiness. */
const isUnreadable = (pixels: AlphaPixels): boolean =>
  pixels.width <= 0 || pixels.height <= 0 || pixels.data.length < pixels.width * pixels.height * 4;

/** True when any pixel has non-zero alpha. Early-exits, so cheaper than {@link alphaBounds}. */
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
 * Tight bounds of the non-transparent pixels, in the BUFFER's own coordinates (the
 * caller translates by its surface origin). Empty rect when every pixel is alpha 0.
 * Edge-inclusive: one opaque pixel yields a 1x1 rect.
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
    // Only scan right of the running maxX; anything inside it is already covered.
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
