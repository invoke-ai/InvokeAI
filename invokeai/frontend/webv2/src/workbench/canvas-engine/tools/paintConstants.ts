/**
 * Shared brush/eraser painting constants and the size-step math both the
 * ctrl+wheel path (`input/wheel.ts`) and the `[`/`]` hotkeys drive through the
 * same helper. Zero React, zero import-time side effects.
 */

import { MAX_BRUSH_SIZE, MIN_BRUSH_SIZE } from '@workbench/canvas-engine/engineStores';

/** Freehand `thinning` applied when brush pressure sensitivity is on. */
export const PRESSURE_THINNING = 0.5;

/** Fraction to grow/shrink the brush/eraser diameter per size-step notch (ctrl+wheel or `[`/`]`). */
export const SIZE_STEP_FACTOR = 0.1;

/** Clamps a brush/eraser diameter to `[MIN_BRUSH_SIZE, MAX_BRUSH_SIZE]`, rounded to the nearest integer. */
export const clampBrushSize = (value: number): number =>
  Math.max(MIN_BRUSH_SIZE, Math.min(MAX_BRUSH_SIZE, Math.round(value)));

/**
 * Steps a brush/eraser diameter by one notch: `+1` grows by `SIZE_STEP_FACTOR`,
 * `-1` shrinks by it, clamped to the valid size range.
 */
export const stepBrushSize = (size: number, direction: 1 | -1): number =>
  clampBrushSize(size * (1 + direction * SIZE_STEP_FACTOR));
