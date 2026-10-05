import { MAX_BRUSH_SIZE, MIN_BRUSH_SIZE } from '@workbench/canvas-engine/engineStores';

/** Freehand `thinning` applied when brush pressure sensitivity is on. */
export const PRESSURE_THINNING = 0.5;

/** Fraction to grow/shrink the brush/eraser diameter per size-step notch (ctrl+wheel or `[`/`]`). */
export const SIZE_STEP_FACTOR = 0.1;

/** Clamps a brush/eraser diameter while retaining hundredth-pixel precision. */
export const clampBrushSize = (value: number): number =>
  Math.max(MIN_BRUSH_SIZE, Math.min(MAX_BRUSH_SIZE, Math.round(value * 100) / 100));

export const stepBrushSize = (size: number, direction: 1 | -1): number =>
  clampBrushSize(size * (1 + direction * SIZE_STEP_FACTOR));
