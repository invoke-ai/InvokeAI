/**
 * Available global composite operations (blend modes) for layers.
 * These are the standard Canvas 2D composite operations.
 * @see https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/globalCompositeOperation
 */
export const COMPOSITE_OPERATIONS = [
  'source-over',
  'source-in',
  'source-out',
  'source-atop',
  'destination-over',
  'destination-in',
  'destination-out',
  'destination-atop',
  'lighter',
  'copy',
  'xor',
  'multiply',
  'screen',
  'overlay',
  'darken',
  'lighten',
  'color-dodge',
  'color-burn',
  'hard-light',
  'soft-light',
  'difference',
  'exclusion',
  'hue',
  'saturation',
  'color',
  'luminosity',
] as const;

export type CompositeOperation = (typeof COMPOSITE_OPERATIONS)[number];

// Subset of color blend modes for UI selection
export const COLOR_BLEND_MODES: CompositeOperation[] = [
  'multiply',
  'screen',
  'overlay',
  'darken',
  'lighten',
  'color-dodge',
  'color-burn',
  'hard-light',
  'soft-light',
  'difference',
  'hue',
  'saturation',
  'color',
  'luminosity',
];
