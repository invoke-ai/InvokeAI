/**
 * Available global composite operations (blend modes) for layers.
 * These are the standard Canvas 2D composite operations.
 * @see https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/globalCompositeOperation
 * NOTE: All of these are supported by canvas layers, but not all are supported by CSS blend modes (live rendering).
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

// Subset of color blend modes for UI selection. All are supported by both Konva and CSS.
export const COLOR_BLEND_MODES: CompositeOperation[] = [
  'color',
  'hue',
  'overlay',
  'soft-light',
  'hard-light',
  'screen',
  'color-burn',
  'color-dodge',
  'multiply',
  'darken',
  'lighten',
  'difference',
  'luminosity',
  'saturation',
];
