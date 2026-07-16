/**
 * Small pure color helpers shared by the engine. Zero React, zero import-time
 * side effects.
 */

/** Clamps a color channel to the `[0, 255]` byte range, rounding to the nearest integer. */
const clampChannel = (value: number): number => Math.max(0, Math.min(255, Math.round(value)));

const toHexByte = (value: number): string => clampChannel(value).toString(16).padStart(2, '0');

/**
 * Formats an opaque color as a lowercase `#rrggbb` CSS hex string. Channels are
 * `[0, 255]`; alpha is intentionally not part of the output — brush/eraser
 * options store an opaque fill color, and the color picker's own transparency
 * check (`sampleDocumentColor` returning `null`) already decides whether a
 * sample is pickable at all.
 */
export const rgbaToHex = (r: number, g: number, b: number): string => `#${toHexByte(r)}${toHexByte(g)}${toHexByte(b)}`;
