/** Clamps a color channel to the `[0, 255]` byte range, rounding to the nearest integer. */
const clampChannel = (value: number): number => Math.max(0, Math.min(255, Math.round(value)));

const toHexByte = (value: number): string => clampChannel(value).toString(16).padStart(2, '0');

/**
 * Formats [0, 255] channels as lowercase `#rrggbb`. Brush colors are opaque; `createColorSampler` handles
 * unpickable transparent samples.
 */
export const rgbaToHex = (r: number, g: number, b: number): string => `#${toHexByte(r)}${toHexByte(g)}${toHexByte(b)}`;
