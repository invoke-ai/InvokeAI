const SWATCHES = [
  { r: 0, g: 0, b: 0, a: 1 }, // black
  { r: 255, g: 255, b: 255, a: 1 }, // white
  { r: 255, g: 90, b: 94, a: 1 }, // red
  { r: 255, g: 146, b: 75, a: 1 }, // orange
  { r: 255, g: 202, b: 59, a: 1 }, // yellow
  { r: 197, g: 202, b: 48, a: 1 }, // lime
  { r: 138, g: 201, b: 38, a: 1 }, // green
  { r: 83, g: 165, b: 117, a: 1 }, // teal
  { r: 23, g: 130, b: 196, a: 1 }, // blue
  { r: 66, g: 103, b: 172, a: 1 }, // indigo
  { r: 107, g: 76, b: 147, a: 1 }, // purple
];

export const RGBA_COLOR_SWATCHES = SWATCHES;
export const RGB_COLOR_SWATCHES = SWATCHES.map(({ r, g, b }) => ({ r, g, b }));
