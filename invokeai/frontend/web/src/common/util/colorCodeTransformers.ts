import type { RgbaColor, RgbColor } from 'react-colorful';

export function rgbaToHex(color: RgbaColor, alpha: boolean = false): string {
  const hex = ((1 << 24) + (color.r << 16) + (color.g << 8) + color.b).toString(16).slice(1);
  const alphaHex = Math.round(color.a * 255)
    .toString(16)
    .padStart(2, '0');
  return alpha ? `#${hex}${alphaHex}` : `#${hex}`;
}

export function hexToRGBA(hex: string, alpha: number) {
  hex = hex.replace(/^#/, '');
  // Expand the #rgb and #rgba shorthands to #rrggbb and #rrggbbaa.
  if (hex.length === 3 || hex.length === 4) {
    hex = hex
      .split('')
      .map((c) => c + c)
      .join('');
  }
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  // A hex with an alpha channel carries its own alpha; otherwise use the one passed in.
  const a = hex.length === 8 ? parseInt(hex.substring(6, 8), 16) / 255 : alpha;
  return { r, g, b, a };
}

export const rgbaColorToString = (color: RgbaColor): string => {
  const { r, g, b, a } = color;
  return `rgba(${r}, ${g}, ${b}, ${a})`;
};

export const rgbColorToString = (color: RgbColor): string => {
  const { r, g, b } = color;
  return `rgba(${r}, ${g}, ${b})`;
};
