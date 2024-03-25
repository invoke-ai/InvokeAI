import type { RgbaColor } from 'react-colorful';

export function rgbaToHex(color: RgbaColor, alpha: boolean = false): string {
  const hex = ((1 << 24) + (color.r << 16) + (color.g << 8) + color.b).toString(16).slice(1);
  const alphaHex = Math.round(color.a * 255)
    .toString(16)
    .padStart(2, '0');
  return alpha ? `#${hex}${alphaHex}` : `#${hex}`;
}

export function hexToRGBA(hex: string, alpha: number) {
  hex = hex.replace(/^#/, '');
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  return { r, g, b, a: alpha };
}
