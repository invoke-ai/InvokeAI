import { RgbaColor, RgbColor } from 'react-colorful';

export const rgbaColorToString = (color: RgbaColor): string => {
  const { r, g, b, a } = color;
  return `rgba(${r}, ${g}, ${b}, ${a})`;
};

export const rgbaColorToRgbString = (color: RgbaColor): string => {
  const { r, g, b } = color;
  return `rgba(${r}, ${g}, ${b})`;
};

export const rgbColorToString = (color: RgbColor): string => {
  const { r, g, b } = color;
  return `rgba(${r}, ${g}, ${b})`;
};
