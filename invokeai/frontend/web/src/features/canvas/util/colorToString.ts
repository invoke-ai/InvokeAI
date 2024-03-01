import type { RgbaColor } from 'react-colorful';

export const rgbaColorToString = (color: RgbaColor): string => {
  const { r, g, b, a } = color;
  return `rgba(${r}, ${g}, ${b}, ${a})`;
};
