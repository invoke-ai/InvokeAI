import { generateColorPalette } from '../util/generateColorPalette';

export const greenTeaThemeColors = {
  base: generateColorPalette(223, 10),
  accent: generateColorPalette(155, 80),
  working: generateColorPalette(47, 68),
  warning: generateColorPalette(28, 75),
  ok: generateColorPalette(122, 49),
  error: generateColorPalette(0, 50),
};
