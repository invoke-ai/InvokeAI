import { InvokeAIThemeColors } from 'theme/themeTypes';
import { generateColorPalette } from '../util/generateColorPalette';

export const greenTeaThemeColors: InvokeAIThemeColors = {
  base: generateColorPalette(223, 10),
  accent: generateColorPalette(155, 80),
  working: generateColorPalette(47, 68),
  warning: generateColorPalette(28, 75),
  ok: generateColorPalette(122, 49),
  error: generateColorPalette(0, 50),
  gridLineColor: 'rgba(255, 255, 255, 0.2)',
};
