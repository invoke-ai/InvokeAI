import { InvokeAIThemeColors } from 'theme/themeTypes';
import { generateColorPalette } from 'theme/util/generateColorPalette';

export const invokeAIThemeColors: InvokeAIThemeColors = {
  base: generateColorPalette(225, 15),
  accent: generateColorPalette(250, 50),
  working: generateColorPalette(47, 67),
  warning: generateColorPalette(28, 75),
  ok: generateColorPalette(113, 70),
  error: generateColorPalette(0, 76),
  gridLineColor: 'rgba(255, 255, 255, 0.2)',
};
