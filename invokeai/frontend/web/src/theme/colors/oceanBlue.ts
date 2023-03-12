import { InvokeAIThemeColors } from 'theme/themeTypes';
import { generateColorPalette } from '../util/generateColorPalette';

export const oceanBlueColors: InvokeAIThemeColors = {
  base: generateColorPalette(220, 30),
  accent: generateColorPalette(210, 80),
  working: generateColorPalette(47, 68),
  warning: generateColorPalette(28, 75),
  ok: generateColorPalette(122, 49),
  error: generateColorPalette(0, 100),
  gridLineColor: 'rgba(136, 148, 184, 0.2)',
};
