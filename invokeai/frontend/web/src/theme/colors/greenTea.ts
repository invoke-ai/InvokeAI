import { InvokeAIThemeColors } from 'theme/themeTypes';
import { generateColorPalette } from '../util/generateColorPalette';

export const greenTeaThemeColors: InvokeAIThemeColors = {
  base: generateColorPalette(223, 10),
  baseAlpha: generateColorPalette(223, 10, false, true),
  accent: generateColorPalette(160, 60),
  accentAlpha: generateColorPalette(160, 60, false, true),
  working: generateColorPalette(47, 68),
  workingAlpha: generateColorPalette(47, 68, false, true),
  warning: generateColorPalette(28, 75),
  warningAlpha: generateColorPalette(28, 75, false, true),
  ok: generateColorPalette(122, 49),
  okAlpha: generateColorPalette(122, 49, false, true),
  error: generateColorPalette(0, 50),
  errorAlpha: generateColorPalette(0, 50, false, true),
  gridLineColor: 'rgba(255, 255, 255, 0.15)',
};
