import { InvokeAIThemeColors } from 'theme/themeTypes';
import { generateColorPalette } from '../util/generateColorPalette';

export const lightThemeColors: InvokeAIThemeColors = {
  base: generateColorPalette(223, 10, true),
  baseAlpha: generateColorPalette(223, 10, true, true),
  accent: generateColorPalette(40, 80, true),
  accentAlpha: generateColorPalette(40, 80, true, true),
  working: generateColorPalette(47, 68, true),
  workingAlpha: generateColorPalette(47, 68, true, true),
  warning: generateColorPalette(28, 75, true),
  warningAlpha: generateColorPalette(28, 75, true, true),
  ok: generateColorPalette(122, 49, true),
  okAlpha: generateColorPalette(122, 49, true, true),
  error: generateColorPalette(0, 50, true),
  errorAlpha: generateColorPalette(0, 50, true, true),
  gridLineColor: 'rgba(0, 0, 0, 0.15)',
};
