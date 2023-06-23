import { InvokeAIThemeColors } from 'theme/themeTypes';
import { generateColorPalette } from 'theme/util/generateColorPalette';

export const invokeAIThemeColors: InvokeAIThemeColors = {
  base: generateColorPalette(220, 15),
  baseAlpha: generateColorPalette(220, 15, false, true),
  accent: generateColorPalette(250, 50),
  accentAlpha: generateColorPalette(250, 50, false, true),
  working: generateColorPalette(47, 67),
  workingAlpha: generateColorPalette(47, 67, false, true),
  warning: generateColorPalette(28, 75),
  warningAlpha: generateColorPalette(28, 75, false, true),
  ok: generateColorPalette(113, 70),
  okAlpha: generateColorPalette(113, 70, false, true),
  error: generateColorPalette(0, 76),
  errorAlpha: generateColorPalette(0, 76, false, true),
  gridLineColor: 'rgba(150, 150, 180, 0.15)',
};
