import { InvokeAIThemeColors } from 'theme/themeTypes';
import { generateColorPalette } from '../util/generateColorPalette';

export const oceanBlueColors: InvokeAIThemeColors = {
  base: generateColorPalette(220, 30),
  baseAlpha: generateColorPalette(220, 30, false, true),
  accent: generateColorPalette(210, 80),
  accentAlpha: generateColorPalette(210, 80, false, true),
  working: generateColorPalette(47, 68),
  workingAlpha: generateColorPalette(47, 68, false, true),
  warning: generateColorPalette(28, 75),
  warningAlpha: generateColorPalette(28, 75, false, true),
  ok: generateColorPalette(122, 49),
  okAlpha: generateColorPalette(122, 49, false, true),
  error: generateColorPalette(0, 100),
  errorAlpha: generateColorPalette(0, 100, false, true),
  gridLineColor: 'rgba(136, 148, 184, 0.15)',
};
