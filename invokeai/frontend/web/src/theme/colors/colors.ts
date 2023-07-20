import { InvokeAIThemeColors } from 'theme/themeTypes';
import { generateColorPalette } from 'theme/util/generateColorPalette';

const BASE = { H: 220, S: 16 };
const ACCENT = { H: 250, S: 42 };
// const ACCENT = { H: 250, S: 52 };
const WORKING = { H: 47, S: 42 };
// const WORKING = { H: 47, S: 50 };
const WARNING = { H: 28, S: 42 };
// const WARNING = { H: 28, S: 50 };
const OK = { H: 113, S: 42 };
// const OK = { H: 113, S: 50 };
const ERROR = { H: 0, S: 42 };
// const ERROR = { H: 0, S: 50 };

export const InvokeAIColors: InvokeAIThemeColors = {
  base: generateColorPalette(BASE.H, BASE.S),
  baseAlpha: generateColorPalette(BASE.H, BASE.S, true),
  accent: generateColorPalette(ACCENT.H, ACCENT.S),
  accentAlpha: generateColorPalette(ACCENT.H, ACCENT.S, true),
  working: generateColorPalette(WORKING.H, WORKING.S),
  workingAlpha: generateColorPalette(WORKING.H, WORKING.S, true),
  warning: generateColorPalette(WARNING.H, WARNING.S),
  warningAlpha: generateColorPalette(WARNING.H, WARNING.S, true),
  ok: generateColorPalette(OK.H, OK.S),
  okAlpha: generateColorPalette(OK.H, OK.S, true),
  error: generateColorPalette(ERROR.H, ERROR.S),
  errorAlpha: generateColorPalette(ERROR.H, ERROR.S, true),
};
