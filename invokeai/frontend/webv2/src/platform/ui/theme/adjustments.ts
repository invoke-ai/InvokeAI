import { previewSwatches, THEMES_BY_ID, type ThemeDefinition, type WorkbenchThemeId } from './themes';

export interface ThemeAdjustment {
  backgroundLightness?: number;
  panelLightness?: number;
  inputLightness?: number;
  inputBorderLightness?: number;
  textLightness?: number;
  brandHex?: string;
  accentHex?: string;
}

export type ThemeAdjustments = Partial<Record<WorkbenchThemeId, ThemeAdjustment>>;

const LIGHTNESS_KEYS = [
  'backgroundLightness',
  'panelLightness',
  'inputLightness',
  'inputBorderLightness',
  'textLightness',
] as const;
const HEX_KEYS = ['brandHex', 'accentHex'] as const;
const HEX_COLOR = /^#[0-9a-fA-F]{6}$/;
const ROOT_COLOR_TOKENS = [
  '--chakra-colors-bg',
  '--chakra-colors-bg.inset',
  '--chakra-colors-bg.subtle',
  '--chakra-colors-bg.muted',
  '--chakra-colors-bg.panel',
  '--chakra-colors-bg.emphasized',
  '--chakra-colors-fg',
  '--chakra-colors-fg.muted',
  '--chakra-colors-fg.subtle',
  '--chakra-colors-gray-fg',
  '--chakra-colors-brand-solid',
  '--chakra-colors-brand-contrast',
  '--chakra-colors-accent-solid',
  '--chakra-colors-accent-contrast',
  '--wb-theme-input-bg',
  '--wb-theme-input-border',
] as const;

/** Discard unknown themes, fields and malformed stored values at the preference boundary. */
export const normalizeThemeAdjustments = (value: unknown): ThemeAdjustments => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {};
  }

  const themes: ThemeAdjustments = {};
  for (const [themeId, candidate] of Object.entries(value)) {
    if (
      !Object.prototype.hasOwnProperty.call(THEMES_BY_ID, themeId) ||
      !candidate ||
      typeof candidate !== 'object' ||
      Array.isArray(candidate)
    ) {
      continue;
    }
    const stored = candidate as Record<string, unknown>;
    const adjustment: ThemeAdjustment = {};
    for (const key of LIGHTNESS_KEYS) {
      const lightness = stored[key];
      if (typeof lightness === 'number' && Number.isInteger(lightness) && lightness >= 0 && lightness <= 100) {
        adjustment[key] = lightness;
      }
    }
    for (const key of HEX_KEYS) {
      const color = stored[key];
      if (typeof color === 'string' && HEX_COLOR.test(color)) {
        adjustment[key] = color.toLowerCase();
      }
    }
    if (Object.keys(adjustment).length) {
      themes[themeId as WorkbenchThemeId] = adjustment;
    }
  }
  return themes;
};

/** An empty adjustment leaves the complete upstream palette untouched. */
export const applyThemeAdjustmentsToRoot = (
  theme: ThemeDefinition,
  adjustment: ThemeAdjustment | undefined,
  highContrast: boolean
): void => {
  const root = document.documentElement;
  const style = root.style;
  for (const name of ROOT_COLOR_TOKENS) {
    style.removeProperty(name);
  }
  root.toggleAttribute('data-theme-input-bg', adjustment?.inputLightness !== undefined && !highContrast);
  root.toggleAttribute('data-theme-input-border', adjustment?.inputBorderLightness !== undefined && !highContrast);
  if (!adjustment || highContrast) {
    return;
  }

  const surface = (lightness: number) => neutralWithLightness(theme, lightness);
  if (adjustment.backgroundLightness !== undefined) {
    const base = adjustment.backgroundLightness;
    const inset = surface(Math.max(0, base - 5));
    style.setProperty('--chakra-colors-bg', inset);
    style.setProperty('--chakra-colors-bg.inset', inset);
    style.setProperty('--chakra-colors-bg.subtle', surface(base));
  }
  if (adjustment.panelLightness !== undefined) {
    const panel = adjustment.panelLightness;
    style.setProperty('--chakra-colors-bg.muted', surface(panel));
    style.setProperty('--chakra-colors-bg.panel', surface(panel));
    style.setProperty(
      '--chakra-colors-bg.emphasized',
      surface(theme.colorScheme === 'light' ? Math.max(0, panel - 4) : Math.min(100, panel + 4))
    );
  }
  if (adjustment.inputLightness !== undefined) {
    style.setProperty('--wb-theme-input-bg', surface(adjustment.inputLightness));
  }
  if (adjustment.inputBorderLightness !== undefined) {
    style.setProperty('--wb-theme-input-border', surface(adjustment.inputBorderLightness));
  }
  if (adjustment.textLightness !== undefined) {
    const lightness = adjustment.textLightness;
    const subdued = (offset: number) =>
      `oklch(${lightness >= 50 ? Math.max(0, lightness - offset) : Math.min(100, lightness + offset)}% 0 0)`;
    const text = `oklch(${lightness}% 0 0)`;
    style.setProperty('--chakra-colors-fg', text);
    style.setProperty('--chakra-colors-gray-fg', text);
    style.setProperty('--chakra-colors-fg.muted', subdued(20));
    style.setProperty('--chakra-colors-fg.subtle', subdued(32));
  }
  if (adjustment.brandHex) {
    style.setProperty('--chakra-colors-brand-solid', adjustment.brandHex);
    style.setProperty('--chakra-colors-brand-contrast', contrastForeground(adjustment.brandHex));
  }
  if (adjustment.accentHex) {
    style.setProperty('--chakra-colors-accent-solid', adjustment.accentHex);
    style.setProperty('--chakra-colors-accent-contrast', contrastForeground(adjustment.accentHex));
  }
};

export const previewAdjustedSwatches = (
  theme: ThemeDefinition,
  adjustment?: ThemeAdjustment
): [string, string, string, string] => {
  const [background, control, brand, accent] = previewSwatches(theme);
  return [
    adjustment?.backgroundLightness === undefined
      ? background
      : neutralWithLightness(theme, adjustment.backgroundLightness),
    adjustment?.panelLightness === undefined ? control : neutralWithLightness(theme, adjustment.panelLightness),
    adjustment?.brandHex ?? brand,
    adjustment?.accentHex ?? accent,
  ];
};

/** The shared color picker accepts hex, while stock theme seeds are OKLCH. */
export const themeSeedToPickerHex = (seed: string): string => {
  const match = /^oklch\(([\d.]+)% ([\d.]+) ([\d.]+)\)$/.exec(seed);
  if (!match) {
    throw new Error(`Expected an OKLCH theme seed, got ${seed}`);
  }

  const lightness = Number(match[1]) / 100;
  const chroma = Number(match[2]);
  const angle = (Number(match[3]) * Math.PI) / 180;
  const a = chroma * Math.cos(angle);
  const b = chroma * Math.sin(angle);
  const l = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (lightness - 0.0894841775 * a - 1.291485548 * b) ** 3;
  const toHex = (linear: number): string => {
    const channel = linear <= 0.0031308 ? 12.92 * linear : 1.055 * linear ** (1 / 2.4) - 0.055;
    return Math.round(Math.max(0, Math.min(1, channel)) * 255)
      .toString(16)
      .padStart(2, '0');
  };

  return `#${toHex(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s)}${toHex(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s)}${toHex(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s)}`;
};

const neutralWithLightness = (theme: ThemeDefinition, lightness: number): string => {
  const seed = theme.colors.neutral[theme.colorScheme === 'light' ? 100 : 900];
  return seed.replace(/^oklch\([\d.]+%/, `oklch(${lightness}%`);
};

/** WCAG relative luminance threshold: choose the stronger black/white text on a custom accent fill. */
const contrastForeground = (hex: string): string => {
  const channels = [1, 3, 5].map((index) => {
    const value = parseInt(hex.slice(index, index + 2), 16) / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  });
  const luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
  return luminance > 0.179 ? '#000000' : '#ffffff';
};
