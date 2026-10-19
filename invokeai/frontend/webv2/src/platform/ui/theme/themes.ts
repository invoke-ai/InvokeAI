export type WorkbenchThemeId = 'classic' | 'light' | 'osakaJade' | 'mono' | 'ultradark' | 'catppuccinMocha';

/** Keep the ramp absolute: 50 lightest, 950 darkest, matching Chakra gray. */
export type NeutralStep = 50 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 950;

/**
 * Themes supply a neutral ramp and hue seeds; inset/fill/grid/control stay explicit because their elevation order
 * varies by theme.
 */
export interface ThemeColors {
  /** Neutral ramp, lightest (`50`) → darkest (`950`). Source of all bg/fg/border. */
  neutral: Record<NeutralStep, string>;
  /** Invoke identity (lime). Two seeds; the palette's other steps derive in `system.ts`. */
  brand: { solid: string; contrast: string };
  /** Selection / focus (blue). */
  accent: { solid: string; contrast: string };
  /** Destructive / error intent. */
  danger: string;
  /** Positive / success intent. */
  success: string;
  /** Caution / warning intent. */
  warning: string;
  /** Recessed work-area floor framed by the chrome (`bg.inset`). Off-ramp. */
  inset: string;
  /** Subtle neutral fill for hovered/ghost controls (`gray.subtle`). Off-ramp. */
  fill: string;
  /** Dot-grid decoration drawn on inset surfaces (`fg.grid`). Off-ramp. */
  grid: string;
  /** Control / chip fill inside panels (`bg.emphasized`). Off-ramp. */
  control: string;
}

export interface ThemeDefinition {
  id: WorkbenchThemeId;
  label: string;
  description: string;
  /** Native color-scheme hint for form controls, scrollbars, and `<select>` popups. */
  colorScheme: 'dark' | 'light';
  colors: ThemeColors;
}

const classic: ThemeColors = {
  neutral: {
    50: 'oklch(96% 0.006 264.5)',
    100: 'oklch(91.5% 0.009 264.5)',
    200: 'oklch(82.5% 0.014 264.5)',
    300: 'oklch(73.5% 0.019 264.4)',
    400: 'oklch(57.5% 0.026 264.3)',
    500: 'oklch(44% 0.024 264.3)',
    600: 'oklch(39% 0.022 264.3)',
    700: 'oklch(34% 0.02 264.3)',
    800: 'oklch(28.8% 0.018 264.3)',
    900: 'oklch(23.8% 0.016 264.35)',
    950: 'oklch(18.8% 0.014 264.4)',
  },
  brand: { solid: 'oklch(92.041% 0.2103 116.59)', contrast: 'oklch(18.8% 0.014 264.4)' },
  accent: { solid: 'oklch(77.738% 0.1 231.76)', contrast: 'oklch(18.8% 0.014 264.4)' },
  danger: 'oklch(70.61% 0.0841 19.38)',
  success: 'oklch(79.8% 0.1132 141.63)',
  warning: 'oklch(76.62% 0.0612 62.9)',
  inset: 'oklch(18.8% 0.014 264.4)',
  // Keep fill visibly above control surfaces so ghost hovers do not disappear.
  fill: 'oklch(36% 0.021 264.3)',
  grid: 'oklch(38.5% 0.022 264.3)',
  control: 'oklch(34% 0.02 264.3)',
};

const light: ThemeColors = {
  neutral: {
    50: 'oklch(99.4% 0.002 264)',
    100: 'oklch(97% 0.003 264)',
    200: 'oklch(96% 0.0038 264)',
    300: 'oklch(90.5% 0.0065 264)',
    400: 'oklch(84% 0.0095 264)',
    500: 'oklch(63% 0.014 264)',
    600: 'oklch(53.5% 0.016 264)',
    700: 'oklch(45% 0.017 264)',
    800: 'oklch(35% 0.015 264)',
    900: 'oklch(27% 0.013 264)',
    950: 'oklch(22.5% 0.013 264)',
  },
  brand: { solid: 'oklch(92.041% 0.2103 116.59)', contrast: 'oklch(22.5% 0.013 264)' },
  accent: { solid: 'oklch(54.615% 0.2152 262.88)', contrast: 'oklch(100% 0 0)' },
  danger: 'oklch(55.509% 0.1707 24.62)',
  success: 'oklch(53% 0.15 150)',
  warning: 'oklch(60% 0.13 72)',
  inset: 'oklch(96% 0.0038 264)',
  fill: 'oklch(95.5% 0.005 264)',
  grid: 'oklch(88.5% 0.0075 264)',
  control: 'oklch(93.5% 0.005 264)',
};

const osakaJade: ThemeColors = {
  neutral: {
    50: 'oklch(92% 0.028 178)',
    100: 'oklch(87.5% 0.03 178)',
    200: 'oklch(79% 0.033 179)',
    300: 'oklch(70.5% 0.036 180)',
    400: 'oklch(53% 0.037 181)',
    500: 'oklch(36.5% 0.038 182)',
    600: 'oklch(28.5% 0.03 183)',
    700: 'oklch(25.5% 0.026 183)',
    800: 'oklch(21% 0.021 184)',
    900: 'oklch(18.5% 0.017 184)',
    950: 'oklch(16.5% 0.015 184)',
  },
  brand: { solid: 'oklch(81% 0.145 165)', contrast: 'oklch(17% 0.028 175)' },
  accent: { solid: 'oklch(72% 0.105 195)', contrast: 'oklch(16.5% 0.024 195)' },
  danger: 'oklch(69.5% 0.147 33)',
  success: 'oklch(78% 0.15 155)',
  warning: 'oklch(80% 0.12 78)',
  inset: 'oklch(16.5% 0.015 184)',
  fill: 'oklch(28.5% 0.032 182)',
  grid: 'oklch(34.5% 0.035 182)',
  control: 'oklch(25.5% 0.026 183)',
};

const mono: ThemeColors = {
  neutral: {
    50: 'oklch(93.1% 0 0)',
    100: 'oklch(88.789% 0 0)',
    200: 'oklch(80.168% 0 0)',
    300: 'oklch(71.547% 0 0)',
    400: 'oklch(53.824% 0 0)',
    500: 'oklch(34.846% 0 0)',
    600: 'oklch(28.908% 0 0)',
    700: 'oklch(27.274% 0 0)',
    800: 'oklch(22.213% 0 0)',
    900: 'oklch(20.019% 0 0)',
    950: 'oklch(18.22% 0 0)',
  },
  brand: { solid: 'oklch(93.1% 0 0)', contrast: 'oklch(18.22% 0 0)' },
  accent: { solid: 'oklch(68.622% 0 0)', contrast: 'oklch(18.22% 0 0)' },
  danger: 'oklch(71.115% 0.0934 19.64)',
  success: 'oklch(76% 0.06 150)',
  warning: 'oklch(79% 0.07 80)',
  inset: 'oklch(18.22% 0 0)',
  fill: 'oklch(28.908% 0 0)',
  grid: 'oklch(32.897% 0 0)',
  control: 'oklch(27.274% 0 0)',
};

const ultradark: ThemeColors = {
  neutral: {
    50: 'oklch(89% 0.008 250)',
    100: 'oklch(84.5% 0.008 250)',
    200: 'oklch(75.5% 0.007 250)',
    300: 'oklch(66.5% 0.007 250)',
    400: 'oklch(47% 0.006 250)',
    500: 'oklch(27% 0.005 250)',
    600: 'oklch(21% 0.004 250)',
    700: 'oklch(19% 0.004 250)',
    800: 'oklch(14.5% 0.003 250)',
    900: 'oklch(11.5% 0.003 250)',
    950: 'oklch(0% 0 0)',
  },
  brand: { solid: 'oklch(93.444% 0.19 125.56)', contrast: 'oklch(15% 0.004 250)' },
  accent: { solid: 'oklch(80.623% 0.1248 228.24)', contrast: 'oklch(17.416% 0.0256 235.84)' },
  danger: 'oklch(71.161% 0.1812 22.84)',
  success: 'oklch(76.5% 0.16 150.5)',
  warning: 'oklch(79.5% 0.13 80)',
  inset: 'oklch(0% 0 0)',
  fill: 'oklch(21.8% 0.004 250)',
  grid: 'oklch(24% 0.004 250)',
  control: 'oklch(19% 0.004 250)',
};

// Catppuccin Mocha palette (https://catppuccin.com), converted from published sRGB; Mauve brand, Blue accent.
const catppuccinMocha: ThemeColors = {
  neutral: {
    50: 'oklch(87.87% 0.0426 272.28)', // Text
    100: 'oklch(81.68% 0.0403 272.86)', // Subtext1
    200: 'oklch(75.1% 0.0396 273.93)', // Subtext0
    300: 'oklch(68.65% 0.0374 274.73)', // Overlay2
    400: 'oklch(61.76% 0.0367 276)', // Overlay1
    500: 'oklch(47.65% 0.034 278.64)', // Surface2
    600: 'oklch(40.37% 0.032 280.15)', // Surface1
    700: 'oklch(32.4% 0.0319 281.98)', // Surface0
    800: 'oklch(24.29% 0.0304 283.91)', // Base
    900: 'oklch(21.55% 0.0254 284.06)', // Mantle
    950: 'oklch(18.28% 0.0204 284.2)', // Crust
  },
  brand: { solid: 'oklch(78.71% 0.1187 304.77)', contrast: 'oklch(18.28% 0.0204 284.2)' }, // Mauve on Crust
  accent: { solid: 'oklch(76.64% 0.1113 259.88)', contrast: 'oklch(18.28% 0.0204 284.2)' }, // Blue on Crust
  danger: 'oklch(75.56% 0.1297 2.76)', // Red
  success: 'oklch(85.77% 0.1092 142.72)', // Green
  warning: 'oklch(82.37% 0.1015 52.63)', // Peach
  inset: 'oklch(18.28% 0.0204 284.2)', // Crust
  // Interpolate fill between Surface0 and Surface1 so hover and badges remain visible on controls.
  fill: 'oklch(36.4% 0.0319 281.06)',
  grid: 'oklch(40.37% 0.032 280.15)', // Surface1
  control: 'oklch(32.4% 0.0319 281.98)', // Surface0
};

/** Registry order is Settings display order. */
export const THEMES: ThemeDefinition[] = [
  {
    id: 'classic',
    label: 'Classic',
    description: 'The legacy blue-on-graphite workbench.',
    colorScheme: 'dark',
    colors: classic,
  },
  {
    id: 'light',
    label: 'Light',
    description: 'Bright, high-contrast surfaces for daylight work.',
    colorScheme: 'light',
    colors: light,
  },
  {
    id: 'ultradark',
    label: 'Ultra Dark',
    description: 'Pure-black OLED surfaces for low-light rooms.',
    colorScheme: 'dark',
    colors: ultradark,
  },
  {
    id: 'mono',
    label: 'Mono',
    description: 'Neutral grayscale with no color cast.',
    colorScheme: 'dark',
    colors: mono,
  },
  {
    id: 'osakaJade',
    label: 'Osaka Jade',
    description: 'Deep jade stone with a cool green accent.',
    colorScheme: 'dark',
    colors: osakaJade,
  },
  {
    id: 'catppuccinMocha',
    label: 'Catppuccin Mocha',
    description: 'Soothing pastel accents on deep violet-gray surfaces.',
    colorScheme: 'dark',
    colors: catppuccinMocha,
  },
];

const defaultTheme = THEMES[0];

if (!defaultTheme) {
  throw new Error('At least one workbench theme must be defined.');
}

/** The default theme. Its ramp/seeds are emitted as the semantic-token `base` value. */
export const DEFAULT_THEME = defaultTheme;

export const DEFAULT_THEME_ID: WorkbenchThemeId = DEFAULT_THEME.id;

export const THEMES_BY_ID: Record<WorkbenchThemeId, ThemeDefinition> = THEMES.reduce(
  (accumulator, theme) => {
    accumulator[theme.id] = theme;
    return accumulator;
  },
  {} as Record<WorkbenchThemeId, ThemeDefinition>
);

export const isWorkbenchThemeId = (value: unknown): value is WorkbenchThemeId =>
  typeof value === 'string' && Object.prototype.hasOwnProperty.call(THEMES_BY_ID, value);

/** Renamed themes keep working for preferences persisted under the old id. */
const LEGACY_THEME_IDS: Record<string, WorkbenchThemeId> = {
  forest: 'osakaJade',
};

/** A stored theme preference → a current theme id, or `null` for junk. */
export const resolveWorkbenchThemeId = (value: unknown): WorkbenchThemeId | null => {
  if (isWorkbenchThemeId(value)) {
    return value;
  }

  return typeof value === 'string' ? (LEGACY_THEME_IDS[value] ?? null) : null;
};

/** The default panel surface of a theme (lightest end in light mode, near-darkest in dark). */
const surfaceOf = (theme: ThemeDefinition): string =>
  theme.colorScheme === 'light' ? theme.colors.neutral[50] : theme.colors.neutral[900];

export const previewSwatches = (theme: ThemeDefinition): [string, string, string, string] => [
  surfaceOf(theme),
  theme.colors.control,
  theme.colors.brand.solid,
  theme.colors.accent.solid,
];
