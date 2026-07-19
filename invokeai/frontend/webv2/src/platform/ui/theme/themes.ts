export type WorkbenchThemeId = 'classic' | 'light' | 'forest' | 'mono' | 'ultradark';

/**
 * Steps of the neutral ramp. `50` is the lightest, `950` the darkest — the same
 * absolute orientation Chakra/Tailwind use, so the ramp can be aliased onto the
 * built-in `gray` palette without surprises.
 */
export type NeutralStep = 50 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 950;

/**
 * One workbench theme, expressed as a small set of concrete OKLch values.
 *
 * The bulk of a theme is the `base` ramp — a single neutral scale from which the
 * semantic-token layer in `system.ts` derives every background, foreground, and
 * border (`bg → neutral.950` in dark mode, `fg → neutral.50`, …). Everything else is a
 * handful of seeds:
 *
 *   - `brand` / `accent` — the two identity hues (lime action, blue selection);
 *   - `danger` / `success` / `warning` — status intents;
 *   - `inset` / `fill` / `grid` / `control` — four neutrals whose *elevation rank*
 *     differs from theme to theme, so they cannot sit on a single shared ramp step
 *     (e.g. `inset` recesses below the app background in the light theme but lifts
 *     above it in the dark themes). They are kept as explicit per-theme values and
 *     consumed by name through `bg.inset`, `fg.grid`, `gray.subtle`, `bg.emphasized`.
 *
 * Adding a theme is therefore: author one ramp + the seeds. No component changes.
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
    50: 'oklch(96.017% 0.0029 264.54)',
    100: 'oklch(91.561% 0.0064 264.52)',
    200: 'oklch(82.648% 0.0133 264.48)',
    300: 'oklch(73.736% 0.0202 264.44)',
    400: 'oklch(57.965% 0.0337 264.27)',
    500: 'oklch(45.106% 0.0251 264.28)',
    600: 'oklch(40.619% 0.0221 264.29)',
    700: 'oklch(36.004% 0.019 264.3)',
    800: 'oklch(31.237% 0.0157 264.32)',
    900: 'oklch(26.279% 0.0123 264.34)',
    950: 'oklch(21.074% 0.0087 264.37)',
  },
  brand: { solid: 'oklch(92.041% 0.2103 116.59)', contrast: 'oklch(21.074% 0.0087 264.37)' },
  accent: { solid: 'oklch(77.738% 0.1 231.76)', contrast: 'oklch(21.074% 0.0087 264.37)' },
  danger: 'oklch(70.61% 0.0841 19.38)',
  success: 'oklch(79.8% 0.1132 141.63)',
  warning: 'oklch(76.62% 0.0612 62.9)',
  inset: 'oklch(21.074% 0.0087 264.37)',
  fill: 'oklch(31.237% 0.0157 264.32)',
  grid: 'oklch(40.619% 0.0221 264.29)',
  control: 'oklch(36.004% 0.019 264.3)',
};

// Cool blue-gray neutrals (hue 264, harmonizing with the blue accent): near-white
// chrome floating on a soft-gray work floor, near-black cool text. Airy and clean.
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

const forest: ThemeColors = {
  neutral: {
    50: 'oklch(90.57% 0.0511 134.45)',
    100: 'oklch(86.78% 0.0524 134.61)',
    200: 'oklch(79.199% 0.0549 134.93)',
    300: 'oklch(71.619% 0.0575 135.25)',
    400: 'oklch(53.577% 0.0538 136.73)',
    500: 'oklch(36.133% 0.0575 141.06)',
    600: 'oklch(27.425% 0.0354 143.04)',
    700: 'oklch(25.62% 0.03 144.64)',
    800: 'oklch(20.99% 0.0219 144.74)',
    900: 'oklch(19.03% 0.0175 144.85)',
    950: 'oklch(17.349% 0.0154 144.88)',
  },
  brand: { solid: 'oklch(79.714% 0.1784 136.37)', contrast: 'oklch(18.298% 0.0314 147.69)' },
  accent: { solid: 'oklch(70.437% 0.1101 178.23)', contrast: 'oklch(17.44% 0.0258 171.89)' },
  danger: 'oklch(69.305% 0.1467 35.44)',
  success: 'oklch(78% 0.16 148)',
  warning: 'oklch(80% 0.12 76)',
  inset: 'oklch(17.349% 0.0154 144.88)',
  fill: 'oklch(28.489% 0.0483 141.22)',
  grid: 'oklch(34.825% 0.0526 141.63)',
  control: 'oklch(25.62% 0.03 144.64)',
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
    50: 'oklch(88.901% 0.0317 120.83)',
    100: 'oklch(84.397% 0.0292 123.06)',
    200: 'oklch(75.39% 0.0242 127.51)',
    300: 'oklch(66.382% 0.0192 131.96)',
    400: 'oklch(47.041% 0.0184 127.12)',
    500: 'oklch(26.862% 0 0)',
    600: 'oklch(20.904% 0 0)',
    700: 'oklch(19.125% 0 0)',
    800: 'oklch(14.479% 0 0)',
    900: 'oklch(11.492% 0 0)',
    950: 'oklch(0% 0 0)',
  },
  brand: { solid: 'oklch(93.444% 0.19 125.56)', contrast: 'oklch(15.913% 0.0233 128.66)' },
  accent: { solid: 'oklch(80.623% 0.1248 228.24)', contrast: 'oklch(17.416% 0.0256 235.84)' },
  danger: 'oklch(71.161% 0.1812 22.84)',
  success: 'oklch(76.5% 0.16 150.5)',
  warning: 'oklch(79.5% 0.13 80)',
  inset: 'oklch(0% 0 0)',
  fill: 'oklch(21.779% 0 0)',
  grid: 'oklch(23.929% 0 0)',
  control: 'oklch(19.125% 0 0)',
};

/**
 * Theme registry. Order here is the display order in the Settings picker.
 * `THEMES` is the single source of truth consumed by both the token builder
 * and the settings UI.
 */
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
    id: 'forest',
    label: 'Forest',
    description: 'Deep greens with a leafy accent.',
    colorScheme: 'dark',
    colors: forest,
  },
  {
    id: 'mono',
    label: 'Mono',
    description: 'Neutral grayscale with no color cast.',
    colorScheme: 'dark',
    colors: mono,
  },
  {
    id: 'ultradark',
    label: 'Ultra Dark',
    description: 'Pure-black OLED surfaces for low-light rooms.',
    colorScheme: 'dark',
    colors: ultradark,
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

/** The default panel surface of a theme (lightest end in light mode, near-darkest in dark). */
const surfaceOf = (theme: ThemeDefinition): string =>
  theme.colorScheme === 'light' ? theme.colors.neutral[50] : theme.colors.neutral[900];

/**
 * The four representative chips shown in the Settings appearance picker:
 * surface, control fill, brand, accent — a compact read of the theme's identity.
 */
export const previewSwatches = (theme: ThemeDefinition): [string, string, string, string] => [
  surfaceOf(theme),
  theme.colors.control,
  theme.colors.brand.solid,
  theme.colors.accent.solid,
];
