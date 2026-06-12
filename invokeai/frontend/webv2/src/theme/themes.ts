import type { WorkbenchThemeId } from '../workbench/types';

/**
 * A single, mode-agnostic set of concrete CSS color values that fully describes
 * one workbench theme. Every theme provides the same slots so the semantic-token
 * builder in `system.ts` can generate a uniform token contract across themes.
 *
 * Slots are named for *elevation* and *intent* — never for the widget or screen
 * region that happens to use them. Components never read these slots directly;
 * they consume the semantic tokens (`bg.subtle`, `fg.muted`, `brand.solid`, …)
 * that map onto them. Adding a new theme is therefore a matter of adding one
 * entry here; no component changes.
 */
export interface ThemeColors {
  /** Deepest backdrop behind every panel (`bg`). */
  base: string;
  /** Default panel / rail surface (`bg.subtle`). */
  surface: string;
  /** Lifted chrome: top bar, headers, popovers, dialogs (`bg.muted`, `bg.panel`). */
  raised: string;
  /** Recessed work-area floor framed by the chrome (`bg.inset`). */
  inset: string;
  /** Control / chip fill inside panels (`bg.emphasized`). */
  control: string;
  /** Subtle neutral fill for hovered/ghost controls (`gray.subtle`). */
  fill: string;
  /** Hairline divider (`border`, `border.subtle`, `border.muted`). */
  line: string;
  /** Emphasized divider / popover stroke (`border.emphasized`). */
  lineStrong: string;
  /** Dot-grid decoration drawn on `inset` surfaces (`fg.grid`). */
  grid: string;
  /** Primary text (`fg`). */
  text: string;
  /** Secondary text (`fg.muted`). */
  textMuted: string;
  /** Tertiary / disabled text (`fg.subtle`). */
  textSubtle: string;
  /** Invoke identity / global action fill (`brand.solid`). */
  brand: string;
  /** Foreground on top of `brand` (`brand.contrast`). */
  brandContrast: string;
  /** Tinted brand surface for identity chips (`brand.subtle`). */
  brandSubtle: string;
  /** Standalone brand foreground, readable on panels (`brand.fg`). */
  brandFg: string;
  /** Selection / focus accent fill (`accent.solid`). */
  accent: string;
  /** Foreground on top of `accent` (`accent.contrast`). */
  accentContrast: string;
  /** Destructive / error (`fg.error`). */
  danger: string;
  /** Positive / success (`fg.success`). */
  success: string;
  /** Caution / warning (`fg.warning`). */
  warning: string;
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
  base: 'oklch(21.074% 0.0087 264.37)',
  surface: 'oklch(26.279% 0.0123 264.34)',
  raised: 'oklch(31.237% 0.0157 264.32)',
  inset: 'oklch(21.074% 0.0087 264.37)',
  control: 'oklch(36.004% 0.019 264.3)',
  fill: 'oklch(31.237% 0.0157 264.32)',
  line: 'oklch(40.619% 0.0221 264.29)',
  lineStrong: 'oklch(45.106% 0.0251 264.28)',
  grid: 'oklch(40.619% 0.0221 264.29)',
  text: 'oklch(96.017% 0.0029 264.54)',
  textMuted: 'oklch(73.736% 0.0202 264.44)',
  textSubtle: 'oklch(57.965% 0.0337 264.27)',
  brand: 'oklch(92.041% 0.2103 116.59)',
  brandContrast: 'oklch(21.074% 0.0087 264.37)',
  brandSubtle: 'oklch(38.737% 0.0865 115.91)',
  brandFg: 'oklch(94.161% 0.1719 114.56)',
  accent: 'oklch(77.738% 0.1 231.76)',
  accentContrast: 'oklch(21.074% 0.0087 264.37)',
  danger: 'oklch(70.61% 0.0841 19.38)',
  success: 'oklch(79.8% 0.1132 141.63)',
  warning: 'oklch(76.62% 0.0612 62.9)',
};

const light: ThemeColors = {
  base: 'oklch(95.222% 0.007 124.45)',
  surface: 'oklch(100% 0 0)',
  raised: 'oklch(98.221% 0.0069 124.45)',
  inset: 'oklch(91.025% 0.0128 126.38)',
  control: 'oklch(96.697% 0.0082 121.63)',
  fill: 'oklch(98.221% 0.0069 124.45)',
  line: 'oklch(90.402% 0.014 124.57)',
  lineStrong: 'oklch(82.866% 0.0254 121.98)',
  grid: 'oklch(85.598% 0.0198 122.88)',
  text: 'oklch(22.5% 0.0139 126.4)',
  textMuted: 'oklch(46.478% 0.0248 125.12)',
  textSubtle: 'oklch(64.728% 0.0229 124.86)',
  brand: 'oklch(92.041% 0.2103 116.59)',
  brandContrast: 'oklch(22.5% 0.0139 126.4)',
  brandSubtle: 'oklch(96.2% 0.055 115.8)',
  brandFg: 'oklch(77.849% 0.1774 116.51)',
  accent: 'oklch(54.615% 0.2152 262.88)',
  accentContrast: 'oklch(100% 0 0)',
  danger: 'oklch(55.509% 0.1707 24.62)',
  success: 'oklch(53% 0.15 150)',
  warning: 'oklch(60% 0.13 72)',
};

const forest: ThemeColors = {
  base: 'oklch(17.349% 0.0154 144.88)',
  surface: 'oklch(19.03% 0.0175 144.85)',
  raised: 'oklch(20.99% 0.0219 144.74)',
  inset: 'oklch(18.457% 0.0201 144.71)',
  control: 'oklch(25.62% 0.03 144.64)',
  fill: 'oklch(28.489% 0.0483 141.22)',
  line: 'oklch(27.425% 0.0354 143.04)',
  lineStrong: 'oklch(36.133% 0.0575 141.06)',
  grid: 'oklch(34.825% 0.0526 141.63)',
  text: 'oklch(90.57% 0.0511 134.45)',
  textMuted: 'oklch(71.619% 0.0575 135.25)',
  textSubtle: 'oklch(53.577% 0.0538 136.73)',
  brand: 'oklch(79.714% 0.1784 136.37)',
  brandContrast: 'oklch(18.298% 0.0314 147.69)',
  brandSubtle: 'oklch(30.516% 0.0451 143.28)',
  brandFg: 'oklch(88.797% 0.1051 134.45)',
  accent: 'oklch(70.437% 0.1101 178.23)',
  accentContrast: 'oklch(17.44% 0.0258 171.89)',
  danger: 'oklch(69.305% 0.1467 35.44)',
  success: 'oklch(78% 0.16 148)',
  warning: 'oklch(80% 0.12 76)',
};

const mono: ThemeColors = {
  base: 'oklch(18.22% 0 0)',
  surface: 'oklch(20.019% 0 0)',
  raised: 'oklch(22.213% 0 0)',
  inset: 'oklch(22.645% 0 0)',
  control: 'oklch(27.274% 0 0)',
  fill: 'oklch(28.908% 0 0)',
  line: 'oklch(28.908% 0 0)',
  lineStrong: 'oklch(34.846% 0 0)',
  grid: 'oklch(32.897% 0 0)',
  text: 'oklch(93.1% 0 0)',
  textMuted: 'oklch(71.547% 0 0)',
  textSubtle: 'oklch(53.824% 0 0)',
  brand: 'oklch(93.1% 0 0)',
  brandContrast: 'oklch(18.22% 0 0)',
  brandSubtle: 'oklch(30.118% 0 0)',
  brandFg: 'oklch(87.61% 0 0)',
  accent: 'oklch(68.622% 0 0)',
  accentContrast: 'oklch(18.22% 0 0)',
  danger: 'oklch(71.115% 0.0934 19.64)',
  success: 'oklch(76% 0.06 150)',
  warning: 'oklch(79% 0.07 80)',
};

const ultradark: ThemeColors = {
  base: 'oklch(0% 0 0)',
  surface: 'oklch(11.492% 0 0)',
  raised: 'oklch(14.479% 0 0)',
  inset: 'oklch(14.958% 0 0)',
  control: 'oklch(19.125% 0 0)',
  fill: 'oklch(21.779% 0 0)',
  line: 'oklch(20.904% 0 0)',
  lineStrong: 'oklch(26.862% 0 0)',
  grid: 'oklch(23.929% 0 0)',
  text: 'oklch(88.901% 0.0317 120.83)',
  textMuted: 'oklch(66.382% 0.0192 131.96)',
  textSubtle: 'oklch(47.041% 0.0184 127.12)',
  brand: 'oklch(93.444% 0.19 125.56)',
  brandContrast: 'oklch(15.913% 0.0233 128.66)',
  brandSubtle: 'oklch(25.602% 0.0333 125.95)',
  brandFg: 'oklch(90.752% 0.134 124.5)',
  accent: 'oklch(80.623% 0.1248 228.24)',
  accentContrast: 'oklch(17.416% 0.0256 235.84)',
  danger: 'oklch(71.161% 0.1812 22.84)',
  success: 'oklch(76.5% 0.16 150.5)',
  warning: 'oklch(79.5% 0.13 80)',
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

/** The default theme. Its palette is emitted as the semantic-token `base` value. */
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
