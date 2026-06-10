import type { WorkbenchThemeId } from '../workbench/types';

/**
 * A single, mode-agnostic set of concrete CSS color values that fully describes
 * one workbench theme. Every theme provides the same slots so the semantic-token
 * builder in `system.ts` can generate a uniform token contract across themes.
 *
 * Components never read these slots directly — they consume the semantic tokens
 * (`bg.surface`, `fg.muted`, `accent.invoke`, …) that map onto them. Adding a new
 * theme is therefore a matter of adding one entry here; no component changes.
 */
export interface ThemeColors {
  /** App backdrop behind every panel. */
  shell: string;
  /** Default raised panel / rail surface. */
  surface: string;
  /** Slightly lifted surface (top bar, headers, popovers). */
  surfaceRaised: string;
  /** Center work-area background. */
  center: string;
  /** Canvas viewport background. */
  canvas: string;
  /** Inset / control surface inside panels. */
  panel: string;
  /** Stroke around the panel surface. */
  panelStroke: string;
  /** Hairline divider color. */
  line: string;
  /** Stronger divider / emphasized border. */
  lineStrong: string;
  /** Canvas dot-grid color. */
  dot: string;
  /** Primary text. */
  fg: string;
  /** Secondary text. */
  fgMuted: string;
  /** Tertiary / disabled text. */
  fgSubtle: string;
  /** Primary action (Invoke) color. */
  accent: string;
  /** Foreground used on top of `accent`. */
  accentFg: string;
  /** Muted accent surface (active widget chip background). */
  accentMuted: string;
  /** Foreground used on top of `accentMuted`. */
  accentMutedFg: string;
  /** Secondary / selection accent (queue, active rail). */
  active: string;
  /** Foreground used on top of `active`. */
  activeFg: string;
  /** Destructive / error foreground. */
  danger: string;
  /** Chakra virtual color-palette subtle surface for component variant fills. */
  paletteSubtle: string;
}

export interface ThemeDefinition {
  id: WorkbenchThemeId;
  label: string;
  description: string;
  /** Native color-scheme hint for form controls, scrollbars, and `<select>` popups. */
  colorScheme: 'dark' | 'light';
  colors: ThemeColors;
}

/** The default theme. Its palette is emitted as the semantic-token `base` value. */
export const DEFAULT_THEME_ID: WorkbenchThemeId = 'dark';

const dark: ThemeColors = {
  shell: 'oklch(19.42% 0.0025 145.48)',
  surface: 'oklch(20.311% 0.0025 145.48)',
  surfaceRaised: 'oklch(21.192% 0.0025 145.49)',
  center: 'oklch(22.064% 0.0024 145.49)',
  canvas: 'oklch(22.927% 0.0024 145.5)',
  panel: 'oklch(27.953% 0.0023 145.51)',
  panelStroke: 'oklch(31.705% 0.0067 145.42)',
  line: 'oklch(30.875% 0.0082 137.81)',
  lineStrong: 'oklch(33.473% 0.0117 135)',
  dot: 'oklch(33.28% 0.0066 145.42)',
  fg: 'oklch(88.901% 0.0317 120.83)',
  fgMuted: 'oklch(74.519% 0.0208 125.92)',
  fgSubtle: 'oklch(55.387% 0.0182 130.58)',
  accent: 'oklch(93.444% 0.19 125.56)',
  accentFg: 'oklch(19.003% 0.0232 131.55)',
  accentMuted: 'oklch(32.866% 0.0228 121.55)',
  accentMutedFg: 'oklch(95.264% 0.1456 121.96)',
  active: 'oklch(80.623% 0.1248 228.24)',
  activeFg: 'oklch(17.553% 0.0195 235.44)',
  danger: 'oklch(71.161% 0.1812 22.84)',
  paletteSubtle: 'oklch(25.368% 0.0035 164.78)',
};

const light: ThemeColors = {
  shell: 'oklch(95.222% 0.007 124.45)',
  surface: 'oklch(100% 0 0)',
  surfaceRaised: 'oklch(98.221% 0.0069 124.45)',
  center: 'oklch(93.28% 0.0099 125.68)',
  canvas: 'oklch(91.025% 0.0128 126.38)',
  panel: 'oklch(96.697% 0.0082 121.63)',
  panelStroke: 'oklch(87.14% 0.0197 122.87)',
  line: 'oklch(91.302% 0.014 124.57)',
  lineStrong: 'oklch(83.866% 0.0254 121.98)',
  dot: 'oklch(85.598% 0.0198 122.88)',
  fg: 'oklch(22.5% 0.0139 126.4)',
  fgMuted: 'oklch(46.478% 0.0248 125.12)',
  fgSubtle: 'oklch(64.728% 0.0229 124.86)',
  accent: 'oklch(65.33% 0.1716 131.16)',
  accentFg: 'oklch(18.537% 0.0379 130.07)',
  accentMuted: 'oklch(91.965% 0.0617 123.52)',
  accentMutedFg: 'oklch(40.391% 0.0959 128.33)',
  active: 'oklch(54.615% 0.2152 262.88)',
  activeFg: 'oklch(100% 0 0)',
  danger: 'oklch(55.509% 0.1707 24.62)',
  paletteSubtle: 'oklch(98.221% 0.0069 124.45)',
};

const forest: ThemeColors = {
  shell: 'oklch(17.349% 0.0154 144.88)',
  surface: 'oklch(19.03% 0.0175 144.85)',
  surfaceRaised: 'oklch(20.99% 0.0219 144.74)',
  center: 'oklch(19.795% 0.0198 144.78)',
  canvas: 'oklch(18.457% 0.0201 144.71)',
  panel: 'oklch(25.62% 0.03 144.64)',
  panelStroke: 'oklch(34.227% 0.0623 137.66)',
  line: 'oklch(27.425% 0.0354 143.04)',
  lineStrong: 'oklch(36.133% 0.0575 141.06)',
  dot: 'oklch(34.825% 0.0526 141.63)',
  fg: 'oklch(90.57% 0.0511 134.45)',
  fgMuted: 'oklch(71.619% 0.0575 135.25)',
  fgSubtle: 'oklch(53.577% 0.0538 136.73)',
  accent: 'oklch(79.714% 0.1784 136.37)',
  accentFg: 'oklch(18.298% 0.0314 147.69)',
  accentMuted: 'oklch(30.516% 0.0451 143.28)',
  accentMutedFg: 'oklch(88.797% 0.1051 134.45)',
  active: 'oklch(70.437% 0.1101 178.23)',
  activeFg: 'oklch(17.44% 0.0258 171.89)',
  danger: 'oklch(69.305% 0.1467 35.44)',
  paletteSubtle: 'oklch(28.489% 0.0483 141.22)',
};

const mono: ThemeColors = {
  shell: 'oklch(18.22% 0 0)',
  surface: 'oklch(20.019% 0 0)',
  surfaceRaised: 'oklch(22.213% 0 0)',
  center: 'oklch(20.904% 0 0)',
  canvas: 'oklch(22.645% 0 0)',
  panel: 'oklch(27.274% 0 0)',
  panelStroke: 'oklch(32.504% 0 0)',
  line: 'oklch(28.908% 0 0)',
  lineStrong: 'oklch(34.846% 0 0)',
  dot: 'oklch(32.897% 0 0)',
  fg: 'oklch(93.1% 0 0)',
  fgMuted: 'oklch(71.547% 0 0)',
  fgSubtle: 'oklch(53.824% 0 0)',
  accent: 'oklch(93.1% 0 0)',
  accentFg: 'oklch(18.22% 0 0)',
  accentMuted: 'oklch(30.118% 0 0)',
  accentMutedFg: 'oklch(87.61% 0 0)',
  active: 'oklch(68.622% 0 0)',
  activeFg: 'oklch(18.22% 0 0)',
  danger: 'oklch(71.115% 0.0934 19.64)',
  paletteSubtle: 'oklch(28.908% 0 0)',
};

const ultradark: ThemeColors = {
  shell: 'oklch(0% 0 0)',
  surface: 'oklch(11.492% 0 0)',
  surfaceRaised: 'oklch(14.479% 0 0)',
  center: 'oklch(12.856% 0 0)',
  canvas: 'oklch(14.958% 0 0)',
  panel: 'oklch(19.125% 0 0)',
  panelStroke: 'oklch(24.353% 0 0)',
  line: 'oklch(20.904% 0 0)',
  lineStrong: 'oklch(26.862% 0 0)',
  dot: 'oklch(23.929% 0 0)',
  fg: 'oklch(88.901% 0.0317 120.83)',
  fgMuted: 'oklch(66.382% 0.0192 131.96)',
  fgSubtle: 'oklch(47.041% 0.0184 127.12)',
  accent: 'oklch(93.444% 0.19 125.56)',
  accentFg: 'oklch(15.913% 0.0233 128.66)',
  accentMuted: 'oklch(25.602% 0.0333 125.95)',
  accentMutedFg: 'oklch(90.752% 0.134 124.5)',
  active: 'oklch(80.623% 0.1248 228.24)',
  activeFg: 'oklch(17.416% 0.0256 235.84)',
  danger: 'oklch(71.161% 0.1812 22.84)',
  paletteSubtle: 'oklch(21.779% 0 0)',
};

/**
 * Theme registry. Order here is the display order in the Settings picker.
 * `THEMES` is the single source of truth consumed by both the token builder
 * and the settings UI.
 */
export const THEMES: ThemeDefinition[] = [
  {
    id: 'dark',
    label: 'Dark',
    description: 'The default lime-on-charcoal workbench.',
    colorScheme: 'dark',
    colors: dark,
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

export const THEMES_BY_ID: Record<WorkbenchThemeId, ThemeDefinition> = THEMES.reduce(
  (accumulator, theme) => {
    accumulator[theme.id] = theme;
    return accumulator;
  },
  {} as Record<WorkbenchThemeId, ThemeDefinition>
);

export const isWorkbenchThemeId = (value: unknown): value is WorkbenchThemeId =>
  typeof value === 'string' && Object.prototype.hasOwnProperty.call(THEMES_BY_ID, value);
