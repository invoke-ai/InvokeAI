import type { WorkbenchThemeId } from '../workbench/types';

/**
 * A single, mode-agnostic set of concrete color values that fully describes one
 * workbench theme. Every theme provides the same slots so the semantic-token
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
  shell: '#141514',
  surface: '#161716',
  surfaceRaised: '#181918',
  center: '#1a1b1a',
  canvas: '#1c1d1c',
  panel: '#282928',
  panelStroke: '#303330',
  line: '#2e312d',
  lineStrong: '#343832',
  dot: '#343734',
  fg: '#d7dec7',
  fgMuted: '#a9afa1',
  fgSubtle: '#6f756a',
  accent: '#cbff63',
  accentFg: '#10160b',
  accentMuted: '#33372a',
  accentMutedFg: '#dfff8c',
  active: '#59cfff',
  activeFg: '#081218',
  danger: '#ff6b6b',
};

const light: ThemeColors = {
  shell: '#eef0eb',
  surface: '#ffffff',
  surfaceRaised: '#f8faf5',
  center: '#e7eae3',
  canvas: '#dfe3da',
  panel: '#f3f5ef',
  panelStroke: '#d2d7c9',
  line: '#e0e4da',
  lineStrong: '#c7cdbb',
  dot: '#cdd2c4',
  fg: '#1a1d16',
  fgMuted: '#565c4d',
  fgSubtle: '#8b9182',
  accent: '#69a417',
  accentFg: '#0d1604',
  accentMuted: '#dcecbf',
  accentMutedFg: '#39520f',
  active: '#2563eb',
  activeFg: '#ffffff',
  danger: '#c33d3d',
};

const forest: ThemeColors = {
  shell: '#0c120c',
  surface: '#0f160f',
  surfaceRaised: '#121b12',
  center: '#101810',
  canvas: '#0d150d',
  panel: '#1a271a',
  panelStroke: '#28401f',
  line: '#1d2c1c',
  lineStrong: '#2c4528',
  dot: '#2a4127',
  fg: '#d2e8c6',
  fgMuted: '#94ac88',
  fgSubtle: '#5f7556',
  accent: '#84d65a',
  accentFg: '#08160a',
  accentMuted: '#213520',
  accentMutedFg: '#bdeaa3',
  active: '#3fb6a0',
  activeFg: '#04140f',
  danger: '#e8775a',
};

const mono: ThemeColors = {
  shell: '#121212',
  surface: '#161616',
  surfaceRaised: '#1b1b1b',
  center: '#181818',
  canvas: '#1c1c1c',
  panel: '#272727',
  panelStroke: '#343434',
  line: '#2b2b2b',
  lineStrong: '#3a3a3a',
  dot: '#353535',
  fg: '#e8e8e8',
  fgMuted: '#a3a3a3',
  fgSubtle: '#6e6e6e',
  accent: '#e8e8e8',
  accentFg: '#121212',
  accentMuted: '#2e2e2e',
  accentMutedFg: '#d6d6d6',
  active: '#9a9a9a',
  activeFg: '#121212',
  danger: '#d68a8a',
};

const ultradark: ThemeColors = {
  shell: '#000000',
  surface: '#050505',
  surfaceRaised: '#0a0a0a',
  center: '#070707',
  canvas: '#0b0b0b',
  panel: '#141414',
  panelStroke: '#202020',
  line: '#181818',
  lineStrong: '#262626',
  dot: '#1f1f1f',
  fg: '#d7dec7',
  fgMuted: '#8f968a',
  fgSubtle: '#585d52',
  accent: '#cbff63',
  accentFg: '#0a0f05',
  accentMuted: '#1f2614',
  accentMutedFg: '#cdf08a',
  active: '#59cfff',
  activeFg: '#05121a',
  danger: '#ff6b6b',
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
