import { createSystem, defaultConfig, defineConfig } from '@chakra-ui/react';

import {
  comboboxSlotRecipe,
  dialogSlotRecipe,
  menuSlotRecipe,
  selectSlotRecipe,
  sliderSlotRecipe,
  tooltipSlotRecipe,
} from './recipes';
import { DEFAULT_THEME, DEFAULT_THEME_ID, type NeutralStep, THEMES, type ThemeDefinition } from './themes';

/**
 * Workbench design system.
 *
 * Each theme (`themes.ts`) is authored as one neutral ramp (`neutral.50…neutral.950`,
 * lightest → darkest) plus a few seeds (brand, accent, status) and four off-ramp
 * neutrals (`inset`, `fill`, `grid`, `control`). This module turns that into two
 * token layers:
 *
 *   1. The **ramp** is emitted verbatim as conditional semantic tokens
 *      (`neutral.*`), one value per theme keyed on a custom `[data-theme=<id>]`
 *      condition. Chakra base `tokens.colors` only hold plain strings, so anything
 *      that varies per theme must live in `semanticTokens`.
 *   2. The **semantic contract** (`bg`, `fg.muted`, `border.emphasized`, the
 *      `gray`/`brand`/`accent` palettes, …) is theme-agnostic: it references ramp
 *      steps and flips by light/dark mode only. `bg` is `neutral.950` in dark mode and
 *      `neutral.200` in light — the light ramp is not a mirror of the dark one, because
 *      light panels go *whiter* than the app background.
 *
 * `ThemeController` sets `data-theme` (and the `.dark`/`.light` class) on `<html>`,
 * so switching themes is a single attribute change with zero React re-render.
 * Components reference only the semantic tokens — never the ramp or a theme.
 *
 * Re-pointing Chakra's neutral `gray` palette at the ramp makes every built-in
 * component (Dialog, Menu, Input, Button, Tooltip, Badge) follow the active theme
 * with no per-component overrides.
 */

const NON_DEFAULT_THEMES = THEMES.filter((theme) => theme.id !== DEFAULT_THEME_ID);

/** `light` -> `themeLight`, `ultradark` -> `themeUltradark`. */
const conditionName = (id: string): string => `theme${id.charAt(0).toUpperCase()}${id.slice(1)}`;

type TokenValue = { value: Record<string, string> };
type Compute = (theme: ThemeDefinition) => string;

/** Build a semantic-token value object: default theme as `base`, the rest as `[data-theme]` conditions. */
const colorToken = (compute: Compute): TokenValue => {
  const value: Record<string, string> = { base: compute(DEFAULT_THEME) };
  for (const theme of NON_DEFAULT_THEMES) {
    value[`_${conditionName(theme.id)}`] = compute(theme);
  }
  return { value };
};

/** Blend `pct`% of one computed color into another — used for derived hover/tint steps. */
const mix = (top: Compute, pct: number, bottom: Compute): TokenValue =>
  colorToken((theme) => `color-mix(in oklab, ${top(theme)} ${pct}%, ${bottom(theme)})`);

const LIGHT_FALLBACK_THEME = THEMES.find((theme) => theme.colorScheme === 'light') ?? DEFAULT_THEME;

/**
 * Like `colorToken`, but additionally shadows Chakra's `_light`/`_dark`
 * class-conditional values. Nested palette tokens (`gray.*`) deep-merge with
 * `defaultConfig` instead of replacing it, so without these keys the default
 * gray values would survive the merge and outrank our zero-specificity `base`
 * value whenever `ThemeController` sets the `.dark`/`.light` class. The explicit
 * `:root[data-theme=…]` conditions still win over both.
 */
const grayToken = (compute: Compute): TokenValue => {
  const token = colorToken(compute);
  token.value._light = compute(LIGHT_FALLBACK_THEME);
  token.value._dark = compute(DEFAULT_THEME);
  return token;
};

/**
 * A token that reads a ramp step, chosen per theme by `colorScheme`. Emitted as
 * per-`[data-theme]` conditions, NOT Chakra's `_light`/`_dark`: the built-in
 * `_light` selector is `:root &, .light &`, and its `:root &` arm matches under
 * EVERY theme — so a mode-flip token leaks its light value into the dark themes.
 * Per-theme conditions sidestep the cascade entirely (this is how the pre-refactor
 * system worked, and why dark themes rendered correctly).
 */
const ref = (step: NeutralStep): string => `{colors.neutral.${step}}`;
const stepRef = (darkStep: NeutralStep, lightStep: NeutralStep): TokenValue =>
  colorToken((theme) => ref(theme.colorScheme === 'light' ? lightStep : darkStep));

const STEPS: NeutralStep[] = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950];

/** The default panel surface of a theme — `bg.subtle`'s step. Used as the floor for tints. */
const surface: Compute = (theme) =>
  theme.colorScheme === 'light' ? theme.colors.neutral[50] : theme.colors.neutral[900];

// Seed accessors.
const danger: Compute = (theme) => theme.colors.danger;
const success: Compute = (theme) => theme.colors.success;
const warning: Compute = (theme) => theme.colors.warning;
const brandSolid: Compute = (theme) => theme.colors.brand.solid;
const accentSolid: Compute = (theme) => theme.colors.accent.solid;

/** The neutral ramp, emitted as `neutral.50…neutral.950`, one value per theme. */
const neutralRamp = Object.fromEntries(STEPS.map((step) => [step, colorToken((theme) => theme.colors.neutral[step])]));

/**
 * The semantic-token contract. Backgrounds/foregrounds/borders reference ramp
 * steps (`stepRef`); the four off-ramp neutrals and the status/identity hues read
 * their per-theme seed directly. Where a Chakra built-in name exists we use it
 * verbatim so built-ins inherit the theme for free.
 */
const semanticColors = {
  neutral: neutralRamp,

  // Surface ladder. Light panels are whiter than the app bg, so the light steps
  // are not a mirror of the dark ones.
  bg: stepRef(950, 200),
  'bg.subtle': stepRef(900, 50),
  'bg.muted': stepRef(800, 100),
  'bg.panel': stepRef(800, 100),
  'bg.emphasized': colorToken((theme) => theme.colors.control),
  'bg.inset': colorToken((theme) => theme.colors.inset),
  // Soft status fills for alerts/banners, mixed into the panel surface.
  'bg.error': mix(danger, 14, surface),
  'bg.success': mix(success, 14, surface),
  'bg.warning': mix(warning, 14, surface),

  // Foreground.
  fg: stepRef(50, 950),
  'fg.muted': stepRef(300, 700),
  'fg.subtle': stepRef(400, 500),
  'fg.grid': colorToken((theme) => theme.colors.grid),
  'fg.error': colorToken(danger),
  'fg.success': colorToken(success),
  'fg.warning': colorToken(warning),

  // Borders.
  border: stepRef(600, 300),
  'border.subtle': stepRef(600, 300),
  'border.muted': stepRef(600, 300),
  'border.emphasized': stepRef(500, 400),
  'border.error': colorToken(danger),

  /**
   * Chakra's default `colorPalette` is `gray`; re-pointing its virtual-palette
   * keys at the ramp makes every un-palettized component (ghost buttons, menu
   * items, badges, …) theme-aware with zero props.
   *
   * Palette tokens must stay NESTED — the `colorPalette` virtual-token map is
   * built from the nested structure and ignores flat dotted keys. Because nesting
   * deep-merges with the defaults, every gray key shadows the default
   * `_light`/`_dark` values via `grayToken`.
   */
  gray: {
    contrast: grayToken((theme) =>
      theme.colorScheme === 'light' ? theme.colors.neutral[200] : theme.colors.neutral[950]
    ),
    fg: grayToken((theme) => (theme.colorScheme === 'light' ? theme.colors.neutral[950] : theme.colors.neutral[50])),
    subtle: grayToken((theme) => theme.colors.fill),
    muted: grayToken((theme) => theme.colors.control),
    emphasized: grayToken((theme) =>
      theme.colorScheme === 'light' ? theme.colors.neutral[400] : theme.colors.neutral[500]
    ),
    solid: grayToken((theme) => (theme.colorScheme === 'light' ? theme.colors.neutral[950] : theme.colors.neutral[50])),
    focusRing: grayToken(accentSolid),
    border: grayToken((theme) =>
      theme.colorScheme === 'light' ? theme.colors.neutral[400] : theme.colors.neutral[500]
    ),
  },
  /**
   * Invoke identity palette (lime). Authored from two seeds (`solid` + `contrast`),
   * like `accent`; the rest derive. NOTE: `brand.fg` is the bright `solid` fill, so
   * `brand.fg` on `brand.subtle` reads well on the dark themes but is low-contrast in
   * the light theme — brand is meant for emphasis fills, not body text.
   */
  brand: {
    solid: colorToken(brandSolid),
    contrast: colorToken((theme) => theme.colors.brand.contrast),
    fg: colorToken(brandSolid),
    subtle: mix(brandSolid, 16, surface),
    muted: mix(brandSolid, 26, surface),
    emphasized: mix(brandSolid, 36, surface),
    focusRing: colorToken(accentSolid),
    border: colorToken(brandSolid),
  },
  /** Selection / focus palette (blue). Use via `accent.solid` or `colorPalette="accent"`. */
  accent: {
    solid: colorToken(accentSolid),
    contrast: colorToken((theme) => theme.colors.accent.contrast),
    fg: colorToken(accentSolid),
    subtle: mix(accentSolid, 16, surface),
    muted: mix(accentSolid, 26, surface),
    emphasized: mix(accentSolid, 36, surface),
    focusRing: colorToken(accentSolid),
    border: colorToken(accentSolid),
  },
};

// `:root` raises specificity above the `.dark`/`.light` colorScheme classes so
// an explicit theme always beats the class-conditional fallback values.
const themeConditions = Object.fromEntries(
  NON_DEFAULT_THEMES.map((theme) => [conditionName(theme.id), `:root[data-theme=${theme.id}]`])
);

const config = defineConfig({
  conditions: themeConditions,
  globalCss: {
    'html, body, #root': {
      height: '100%',
    },
    body: {
      bg: 'bg',
      color: 'fg',
      fontFamily: 'body',
      margin: 0,
      minHeight: '720px',
      minWidth: '960px',
      overflow: 'hidden',
    },
    // Respect the "reduce motion" preference by neutralizing transitions/animations.
    '[data-reduce-motion="true"], [data-reduce-motion="true"] *, [data-reduce-motion="true"] *::before, [data-reduce-motion="true"] *::after':
      {
        animationDuration: '0.001ms !important',
        animationIterationCount: '1 !important',
        scrollBehavior: 'auto !important',
        transitionDuration: '0.001ms !important',
      },
  },
  theme: {
    tokens: {
      fonts: {
        body: {
          value: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        },
        heading: {
          value: "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif",
        },
      },
    },
    semanticTokens: {
      colors: semanticColors,
    },
    // Chrome-level overrides for Chakra's built-in components, so popover and
    // dialog surfaces are consistent everywhere without per-instance props.
    slotRecipes: {
      combobox: comboboxSlotRecipe,
      dialog: dialogSlotRecipe,
      menu: menuSlotRecipe,
      select: selectSlotRecipe,
      slider: sliderSlotRecipe,
      tooltip: tooltipSlotRecipe,
    },
  },
});

export const system = createSystem(defaultConfig, config);

/** Theme metadata re-exported so UI can import a single module. */
export { THEMES, THEMES_BY_ID, DEFAULT_THEME, DEFAULT_THEME_ID, previewSwatches } from './themes';
export type { ThemeColors, ThemeDefinition, NeutralStep } from './themes';
