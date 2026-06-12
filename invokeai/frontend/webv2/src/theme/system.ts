import { createSystem, defaultConfig, defineConfig } from '@chakra-ui/react';

import { dialogSlotRecipe, menuSlotRecipe, tooltipSlotRecipe } from './recipes';
import { DEFAULT_THEME, DEFAULT_THEME_ID, THEMES, THEMES_BY_ID, type ThemeColors } from './themes';

/**
 * Workbench design system.
 *
 * The shell is theme-able across several presets (dark, light, forest, mono,
 * ultra dark). Rather than hand-author one token set per theme, we derive the
 * semantic-token layer from the theme registry in `themes.ts`:
 *
 *   - the default theme (`DEFAULT_THEME_ID`) becomes each token's `base` value;
 *   - every other theme contributes a conditional value keyed on a custom
 *     `[data-theme=<id>]` condition.
 *
 * `ThemeController` sets `data-theme` on `<html>`, so flipping themes is a single
 * attribute change with zero re-render of the React tree. Components reference
 * only the semantic tokens below — they never know which theme is active.
 *
 * The token contract deliberately reuses Chakra's own semantic names (`bg`,
 * `bg.subtle`, `fg.muted`, `border.emphasized`, …) and re-points its neutral
 * `gray` palette at the theme ladder, so built-in components (Dialog, Menu,
 * Input, Button, Tooltip) follow the active theme with no per-component
 * overrides. Workbench-specific additions are intent-based (`bg.inset`,
 * `fg.grid`, the `brand` and `accent` palettes) — never named after a widget.
 */

const NON_DEFAULT_THEMES = THEMES.filter((theme) => theme.id !== DEFAULT_THEME_ID);

/** `light` -> `themeLight`, `ultradark` -> `themeUltradark`. */
const conditionName = (id: string): string => `theme${id.charAt(0).toUpperCase()}${id.slice(1)}`;

type TokenValue = { value: Record<string, string> };

/** Build a semantic-token value object: default theme as `base`, the rest as conditions. */
const buildToken = (compute: (colors: ThemeColors) => string): TokenValue => {
  const value: Record<string, string> = { base: compute(DEFAULT_THEME.colors) };

  for (const theme of NON_DEFAULT_THEMES) {
    value[`_${conditionName(theme.id)}`] = compute(theme.colors);
  }

  return { value };
};

const colorToken = (slot: keyof ThemeColors): TokenValue => buildToken((colors) => colors[slot]);

/** Blend `pct`% of a slot color into another slot — used for derived hover/tint steps. */
const mixToken = (slot: keyof ThemeColors, pct: number, baseSlot: keyof ThemeColors): TokenValue =>
  buildToken((colors) => `color-mix(in oklab, ${colors[slot]} ${pct}%, ${colors[baseSlot]})`);

const LIGHT_FALLBACK_THEME = THEMES.find((theme) => theme.colorScheme === 'light') ?? THEMES_BY_ID[DEFAULT_THEME_ID];

/**
 * Like `colorToken`, but additionally shadows Chakra's `_light`/`_dark`
 * class-conditional values. Nested palette tokens (`gray.*`) deep-merge with
 * `defaultConfig` instead of replacing it, so without these keys the default
 * gray values would survive the merge and outrank our zero-specificity `base`
 * value whenever `ThemeController` sets the `.dark`/`.light` class. The
 * explicit `:root[data-theme=…]` conditions still win over both.
 */
const colorTokenWithModeFallback = (slot: keyof ThemeColors): TokenValue => {
  const token = colorToken(slot);
  token.value._light = LIGHT_FALLBACK_THEME.colors[slot];
  token.value._dark = DEFAULT_THEME.colors[slot];
  return token;
};

/**
 * The semantic-token contract. Where a Chakra built-in name exists for a
 * concept we use it verbatim; the handful of workbench-specific tokens
 * (`bg.inset`, `fg.grid`) describe elevation or intent, never a widget.
 */
const colorSlotByToken: Record<string, keyof ThemeColors> = {
  // Surface ladder, deepest to most lifted
  bg: 'base',
  'bg.subtle': 'surface',
  'bg.muted': 'raised',
  'bg.emphasized': 'control',
  // Popover / dialog surface used by Chakra's built-in components
  'bg.panel': 'raised',
  // Recessed work-area floor framed by the chrome
  'bg.inset': 'inset',
  // Foreground
  fg: 'text',
  'fg.muted': 'textMuted',
  'fg.subtle': 'textSubtle',
  // Dot-grid decoration drawn on inset surfaces
  'fg.grid': 'grid',
  // Borders
  border: 'line',
  'border.subtle': 'line',
  'border.muted': 'line',
  'border.emphasized': 'lineStrong',
  // Status intent
  'fg.error': 'danger',
  'border.error': 'danger',
  'fg.success': 'success',
  'fg.warning': 'warning',
};

const semanticColors = {
  ...Object.fromEntries(Object.entries(colorSlotByToken).map(([token, slot]) => [token, colorToken(slot)])),
  // Soft status fills for alerts/banners, derived from each theme's palette.
  'bg.error': mixToken('danger', 14, 'surface'),
  'bg.success': mixToken('success', 14, 'surface'),
  'bg.warning': mixToken('warning', 14, 'surface'),
  /**
   * Chakra's default `colorPalette` is `gray`; re-pointing its virtual-palette
   * keys at the theme ladder makes every un-palettized component (ghost
   * buttons, menu items, badges, …) theme-aware with zero props.
   *
   * Palette tokens must stay NESTED — the `colorPalette` virtual-token map is
   * built from the nested structure and ignores flat dotted keys. Because
   * nesting deep-merges with the defaults, every gray key shadows the default
   * `_light`/`_dark` values via `colorTokenWithModeFallback`.
   */
  gray: {
    contrast: colorTokenWithModeFallback('base'),
    fg: colorTokenWithModeFallback('text'),
    subtle: colorTokenWithModeFallback('fill'),
    muted: colorTokenWithModeFallback('control'),
    emphasized: colorTokenWithModeFallback('lineStrong'),
    solid: colorTokenWithModeFallback('text'),
    focusRing: colorTokenWithModeFallback('accent'),
    border: colorTokenWithModeFallback('lineStrong'),
  },
  /** Invoke identity palette. Use sparingly for the logo and global Invoke action. */
  brand: {
    solid: colorToken('brand'),
    contrast: colorToken('brandContrast'),
    fg: colorToken('brandFg'),
    subtle: colorToken('brandSubtle'),
    muted: mixToken('brandFg', 12, 'brandSubtle'),
    emphasized: mixToken('brandFg', 22, 'brandSubtle'),
    focusRing: colorToken('accent'),
    border: mixToken('brandFg', 50, 'surface'),
  },
  /** Selection / focus palette (blue). Use via `accent.solid` or `colorPalette="accent"`. */
  accent: {
    solid: colorToken('accent'),
    contrast: colorToken('accentContrast'),
    fg: colorToken('accent'),
    subtle: mixToken('accent', 16, 'surface'),
    muted: mixToken('accent', 26, 'surface'),
    emphasized: mixToken('accent', 36, 'surface'),
    focusRing: colorToken('accent'),
    border: colorToken('accent'),
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
      dialog: dialogSlotRecipe,
      menu: menuSlotRecipe,
      tooltip: tooltipSlotRecipe,
    },
  },
});

export const system = createSystem(defaultConfig, config);

/** Theme metadata re-exported so UI can import a single module. */
export { THEMES, THEMES_BY_ID, DEFAULT_THEME, DEFAULT_THEME_ID } from './themes';
export type { ThemeColors, ThemeDefinition } from './themes';
