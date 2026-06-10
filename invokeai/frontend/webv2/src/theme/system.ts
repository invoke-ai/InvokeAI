import { createSystem, defaultConfig, defineConfig } from '@chakra-ui/react';

import { DEFAULT_THEME_ID, THEMES, THEMES_BY_ID, type ThemeColors } from './themes';

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
 * We deliberately override Chakra's own neutral tokens (`bg`, `fg`, `border`, …)
 * as well as the workbench-specific ones, so built-in components (Dialog, Input,
 * Switch, Button) follow the active theme without per-component overrides.
 */

const NON_DEFAULT_THEMES = THEMES.filter((theme) => theme.id !== DEFAULT_THEME_ID);

/** `light` -> `themeLight`, `ultradark` -> `themeUltradark`. */
const conditionName = (id: string): string => `theme${id.charAt(0).toUpperCase()}${id.slice(1)}`;

/** Build a semantic-token value object: default theme as `base`, the rest as conditions. */
const colorToken = (slot: keyof ThemeColors): { value: Record<string, string> } => {
  const value: Record<string, string> = { base: THEMES_BY_ID[DEFAULT_THEME_ID].colors[slot] };

  for (const theme of NON_DEFAULT_THEMES) {
    value[`_${conditionName(theme.id)}`] = theme.colors[slot];
  }

  return { value };
};

/**
 * The semantic-token contract. Workbench tokens (`bg.shell`, `accent.invoke`, …)
 * are the names components consume; the second group re-points Chakra's built-in
 * neutral tokens at the same palette so library components stay on-theme.
 */
const colorSlotByToken: Record<string, keyof ThemeColors> = {
  // Workbench surfaces
  'bg.shell': 'shell',
  'bg.surface': 'surface',
  'bg.surfaceRaised': 'surfaceRaised',
  'bg.center': 'center',
  'bg.canvas': 'canvas',
  'bg.panel': 'panel',
  // Workbench borders
  'border.subtle': 'line',
  'border.emphasis': 'lineStrong',
  'border.panel': 'panelStroke',
  // Workbench foreground
  'fg.default': 'fg',
  'fg.muted': 'fgMuted',
  'fg.subtle': 'fgSubtle',
  // Accents
  'accent.invoke': 'accent',
  'accent.invokeFg': 'accentFg',
  'accent.widget': 'accentMuted',
  'accent.widgetFg': 'accentMutedFg',
  'accent.active': 'active',
  'accent.activeFg': 'activeFg',
  'canvas.dot': 'dot',
  // Chakra built-in neutrals, re-pointed at the theme palette
  bg: 'surface',
  'bg.subtle': 'surfaceRaised',
  'bg.muted': 'panel',
  'bg.emphasized': 'lineStrong',
  fg: 'fg',
  border: 'line',
  'border.muted': 'line',
  'border.emphasized': 'lineStrong',
  'fg.error': 'danger',
  // Chakra's default recipes style controls through colorPalette.* even when
  // callers don't pass a colorPalette prop. Point that virtual palette at the
  // active workbench theme so Button/Input/Switch/etc. don't fall back to gray.
  'colorPalette.fg': 'fg',
  'colorPalette.subtle': 'surfaceRaised',
  'colorPalette.muted': 'panel',
  'colorPalette.emphasized': 'lineStrong',
  'colorPalette.border': 'lineStrong',
  'colorPalette.solid': 'accent',
  'colorPalette.contrast': 'accentFg',
  'colorPalette.focusRing': 'active',
};

const semanticColors = Object.fromEntries(
  Object.entries(colorSlotByToken).map(([token, slot]) => [token, colorToken(slot)])
);

const themeConditions = Object.fromEntries(
  NON_DEFAULT_THEMES.map((theme) => [conditionName(theme.id), `[data-theme=${theme.id}]`])
);

const config = defineConfig({
  conditions: themeConditions,
  globalCss: {
    'html, body, #root': {
      height: '100%',
    },
    body: {
      bg: 'bg.shell',
      color: 'fg.default',
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
  },
});

export const system = createSystem(defaultConfig, config);

/** Theme metadata re-exported so UI can import a single module. */
export { THEMES, THEMES_BY_ID, DEFAULT_THEME_ID } from './themes';
export type { ThemeColors, ThemeDefinition } from './themes';
