// Color tokens for the docs, generated from the app's Classic theme so the two never drift apart. The theme module is
// `invokeai/frontend/webv2/src/platform/ui/theme/themes.ts`; `astro.config.mjs` imports it and hands it to this
// integration, which serves its colors as `--invoke-*` custom properties from a virtual CSS module.
// `src/styles/custom.css` maps those onto Starlight's variables and the docs' semantic tokens.

import { fileURLToPath } from 'node:url';

export const THEME_TOKENS_MODULE_ID = 'virtual:invoke-theme-tokens.css';
const RESOLVED_MODULE_ID = `\0${THEME_TOKENS_MODULE_ID}`;

const THEME_ID = 'classic';
const NEUTRAL_STEPS = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950];
const PAIRED_COLORS = ['brand', 'accent'];
const SINGLE_COLORS = ['danger', 'success', 'warning', 'inset', 'fill', 'grid', 'control'];
// A CSS color function or hex color: anything else would need a matching change here and in `custom.css`.
const COLOR_RE = /^(?:(?:oklch|oklab|lab|lch|rgba?|hsla?|color)\([^;{}]+\)|#[\da-f]{3,8})$/i;
const UPDATE_HINT = 'Update docs/plugins/theme-tokens.mjs and docs/src/styles/custom.css to match.';

/**
 * Reads the Classic theme's colors as token name -> CSS color, e.g. `neutral-950`, `accent`, `accent-contrast`.
 * Throws, failing the build, when the theme module no longer has the shape the docs styles are built on.
 * @param {unknown} themesModule the namespace of the app's `themes.ts`
 * @returns {Record<string, string>}
 */
export const extractThemeTokens = (themesModule) => {
  const themes = /** @type {{ THEMES?: unknown }} */ (themesModule)?.THEMES;
  const theme = Array.isArray(themes) ? themes.find((candidate) => candidate?.id === THEME_ID) : undefined;
  if (!theme?.colors) {
    throw new Error(`themes.ts no longer exports a THEMES entry with id "${THEME_ID}" and colors. ${UPDATE_HINT}`);
  }

  /** @type {Record<string, string>} */
  const tokens = {};
  const problems = [];
  /** @param {string} name @param {string} path @param {unknown} value */
  const add = (name, path, value) => {
    if (typeof value === 'string' && COLOR_RE.test(value.trim())) tokens[name] = value.trim();
    else problems.push(`colors.${path} is ${JSON.stringify(value)}`);
  };

  const { colors } = theme;
  for (const step of NEUTRAL_STEPS) add(`neutral-${step}`, `neutral.${step}`, colors.neutral?.[step]);
  for (const name of PAIRED_COLORS) {
    add(name, `${name}.solid`, colors[name]?.solid);
    add(`${name}-contrast`, `${name}.contrast`, colors[name]?.contrast);
  }
  for (const name of SINGLE_COLORS) add(name, name, colors[name]);

  if (problems.length > 0) {
    throw new Error(`The "${THEME_ID}" theme in themes.ts has an unexpected shape: ${problems.join('; ')}. ${UPDATE_HINT}`);
  }
  return tokens;
};

/**
 * The app's derived interaction and status fills (`invokeai/frontend/webv2/src/platform/ui/theme/system.ts`), for a
 * dark theme, written over the theme colors so they follow them.
 */
const DERIVED_TOKENS = {
  // `bg.subtle`: the panel surface the fills below are mixed into.
  surface: 'var(--invoke-neutral-900)',
  // `gray.hoverTint`: the foreground pulled toward the accent, applied at low alpha for hover fills.
  'hover-tint': 'color-mix(in oklab, var(--invoke-accent) 55%, var(--invoke-neutral-50))',
  // `bg.hover` (`gray.hoverTint/10`): the hovered fill of tabs, rows and ghost buttons.
  'bg-hover': 'color-mix(in oklab, var(--invoke-hover-tint) 10%, transparent)',
  // The shown tab's fill: `gray.hoverTint/22`, the app's subtle button fill. The segment tabs' own `/15`
  // (`SegmentTabs.tsx`) all but disappears on the docs' tinted asides; `/22` keeps a 1.7:1 step on every surface.
  'bg-shown': 'color-mix(in oklab, var(--invoke-hover-tint) 22%, transparent)',
  // `border.emphasized`: segment tab dividers.
  'border-emphasized': 'var(--invoke-neutral-500)',
};

/**
 * Declares the tokens on `::backdrop` too, which does not inherit from `:root` and is where Starlight resolves its
 * dialog backdrop color.
 * @param {Record<string, string>} tokens
 */
export const themeTokensCss = (tokens) =>
  `:root,\n::backdrop {\n${Object.entries({ ...tokens, ...DERIVED_TOKENS })
    .map(([name, value]) => `  --invoke-${name}: ${value};\n`)
    .join('')}}\n`;

/**
 * Vite plugin serving the tokens as `virtual:invoke-theme-tokens.css`.
 * @param {unknown} themesModule the namespace of the app's `themes.ts`
 * @param {string} [themesPath] the theme file, for the dev server to watch: it is outside the docs root
 */
export const themeTokensVitePlugin = (themesModule, themesPath) => {
  // Extract eagerly so a theme change that breaks the contract fails as soon as the config loads.
  const css = themeTokensCss(extractThemeTokens(themesModule));
  return {
    name: 'invoke-theme-tokens',
    resolveId: (id) => (id === THEME_TOKENS_MODULE_ID ? RESOLVED_MODULE_ID : undefined),
    load: (id) => (id === RESOLVED_MODULE_ID ? css : undefined),
    configureServer: (server) => {
      if (themesPath) server.watcher.add(themesPath);
    },
  };
};

/**
 * Astro integration adding the tokens' Vite plugin. `astro dev` restarts when the theme file changes (Astro restarts
 * for files added with `addWatchFile`, once the dev server's watcher sees them), which reloads the config and with it
 * a fresh copy of the theme module, so theme edits show without a manual restart. An edit to this file still needs
 * one.
 * @param {{ themesModule: unknown, themesPath: URL }} options the namespace of the app's `themes.ts`, and its location
 * @returns {import('astro').AstroIntegration}
 */
export const themeTokens = ({ themesModule, themesPath }) => ({
  name: 'invoke-theme-tokens',
  hooks: {
    'astro:config:setup': ({ addWatchFile, updateConfig }) => {
      addWatchFile(themesPath);
      updateConfig({ vite: { plugins: [themeTokensVitePlugin(themesModule, fileURLToPath(themesPath))] } });
    },
  },
});
