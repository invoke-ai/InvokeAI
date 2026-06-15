/**
 * Test-only helper: resolve every semantic color token to its FINAL literal value
 * (following `var(--chakra-colors-…)` references) for each workbench theme, straight
 * from `system.getTokenCss()`. This mirrors what the browser computes for a given
 * `<html data-theme=… class="dark|light">`, so it is the ground truth used by the
 * legacy-value gate in `system.test.ts`.
 *
 * Not imported by any app entry — excluded from the production bundle.
 */

type Block = Record<string, string>;
type Layer = Record<string, Block>;

export interface TokenSystem {
  getTokenCss: () => Record<string, Layer>;
  tokens: { getByName: (name: string) => { extensions: { cssVar: { var: string } } } | undefined };
}

const BASE_SEL = '&:where(html, .chakra-theme)';
const DARK_SEL = '.dark &, .dark .chakra-theme:not(.light) &';
const LIGHT_SEL = ':root &, .light &';
const dataSel = (id: string): string => `&:root[data-theme=${id}]`;

/** Selectors that apply to each theme, HIGHEST CSS priority first. */
export const THEME_SELECTORS: Record<string, string[]> = {
  classic: [DARK_SEL, BASE_SEL], // default theme: no [data-theme=classic] rule
  light: [dataSel('light'), LIGHT_SEL, BASE_SEL],
  forest: [dataSel('forest'), DARK_SEL, BASE_SEL],
  mono: [dataSel('mono'), DARK_SEL, BASE_SEL],
  ultradark: [dataSel('ultradark'), DARK_SEL, BASE_SEL],
};

export const THEMES = Object.keys(THEME_SELECTORS);

const VAR_REF = /var\((--chakra-colors-[A-Za-z0-9\\.-]+)\)/g;

const layerOf = (sys: TokenSystem): Layer => sys.getTokenCss()['@layer tokens'];

const resolveVar = (layer: Layer, theme: string, varName: string, seen: Set<string>): string | undefined => {
  for (const sel of THEME_SELECTORS[theme]) {
    const block = layer[sel];
    if (block && varName in block) {
      return resolveValue(layer, theme, block[varName], seen);
    }
  }
  return undefined;
};

const resolveValue = (layer: Layer, theme: string, value: string, seen: Set<string>): string =>
  value.replace(VAR_REF, (whole, varName: string) => {
    if (seen.has(varName)) {
      return whole;
    }
    const resolved = resolveVar(layer, theme, varName, new Set(seen).add(varName));
    return resolved ?? whole;
  });

/** Final literal value of `colors.<tokenName>` in `theme`. */
export const resolveToken = (sys: TokenSystem, theme: string, tokenName: string): string => {
  const token = sys.tokens.getByName(`colors.${tokenName}`);
  if (!token) {
    throw new Error(`token not found: colors.${tokenName}`);
  }
  const value = resolveVar(layerOf(sys), theme, token.extensions.cssVar.var, new Set());
  if (value === undefined) {
    throw new Error(`could not resolve colors.${tokenName} for theme ${theme}`);
  }
  return value;
};

/** Resolve every token across every theme → { [theme]: { [token]: value } }. */
export const resolveAll = (sys: TokenSystem, tokens: readonly string[]): Record<string, Record<string, string>> => {
  const out: Record<string, Record<string, string>> = {};
  for (const theme of THEMES) {
    out[theme] = {};
    for (const token of tokens) {
      out[theme][token] = resolveToken(sys, theme, token);
    }
  }
  return out;
};

/** The frozen public color-token contract — every name a component/recipe may consume. */
export const CONSUMER_TOKENS = [
  'bg',
  'bg.subtle',
  'bg.muted',
  'bg.panel',
  'bg.emphasized',
  'bg.inset',
  'bg.error',
  'bg.success',
  'bg.warning',
  'fg',
  'fg.muted',
  'fg.subtle',
  'fg.grid',
  'fg.error',
  'fg.success',
  'fg.warning',
  'border',
  'border.subtle',
  'border.muted',
  'border.emphasized',
  'border.error',
  'gray.contrast',
  'gray.fg',
  'gray.subtle',
  'gray.muted',
  'gray.emphasized',
  'gray.solid',
  'gray.focusRing',
  'gray.border',
  'brand.solid',
  'brand.contrast',
  'brand.fg',
  'brand.subtle',
  'brand.muted',
  'brand.emphasized',
  'brand.focusRing',
  'brand.border',
  'accent.solid',
  'accent.contrast',
  'accent.fg',
  'accent.subtle',
  'accent.muted',
  'accent.emphasized',
  'accent.focusRing',
  'accent.border',
] as const;
