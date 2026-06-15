import { describe, expect, it } from 'vitest';

import legacyBaseline from './__fixtures__/legacyTokenBaseline.json';
import { CONSUMER_TOKENS, resolveToken, THEMES, type TokenSystem } from './__tokenResolve';
import { system } from './system';

const sys = system as unknown as TokenSystem;
const STEPS = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950] as const;

const lightnessOf = (oklch: string): number => {
  const match = /oklch\(\s*([\d.]+)%/.exec(oklch);
  if (!match) {
    throw new Error(`not a plain oklch value: ${oklch}`);
  }
  return Number(match[1]);
};

describe('color token contract — legacy-value gate', () => {
  // Primary gate: every consumer token must resolve to the EXACT value it had
  // before the ramp refactor. The baseline was captured from the previous system
  // on the working tree before any change (see __fixtures__/legacyTokenBaseline.json).
  for (const theme of THEMES) {
    for (const token of CONSUMER_TOKENS) {
      it(`${theme}: ${token} is unchanged`, () => {
        const expected = (legacyBaseline as Record<string, Record<string, string>>)[theme][token];
        expect(resolveToken(sys, theme, token)).toBe(expected);
      });
    }
  }
});

describe('ramp + mapping structure', () => {
  it('emits every neutral ramp step for every theme', () => {
    for (const theme of THEMES) {
      for (const step of STEPS) {
        expect(resolveToken(sys, theme, `neutral.${step}`)).toMatch(/^oklch\(/);
      }
    }
  });

  it('maps the light surface ladder to the corrected (non-mirrored) steps', () => {
    // The point-1 fix: light bg sits at neutral.200, not the lightest step.
    expect(resolveToken(sys, 'light', 'bg')).toBe(resolveToken(sys, 'light', 'neutral.200'));
    expect(resolveToken(sys, 'light', 'bg.subtle')).toBe(resolveToken(sys, 'light', 'neutral.50'));
    expect(resolveToken(sys, 'light', 'bg.muted')).toBe(resolveToken(sys, 'light', 'neutral.100'));
    expect(resolveToken(sys, 'light', 'border')).toBe(resolveToken(sys, 'light', 'neutral.300'));
    expect(resolveToken(sys, 'light', 'border.emphasized')).toBe(resolveToken(sys, 'light', 'neutral.400'));
  });

  it('maps the dark surface ladder onto the ramp', () => {
    expect(resolveToken(sys, 'classic', 'bg')).toBe(resolveToken(sys, 'classic', 'neutral.950'));
    expect(resolveToken(sys, 'classic', 'bg.subtle')).toBe(resolveToken(sys, 'classic', 'neutral.900'));
    expect(resolveToken(sys, 'classic', 'fg')).toBe(resolveToken(sys, 'classic', 'neutral.50'));
    expect(resolveToken(sys, 'classic', 'border')).toBe(resolveToken(sys, 'classic', 'neutral.600'));
  });

  it('emits surface/text/border tokens via per-theme conditions only — never the leaky mode selectors', () => {
    // Chakra's `_light` selector (`:root &, .light &`) matches under EVERY theme via its
    // `:root &` arm. A surface token placed there leaks its light value into the dark
    // themes (the classic-renders-light regression). These tokens must live ONLY in the
    // base + `[data-theme=…]` selectors.
    const layer = sys.getTokenCss()['@layer tokens'];
    const LEAKY = [':root &, .light &', '.dark &, .dark .chakra-theme:not(.light) &'];
    const surfaceTokens = [
      'bg',
      'bg.subtle',
      'bg.muted',
      'bg.panel',
      'fg',
      'fg.muted',
      'fg.subtle',
      'border',
      'border.subtle',
      'border.muted',
      'border.emphasized',
    ];
    for (const name of surfaceTokens) {
      const token = sys.tokens.getByName(`colors.${name}`);
      const cssVar = token?.extensions.cssVar.var ?? '';
      for (const selector of LEAKY) {
        expect(layer[selector]?.[cssVar], `${name} must not be emitted under "${selector}"`).toBeUndefined();
      }
    }
  });

  it('keeps each ramp strictly decreasing in lightness 50 → 950', () => {
    for (const theme of THEMES) {
      const ls = STEPS.map((step) => lightnessOf(resolveToken(sys, theme, `neutral.${step}`)));
      for (let i = 1; i < ls.length; i++) {
        expect(ls[i], `${theme} neutral.${STEPS[i]} should be darker than neutral.${STEPS[i - 1]}`).toBeLessThan(
          ls[i - 1]
        );
      }
    }
  });
});

describe('native Chakra integration', () => {
  it('resolves every gray virtual-palette key under both dark and light', () => {
    const keys = ['contrast', 'fg', 'subtle', 'muted', 'emphasized', 'solid', 'focusRing', 'border'];
    for (const theme of ['classic', 'light']) {
      for (const key of keys) {
        expect(resolveToken(sys, theme, `gray.${key}`)).toBeTruthy();
      }
    }
  });

  it('leaves stock Chakra hue palettes untouched in Phase 1', () => {
    // Phase 2 will theme these; for now they must remain Chakra defaults so model
    // badges / destructive actions keep their stock colors.
    expect(sys.tokens.getByName('colors.red.500')).toBeTruthy();
    expect(sys.tokens.getByName('colors.blue.500')).toBeTruthy();
    expect(sys.tokens.getByName('colors.purple.500')).toBeTruthy();
  });
});

describe('brand palette derives from its two seeds (like accent)', () => {
  it('fg/border equal solid; subtle/muted/emphasized mix solid into the surface', () => {
    for (const theme of THEMES) {
      const solid = resolveToken(sys, theme, 'brand.solid');
      const surface = resolveToken(sys, theme, 'bg.subtle');
      expect(resolveToken(sys, theme, 'brand.fg')).toBe(solid);
      expect(resolveToken(sys, theme, 'brand.border')).toBe(solid);
      for (const [key, pct] of [
        ['subtle', 16],
        ['muted', 26],
        ['emphasized', 36],
      ] as const) {
        expect(resolveToken(sys, theme, `brand.${key}`)).toBe(`color-mix(in oklab, ${solid} ${pct}%, ${surface})`);
      }
    }
  });
});
