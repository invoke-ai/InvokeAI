import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { extractThemeTokens, THEME_TOKENS_MODULE_ID, themeTokensVitePlugin } from './theme-tokens.mjs';

const steps = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950];

/** A theme module shaped like the app's `themes.ts`, with a distinct color per slot. */
const makeThemes = () => {
  const colors = {
    neutral: Object.fromEntries(steps.map((step) => [step, `oklch(${step / 10}% 0 0)`])),
    brand: { solid: '#e4fd1d', contrast: 'oklch(1% 0 0)' },
    accent: { solid: 'oklch(77% 0.1 231)', contrast: 'oklch(2% 0 0)' },
    danger: 'rgb(200 0 0)',
    success: 'hsl(120 50% 50%)',
    warning: 'oklch(70% 0.1 60)',
    inset: 'oklch(3% 0 0)',
    fill: 'oklch(4% 0 0)',
    grid: 'oklch(5% 0 0)',
    control: 'oklch(6% 0 0)',
  };
  return {
    THEMES: [
      { id: 'light', colors: { ...colors, accent: { solid: 'oklch(50% 0.2 260)', contrast: '#fff' } } },
      { id: 'classic', colors },
    ],
  };
};

describe('extractThemeTokens', () => {
  it("accepts the app's current themes.ts", async () => {
    // The docs require a Node version that runs TypeScript without a build step.
    const themes = await import('../../invokeai/frontend/webv2/src/platform/ui/theme/themes.ts');
    assert.equal(Object.keys(extractThemeTokens(themes)).length, steps.length + 4 + 7);
  });

  it('reads every Classic color, whatever its position in the theme list', () => {
    const tokens = extractThemeTokens(makeThemes());
    assert.equal(Object.keys(tokens).length, steps.length + 4 + 7);
    assert.equal(tokens['neutral-950'], 'oklch(95% 0 0)');
    assert.equal(tokens.accent, 'oklch(77% 0.1 231)');
    assert.equal(tokens['accent-contrast'], 'oklch(2% 0 0)');
    assert.equal(tokens.brand, '#e4fd1d');
    assert.equal(tokens.control, 'oklch(6% 0 0)');
  });

  it('fails when the Classic theme is gone', () => {
    const themes = makeThemes();
    themes.THEMES = themes.THEMES.filter((theme) => theme.id !== 'classic');
    assert.throws(() => extractThemeTokens(themes), /id "classic"/);
    assert.throws(() => extractThemeTokens({}), /id "classic"/);
  });

  it('names every missing or malformed color', () => {
    const themes = makeThemes();
    const { colors } = themes.THEMES[1];
    delete colors.neutral[500];
    colors.accent = { solid: 'oklch(77% 0.1 231)' };
    colors.grid = 'oklch(5% 0 0); color: red';
    assert.throws(
      () => extractThemeTokens(themes),
      (error) =>
        ['colors.neutral.500', 'colors.accent.contrast', 'colors.grid'].every((path) => error.message.includes(path))
    );
  });
});

describe('themeTokensVitePlugin', () => {
  it('serves the tokens as custom properties from the virtual module', () => {
    const plugin = themeTokensVitePlugin(makeThemes());
    const resolved = plugin.resolveId(THEME_TOKENS_MODULE_ID);
    const css = plugin.load(resolved);
    assert.match(css, /^:root,\n::backdrop \{\n/);
    assert.match(css, /\n {2}--invoke-accent: oklch\(77% 0\.1 231\);\n/);
    assert.match(css, /\n {2}--invoke-neutral-50: oklch\(5% 0 0\);\n/);
    // The app's derived fills build on the theme colors rather than copying them.
    assert.match(css, /\n {2}--invoke-bg-shown: color-mix\(in oklab, var\(--invoke-hover-tint\) 22%, transparent\);\n/);
    assert.equal(plugin.resolveId('virtual:other.css'), undefined);
    assert.equal(plugin.load('other'), undefined);
  });

  it('fails as soon as it is created from a broken theme', () => {
    assert.throws(() => themeTokensVitePlugin({ THEMES: [] }), /id "classic"/);
  });
});
