import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  applyThemeAdjustmentsToRoot,
  normalizeThemeAdjustments,
  previewAdjustedSwatches,
  themeSeedToPickerHex,
} from './adjustments';
import { THEMES_BY_ID } from './themes';

afterEach(() => vi.unstubAllGlobals());

describe('theme adjustments', () => {
  it('keeps only bounded values for known themes and never inherits unexpected keys', () => {
    expect(
      normalizeThemeAdjustments({
        classic: { backgroundLightness: 0, textLightness: 100, brandHex: '#ABCDEF', extra: 'ignored' },
        light: { textLightness: 101, accentHex: 'invalid' },
        unknown: { textLightness: 30 },
        ['__proto__']: { textLightness: 50 },
      })
    ).toEqual({ classic: { backgroundLightness: 0, textLightness: 100, brandHex: '#abcdef' } });
  });

  it('applies only opted-in tokens, restores the native palette on reset or high contrast', () => {
    const values = new Map<string, string>();
    const attributes = new Set<string>();
    vi.stubGlobal('document', {
      documentElement: {
        style: {
          removeProperty: (key: string) => values.delete(key),
          setProperty: (key: string, value: string) => values.set(key, value),
        },
        toggleAttribute: (key: string, on: boolean) => (on ? attributes.add(key) : attributes.delete(key)),
      },
    });

    const classic = THEMES_BY_ID.classic;
    const light = THEMES_BY_ID.light;
    applyThemeAdjustmentsToRoot(classic, undefined, false);
    expect(values.size).toBe(0);

    applyThemeAdjustmentsToRoot(classic, { textLightness: 0, inputLightness: 45, brandHex: '#ffffff' }, false);
    expect(values.get('--chakra-colors-fg')).toBe('oklch(0% 0 0)');
    expect(values.get('--chakra-colors-brand-contrast')).toBe('#000000');
    expect(attributes.has('data-theme-input-bg')).toBe(true);

    applyThemeAdjustmentsToRoot(light, { textLightness: 100, accentHex: '#000000' }, false);
    expect(values.get('--chakra-colors-fg')).toBe('oklch(100% 0 0)');
    expect(values.get('--chakra-colors-accent-contrast')).toBe('#ffffff');
    expect(attributes.has('data-theme-input-bg')).toBe(false);

    applyThemeAdjustmentsToRoot(light, { textLightness: 100 }, true);
    expect(values.size).toBe(0);
    applyThemeAdjustmentsToRoot(light, undefined, false);
    expect(values.size).toBe(0);

    // Account B loads the same theme in the same browser after A signs out.
    applyThemeAdjustmentsToRoot(classic, { textLightness: 0 }, false);
    applyThemeAdjustmentsToRoot(classic, undefined, false);
    expect(values.has('--chakra-colors-fg')).toBe(false);
    applyThemeAdjustmentsToRoot(classic, { textLightness: 100 }, false);
    expect(values.get('--chakra-colors-fg')).toBe('oklch(100% 0 0)');
  });

  it('previews each theme without changing the stock swatches', () => {
    const original = previewAdjustedSwatches(THEMES_BY_ID.classic);
    const customized = previewAdjustedSwatches(THEMES_BY_ID.classic, { brandHex: '#112233' });
    expect(customized[0]).toBe(original[0]);
    expect(customized[2]).toBe('#112233');
  });

  it('starts the color picker from the selected theme instead of a separate color table', () => {
    expect(themeSeedToPickerHex('oklch(0% 0 0)')).toBe('#000000');
    expect(themeSeedToPickerHex('oklch(100% 0 0)')).toBe('#ffffff');
    expect(themeSeedToPickerHex(THEMES_BY_ID.classic.colors.brand.solid)).toMatch(/^#[\da-f]{6}$/);
  });
});
