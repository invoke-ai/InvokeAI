import { describe, expect, it } from 'vitest';

import { resolveWorkbenchThemeId, THEMES, THEMES_BY_ID } from './themes';

const indexHtmlModules = import.meta.glob('../../../../index.html', { eager: true, import: 'default', query: '?raw' });

/** The `themeColorSchemes` literal the pre-paint script in index.html keeps by hand. */
const readPrePaintThemeMap = (): Record<string, string> => {
  const html = Object.values(indexHtmlModules)[0];
  if (typeof html !== 'string') {
    throw new TypeError('index.html was not loaded');
  }
  const literal = /var themeColorSchemes = \{([^}]*)\}/.exec(html)?.[1];
  if (!literal) {
    throw new Error('index.html no longer declares themeColorSchemes');
  }
  return Object.fromEntries(
    [...literal.matchAll(/(\w+):\s*'(\w+)'/g)].map((match) => [match[1] as string, match[2] as string])
  );
};

describe('pre-paint theme map', () => {
  const map = readPrePaintThemeMap();

  it('lists every registered theme with its color scheme', () => {
    for (const theme of THEMES) {
      expect(map[theme.id], theme.id).toBe(theme.colorScheme);
    }
  });

  it('maps every listed id, legacy ids included, to a registered theme with the same color scheme', () => {
    for (const [id, colorScheme] of Object.entries(map)) {
      const themeId = resolveWorkbenchThemeId(id);
      expect(themeId, id).not.toBeNull();
      expect(colorScheme, id).toBe(THEMES_BY_ID[themeId!].colorScheme);
    }
  });
});
