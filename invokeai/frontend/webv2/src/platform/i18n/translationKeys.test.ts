import { describe, expect, it } from 'vitest';

/**
 * Guards against translation keys that render as raw dotted paths. Every static
 * `t('…')` literal in production source must resolve in the English catalog —
 * webv2 ships English-only today (the other locales are empty stubs behind
 * `fallbackLng: 'en'`), so en.json is the whole contract.
 */

const enModules = import.meta.glob('../../../public/locales/en.json', { eager: true, import: 'default' });
const sourceModules = import.meta.glob('../../**/*.{ts,tsx}', {
  eager: true,
  import: 'default',
  query: '?raw',
});

const en = Object.values(enModules)[0] as Record<string, unknown>;

/** Top-level namespaces in en.json. Anchoring on these keeps the scan from matching arbitrary dotted strings. */
const NAMESPACES = Object.keys(en);

/**
 * Key prefixes built by interpolation and always called with a `defaultValue`,
 * so a miss degrades to readable English rather than a dotted path.
 * `commandPalette.commands.*`: see `entries.ts` — titles fall back to the
 * command definition's own title.
 */
export const DYNAMIC_KEY_PREFIXES = ['commandPalette.commands.'];

const isTestModule = (path: string): boolean => /\.test\.|\.type-test\.|\.testing\./.test(path);

/** Flattens the nested catalog to dotted paths, keeping plural suffixes intact. */
const flattenCatalog = (value: unknown, prefix = '', keys = new Set<string>()): Set<string> => {
  if (typeof value !== 'object' || value === null) {
    return keys;
  }

  for (const [segment, child] of Object.entries(value as Record<string, unknown>)) {
    const path = prefix ? `${prefix}.${segment}` : segment;

    if (typeof child === 'object' && child !== null) {
      flattenCatalog(child, path, keys);
      continue;
    }

    keys.add(path);
    // i18next resolves `t('x', { count })` against `x_one` / `x_other`; index the stem too.
    const pluralMatch = /^(.*)_(?:zero|one|two|few|many|other)$/.exec(path);

    if (pluralMatch?.[1]) {
      keys.add(pluralMatch[1]);
    }
  }

  return keys;
};

const catalogKeys = flattenCatalog(en);

const KEY_PATTERN = '[A-Za-z0-9_]+(?:\\.[A-Za-z0-9_]+)+';
// `t('a.b')` — the translator call itself.
const T_CALL = new RegExp(String.raw`\bt\(\s*['"](${KEY_PATTERN})['"]`, 'g');
// `i18nKey="a.b"` — the <Trans> form.
const I18N_KEY_PROP = new RegExp(String.raw`\bi18nKey\s*=\s*['"](${KEY_PATTERN})['"]`, 'g');
// `labelKey: 'a.b'` / `titleKey: 'a.b'` — keys held in const maps and passed to `t()` indirectly.
const KEY_PROPERTY = new RegExp(String.raw`\b\w*[Kk]ey\s*:\s*['"](${KEY_PATTERN})['"]`, 'g');

const collectKeys = (source: string): string[] => {
  const found: string[] = [];

  for (const pattern of [T_CALL, I18N_KEY_PROP, KEY_PROPERTY]) {
    pattern.lastIndex = 0;

    for (const match of source.matchAll(pattern)) {
      const key = match[1];

      if (key && NAMESPACES.includes(key.split('.')[0] ?? '')) {
        found.push(key);
      }
    }
  }

  return found;
};

const usages = Object.entries(sourceModules)
  .filter(([path]) => !isTestModule(path))
  .flatMap(([path, source]) => collectKeys(source as string).map((key) => ({ key, path })));

describe('translation keys', () => {
  it('scans the source tree', () => {
    // A broken glob would make every other assertion vacuously pass.
    expect(usages.length).toBeGreaterThan(500);
  });

  it('resolves every static key against the English catalog', () => {
    const missing = usages
      .filter(({ key }) => !catalogKeys.has(key))
      .filter(({ key }) => !DYNAMIC_KEY_PREFIXES.some((prefix) => key.startsWith(prefix)))
      .map(({ key, path }) => `${key} (${path.replace('../../', 'src/')})`);

    expect([...new Set(missing)].sort()).toEqual([]);
  });
});
