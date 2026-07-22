import { describe, expect, it } from 'vitest';

const enModules = import.meta.glob('../../../public/locales/en.json', { eager: true, import: 'default' });
const enGbModules = import.meta.glob('../../../public/locales/en-GB.json', { eager: true, import: 'default' });
const en = Object.values(enModules)[0] as { commandPalette: Record<string, unknown> };
const enGb = Object.values(enGbModules)[0] as { commandPalette: Record<string, unknown> };

const flattenKeys = (value: Record<string, unknown>, prefix = ''): string[] =>
  Object.entries(value).flatMap(([key, child]) => {
    const path = prefix ? `${prefix}.${key}` : key;

    return child && typeof child === 'object' && !Array.isArray(child)
      ? flattenKeys(child as Record<string, unknown>, path)
      : [path];
  });

describe('command palette translations', () => {
  it('keeps English and British-English command palette keys in parity', () => {
    const enCommandPalette = en.commandPalette;
    const enGbCommandPalette = enGb.commandPalette;

    expect(flattenKeys(enGbCommandPalette).sort()).toEqual(flattenKeys(enCommandPalette).sort());
  });
});
