import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

const FILES = ['OpenAIProviderOptions.tsx', 'GeminiProviderOptions.tsx', 'SeedreamProviderOptions.tsx'] as const;

const LABEL_RE = /<FormLabel[^>]*>([\s\S]*?)<\/FormLabel>/g;
const OPTION_RE = /<option\b[^>]*>([\s\S]*?)<\/option>/g;

const isTranslatedExpression = (inner: string): boolean => {
  const trimmed = inner.trim();
  if (trimmed === '') {
    return true;
  }
  return /^\{\s*t\s*\(/.test(trimmed) && /\)\s*\}$/.test(trimmed);
};

const collectOffenders = (source: string, re: RegExp): string[] => {
  const offenders: string[] = [];
  for (const match of source.matchAll(re)) {
    const inner = match[1] ?? '';
    if (!isTranslatedExpression(inner)) {
      offenders.push(match[0]);
    }
  }
  return offenders;
};

describe('External provider option components are fully localised', () => {
  for (const file of FILES) {
    const source = readFileSync(new URL(`./${file}`, import.meta.url), 'utf8');

    it(`${file}: every <FormLabel> child is a t(...) expression`, () => {
      const offenders = collectOffenders(source, LABEL_RE);
      expect(
        offenders,
        `Found <FormLabel> nodes whose visible text is not wrapped in t(...). ` +
          `External provider option labels must be localised so non-English users see translated text. ` +
          `Offenders:\n${offenders.join('\n')}`
      ).toEqual([]);
    });

    it(`${file}: every <option> child is a t(...) expression`, () => {
      const offenders = collectOffenders(source, OPTION_RE);
      expect(
        offenders,
        `Found <option> nodes whose visible text is a raw literal. ` +
          `Select option labels under External/ must be wrapped in t(...) so they translate ` +
          `alongside their <FormLabel>. Offenders:\n${offenders.join('\n')}`
      ).toEqual([]);
    });
  }
});
