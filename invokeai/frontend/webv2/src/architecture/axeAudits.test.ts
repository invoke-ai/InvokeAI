import { describe, expect, it } from 'vitest';

/**
 * Require settled UI before axe to avoid transient contrast failures. This scans src only; serialized release
 * journeys must implement the same wait separately.
 */
const sources = import.meta.glob('../**/*.{ts,tsx}', {
  eager: true,
  import: 'default',
  query: '?raw',
}) as Record<string, string>;

const ALLOWED = new Set([
  '../platform/browser/auditAccessibility.testing.ts',
  // Audits un-settled UI on purpose, to hold on to a reproduction of the bug the helper fixes.
  '../platform/browser/auditAccessibility.browser.test.ts',
]);
const AXE_RUN = /\baxe\s*\.\s*run\s*\(/;

describe('accessibility audits', () => {
  it('routes every audit under src/ through the helper that settles animations first', () => {
    const offenders = Object.entries(sources)
      .filter(([path]) => !ALLOWED.has(path))
      .filter(([, text]) => AXE_RUN.test(text))
      .map(([path]) => path.replace(/^\.\.\//, ''));

    expect(offenders).toEqual([]);
  });
});
