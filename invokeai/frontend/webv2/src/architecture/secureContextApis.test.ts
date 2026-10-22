import { describe, expect, it } from 'vitest';

/**
 * Guard secure-context APIs. Browsers expose them only over HTTPS or on localhost, so on a plain-HTTP origin (another
 * computer on the LAN) each is missing; one platform helper per API decides what happens then, and every other
 * production module goes through it.
 */
const sources = import.meta.glob('../**/*.{ts,tsx}', {
  eager: true,
  import: 'default',
  query: '?raw',
}) as Record<string, string>;

const isTestFile = (path: string): boolean => /\.(test|spec|stories)\.tsx?$/.test(path);

/** `navigator.x` or `navigator?.x`, whatever the spacing. */
const navigatorMember = (member: string): RegExp => new RegExp(String.raw`\bnavigator\s*\??\.\s*${member}\b`);

const RULES = [
  {
    allowed: ['../platform/browser/randomUuid.ts', '../platform/browser/sha256.ts'],
    name: 'randomUUID and SubtleCrypto',
    pattern: /\brandomUUID\b|\bcrypto\s*\??\.\s*subtle\b|\bsubtle\s*\??\.\s*digest\b/,
  },
  { allowed: ['../platform/browser/webLocks.ts'], name: 'navigator.locks', pattern: navigatorMember('locks') },
  {
    allowed: ['../platform/browser/clipboard.ts'],
    name: 'navigator.clipboard and runtime ClipboardItem',
    // Type-only `ClipboardItem` references stay allowed; constructing or reading the global does not.
    pattern: new RegExp(
      [
        navigatorMember('clipboard').source,
        String.raw`\bnew\s+ClipboardItem\b`,
        String.raw`\b(?:globalThis|window|self)\s*\??\.\s*ClipboardItem\b`,
        String.raw`\bClipboardItem\s*\??\.\s*\w`,
      ].join('|')
    ),
  },
  {
    allowed: ['../platform/pwa/registerServiceWorker.ts'],
    name: 'navigator.serviceWorker',
    pattern: navigatorMember('serviceWorker'),
  },
  // Unused today: a helper that needs it is added here on purpose.
  { allowed: [], name: 'navigator.storage', pattern: navigatorMember('storage') },
] as const satisfies readonly { allowed: readonly string[]; name: string; pattern: RegExp }[];

describe('secure-context web APIs', () => {
  it('routes every production use of a secure-context API through its platform helper', () => {
    const offenders = Object.fromEntries(
      RULES.map(({ allowed, name, pattern }) => [
        name,
        Object.entries(sources)
          .filter(
            ([path, text]) => !isTestFile(path) && !(allowed as readonly string[]).includes(path) && pattern.test(text)
          )
          .map(([path]) => path.replace(/^\.\.\//, '')),
      ])
    );

    expect(offenders).toEqual(Object.fromEntries(RULES.map(({ name }) => [name, []])));
  });

  it('recognizes each API however it is spelled, and leaves type-only ClipboardItem references alone', () => {
    const matches = (name: (typeof RULES)[number]['name'], text: string): boolean =>
      RULES.find((rule) => rule.name === name)!.pattern.test(text);

    expect(matches('navigator.locks', 'navigator?.locks.request(name)')).toBe(true);
    expect(matches('navigator.locks', 'navigator . locks')).toBe(true);
    expect(matches('navigator.clipboard and runtime ClipboardItem', 'navigator.clipboard.writeText(x)')).toBe(true);
    expect(matches('navigator.clipboard and runtime ClipboardItem', 'new ClipboardItem({ [type]: blob })')).toBe(true);
    expect(matches('navigator.clipboard and runtime ClipboardItem', 'globalThis?.ClipboardItem')).toBe(true);
    expect(matches('navigator.clipboard and runtime ClipboardItem', "ClipboardItem.supports('image/png')")).toBe(true);
    expect(matches('navigator.clipboard and runtime ClipboardItem', 'let items: readonly ClipboardItem[];')).toBe(
      false
    );
    expect(matches('navigator.serviceWorker', 'navigator.serviceWorker.register(url)')).toBe(true);
    expect(matches('navigator.storage', 'navigator.storage.persist()')).toBe(true);
    expect(matches('randomUUID and SubtleCrypto', 'crypto.subtle.digest(data)')).toBe(true);
  });
});
