import { describe, expect, it } from 'vitest';

const html = Object.values(
  import.meta.glob('/index.html', { eager: true, query: '?raw', import: 'default' })
)[0] as string;

const script = html.match(/<script>([\s\S]*?)<\/script>/i)?.[1]?.replace('__READY_CONTENT_LANGUAGES__', '["en"]');

describe('before-paint language hint', () => {
  it.each(['ar', 'he', 'fr', 'ua', 'obsolete'])('does not label fallback English as %s', (language) => {
    const attributes = new Map<string, string>();
    const root = {
      setAttribute: (name: string, value: string) => attributes.set(name, value),
      style: {},
      classList: { toggle: () => {} },
    };
    new Function('document', 'localStorage', script!)(
      { documentElement: root },
      { getItem: (key: string) => (key.endsWith(':language') ? language : null) }
    );
    expect(attributes.get('lang')).toBe('en');
    expect(attributes.get('dir')).toBe('ltr');
  });
});

it('uses RTL only when that content is explicitly ready in a fixture', () => {
  const attributes = new Map<string, string>();
  const root = {
    setAttribute: (key: string, value: string) => attributes.set(key, value),
    style: {},
    classList: { toggle: () => {} },
  };
  const fixture = script!.replace('["en"]', '["en","ar"]');
  new Function('document', 'localStorage', fixture)(
    { documentElement: root },
    { getItem: (key: string) => (key.endsWith(':language') ? 'ar' : null) }
  );
  expect(attributes.get('lang')).toBe('ar');
  expect(attributes.get('dir')).toBe('rtl');
});
