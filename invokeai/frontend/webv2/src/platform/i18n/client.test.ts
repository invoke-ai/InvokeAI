import type * as I18nextModule from 'i18next';

import { afterEach, expect, it, vi } from 'vitest';

vi.mock('i18next', async (importOriginal) => {
  const actual = await importOriginal<typeof I18nextModule>();
  return { ...actual, default: actual.createInstance() };
});
vi.mock('@platform/transport/deploymentBase', () => ({
  getDeploymentBaseUrl: () => 'https://invoke.example/studio',
}));

afterEach(() => {
  vi.unstubAllGlobals();
});

it('loads cold namespaces only on demand under the deployment prefix with English and root fallbacks', async () => {
  const resources: Record<string, unknown> = {
    'en.json': { common: { copy: 'Copy' } },
    'fr.json': { common: { copy: 'Copier' } },
    'en.metadata.json': { widgets: { preview: { metadataLabels: { control_layers: 'Control layers' } } } },
    'fr.metadata.json': {},
    'en.fonts.json': { fonts: { title: 'Fonts' } },
    'fr.fonts.json': {},
  };
  const request = vi.fn((input: RequestInfo | URL) => {
    const url = new URL(String(input));
    expect(url.origin).toBe('https://invoke.example');
    expect(url.pathname).toMatch(/^\/studio\/locales\//u);
    expect(url.searchParams.get('v')).toMatch(/^[a-f\d]{12}$/u);
    const value = resources[url.pathname.split('/').at(-1)!];
    return Promise.resolve(new Response(JSON.stringify(value), { status: value ? 200 : 404 }));
  });
  vi.stubGlobal('fetch', request);

  const { default: i18n } = await import('./client');
  await vi.waitFor(() => expect(i18n.isInitialized).toBe(true));
  await i18n.changeLanguage('fr');
  expect(request.mock.calls.map(([url]) => new URL(String(url)).pathname)).toEqual([
    '/studio/locales/en.json',
    '/studio/locales/fr.json',
  ]);

  await i18n.loadNamespaces('metadata');
  const t = i18n.getFixedT('fr', 'metadata');
  expect(t('widgets.preview.metadataLabels.control_layers')).toBe('Control layers');
  expect(t('common.copy')).toBe('Copier');
  expect(request.mock.calls.slice(2).map(([url]) => new URL(String(url)).pathname)).toEqual([
    '/studio/locales/fr.metadata.json',
    '/studio/locales/en.metadata.json',
  ]);

  await i18n.loadNamespaces('fonts');
  expect(i18n.getFixedT('fr', 'fonts')('fonts.title')).toBe('Fonts');
  expect(request.mock.calls.slice(-2).map(([url]) => new URL(String(url)).pathname)).toEqual([
    '/studio/locales/fr.fonts.json',
    '/studio/locales/en.fonts.json',
  ]);
});
