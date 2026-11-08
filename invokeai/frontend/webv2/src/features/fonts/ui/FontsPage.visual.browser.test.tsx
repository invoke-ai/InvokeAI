import { ChakraProvider } from '@chakra-ui/react';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { settleAnimations } from '@platform/browser/settleAnimations.testing';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

const dependencies = vi.hoisted(() => ({
  deleteFont: vi.fn(),
  ensure: vi.fn(),
  fontsQueryOptions: vi.fn(),
  rescanFonts: vi.fn(),
  uploadFont: vi.fn(),
}));

vi.mock('@features/fonts', () => ({
  deleteFont: dependencies.deleteFont,
  fontKeys: { all: ['fonts'] },
  fontsQueryOptions: dependencies.fontsQueryOptions,
  getFontRuntimeKey: (reference: { id: string }) => reference.id,
  rescanFonts: dependencies.rescanFonts,
  uploadFont: dependencies.uploadFont,
}));
vi.mock('@features/fonts/react', () => ({
  useFontRuntime: () => ({
    ensure: dependencies.ensure,
    retain: () => () => undefined,
    resolveFamily: () => 'Example Sans',
    subscribe: () => vi.fn(),
    getSnapshot: () => ({ generation: 0, states: new Map() }),
  }),
  useFontRuntimeSnapshot: () => ({ generation: 0, states: new Map() }),
}));
vi.mock('@features/identity', () => ({ useCapabilities: () => ({ canManageSharedFonts: false }) }));

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  lng: 'en',
  fallbackNS: 'translation',
  resources: {
    en: {
      translation: await fetch('/locales/en.json').then((response) => response.json()),
      fonts: await fetch('/locales/en.fonts.json').then((response) => response.json()),
    },
  },
});

const font = {
  axes: [{ default: 400, hidden: false, label: 'Weight', maximum: 700, minimum: 100, tag: 'wght' }],
  byteSize: 1024 * 32,
  contentHash: 'a'.repeat(64),
  family: 'Example Sans',
  filename: 'ExampleSans-Variable.ttf',
  id: 'font-a',
  instances: [{ coordinates: { wght: 400 }, name: 'Regular' }],
  label: 'Example Sans Variable',
  scope: 'private',
  source: 'uploaded',
  style: 'normal',
  url: '/api/v1/fonts/font-a/file',
  weight: 400,
};

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | undefined;
let root: Root | undefined;
let queryClient: QueryClient | undefined;

const renderPage = async (): Promise<void> => {
  await page.viewport(1280, 900);
  host = document.createElement('div');
  host.style.height = '720px';
  host.style.width = '960px';
  document.body.append(host);
  root = createRoot(host);
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const { FontsPage } = await import('./FontsPage');

  await act(() => {
    root!.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <QueryClientProvider client={queryClient!}>
            <FontsPage />
          </QueryClientProvider>
        </I18nextProvider>
      </ChakraProvider>
    );
  });
};

afterEach(async () => {
  await act(() => root?.unmount());
  queryClient?.clear();
  host?.remove();
  root = undefined;
  queryClient = undefined;
  host = undefined;
});

it('renders the font library with the real English locale', async () => {
  dependencies.ensure.mockReset().mockResolvedValue('Example Sans');
  dependencies.fontsQueryOptions.mockReset().mockImplementation((params: unknown) => ({
    queryFn: () => Promise.resolve({ items: [font], limit: 100, offset: 0, total: 1 }),
    queryKey: ['fonts', params],
  }));

  await renderPage();

  await expect.element(page.getByRole('heading', { name: 'Fonts' })).toBeVisible();
  await expect.element(page.getByText('Example Sans Variable')).toBeVisible();
  await page.getByRole('button', { name: /Example Sans Variable/ }).click();
  await expect.element(page.getByRole('heading', { name: 'Example Sans Variable' })).toBeVisible();
  await expect.element(page.getByText('Variation axes', { exact: true })).toBeVisible();
  await settleAnimations();
  await page.screenshot({ path: '../../../../artifacts/fonts/fonts-page-real-en.png' });
  await page.getByRole('button', { name: 'Filter fonts' }).click();
  await expect
    .element(page.getByRole('menuitemradio', { name: 'All', exact: true }))
    .toHaveAttribute('aria-checked', 'true');
  const menu = document.querySelector('[role="menu"]')!;

  expect(await auditAccessibility(host!)).toEqual([]);
  expect(await auditAccessibility(menu)).toEqual([]);
  await page.screenshot({ path: '../../../../artifacts/fonts/fonts-filter-real-en.png' });
  await page.getByRole('menuitemradio', { name: 'My fonts', exact: true }).click();
  await vi.waitFor(() =>
    expect(dependencies.fontsQueryOptions).toHaveBeenCalledWith({ limit: 100, scope: 'private', search: '' })
  );
  await expect
    .element(page.getByRole('menuitemradio', { name: 'My fonts', exact: true }))
    .toHaveAttribute('aria-checked', 'true');
  await userEvent.keyboard('{Escape}');
  await expect.element(page.getByRole('button', { name: 'Filter fonts' })).toHaveFocus();
  await page.getByRole('tab', { name: 'Add Fonts' }).click();
  await expect.element(page.getByRole('button', { name: 'Upload fonts' })).toBeVisible();
  await settleAnimations();
  await page.screenshot({ path: '../../../../artifacts/fonts/fonts-add-real-en.png' });
});

it('uses the manager empty state and opens Add Fonts from its action', async () => {
  dependencies.fontsQueryOptions.mockReset().mockImplementation((params: unknown) => ({
    queryFn: () => Promise.resolve({ items: [], limit: 100, offset: 0, total: 0 }),
    queryKey: ['fonts', params],
  }));
  await renderPage();
  await expect.element(page.getByText('No fonts available', { exact: true })).toBeVisible();
  expect(await auditAccessibility(host!)).toEqual([]);
  await page.screenshot({ path: '../../../../artifacts/fonts/fonts-empty-real-en.png' });
  await page.getByRole('button', { name: 'Add Fonts', exact: true }).click();
  await expect.element(page.getByRole('button', { name: 'Upload fonts', exact: true })).toBeVisible();
});
