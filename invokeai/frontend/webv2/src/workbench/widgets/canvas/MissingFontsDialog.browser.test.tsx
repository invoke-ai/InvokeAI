import type * as FontsFeature from '@features/fonts';
import type { CanvasEngine } from '@workbench/canvas-engine/api';

import { ChakraProvider } from '@chakra-ui/react';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createEmptyCanvasState } from '@workbench/canvasMigration';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import { MissingFontsDialog } from './MissingFontsDialog';

const api = vi.hoisted(() => ({ getFont: vi.fn(), listFonts: vi.fn() }));
vi.mock('@features/fonts', async (original) => ({ ...(await original<typeof FontsFeature>()), ...api }));
const i18n = createInstance();
await i18n.use(initReactI18next).init({
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
let root: Root | undefined;
let host: HTMLDivElement | undefined;
let client: QueryClient | undefined;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const missing = { id: 'missing', contentHash: 'a'.repeat(64), family: 'Lost Sans', label: 'Lost Sans Regular' };
const replacement = {
  id: 'replacement',
  contentHash: 'b'.repeat(64),
  family: 'Available Sans',
  label: 'Available Sans Variable',
  style: 'italic',
  weight: 700,
  axes: [{ tag: 'wdth', minimum: 50, maximum: 150, default: 100 }],
};
const replaceAllReferences = vi.fn(() => ({ status: 'committed' }));
const engine = {
  fonts: {
    collectReferences: () => [{ fontRef: missing, count: 3, layerIds: ['a', 'b', 'c'] }],
    replaceAllReferences,
  },
} as unknown as CanvasEngine;

const render = async (total = 1) => {
  await page.viewport(1280, 900);
  accountLifecycle.activate('font-recovery');
  replaceAllReferences.mockClear();
  api.getFont.mockRejectedValue(new ApiError('Missing font', 404));
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(['fonts', 'catalog', { limit: 50, search: '' }], { items: [replacement], total });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root!.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <QueryClientProvider client={client!}>
            <MissingFontsDialog
              engine={engine}
              groups={engine.fonts.collectReferences(createEmptyCanvasState().document)}
            />
          </QueryClientProvider>
        </I18nextProvider>
      </ChakraProvider>
    )
  );
};
afterEach(async () => {
  await act(() => root?.unmount());
  client?.clear();
  host?.remove();
  accountLifecycle.invalidate();
});

it('groups affected layers, keeps a warning after dismissal, and reopens recovery', async () => {
  await render();
  await expect.element(page.getByRole('dialog', { name: 'Missing fonts' })).toBeVisible();
  await expect.element(page.getByText('Used by 3 text layers')).toBeVisible();
  await expect.element(page.getByRole('button', { name: 'Replace all uses' })).toBeDisabled();
  expect(await auditAccessibility(document.querySelector('[role="dialog"]')!)).toEqual([]);
  await page.screenshot({ path: '../../../../artifacts/fonts/missing-fonts-dialog.png' });
  await act(() => page.getByRole('button', { name: 'Continue with previews' }).click());
  await expect.element(page.getByRole('dialog')).not.toBeInTheDocument();
  await act(() => page.getByRole('button', { name: 'Missing font — review' }).click());
  await expect.element(page.getByRole('dialog')).toBeVisible();
});

it('replaces every use through the undoable Canvas capability with target axis limits', async () => {
  await render();
  await act(() => page.getByRole('combobox', { name: 'Replacement font' }).selectOptions('replacement'));
  await act(() => page.getByRole('button', { name: 'Replace all uses' }).click());
  expect(replaceAllReferences).toHaveBeenCalledWith(missing, {
    fontRef: {
      id: replacement.id,
      contentHash: replacement.contentHash,
      family: replacement.family,
      label: replacement.label,
    },
    axes: replacement.axes,
    style: replacement.style,
    weight: replacement.weight,
  });
});

it('browses replacements beyond the first catalog page and preserves selection across pages', async () => {
  const later = { ...replacement, id: 'later', label: 'Later page font' };
  await render(101);
  client!.setQueryData(['fonts', 'catalog', { limit: 50, search: '', offset: 50 }], {
    items: [later],
    total: 101,
    offset: 50,
    limit: 50,
  });
  await act(() => page.getByRole('button', { name: 'Next page' }).click());
  await expect.element(page.getByRole('option', { name: later.label })).toBeInTheDocument();
  await act(() => page.getByRole('combobox', { name: 'Replacement font' }).selectOptions('later'));
  await act(() => page.getByRole('button', { name: 'Previous page' }).click());
  await expect.element(page.getByRole('combobox', { name: 'Replacement font' })).toHaveValue('later');
  await act(() => page.getByRole('button', { name: 'Replace all uses' }).click());
  expect(replaceAllReferences).toHaveBeenCalledWith(
    missing,
    expect.objectContaining({ fontRef: expect.objectContaining({ id: 'later' }) })
  );
});
