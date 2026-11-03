/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { ModelInstallJob, StarterModelResponse } from '@features/models/core/types';
import type { ModelsUiAdapter } from '@features/models/ui/ModelsUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { ModelsUiProvider } from '@features/models/ui/ModelsUiContext';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { createMemoryHistory, createRootRoute, createRouter, RouterProvider } from '@tanstack/react-router';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import type * as ImportQueueModelModule from './importQueueModel';

import { ModelsNotice } from './ModelsNotice';

// Its own file: the notice keeps the install queue model's load for the page, failed or not. The first load fails as a
// chunk that did not arrive; any later one succeeds.
const queueModel = vi.hoisted(() => ({ loads: 0 }));

vi.mock('./importQueueModel', async (importOriginal) => {
  const { importQueueModel } = await importOriginal<typeof ImportQueueModelModule>();

  return {
    importQueueModel: () => {
      queueModel.loads += 1;

      return queueModel.loads === 1
        ? Promise.reject(new TypeError('Failed to fetch dynamically imported module'))
        : importQueueModel();
    },
  };
});

vi.mock('@features/models/data/api', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getModelsDir: () => Promise.resolve('/models'),
  getStarterModels: () =>
    Promise.resolve({
      starter_bundles: {
        sdxl: {
          models: [{ base: 'sdxl', name: 'SDXL base', source: 'sdxl-source', type: 'main' }],
          name: 'SDXL',
        },
      },
      starter_models: [],
    } as unknown as StarterModelResponse),
  listMissingModels: () => Promise.resolve([]),
  listModelInstalls: () =>
    Promise.resolve([
      { error: 'boom', id: 1, source: 'https://example.com/a.safetensors', status: 'error' },
      { error: 'boom', id: 2, source: 'https://example.com/b.safetensors', status: 'error' },
    ] as ModelInstallJob[]),
  listModels: () => Promise.resolve([]),
}));

vi.mock('@features/models/ui/useModelsNotify', () => ({
  useNotify: () => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }),
}));

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const ADAPTER: ModelsUiAdapter = {
  canManageModels: true,
  enableModelDescriptions: true,
  installInPlace: true,
  isProjectActive: () => false,
  managerProjectId: null,
  schedulerPolicy: {
    getSnapshot: () => ({ status: 'loading' }),
    load: () => undefined,
    subscribe: () => () => undefined,
  },
  setInstallInPlace: () => undefined,
};

const host = document.createElement('div');
const root = createRoot(host);

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

it('keeps the notice up when the attention count cannot load, without retrying on its own, and Retry recovers it', async () => {
  accountLifecycle.activate('models-notice-attention-load-test');
  document.body.append(host);
  const rootRoute = createRootRoute({
    component: () => (
      <ModelsUiProvider adapter={ADAPTER}>
        <ModelsNotice />
      </ModelsUiProvider>
    ),
  });
  const router = createRouter({ history: createMemoryHistory({ initialEntries: ['/'] }), routeTree: rootRoute });
  await act(async () => {
    root.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <RouterProvider router={router} />
        </I18nextProvider>
      </ChakraProvider>
    );
    await router.load();
  });

  await expect.element(page.getByText(i18n.t('models.launchpad.installsCheckFailed'))).toBeVisible();
  // The rest of the notice is untouched: the bundle install is still offered.
  await expect.element(page.getByRole('button', { exact: true, name: 'Install SDXL' })).toBeVisible();
  // A failed load is not retried behind the viewer's back.
  await act(
    () =>
      new Promise<void>((resolve) => {
        window.setTimeout(resolve, 300);
      })
  );
  expect(queueModel.loads).toBe(1);

  await userEvent.click(page.getByRole('button', { name: i18n.t('common.retry') }));

  await expect.element(page.getByText('2 installs failed or need attention.')).toBeVisible();
  expect(page.getByText(i18n.t('models.launchpad.installsCheckFailed')).query()).toBeNull();
  expect(queueModel.loads).toBe(2);
});
