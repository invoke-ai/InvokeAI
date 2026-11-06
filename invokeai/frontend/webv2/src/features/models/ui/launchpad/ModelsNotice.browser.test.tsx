/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { ModelConfig, ModelInstallJob, StarterModelResponse } from '@features/models/core/types';
import type { ModelsUiAdapter } from '@features/models/ui/ModelsUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { ModelsUiProvider } from '@features/models/ui/ModelsUiContext';
import { getModelsUiSnapshotForTests } from '@features/models/ui/uiStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import {
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
  Outlet,
  RouterProvider,
} from '@tanstack/react-router';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { ModelsNotice, ModelsRequiredHint } from './ModelsNotice';

const api = vi.hoisted(() => ({
  getModelsDir: vi.fn(),
  getStarterModels: vi.fn(),
  installModel: vi.fn(),
  listMissingModels: vi.fn(),
  listModelInstalls: vi.fn(),
  listModels: vi.fn(),
}));

vi.mock('@features/models/data/api', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ...api,
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

const starter = (name: string, base: string, source: string) => ({
  base,
  description: name,
  format: 'diffusers',
  is_installed: false,
  name,
  source,
  type: 'main',
});

const STARTERS = {
  starter_bundles: {
    'sd-1': { models: [starter('SD 1.5 base', 'sd-1', 'sd1-source')], name: 'SD 1.5' },
    sdxl: {
      models: [starter('SDXL base', 'sdxl', 'sdxl-source'), starter('SDXL VAE', 'sdxl', 'sdxl-vae-source')],
      name: 'SDXL',
    },
  },
  starter_models: [],
} as unknown as StarterModelResponse;

const ENCODER_ONLY = [{ base: 'any', key: 'encoder', name: 'T5', type: 't5_encoder' }] as unknown as ModelConfig[];
const MAIN_MODEL = [{ base: 'sdxl', key: 'main', name: 'SDXL base', type: 'main' }] as unknown as ModelConfig[];

let host: HTMLDivElement;
let root: Root;

const adapter = (canManageModels: boolean): ModelsUiAdapter => ({
  canManageModels,
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
});

const render = async (canManageModels: boolean) => {
  const rootRoute = createRootRoute({
    component: () => (
      <ModelsUiProvider adapter={adapter(canManageModels)}>
        <ModelsRequiredHint />
        <ModelsNotice />
        <Outlet />
      </ModelsUiProvider>
    ),
  });
  const router = createRouter({
    history: createMemoryHistory({ initialEntries: ['/'] }),
    routeTree: rootRoute.addChildren([
      createRoute({ getParentRoute: () => rootRoute, path: '/' }),
      createRoute({ getParentRoute: () => rootRoute, path: 'models' }),
    ]),
  });
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
  return router;
};

beforeEach(() => {
  accountLifecycle.invalidate();
  accountLifecycle.activate('models-notice-test');
  for (const mock of Object.values(api)) {
    mock.mockReset();
  }
  api.getModelsDir.mockResolvedValue('/models');
  api.listMissingModels.mockResolvedValue([]);
  api.listModels.mockResolvedValue([]);
  api.getStarterModels.mockResolvedValue(STARTERS);
  api.listModelInstalls.mockResolvedValue([]);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

describe('Home models notice', () => {
  it('tells accounts that cannot manage models who can add them, without asking for the admin-only catalog', async () => {
    await render(false);

    await expect.element(page.getByText(i18n.t('models.launchpad.noModelsAskAdmin'))).toBeVisible();
    await expect.element(page.getByText(i18n.t('models.launchpad.startNeedsModelAskAdmin'))).toBeVisible();
    expect(page.getByRole('button').query()).toBeNull();
    expect(api.getStarterModels).not.toHaveBeenCalled();
    expect(api.listModelInstalls).not.toHaveBeenCalled();
  });

  it('counts only models that can generate: encoders alone still leave Invoke without one', async () => {
    api.listModels.mockResolvedValue(ENCODER_ONLY);
    await render(false);
    await expect.element(page.getByText(i18n.t('models.launchpad.noModelsAskAdmin'))).toBeVisible();

    await act(() => root.unmount());
    root = createRoot(host);
    accountLifecycle.invalidate();
    accountLifecycle.activate('models-notice-test-main');
    api.listModels.mockResolvedValue(MAIN_MODEL);
    await render(false);
    await expect.poll(() => api.listModels.mock.calls.length).toBe(2);
    expect(host.textContent).toBe('');
  });

  it('installs the recommended bundle in one click and says what it suits', async () => {
    api.installModel.mockImplementation((request: { source: string }) =>
      Promise.resolve({ id: request.source.length, source: request.source, status: 'waiting' } as ModelInstallJob)
    );
    const router = await render(true);

    const install = page.getByRole('button', { exact: true, name: 'Install SDXL' });
    await expect.element(install).toBeVisible();
    await expect.element(page.getByText(i18n.t('models.launchpad.bundleGoodFor.sdxl'))).toBeVisible();

    await userEvent.click(install);
    await expect
      .poll(() => api.installModel.mock.calls.map(([request]) => request.source))
      .toEqual(['sdxl-source', 'sdxl-vae-source']);
    await expect.element(page.getByText('2 models installing…')).toBeVisible();
    // The button went away with the bundles; focus stays in the panel, and the start hint stops asking for an install.
    await expect.element(page.getByRole('heading', { name: i18n.t('models.launchpad.noModelsTitle') })).toHaveFocus();
    await expect.element(page.getByText(i18n.t('models.launchpad.startAfterInstall'))).toBeVisible();
    expect(router.state.location.pathname).toBe('/');

    await userEvent.click(page.getByRole('button', { name: i18n.t('models.launchpad.openQueue') }));
    await expect.poll(() => router.state.location.pathname).toBe('/models');
    expect(getModelsUiSnapshotForTests().queueExpanded).toBe(true);
  });

  it('says how many installs need attention and opens the install queue to review them', async () => {
    api.listModels.mockResolvedValue(MAIN_MODEL);
    api.listModelInstalls.mockResolvedValue([
      { error: 'boom', id: 1, source: 'https://example.com/a.safetensors', status: 'error' },
      { error: 'boom', id: 2, source: 'https://example.com/b.safetensors', status: 'error' },
    ] as ModelInstallJob[]);
    const router = await render(true);

    await expect.element(page.getByText('2 installs failed or need attention.')).toBeVisible();
    await userEvent.click(page.getByRole('button', { name: i18n.t('models.launchpad.reviewInstalls') }));

    await expect.poll(() => router.state.location.pathname).toBe('/models');
    expect(getModelsUiSnapshotForTests().queueExpanded).toBe(true);
  });
});
