import type { ArchitectureCapabilitiesRow } from '@features/generation/core/architectureCapabilities';
import type { ModelConfig } from '@features/models/core/types';
import type * as ModelsApi from '@features/models/data/api';
import type { DefaultSettingsModel } from '@features/models/ui/detail/defaultSettingsFields';
import type { ReactNode } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { AddModelsView } from '@features/models/ui/add-models/AddModelsView';
import { DefaultSettingsSection } from '@features/models/ui/detail/DefaultSettingsSection';
import { DetailPane } from '@features/models/ui/manager/DetailPane';
import { openModelDetail, setAddModelsQuery } from '@features/models/ui/uiStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { AccountModelsUiProvider } from './router';

/**
 * The Model Manager renders on the Launchpad, outside the editor's ports, so the authenticated layout's account Models
 * port is what its Default settings, re-identify notice and Install in place read: real capabilities, account
 * preferences, and Generate's scheduler rules loaded on demand.
 */

const api = vi.hoisted(() => ({
  getFp8StorageSupport: vi.fn(() => Promise.resolve([])),
  getModelsDir: vi.fn(() => Promise.resolve('')),
  listMissingModels: vi.fn(() => Promise.resolve([])),
  listModels: vi.fn(() => Promise.resolve([])),
  updateModel: vi.fn(),
}));
const capabilitiesApi = vi.hoisted(() => ({ get: vi.fn<() => Promise<ArchitectureCapabilitiesRow[]>>() }));
const session = vi.hoisted(() => ({ canManageModels: true }));
const preferences = vi.hoisted(() => ({ patch: vi.fn() }));

vi.mock('@features/models/data/api', async (importOriginal) => ({
  ...(await importOriginal<typeof ModelsApi>()),
  ...api,
}));
vi.mock('@features/generation/data/architectureCapabilitiesApi', () => ({
  getArchitectureCapabilities: () => capabilitiesApi.get(),
}));
vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useCapabilities: () => ({ canManageModels: session.canManageModels }),
}));

// The account preferences, kept beyond each mount as the settings store keeps them.
const accountPreferences = createExternalStore({ enableModelDescriptions: true, installInPlace: true });
vi.mock('@workbench/settings/store', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  patchWorkbenchPreferences: (next: Record<string, unknown>) => {
    preferences.patch(next);
    accountPreferences.patchSnapshot(next);
    return Promise.resolve();
  },
  useWorkbenchPreferenceSelector: (select: (value: object) => unknown) => accountPreferences.useSelector(select),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, string>) => (values ? `${key}:${Object.values(values).join(',')}` : key),
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const SCHEDULER_LABEL = 'models.defaultFields.scheduler';

const settingsModel = (overrides: Partial<DefaultSettingsModel>): DefaultSettingsModel => ({
  base: 'anima',
  default_settings: null,
  format: 'checkpoint',
  key: 'model-key',
  type: 'main',
  variant: null,
  ...overrides,
});

const schedulerSwitch = (host: HTMLElement) =>
  [...host.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')].find((input) =>
    input.closest('label')?.textContent?.includes(`models.customizeDefaultField:${SCHEDULER_LABEL}`)
  );

const schedulerInput = () => document.querySelector<HTMLInputElement>(`input[aria-label="${SCHEDULER_LABEL}"]`);

const openOptions = async (): Promise<string[]> => {
  await act(() => schedulerInput()!.click());
  await expect.poll(() => document.querySelectorAll('[role="option"]').length).toBeGreaterThan(0);

  return [...document.querySelectorAll('[role="option"]')].map((option) => option.textContent ?? '');
};

const clickButton = async (host: HTMLElement, label: string) => {
  const button = [...host.querySelectorAll('button')].find((candidate) => candidate.textContent === label);
  await act(async () => {
    button!.click();
    await Promise.resolve();
  });
};

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });

  return { promise, resolve };
};

describe('Account Models port on the Launchpad', () => {
  let host: HTMLDivElement;
  let root: Root;

  const render = async (children: ReactNode) => {
    await act(async () => {
      root.render(
        <ChakraProvider value={system}>
          <AccountModelsUiProvider>{children}</AccountModelsUiProvider>
        </ChakraProvider>
      );
      await Promise.resolve();
    });
  };
  const renderDefaults = (model: DefaultSettingsModel) =>
    render(<DefaultSettingsSection model={model} onError={vi.fn()} onSaved={vi.fn()} />);

  beforeEach(() => {
    session.canManageModels = true;
    accountPreferences.setSnapshot({ enableModelDescriptions: true, installInPlace: true });
    preferences.patch.mockReset();
    capabilitiesApi.get.mockReset();
    capabilitiesApi.get.mockResolvedValue(architectureCapabilitiesFixture);
    // After the API stubs: a table requested earlier in the page is fetched again for the new account.
    accountLifecycle.activate('launchpad-models', ':user:launchpad-models');
    api.updateModel.mockReset();
    api.updateModel.mockImplementation((key: string, changes: { default_settings: object }) =>
      Promise.resolve({ ...settingsModel({ key }), ...changes })
    );
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    setAddModelsQuery('');
    // Clears the architecture table and the models store, as an account change does.
    accountLifecycle.invalidate();
  });

  describe('Default settings scheduler', () => {
    // The rules are a lazy chunk (the reader and the Generation modules it imports); load it up front so the first
    // test is not timing a cold transform. Importing it starts nothing: the section's load does.
    beforeAll(async () => {
      await import('./modelSchedulerPolicyReader');
    });

    it('loads Generate’s rules on demand and offers its Anima schedulers once the table arrives', async () => {
      const table = deferred<ArchitectureCapabilitiesRow[]>();
      capabilitiesApi.get.mockReturnValue(table.promise);
      await renderDefaults(settingsModel({}));

      // Until the table arrives the row holds its place, disabled and saying why.
      await expect.poll(() => capabilitiesApi.get.mock.calls.length).toBe(1);
      expect(schedulerSwitch(host)?.disabled).toBe(true);
      expect(host.textContent).toContain('models.schedulerListLoading');

      await act(async () => {
        table.resolve(architectureCapabilitiesFixture);
        await table.promise;
      });

      await expect.poll(() => schedulerSwitch(host)?.disabled).toBe(false);
      await act(() => schedulerSwitch(host)!.click());
      // Switched on at Anima's own default rather than an SD scheduler it cannot run.
      expect(schedulerInput()?.value).toBe('Euler');
      expect(await openOptions()).toEqual(['Euler', 'Heun (2nd order)', 'DPM++ 2M', 'DPM++ 2M SDE', 'ER-SDE', 'LCM']);
    });

    it('saves ER-SDE chosen for an SDXL model', async () => {
      await renderDefaults(settingsModel({ base: 'sdxl' }));
      await expect.poll(() => schedulerSwitch(host)?.disabled).toBe(false);

      await act(() => schedulerSwitch(host)!.click());
      expect(schedulerInput()?.value).toBe('Euler Ancestral');
      expect(await openOptions()).toContain('ER-SDE');
      const erSde = [...document.querySelectorAll<HTMLElement>('[role="option"]')].find(
        (option) => option.textContent === 'ER-SDE'
      );
      await act(() => erSde!.click());
      await clickButton(host, 'models.saveDefaults');

      expect(api.updateModel).toHaveBeenCalledWith(
        'model-key',
        { default_settings: expect.objectContaining({ scheduler: 'er_sde' }) },
        expect.anything()
      );
    });

    it('keeps a stored scheduler outside the base’s set visible until it is replaced', async () => {
      await renderDefaults(settingsModel({ base: 'flux', default_settings: { scheduler: 'dpmpp_2m_k' } }));
      await expect.poll(() => schedulerInput()?.value).toBe('dpmpp_2m_k');

      expect(await openOptions()).toEqual(['dpmpp_2m_k', 'Euler', 'Heun (2nd order)', 'LCM']);
    });

    it('shows the row on a base whose graph ignores the scheduler only to clear a stored value', async () => {
      await renderDefaults(settingsModel({ base: 'qwen-image', default_settings: { scheduler: 'euler' } }));
      await expect.poll(() => schedulerSwitch(host)?.checked).toBe(true);

      await act(() => schedulerSwitch(host)!.click());
      await clickButton(host, 'models.saveDefaults');
      expect(api.updateModel).toHaveBeenCalledWith(
        'model-key',
        { default_settings: expect.objectContaining({ scheduler: null }) },
        expect.anything()
      );

      await act(() => root.unmount());
      root = createRoot(host);
      await renderDefaults(settingsModel({ base: 'qwen-image' }));
      await expect.poll(() => schedulerSwitch(host)).toBeUndefined();
      expect(host.textContent).toContain('models.defaultFields.vae');
    });

    it('says the list is unavailable when the table fails, keeps a stored value, and retries', async () => {
      capabilitiesApi.get.mockRejectedValue(new Error('offline'));
      await act(() => accountLifecycle.activate('launchpad-models-offline', ':user:launchpad-models-offline'));
      await renderDefaults(settingsModel({ base: 'sdxl', default_settings: { scheduler: 'er_sde' } }));

      await expect.poll(() => host.textContent).toContain('models.schedulerListUnavailable');
      // The stored choice stays named and can still be cleared; nothing can be picked without the rules.
      expect(schedulerInput()?.value).toBe('er_sde');
      expect(schedulerInput()?.disabled).toBe(true);
      expect(schedulerSwitch(host)?.disabled).toBe(false);

      const failedRequests = capabilitiesApi.get.mock.calls.length;
      const table = deferred<ArchitectureCapabilitiesRow[]>();
      capabilitiesApi.get.mockReturnValue(table.promise);
      const retry = [...host.querySelectorAll('button')].find((button) => button.textContent === 'common.retry')!;
      await act(() => userEvent.click(retry));

      // While the retry loads, the same button stays focused, marked busy rather than disabled.
      await expect.poll(() => retry.textContent).toBe('models.schedulerListRetrying');
      expect(retry.getAttribute('aria-disabled')).toBe('true');
      expect(document.activeElement).toBe(retry);

      await act(async () => {
        table.resolve(architectureCapabilitiesFixture);
        await table.promise;
      });

      await expect.poll(() => schedulerInput()?.disabled).toBe(false);
      expect(capabilitiesApi.get).toHaveBeenCalledTimes(failedRequests + 1);
      expect(host.textContent).not.toContain('models.schedulerListUnavailable');
      expect(schedulerInput()?.value).toBe('ER-SDE');
      // The button left with the note; its focus moved to the scheduler row rather than the page body.
      await expect.poll(() => document.activeElement).toBe(schedulerSwitch(host));

      // The retry is over: a later ordinary load (another account here) says it is loading, not retrying.
      capabilitiesApi.get.mockReturnValue(deferred<ArchitectureCapabilitiesRow[]>().promise);
      await act(() => accountLifecycle.activate('launchpad-models-next', ':user:launchpad-models-next'));
      await expect.poll(() => host.textContent).toContain('models.schedulerListLoading');
      expect(host.textContent).not.toContain('models.schedulerListRetrying');
    });

    it('hands the Retry focus to the section heading when the recovered base has no scheduler row', async () => {
      capabilitiesApi.get.mockRejectedValue(new Error('offline'));
      await act(() => accountLifecycle.activate('launchpad-models-offline', ':user:launchpad-models-offline'));
      // Qwen Image's graph ignores the scheduler and nothing is stored: once the rules arrive the row goes.
      await renderDefaults(settingsModel({ base: 'qwen-image' }));
      await expect.poll(() => host.textContent).toContain('models.schedulerListUnavailable');

      capabilitiesApi.get.mockResolvedValue(architectureCapabilitiesFixture);
      const retry = [...host.querySelectorAll('button')].find((button) => button.textContent === 'common.retry')!;
      await act(() => userEvent.click(retry));

      await expect.poll(() => schedulerSwitch(host)).toBeUndefined();
      const heading = [...host.querySelectorAll<HTMLElement>('[tabindex="-1"]')].find((element) =>
        element.textContent?.startsWith('models.defaultSettings')
      );
      await expect.poll(() => document.activeElement).toBe(heading);
    });
  });

  describe('Model Manager surfaces by capability', () => {
    const UNIDENTIFIED_LLLITE = {
      base: 'anima',
      cond_in_channels: null,
      file_size: 1024,
      format: 'checkpoint',
      hash: 'hash-lllite',
      key: 'lllite-1',
      name: 'Anima LLLite',
      path: '/models/lllite.safetensors',
      source: '/models/lllite.safetensors',
      source_type: 'path',
      type: 'controlnet',
    } as ModelConfig;

    const showDetail = async () => {
      setModelsSnapshotForTests({ models: [UNIDENTIFIED_LLLITE], status: 'loaded' });
      openModelDetail(UNIDENTIFIED_LLLITE.key);
      await render(<DetailPane />);
    };
    const hasReidentifyButton = () =>
      [...host.querySelectorAll('button')].some((button) => button.textContent === 'models.reidentify');

    it('offers Re-identify on the notice to a user who manages models', async () => {
      await showDetail();

      expect(host.textContent).toContain('models.adapterTypeUnknown');
      expect(host.textContent).not.toContain('models.reidentifyAskAdmin');
      expect(hasReidentifyButton()).toBe(true);
    });

    it('shows a user without model management the notice and who to ask, and no button', async () => {
      session.canManageModels = false;
      await showDetail();

      expect(host.textContent).toContain('models.reidentifyAskAdmin');
      expect(hasReidentifyButton()).toBe(false);
    });

    it('saves Install in place to the account and reads it back on the next visit', async () => {
      setModelsSnapshotForTests({ models: [], modelsDir: '/opt/invokeai/models', status: 'loaded' });
      await render(<AddModelsView />);
      const inplaceInput = () =>
        [...host.querySelectorAll('label')]
          .find((label) => label.textContent === 'models.installInPlace')
          ?.querySelector('input');

      await act(() =>
        userEvent.fill(host.querySelector<HTMLInputElement>('input[aria-label="models.searchOrAdd"]')!, '/tmp/x.ckpt')
      );
      await act(() => userEvent.click(inplaceInput()!.closest('label')!));
      expect(preferences.patch).toHaveBeenCalledExactlyOnceWith({ installInPlace: false });

      await act(() => root.unmount());
      root = createRoot(host);
      await render(<AddModelsView />);
      expect(inplaceInput()?.checked).toBe(false);
    });
  });
});
