import type { ModelConfig, ModelInstallJob } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { handleModelInstallSocketEvent } from '@features/models/data/installsStore';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { getModelsUiSnapshotForTests, updateModelsUi } from '@features/models/ui/uiStore';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { setConnectionStatus } from '@platform/transport/connectionStore';
import { applyThemeToRoot } from '@theme/applyTheme';
import { DEFAULT_THEME_ID, system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import { InstallQueueBar } from './InstallQueueBar';

const api = vi.hoisted(() => ({
  cancelModelInstall: vi.fn(),
  installModel: vi.fn(),
  listModelInstalls: vi.fn(),
  pauseModelInstall: vi.fn(),
  pruneCompletedModelInstalls: vi.fn(),
  restartFailedModelInstall: vi.fn(),
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
  fallbackNS: 'translation',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const JOBS: ModelInstallJob[] = [
  {
    bytes: 0,
    id: 1,
    source: { repo_id: 'vantagewithai/Krea-2-Turbo-GGUF', subfolder: 'krea2_turbo-Q4_K_M.gguf', type: 'hf' },
    status: 'downloading',
    total_bytes: 7_000_000_000,
  },
  {
    id: 2,
    source: { repo_id: 'unsloth/FLUX.1-Kontext-dev-GGUF', subfolder: 'flux1-kontext-dev-Q4_K_M.gguf', type: 'hf' },
    status: 'waiting',
    total_bytes: 6_800_000_000,
  },
  {
    error: 'A model with this path is already installed.',
    error_reason: 'DuplicateModelException',
    id: 3,
    source: { path: '/mnt/models/animosity_krea2Ver10FP8.safetensors', type: 'local' },
    status: 'error',
  },
  {
    error: 'Unauthorized',
    error_reason: 'HTTPError',
    id: 4,
    source: { repo_id: 'black-forest-labs/FLUX.1-schnell', subfolder: 'ae.safetensors', type: 'hf' },
    status: 'error',
    total_bytes: 335_000_000,
  },
  {
    config_out: {
      base: 'flux',
      key: 'lora-key',
      name: 'Elegance_Niji_KR2',
      type: 'lora',
    } as ModelInstallJob['config_out'],
    id: 5,
    source: { path: '/mnt/models/Lora/Elegance_Niji_KR2.safetensors', type: 'local' },
    status: 'completed',
  },
  {
    bytes: 12_000_000,
    id: 6,
    source: 'https://example.com/old.safetensors',
    status: 'cancelled',
    total_bytes: 90_000_000,
  },
];

const INSTALLED_LORA = {
  base: 'flux',
  key: 'lora-key',
  name: 'Elegance_Niji_KR2',
  type: 'lora',
} as unknown as ModelConfig;

let root: Root | undefined;
let host: HTMLDivElement | undefined;

const settle = () =>
  act(async () => {
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 20);
    });
  });

const render = async () => {
  host = document.createElement('div');
  host.style.cssText = 'display:flex;flex-direction:column;height:600px;width:1000px';
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <InstallQueueBar />
        </ChakraProvider>
      </I18nextProvider>
    );
  });
  await settle();
};

const rows = () => [...(host?.querySelectorAll<HTMLTableRowElement>('tbody tr[data-install-status]') ?? [])];
const rowByStatus = (status: string) => rows().find((row) => row.dataset.installStatus === status);
const click = (name: string, within: ParentNode | undefined = host) =>
  act(() => {
    const button = [...(within?.querySelectorAll<HTMLButtonElement>('button') ?? [])].find(
      (candidate) => candidate.getAttribute('aria-label') === name || candidate.textContent?.trim() === name
    );

    if (!button) {
      throw new Error(`No button named ${name}`);
    }

    button.click();
  });

beforeEach(async () => {
  await page.viewport(1100, 700);
  applyThemeToRoot(DEFAULT_THEME_ID);
  accountLifecycle.invalidate();
  accountLifecycle.activate('tester');
  setConnectionStatus('connected');
  setModelsSnapshotForTests({
    modelsByKey: new Map([['lora-key', INSTALLED_LORA]]),
  });
  for (const mock of Object.values(api)) {
    mock.mockReset();
  }
  api.listModelInstalls.mockResolvedValue(JOBS);
  api.pruneCompletedModelInstalls.mockResolvedValue(undefined);
  api.cancelModelInstall.mockResolvedValue(undefined);
  api.pauseModelInstall.mockImplementation((id: number) =>
    Promise.resolve({ ...JOBS.find((job) => job.id === id)!, status: 'paused' })
  );
  api.installModel.mockImplementation(() => {
    const created: ModelInstallJob = { id: 7, source: 'https://example.com/old.safetensors', status: 'waiting' };

    // The follow-up refresh must see the new job, as the real backend lists it immediately.
    api.listModelInstalls.mockResolvedValue([...JOBS, created]);

    return Promise.resolve(created);
  });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  root = undefined;
  host = undefined;
});

describe('InstallQueueBar', () => {
  it('renders one row per job with grouped ordering, captions, and status badges', async () => {
    updateModelsUi({ queueExpanded: true, queueMaximized: false });
    await render();

    expect(rows().map((row) => row.dataset.installStatus)).toEqual([
      'downloading',
      'queued',
      'unauthorized',
      'failed',
      'installed',
      'cancelled',
    ]);

    const header = host?.querySelector('[aria-expanded="true"]')?.parentElement;
    expect(header?.textContent).toContain('1 downloading');
    expect(header?.textContent).toContain('1 queued');
    expect(header?.textContent).toContain('2 need attention');
    expect(header?.textContent).toContain('1 installed');

    expect(rowByStatus('downloading')?.textContent).toContain('krea2_turbo-Q4_K_M.gguf');
    expect(rowByStatus('downloading')?.textContent).toContain(
      'vantagewithai/Krea-2-Turbo-GGUF :: krea2_turbo-Q4_K_M.gguf'
    );
    expect(rowByStatus('queued')?.textContent).toContain('Waiting · #1 in queue · 6.8 GB');
    expect(rowByStatus('failed')?.textContent).toContain('DuplicateModelException');
    expect(rowByStatus('failed')?.textContent).toContain('A model with this path is already installed.');
    expect(rowByStatus('failed')?.textContent).toContain('Stopped · 0 bytes copied');
    expect(rowByStatus('unauthorized')?.textContent).toContain(
      'This repo is gated. Add a Hugging Face token to continue.'
    );
    expect(rowByStatus('unauthorized')?.textContent).toContain('Not started · 335.0 MB');
    expect(rowByStatus('installed')?.textContent).toContain('Done · LoRA');
    expect(rowByStatus('cancelled')?.textContent).toContain('Cancelled · 12.0 MB copied');

    expect(await auditAccessibility(host!)).toEqual([]);
  });

  it('reflects live progress with a rate and ETA on the downloading row and the collapsed bar', async () => {
    updateModelsUi({ queueExpanded: true, queueMaximized: false });
    await render();

    await act(() => {
      handleModelInstallSocketEvent('model_install_download_progress', {
        bytes: 2_660_000_000,
        id: 1,
        total_bytes: 7_000_000_000,
      });
    });
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 600);
    });
    await act(() => {
      handleModelInstallSocketEvent('model_install_download_progress', {
        bytes: 2_690_000_000,
        id: 1,
        total_bytes: 7_000_000_000,
      });
    });
    await settle();

    const caption = rowByStatus('downloading')?.textContent ?? '';
    expect(caption).toContain('38% · 2.7 GB / 7.0 GB');
    expect(caption).toMatch(/MB\/s · ~\d+ (s|min|h) left/);

    await click('Install Queue');
    await settle();
    expect(rows()).toHaveLength(0);
    const bar = host?.textContent ?? '';
    expect(bar).toContain('krea2_turbo-Q4_K_M.gguf');
    expect(bar).toContain('38% · 2.7 GB / 7.0 GB');
    expect(bar).toContain('+1 queued');
    expect(bar).toContain('2 need attention');

    await click('Pause download');
    await settle();
    expect(api.pauseModelInstall).toHaveBeenCalledWith(1, expect.any(AbortSignal));
  });

  it('routes the unauthorized fix to the API keys tab and resubmits or dismisses settled jobs', async () => {
    updateModelsUi({ activeTab: 'add', queueExpanded: true, queueMaximized: false });
    await render();

    await click('Add token', rowByStatus('unauthorized'));
    expect(getModelsUiSnapshotForTests().activeTab).toBe('keys');

    await click('Retry install', rowByStatus('cancelled'));
    await settle();
    expect(api.installModel).toHaveBeenCalledWith(
      expect.objectContaining({ source: 'https://example.com/old.safetensors' }),
      expect.any(AbortSignal)
    );
    expect(rowByStatus('cancelled')).toBeUndefined();
    expect(rows().filter((row) => row.dataset.installStatus === 'queued')).toHaveLength(2);

    await click('Remove from list', rowByStatus('installed'));
    await settle();
    expect(rowByStatus('installed')).toBeUndefined();
    expect(api.pruneCompletedModelInstalls).not.toHaveBeenCalled();

    await click('Clear finished');
    await settle();
    expect(api.pruneCompletedModelInstalls).toHaveBeenCalledTimes(1);
  });

  it('expands to fill the pane and restores', async () => {
    updateModelsUi({ queueExpanded: true, queueMaximized: false });
    await render();

    await click('Expand');
    expect(getModelsUiSnapshotForTests().queueMaximized).toBe(true);
    await settle();
    await click('Restore');
    expect(getModelsUiSnapshotForTests().queueMaximized).toBe(false);

    // Row navigation must uncover the pane it targets.
    await click('Expand');
    await settle();
    await click('Add token', rowByStatus('unauthorized'));
    expect(getModelsUiSnapshotForTests()).toMatchObject({ activeTab: 'keys', queueMaximized: false });
  });
});
