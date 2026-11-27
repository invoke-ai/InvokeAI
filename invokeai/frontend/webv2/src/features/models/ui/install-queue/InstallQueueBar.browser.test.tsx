import type { ModelConfig, ModelInstallJob } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { handleModelInstallSocketEvent, refreshInstalls } from '@features/models/data/installsStore';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { updateModelsUi } from '@features/models/ui/uiStore';
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
import { page, userEvent } from 'vitest/browser';

import { InstallQueueBar } from './InstallQueueBar';

const api = vi.hoisted(() => ({
  cancelModelInstall: vi.fn(),
  installModel: vi.fn(),
  listModelInstalls: vi.fn(),
  pauseModelInstall: vi.fn(),
  pruneCompletedModelInstalls: vi.fn(),
  restartFailedModelInstall: vi.fn(),
  resumeModelInstall: vi.fn(),
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
  api.pruneCompletedModelInstalls.mockResolvedValue([3, 4, 5, 6]);
  api.cancelModelInstall.mockImplementation((id: number) =>
    Promise.resolve({ ...JOBS.find((job) => job.id === id)!, status: 'cancelled' })
  );
  api.pauseModelInstall.mockImplementation((id: number) =>
    Promise.resolve({ ...JOBS.find((job) => job.id === id)!, status: 'paused' })
  );
  api.installModel.mockImplementation(() => {
    const created: ModelInstallJob = { id: 7, source: 'https://example.com/old.safetensors', status: 'waiting' };

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

  it('refetches an auto-paused download to show the failed file and its cause, and resumes it', async () => {
    updateModelsUi({ queueExpanded: true, queueMaximized: false });
    const healthyPart = { bytes: 4, source: 'https://example.com/weights.safetensors', total_bytes: 100 };
    api.listModelInstalls.mockResolvedValue(
      JOBS.map((job) => (job.id === 1 ? { ...job, download_parts: [{ ...healthyPart, status: 'running' }] } : job))
    );
    api.resumeModelInstall.mockResolvedValue({ ...JOBS[0], status: 'waiting' });
    await render();
    expect(api.listModelInstalls).toHaveBeenCalledTimes(1);
    expect(rowByStatus('downloading')?.textContent).not.toContain('weights.safetensors');

    api.listModelInstalls.mockResolvedValue(
      JOBS.map((job) =>
        job.id === 1
          ? {
              ...job,
              download_parts: [
                {
                  ...healthyPart,
                  error: 'Traceback (most recent call last):\nConnectionError: reset by peer',
                  error_type: 'ConnectionError(reset by peer)',
                  status: 'error',
                },
              ],
              status: 'paused' as const,
            }
          : job
      )
    );
    await act(() => {
      handleModelInstallSocketEvent('model_install_paused', { id: 1, source: JOBS[0].source });
    });
    await vi.waitFor(() => expect(api.listModelInstalls).toHaveBeenCalledTimes(2));
    await settle();

    expect(rowByStatus('downloading')).toBeUndefined();
    const pausedRow = rowByStatus('paused');
    expect(pausedRow?.textContent).toContain('weights.safetensors');
    expect(pausedRow?.textContent).toContain('ConnectionError(reset by peer)');
    expect(pausedRow?.textContent).not.toContain('Traceback');
    expect(pausedRow?.querySelector('button[aria-label="Resume download"]')).not.toBeNull();
    expect(pausedRow?.querySelector('button[aria-label="Restart this file"]')).not.toBeNull();
    expect(pausedRow?.querySelector('button[aria-label="Retry failed download"]')).not.toBeNull();
    const header = host?.querySelector('[aria-expanded="true"]')?.parentElement;
    expect(header?.textContent).toContain('3 need attention');
    expect(header?.textContent).not.toContain('paused');
    expect(await auditAccessibility(host!)).toEqual([]);

    await click('Resume download', pausedRow);
    await settle();

    expect(api.resumeModelInstall).toHaveBeenCalledWith(1, expect.any(AbortSignal));
    expect(api.restartFailedModelInstall).not.toHaveBeenCalled();
  });

  it('does not restore finished rows from a list request started before successful pruning', async () => {
    updateModelsUi({ queueExpanded: true, queueMaximized: false });
    let resolveStale!: (jobs: ModelInstallJob[]) => void;
    const stale = new Promise<ModelInstallJob[]>((resolve) => {
      resolveStale = resolve;
    });
    api.listModelInstalls
      .mockReset()
      .mockResolvedValueOnce(JOBS)
      .mockReturnValueOnce(stale)
      .mockRejectedValueOnce(new Error('temporary list failure'));
    api.pruneCompletedModelInstalls.mockResolvedValue([1, 3, 4, 5, 6]);
    await render();

    act(() => {
      void refreshInstalls();
    });
    expect(api.listModelInstalls).toHaveBeenCalledTimes(2);
    await click('Clear finished');
    await settle();
    expect(api.pruneCompletedModelInstalls).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveStale(JOBS);
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 20);
      });
    });

    expect(api.listModelInstalls).toHaveBeenCalledTimes(3);
    expect(
      rows().some((row) =>
        ['installed', 'cancelled', 'failed', 'unauthorized'].includes(row.dataset.installStatus ?? '')
      )
    ).toBe(false);
  });

  it('keeps a job that another client restarted before pruning', async () => {
    updateModelsUi({ queueExpanded: true, queueMaximized: false });
    api.listModelInstalls.mockResolvedValue(
      JOBS.map((job) => (job.id === 3 ? { ...job, status: 'waiting' as const } : job))
    );
    api.pruneCompletedModelInstalls.mockResolvedValue([4, 5, 6]);
    await render();

    await click('Clear finished');
    await settle();

    expect(rowByStatus('queued')).toBeDefined();
    expect(rowByStatus('failed')).toBeUndefined();
  });

  it('keeps a completed cancel response over an older list row when the refresh fails', async () => {
    updateModelsUi({ queueExpanded: true, queueMaximized: false });
    let resolveStale!: (jobs: ModelInstallJob[]) => void;
    const stale = new Promise<ModelInstallJob[]>((resolve) => {
      resolveStale = resolve;
    });
    api.listModelInstalls
      .mockReset()
      .mockResolvedValueOnce(JOBS)
      .mockReturnValueOnce(stale)
      .mockRejectedValueOnce(new Error('temporary list failure'));
    api.cancelModelInstall.mockResolvedValue({ ...JOBS[0]!, config_out: INSTALLED_LORA, status: 'completed' });
    await render();

    act(() => {
      void refreshInstalls();
    });
    expect(api.listModelInstalls).toHaveBeenCalledTimes(2);
    await click('Cancel install', rowByStatus('downloading'));
    await settle();

    await act(async () => {
      resolveStale(JOBS);
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 20);
      });
    });

    expect(api.cancelModelInstall).toHaveBeenCalledWith(1, expect.any(AbortSignal));
    expect(api.listModelInstalls).toHaveBeenCalledTimes(3);
    expect(rowByStatus('installed')).toBeDefined();
    expect(rowByStatus('downloading')).toBeUndefined();
  });

  it('stops with a stop icon, removes with an X, and asks before discarding a download that has progress', async () => {
    updateModelsUi({ queueExpanded: true, queueMaximized: false });
    api.listModelInstalls.mockResolvedValue(JOBS.map((job) => (job.id === 1 ? { ...job, bytes: 2_000_000_000 } : job)));
    await render();

    const cancel = rowByStatus('downloading')?.querySelector('button[aria-label="Cancel install"] svg');
    const remove = rowByStatus('cancelled')?.querySelector('button[aria-label="Remove from list"] svg');
    expect(cancel?.innerHTML).toBeTruthy();
    expect(cancel?.innerHTML).not.toBe(remove?.innerHTML);

    await click('Cancel install', rowByStatus('downloading'));
    await expect
      .poll(() => document.querySelector('[role="alertdialog"]')?.textContent ?? '')
      .toContain('has downloaded 2.0 GB of 7.0 GB');
    await click('Keep downloading', document.body);
    await settle();
    expect(api.cancelModelInstall).not.toHaveBeenCalled();

    await click('Cancel install', rowByStatus('downloading'));
    await expect.poll(() => document.querySelector('[role="alertdialog"]')).not.toBeNull();
    await click('Cancel download', document.querySelector('[role="alertdialog"]')!);
    await settle();
    expect(api.cancelModelInstall).toHaveBeenCalledWith(1, expect.any(AbortSignal));
  });

  it('confirms Cancel all with the number of installs it stops, and stops exactly those', async () => {
    updateModelsUi({ queueExpanded: true, queueMaximized: false });
    await render();

    await click('More queue actions');
    await userEvent.click(page.getByRole('menuitem', { exact: true, name: 'Cancel all…' }));
    await expect
      .poll(() => document.querySelector('[role="alertdialog"]')?.textContent ?? '')
      .toContain('Cancel 2 installs?');
    expect(api.cancelModelInstall).not.toHaveBeenCalled();

    await click('Cancel 2 installs', document.querySelector('[role="alertdialog"]')!);
    await settle();
    expect(api.cancelModelInstall.mock.calls.map(([id]) => id).sort()).toEqual([1, 2]);
  });
});
