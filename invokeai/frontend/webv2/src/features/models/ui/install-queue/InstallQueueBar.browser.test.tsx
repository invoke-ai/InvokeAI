import type { ModelConfig, ModelInstallJob } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { handleModelInstallSocketEvent, refreshInstalls } from '@features/models/data/installsStore';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { getModelsUiSnapshotForTests, updateModelsUi } from '@features/models/ui/uiStore';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { setConnectionStatus } from '@platform/transport/connectionStore';
import { ApiError } from '@platform/transport/http';
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

    // The backend auto-paused the job after the part failed; only the refetch carries the errored part.
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
    // A transient failure keeps its partial file: resume first, restarts stay available but explicit.
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

  it('offers only a restart for a paused download whose server refused to resume', async () => {
    updateModelsUi({ queueExpanded: true, queueMaximized: false });
    api.listModelInstalls.mockResolvedValue(
      JOBS.map((job) =>
        job.id === 1
          ? {
              ...job,
              download_parts: [
                {
                  bytes: 4,
                  resume_message: 'Resume refused by server. Restart required.',
                  resume_required: true,
                  source: 'https://example.com/weights.safetensors',
                  status: 'paused',
                  total_bytes: 100,
                },
              ],
              status: 'paused' as const,
            }
          : job
      )
    );
    api.restartFailedModelInstall.mockResolvedValue({ ...JOBS[0], status: 'waiting' });
    await render();

    const pausedRow = rowByStatus('paused');
    expect(pausedRow?.textContent).toContain('Resume refused by server. Restart required.');
    expect(pausedRow?.querySelector('button[aria-label="Resume download"]')).toBeNull();
    await click('Retry failed download', pausedRow);
    await settle();

    expect(api.restartFailedModelInstall).toHaveBeenCalledWith(1, expect.any(AbortSignal));
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

  it('dismisses jobs the server prunes even when their completion socket event was missed', async () => {
    updateModelsUi({ queueExpanded: true, queueMaximized: false });
    api.pruneCompletedModelInstalls.mockResolvedValue([1, 3, 4, 5, 6]);
    await render();

    await click('Clear finished');
    await settle();

    expect(rowByStatus('installed')).toBeUndefined();
    expect(api.pruneCompletedModelInstalls).toHaveBeenCalledTimes(1);
  });

  it('removes finished rows when pruning succeeds on the server but its response is lost', async () => {
    updateModelsUi({ queueExpanded: true, queueMaximized: false });
    api.listModelInstalls
      .mockReset()
      .mockResolvedValueOnce(JOBS)
      .mockResolvedValueOnce(JOBS.filter(({ status }) => !['completed', 'error'].includes(status)));
    api.pruneCompletedModelInstalls.mockRejectedValue(new Error('Prune response lost'));
    await render();
    expect(rows()).toHaveLength(JOBS.length);

    await click('Clear finished');
    await settle();

    expect(rows().map((row) => row.dataset.installStatus)).toEqual(['downloading', 'queued', 'cancelled']);
    expect(api.pruneCompletedModelInstalls).toHaveBeenCalledTimes(1);
    expect(api.listModelInstalls).toHaveBeenCalledTimes(2);
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

  it('keeps jobs that settle after a prune request started', async () => {
    updateModelsUi({ queueExpanded: true, queueMaximized: false });
    let resolvePrune!: (jobIds: number[]) => void;
    const afterPrune = [{ ...JOBS[0]!, status: 'completed' as const }, JOBS[1]!];
    api.listModelInstalls.mockReset().mockResolvedValueOnce(JOBS).mockResolvedValueOnce(afterPrune);
    api.pruneCompletedModelInstalls.mockReturnValue(
      new Promise<number[]>((resolve) => {
        resolvePrune = resolve;
      })
    );
    await render();

    await click('Clear finished');
    act(() => handleModelInstallSocketEvent('model_install_complete', { id: 1, source: 'org/model', config: {} }));
    await act(async () => {
      resolvePrune([3, 4, 5, 6]);
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 20);
      });
    });

    expect(rowByStatus('installed')).toBeDefined();
    expect(rows()).toHaveLength(2);
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

  it('projects an active cancel response as cancelled over an older list when refresh fails', async () => {
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
    api.cancelModelInstall.mockResolvedValue({ ...JOBS[0]!, status: 'downloading' });
    await render();

    act(() => {
      void refreshInstalls();
    });
    await click('Cancel install', rowByStatus('downloading'));
    await settle();
    await act(async () => {
      resolveStale(JOBS);
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 20);
      });
    });

    expect(rowByStatus('cancelled')).toBeDefined();
    expect(rowByStatus('downloading')).toBeUndefined();
  });

  it('refreshes after cancel succeeds but its follow-up state fetch fails', async () => {
    updateModelsUi({ queueExpanded: true, queueMaximized: false });
    api.listModelInstalls
      .mockReset()
      .mockResolvedValueOnce(JOBS)
      .mockResolvedValueOnce(JOBS.map((job) => (job.id === 1 ? { ...job, status: 'cancelled' as const } : job)));
    api.cancelModelInstall.mockRejectedValue(new ApiError('Could not fetch cancel state', 500));
    await render();

    await click('Cancel install', rowByStatus('downloading'));
    await settle();

    expect(api.listModelInstalls).toHaveBeenCalledTimes(2);
    expect(rowByStatus('cancelled')).toBeDefined();
    expect(rowByStatus('downloading')).toBeUndefined();
  });

  it('refreshes after a concurrent prune makes cancel return 415', async () => {
    updateModelsUi({ queueExpanded: true, queueMaximized: false });
    api.listModelInstalls.mockReset().mockResolvedValueOnce(JOBS).mockResolvedValueOnce(JOBS.slice(1));
    api.cancelModelInstall.mockRejectedValue(new ApiError('Install job is no longer available', 415));
    await render();

    await click('Cancel install', rowByStatus('downloading'));
    await settle();

    expect(api.listModelInstalls).toHaveBeenCalledTimes(2);
    expect(rowByStatus('downloading')).toBeUndefined();
    expect(rows()).toHaveLength(JOBS.length - 1);
  });

  it('stops with a stop icon, removes with an X, and asks before discarding a download that has progress', async () => {
    updateModelsUi({ queueExpanded: true, queueMaximized: false });
    api.listModelInstalls.mockResolvedValue(JOBS.map((job) => (job.id === 1 ? { ...job, bytes: 2_000_000_000 } : job)));
    await render();

    const cancel = rowByStatus('downloading')?.querySelector('button[aria-label="Cancel install"] svg');
    const remove = rowByStatus('cancelled')?.querySelector('button[aria-label="Remove from list"] svg');
    // Stopping work and tidying the list must not look alike.
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

  it('refreshes after a 409 resume response that may have changed backend status', async () => {
    updateModelsUi({ queueExpanded: true, queueMaximized: false });
    api.listModelInstalls
      .mockReset()
      .mockResolvedValueOnce(JOBS.map((job) => (job.id === 6 ? { ...job, status: 'paused' } : job)))
      .mockResolvedValue(JOBS.map((job) => (job.id === 6 ? { ...job, status: 'error' } : job)));
    api.resumeModelInstall.mockRejectedValue(new ApiError('Install changed state', 409));
    await render();

    const before = api.listModelInstalls.mock.calls.length;
    await click('Resume download', rowByStatus('paused'));
    await settle();

    expect(api.listModelInstalls.mock.calls.length).toBeGreaterThan(before);
    expect(rowByStatus('failed')).toBeDefined();
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
