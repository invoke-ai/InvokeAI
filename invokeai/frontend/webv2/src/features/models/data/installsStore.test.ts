import { beforeEach, describe, expect, it, vi } from 'vitest';

const dependencies = vi.hoisted(() => ({
  listModelInstalls: vi.fn(),
  refreshModels: vi.fn(),
  refreshStartersIfLoaded: vi.fn(),
}));

vi.mock('./api', () => ({ listModelInstalls: dependencies.listModelInstalls }));
vi.mock('./modelsStore', () => ({ refreshModels: dependencies.refreshModels }));
vi.mock('./startersStore', () => ({ refreshStartersIfLoaded: dependencies.refreshStartersIfLoaded }));

beforeEach(() => {
  vi.resetModules();
  vi.useFakeTimers();
  dependencies.listModelInstalls.mockReset().mockResolvedValue([]);
  dependencies.refreshModels.mockReset().mockResolvedValue(undefined);
  dependencies.refreshStartersIfLoaded.mockReset();
});

describe('model install event interpretation', () => {
  it('projects download progress immediately and coalesces the REST refresh', async () => {
    const store = await import('./installsStore');

    store.addInstallJob({ id: 7, source: 'org/model', status: 'waiting' });
    store.handleModelInstallSocketEvent('model_install_download_progress', {
      bytes: 25,
      id: 7,
      total_bytes: 100,
    });

    expect(store.getInstallsSnapshot().jobs[0]?.status).toBe('downloading');
    expect(store.getInstallProgress(7)).toEqual({ bytes: 25, totalBytes: 100 });

    store.handleModelInstallSocketEvent('model_install_started', { id: 7 });
    store.handleModelInstallSocketEvent('model_install_download_started', { id: 7 });
    await vi.advanceTimersByTimeAsync(250);

    expect(dependencies.listModelInstalls).toHaveBeenCalledTimes(1);
  });

  it('records completion identity and refreshes catalog capabilities', async () => {
    const store = await import('./installsStore');

    store.handleModelInstallSocketEvent('model_install_complete', {
      config: { name: 'Installed Model' },
      id: 8,
      source: { repo_id: 'org/model' },
    });

    expect(store.getInstallOutcomes()[0]).toMatchObject({
      error: null,
      jobId: 8,
      kind: 'completed',
      modelName: 'Installed Model',
      source: 'org/model',
    });
    expect(dependencies.refreshModels).toHaveBeenCalledTimes(1);
    expect(dependencies.refreshStartersIfLoaded).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(250);
    expect(dependencies.listModelInstalls).toHaveBeenCalledTimes(1);
  });
});
