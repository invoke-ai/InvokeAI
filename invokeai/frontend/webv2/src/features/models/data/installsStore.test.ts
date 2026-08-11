import { beforeEach, describe, expect, it, vi } from 'vitest';

const deferred = <T>(): { promise: Promise<T>; resolve: (value: T) => void } => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((settle) => {
    resolve = settle;
  });

  return { promise, resolve };
};

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
  it("prunes a job's transient progress when it settles", async () => {
    const store = await import('./installsStore');

    store.handleModelInstallSocketEvent('model_install_download_progress', { bytes: 25, id: 7, total_bytes: 100 });
    expect(store.getInstallProgress(7)).toEqual({ bytes: 25, totalBytes: 100 });

    store.handleModelInstallSocketEvent('model_install_complete', { config: {}, id: 7, source: 'org/model' });

    expect(store.getInstallProgress(7)).toBeNull();
  });

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
    // Catalog refreshes ride the same coalescing window as the jobs list.
    expect(dependencies.refreshModels).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(250);
    expect(dependencies.refreshModels).toHaveBeenCalledTimes(1);
    expect(dependencies.refreshStartersIfLoaded).toHaveBeenCalledTimes(1);
    expect(dependencies.listModelInstalls).toHaveBeenCalledTimes(1);
  });

  it('coalesces a burst of completions into one library refetch', async () => {
    const store = await import('./installsStore');

    for (const id of [1, 2, 3, 4]) {
      store.handleModelInstallSocketEvent('model_install_complete', { config: {}, id, source: 'org/model' });
    }

    await vi.advanceTimersByTimeAsync(250);

    expect(dependencies.refreshModels).toHaveBeenCalledTimes(1);
    expect(dependencies.refreshStartersIfLoaded).toHaveBeenCalledTimes(1);
  });

  it('drops a stale refresh without releasing the replacement account refresh', async () => {
    const first = deferred<Array<{ id: number; source: string; status: 'waiting' }>>();
    const second = deferred<Array<{ id: number; source: string; status: 'waiting' }>>();

    dependencies.listModelInstalls
      .mockReset()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise)
      // The join during the replacement flight queues a trailing rerun.
      .mockResolvedValueOnce([{ id: 2, source: 'new/account', status: 'waiting' }]);

    const store = await import('./installsStore');
    const { accountLifecycle } = await import('@platform/state/accountLifecycle');
    const staleRefresh = store.refreshInstalls();

    accountLifecycle.invalidate();

    expect(store.getInstallsSnapshot()).toEqual({ error: null, jobs: [], status: 'idle' });

    const currentRefresh = store.refreshInstalls();

    first.resolve([{ id: 1, source: 'old/account', status: 'waiting' }]);
    await staleRefresh;

    expect(store.getInstallsSnapshot()).toEqual({ error: null, jobs: [], status: 'loading' });
    expect(store.refreshInstalls()).toBe(currentRefresh);
    expect(dependencies.listModelInstalls).toHaveBeenCalledTimes(2);

    second.resolve([{ id: 2, source: 'new/account', status: 'waiting' }]);
    await currentRefresh;
    await vi.advanceTimersByTimeAsync(0);

    expect(dependencies.listModelInstalls).toHaveBeenCalledTimes(3);
    expect(store.getInstallsSnapshot()).toEqual({
      error: null,
      jobs: [{ id: 2, source: 'new/account', status: 'waiting' }],
      status: 'loaded',
    });
  });

  it('re-fetches once more when a refresh lands while another is in flight', async () => {
    const stale = deferred<Array<{ id: number; source: string; status: string }>>();

    dependencies.listModelInstalls
      .mockReset()
      .mockReturnValueOnce(stale.promise)
      .mockResolvedValueOnce([{ id: 1, source: 'org/model', status: 'completed' }]);

    const store = await import('./installsStore');

    // First completion schedules the coalesced refresh, which starts and hangs.
    store.handleModelInstallSocketEvent('model_install_complete', { config: {}, id: 1, source: 'org/model' });
    await vi.advanceTimersByTimeAsync(250);
    expect(dependencies.listModelInstalls).toHaveBeenCalledTimes(1);

    // Second completion's refresh joins the in-flight request...
    store.handleModelInstallSocketEvent('model_install_complete', { config: {}, id: 1, source: 'org/model' });
    await vi.advanceTimersByTimeAsync(250);

    // ...so when the stale response lands, one trailing re-fetch brings the fresh list.
    stale.resolve([{ id: 1, source: 'org/model', status: 'running' }]);
    await vi.advanceTimersByTimeAsync(0);

    expect(dependencies.listModelInstalls).toHaveBeenCalledTimes(2);
    expect(store.getInstallsSnapshot().jobs).toEqual([{ id: 1, source: 'org/model', status: 'completed' }]);
  });

  it('retries a failed load on the next ensure instead of sticking in error', async () => {
    dependencies.listModelInstalls
      .mockReset()
      .mockRejectedValueOnce(new Error('outage'))
      .mockResolvedValueOnce([{ id: 1, source: 'org/model', status: 'waiting' }]);

    const store = await import('./installsStore');

    store.ensureInstallsLoaded();
    await vi.advanceTimersByTimeAsync(0);
    expect(store.getInstallsSnapshot().status).toBe('error');

    store.ensureInstallsLoaded();
    await vi.advanceTimersByTimeAsync(0);
    expect(store.getInstallsSnapshot()).toEqual({
      error: null,
      jobs: [{ id: 1, source: 'org/model', status: 'waiting' }],
      status: 'loaded',
    });
    expect(dependencies.listModelInstalls).toHaveBeenCalledTimes(2);
  });

  it('ignores socket events owned by an expired account scope', async () => {
    const store = await import('./installsStore');
    const { accountLifecycle } = await import('@platform/state/accountLifecycle');
    const owner = accountLifecycle.capture();

    accountLifecycle.invalidate();
    store.handleModelInstallSocketEvent(
      'model_install_download_progress',
      { bytes: 25, id: 7, total_bytes: 100 },
      owner
    );

    expect(store.getInstallProgress(7)).toBeNull();
    expect(store.getInstallsSnapshot()).toEqual({ error: null, jobs: [], status: 'idle' });
  });
});

describe('getInstallSourceLabel', () => {
  it('passes strings through and extracts structured source fields in order', async () => {
    const store = await import('./installsStore');

    expect(store.getInstallSourceLabel('https://example.com/model.safetensors')).toBe(
      'https://example.com/model.safetensors'
    );
    expect(store.getInstallSourceLabel({ repo_id: 'owner/repo', url: 'https://x' })).toBe('owner/repo');
    expect(store.getInstallSourceLabel({ url: 'https://x' })).toBe('https://x');
    expect(store.getInstallSourceLabel({ path: '/models/x' })).toBe('/models/x');
  });

  it('falls back to a generic label for unrecognized payloads', async () => {
    const store = await import('./installsStore');

    expect(store.getInstallSourceLabel({ something: 'else' })).toBe('model');
    expect(store.getInstallSourceLabel(undefined)).toBe('model');
    expect(store.getInstallSourceLabel(42)).toBe('model');
  });
});
