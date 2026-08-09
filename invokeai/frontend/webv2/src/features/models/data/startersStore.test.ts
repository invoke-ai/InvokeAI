import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({ getStarterModels: vi.fn() }));

vi.mock('./api', () => api);

const response = { starter_bundles: {}, starter_models: [] };

describe('starters store', () => {
  beforeEach(() => {
    vi.resetModules();
    api.getStarterModels.mockReset();
  });

  it('dedupes concurrent refreshes and keeps the catalog on later failures', async () => {
    api.getStarterModels.mockResolvedValueOnce(response).mockRejectedValueOnce(new Error('outage'));
    const store = await import('./startersStore');

    const first = store.refreshStarters();
    expect(store.refreshStarters()).toBe(first);
    await first;
    expect(store.getStartersSnapshot()).toMatchObject({ error: null, response, status: 'loaded' });

    await store.refreshStarters();
    expect(store.getStartersSnapshot()).toMatchObject({ error: 'outage', response, status: 'loaded' });
    expect(api.getStarterModels).toHaveBeenCalledTimes(2);
  });

  it('ensures once and revalidates only when already loaded', async () => {
    api.getStarterModels.mockResolvedValue(response);
    const store = await import('./startersStore');

    store.refreshStartersIfLoaded();
    expect(api.getStarterModels).not.toHaveBeenCalled();

    store.ensureStartersLoaded();
    store.ensureStartersLoaded();
    await Promise.resolve();
    expect(api.getStarterModels).toHaveBeenCalledTimes(1);
  });

  it('clears the catalog on account switch', async () => {
    const account = await import('@platform/state/accountLifecycle');
    account.accountLifecycle.activate('user-a');
    api.getStarterModels.mockResolvedValue(response);
    const store = await import('./startersStore');

    await store.refreshStarters();
    expect(store.getStartersSnapshot().status).toBe('loaded');

    account.accountLifecycle.invalidate();
    account.accountLifecycle.activate('user-b');

    expect(store.getStartersSnapshot()).toMatchObject({ response: null, status: 'idle' });
  });
});
