import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({
  getExternalProviderConfigs: vi.fn(),
  resetExternalProviderConfig: vi.fn(),
  setExternalProviderConfig: vi.fn(),
}));

vi.mock('./api', () => api);

const config = (providerId: string, configured = false) => ({
  api_key_configured: configured,
  base_url: null,
  provider_id: providerId,
});

describe('external providers store', () => {
  beforeEach(() => {
    vi.resetModules();
    api.getExternalProviderConfigs.mockReset();
    api.resetExternalProviderConfig.mockReset();
    api.setExternalProviderConfig.mockReset();
  });

  it('shares one request and retries after an initial failure', async () => {
    api.getExternalProviderConfigs
      .mockRejectedValueOnce(new Error('temporary outage'))
      .mockResolvedValueOnce([config('openai')]);
    const store = await import('./externalProvidersStore');

    const first = store.ensureExternalProvidersLoaded();
    expect(store.ensureExternalProvidersLoaded()).toBe(first);
    await expect(first).rejects.toThrow('temporary outage');
    expect(store.getExternalProvidersSnapshot()).toMatchObject({ error: 'temporary outage', status: 'error' });

    await store.ensureExternalProvidersLoaded();
    expect(store.getExternalProvidersSnapshot()).toMatchObject({
      configs: [config('openai')],
      error: null,
      status: 'loaded',
    });
    expect(api.getExternalProviderConfigs).toHaveBeenCalledTimes(2);
  });

  it('patches the snapshot when a config is saved or cleared', async () => {
    api.getExternalProviderConfigs.mockResolvedValue([config('openai'), config('gemini')]);
    api.setExternalProviderConfig.mockResolvedValue(config('openai', true));
    api.resetExternalProviderConfig.mockResolvedValue(config('openai'));
    const store = await import('./externalProvidersStore');

    await store.ensureExternalProvidersLoaded();

    await expect(store.saveExternalProviderConfig('openai', { api_key: 'sk-test' })).resolves.toMatchObject({
      api_key_configured: true,
    });
    expect(store.getExternalProvidersSnapshot().configs).toEqual([config('openai', true), config('gemini')]);

    await store.clearExternalProviderConfig('openai');
    expect(store.getExternalProvidersSnapshot().configs).toEqual([config('openai'), config('gemini')]);
  });

  it('clears the cache on account switch', async () => {
    const account = await import('@platform/state/accountLifecycle');
    account.accountLifecycle.activate('user-a');
    api.getExternalProviderConfigs.mockResolvedValue([config('openai')]);
    const store = await import('./externalProvidersStore');

    await store.ensureExternalProvidersLoaded();
    expect(store.getExternalProvidersSnapshot().status).toBe('loaded');

    account.accountLifecycle.invalidate();
    account.accountLifecycle.activate('user-b');

    expect(store.getExternalProvidersSnapshot()).toMatchObject({ configs: null, status: 'idle' });
  });
});
