import fixture from '@features/generation/core/__fixtures__/architectureCapabilities.json';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({ getArchitectureCapabilities: vi.fn() }));

vi.mock('./architectureCapabilitiesApi', () => api);

const rows = fixture as unknown[];

describe('architecture capabilities store', () => {
  beforeEach(() => {
    vi.resetModules();
    api.getArchitectureCapabilities.mockReset();
  });

  it('fills the core registry rather than the snapshot', async () => {
    // The core registry is the sole synchronous table authority.
    api.getArchitectureCapabilities.mockResolvedValue(rows);
    const store = await import('./architectureCapabilitiesStore');
    const registry = await import('@features/generation/core/architectureCapabilities');

    expect(registry.hasArchitectureCapabilities()).toBe(false);
    await store.refreshArchitectureCapabilities();

    expect(store.getArchitectureCapabilitiesSnapshot()).toEqual({ error: null, revision: 1, status: 'loaded' });
    expect(registry.getArchitectureGenerationConfig('sd-1')?.defaults.steps).toBe(30);
  });

  it('dedupes concurrent refreshes', async () => {
    api.getArchitectureCapabilities.mockResolvedValue(rows);
    const store = await import('./architectureCapabilitiesStore');

    const first = store.refreshArchitectureCapabilities();
    expect(store.refreshArchitectureCapabilities()).toBe(first);
    await first;

    expect(store.getArchitectureCapabilitiesSnapshot().status).toBe('loaded');
  });

  it('reports a failure and leaves the registry empty', async () => {
    api.getArchitectureCapabilities.mockRejectedValue(new Error('outage'));
    const store = await import('./architectureCapabilitiesStore');
    const registry = await import('@features/generation/core/architectureCapabilities');

    await store.refreshArchitectureCapabilities();

    expect(store.getArchitectureCapabilitiesSnapshot()).toEqual({ error: 'outage', revision: 0, status: 'error' });
    expect(registry.hasArchitectureCapabilities()).toBe(false);
  });

  it('retries after an error but not after success', async () => {
    api.getArchitectureCapabilities.mockRejectedValueOnce(new Error('outage')).mockResolvedValue(rows);
    const store = await import('./architectureCapabilitiesStore');

    store.ensureArchitectureCapabilitiesLoaded();
    await Promise.resolve();
    await Promise.resolve();
    expect(store.getArchitectureCapabilitiesSnapshot().status).toBe('error');

    // One failed load must not stick...
    await store.refreshArchitectureCapabilities();
    expect(store.getArchitectureCapabilitiesSnapshot().status).toBe('loaded');

    // ...and a loaded table is never refetched: it is static per backend build.
    store.ensureArchitectureCapabilitiesLoaded();
    expect(api.getArchitectureCapabilities).toHaveBeenCalledTimes(2);
  });

  it('clears the registry as well as the snapshot when the account changes', async () => {
    // Clear registry rows with store status so previous-account data cannot survive.
    api.getArchitectureCapabilities.mockResolvedValue(rows);
    const lifecycle = await import('@platform/state/accountLifecycle');
    const store = await import('./architectureCapabilitiesStore');
    const registry = await import('@features/generation/core/architectureCapabilities');

    await store.refreshArchitectureCapabilities();
    expect(registry.hasArchitectureCapabilities()).toBe(true);

    lifecycle.accountLifecycle.invalidate();

    expect(store.getArchitectureCapabilitiesSnapshot()).toEqual({ error: null, revision: 0, status: 'idle' });
    expect(registry.hasArchitectureCapabilities()).toBe(false);
  });

  it('publishes the registry rather than its own request state', async () => {
    // Subscribe to the registry, not fetch status, so every table writer notifies readers.
    api.getArchitectureCapabilities.mockResolvedValue(rows);
    const store = await import('./architectureCapabilitiesStore');
    const registry = await import('@features/generation/core/architectureCapabilities');
    const seen: number[] = [];
    const unsubscribe = store.subscribeArchitectureCapabilities(() =>
      seen.push(store.getArchitectureCapabilitiesSnapshot().revision)
    );

    registry.setArchitectureCapabilities(rows as never);
    expect(store.getArchitectureCapabilitiesSnapshot().revision).toBe(1);

    registry.resetArchitectureCapabilities();
    unsubscribe();

    expect(seen).toContain(1);
    expect(store.getArchitectureCapabilitiesSnapshot().revision).toBe(0);
  });

  it('reloads the table for the account that replaces the one it was cleared for', async () => {
    // Rearm after account changes; boot effects and subscribers do not remount.
    api.getArchitectureCapabilities.mockResolvedValue(rows);
    const lifecycle = await import('@platform/state/accountLifecycle');
    const store = await import('./architectureCapabilitiesStore');
    const registry = await import('@features/generation/core/architectureCapabilities');

    store.ensureArchitectureCapabilitiesLoaded();
    await store.refreshArchitectureCapabilities();
    expect(registry.hasArchitectureCapabilities()).toBe(true);

    lifecycle.accountLifecycle.activate('account-2');
    expect(registry.hasArchitectureCapabilities()).toBe(false);

    await Promise.resolve();
    await Promise.resolve();
    expect(store.getArchitectureCapabilitiesSnapshot().status).toBe('loaded');
    expect(registry.hasArchitectureCapabilities()).toBe(true);

    lifecycle.accountLifecycle.invalidate();
  });

  it('does not fetch for a signed-out lifetime', async () => {
    api.getArchitectureCapabilities.mockResolvedValue(rows);
    const lifecycle = await import('@platform/state/accountLifecycle');
    const store = await import('./architectureCapabilitiesStore');

    store.ensureArchitectureCapabilitiesLoaded();
    await store.refreshArchitectureCapabilities();
    api.getArchitectureCapabilities.mockClear();

    lifecycle.accountLifecycle.invalidate();

    await Promise.resolve();
    expect(api.getArchitectureCapabilities).not.toHaveBeenCalled();
    expect(store.getArchitectureCapabilitiesSnapshot().status).toBe('idle');
  });
});
