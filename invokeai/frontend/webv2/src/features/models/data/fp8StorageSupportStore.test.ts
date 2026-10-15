import type { ModelConfig } from '@features/models/core/types';

import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({ getFp8StorageSupport: vi.fn() }));

vi.mock('./api', () => api);

const row = (base: string, type: string, format: string, supported: boolean) => ({ base, format, supported, type });

const model = (base: string, type: string, format: string) =>
  ({ base, format, type }) as Pick<ModelConfig, 'base' | 'format' | 'type'>;

/** The two rows that make the case for the key: same base, same type, opposite answers by format. */
const WAN_ROWS = [
  row('wan', 'main', 'checkpoint', true),
  row('wan', 'main', 'gguf_quantized', false),
  row('flux', 'controlnet', 'checkpoint', false),
];

describe('fp8 storage support store', () => {
  beforeEach(() => {
    vi.resetModules();
    api.getFp8StorageSupport.mockReset();
  });

  it('answers per (base, type, format) once the table is loaded', async () => {
    api.getFp8StorageSupport.mockResolvedValue(WAN_ROWS);
    const store = await import('./fp8StorageSupportStore');

    await store.ensureFp8StorageSupportLoaded();
    const snapshot = store.getFp8StorageSupportSnapshot();

    expect(snapshot.status).toBe('loaded');
    expect(store.isFp8StorageSupported(snapshot, model('wan', 'main', 'checkpoint'))).toBe(true);
    // The format alone flips it: the same architecture's packed weights must never be re-encoded.
    expect(store.isFp8StorageSupported(snapshot, model('wan', 'main', 'gguf_quantized'))).toBe(false);
    // And the type alone flips it for FLUX, which is why neither half of the key can be dropped.
    expect(store.isFp8StorageSupported(snapshot, model('flux', 'controlnet', 'checkpoint'))).toBe(false);
  });

  it('resolves a wildcard row the way the backend resolves a wildcard loader', async () => {
    api.getFp8StorageSupport.mockResolvedValue([row('any', 't2i_adapter', 'diffusers', true)]);
    const store = await import('./fp8StorageSupportStore');

    await store.ensureFp8StorageSupportLoaded();
    const snapshot = store.getFp8StorageSupportSnapshot();

    // Every T2I adapter is served by one `any`-base registration, and no record carries `any` as its
    // own base. Exact key first, then the wildcard — the order `get_implementation` uses.
    expect(store.isFp8StorageSupported(snapshot, model('sd-1', 't2i_adapter', 'diffusers'))).toBe(true);
    expect(store.isFp8StorageSupported(snapshot, model('sdxl', 't2i_adapter', 'diffusers'))).toBe(true);
    // The wildcard does not leak across the rest of the key.
    expect(store.isFp8StorageSupported(snapshot, model('sd-1', 'main', 'diffusers'))).toBe(false);
  });

  it('prefers an exact row over the wildcard', async () => {
    api.getFp8StorageSupport.mockResolvedValue([
      row('any', 't2i_adapter', 'diffusers', true),
      row('flux', 't2i_adapter', 'diffusers', false),
    ]);
    const store = await import('./fp8StorageSupportStore');

    await store.ensureFp8StorageSupportLoaded();
    const snapshot = store.getFp8StorageSupportSnapshot();

    // A loader registered for one architecture overrides the generic one, so its answer has to win —
    // including when it is the more restrictive of the two.
    expect(store.isFp8StorageSupported(snapshot, model('flux', 't2i_adapter', 'diffusers'))).toBe(false);
    expect(store.isFp8StorageSupported(snapshot, model('sd-1', 't2i_adapter', 'diffusers'))).toBe(true);
  });

  it('answers no for a model the table does not mention', async () => {
    api.getFp8StorageSupport.mockResolvedValue(WAN_ROWS);
    const store = await import('./fp8StorageSupportStore');

    await store.ensureFp8StorageSupportLoaded();

    // A VAE gets no row, and neither does a loader key the backend does not serve. Both mean the
    // control would do nothing, which is the same conclusion the server reaches for them.
    expect(store.isFp8StorageSupported(store.getFp8StorageSupportSnapshot(), model('sdxl', 'vae', 'diffusers'))).toBe(
      false
    );
  });

  it('answers no before the table has arrived', async () => {
    api.getFp8StorageSupport.mockResolvedValue(WAN_ROWS);
    const store = await import('./fp8StorageSupportStore');

    // A control that appears a moment late is better than one that turns out to be inert, so unknown
    // reads as unsupported rather than optimistically as supported.
    expect(store.isFp8StorageSupported(store.getFp8StorageSupportSnapshot(), model('wan', 'main', 'checkpoint'))).toBe(
      false
    );
  });

  it('shares one request and retries after a failure', async () => {
    api.getFp8StorageSupport.mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce(WAN_ROWS);
    const store = await import('./fp8StorageSupportStore');

    const first = store.ensureFp8StorageSupportLoaded();
    expect(store.ensureFp8StorageSupportLoaded()).toBe(first);
    // Swallowed rather than rethrown: the consumer hides the row and has no message to show. The
    // status is what lets the next mount try again instead of waiting forever.
    await expect(first).resolves.toBeUndefined();
    expect(store.getFp8StorageSupportSnapshot().status).toBe('error');

    await store.ensureFp8StorageSupportLoaded();
    expect(store.getFp8StorageSupportSnapshot().status).toBe('loaded');
    expect(api.getFp8StorageSupport).toHaveBeenCalledTimes(2);
  });

  it('does not refetch a table it already holds', async () => {
    api.getFp8StorageSupport.mockResolvedValue(WAN_ROWS);
    const store = await import('./fp8StorageSupportStore');

    await store.ensureFp8StorageSupportLoaded();
    await store.ensureFp8StorageSupportLoaded();

    // Static per backend build: a second mount of the detail view must not hit the network again.
    expect(api.getFp8StorageSupport).toHaveBeenCalledTimes(1);
  });
});
