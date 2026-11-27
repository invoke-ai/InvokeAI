import type { ModelConfig } from '@features/models/core/types';

import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({ getFp8StorageSupport: vi.fn() }));

vi.mock('./api', () => api);

const row = (base: string, type: string, format: string, supported: boolean) => ({ base, format, supported, type });

const model = (base: string, type: string, format: string) =>
  ({ base, format, type }) as Pick<ModelConfig, 'base' | 'format' | 'type'>;

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
    expect(store.isFp8StorageSupported(snapshot, model('wan', 'main', 'gguf_quantized'))).toBe(false);
    expect(store.isFp8StorageSupported(snapshot, model('flux', 'controlnet', 'checkpoint'))).toBe(false);
  });

  it('resolves a wildcard row the way the backend resolves a wildcard loader', async () => {
    api.getFp8StorageSupport.mockResolvedValue([row('any', 't2i_adapter', 'diffusers', true)]);
    const store = await import('./fp8StorageSupportStore');

    await store.ensureFp8StorageSupportLoaded();
    const snapshot = store.getFp8StorageSupportSnapshot();

    // Every T2I adapter is served by one `any`-base registration, and no record carries `any` as its own base. Exact
    // key first, then the wildcard — the order `get_implementation` uses.
    expect(store.isFp8StorageSupported(snapshot, model('sd-1', 't2i_adapter', 'diffusers'))).toBe(true);
    expect(store.isFp8StorageSupported(snapshot, model('sdxl', 't2i_adapter', 'diffusers'))).toBe(true);
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

    expect(store.isFp8StorageSupported(snapshot, model('flux', 't2i_adapter', 'diffusers'))).toBe(false);
    expect(store.isFp8StorageSupported(snapshot, model('sd-1', 't2i_adapter', 'diffusers'))).toBe(true);
  });

  it('answers no for a model the table does not mention', async () => {
    api.getFp8StorageSupport.mockResolvedValue(WAN_ROWS);
    const store = await import('./fp8StorageSupportStore');

    await store.ensureFp8StorageSupportLoaded();

    expect(store.isFp8StorageSupported(store.getFp8StorageSupportSnapshot(), model('sdxl', 'vae', 'diffusers'))).toBe(
      false
    );
  });

  it('answers no before the table has arrived', async () => {
    api.getFp8StorageSupport.mockResolvedValue(WAN_ROWS);
    const store = await import('./fp8StorageSupportStore');

    expect(store.isFp8StorageSupported(store.getFp8StorageSupportSnapshot(), model('wan', 'main', 'checkpoint'))).toBe(
      false
    );
  });

  it('shares one request and retries after a failure', async () => {
    api.getFp8StorageSupport.mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce(WAN_ROWS);
    const store = await import('./fp8StorageSupportStore');

    const first = store.ensureFp8StorageSupportLoaded();
    expect(store.ensureFp8StorageSupportLoaded()).toBe(first);
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

    expect(api.getFp8StorageSupport).toHaveBeenCalledTimes(1);
  });
});
