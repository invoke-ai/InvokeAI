import { describe, expect, it, vi } from 'vitest';

import { createMenuPartLoader } from './Menu';

const settle = () =>
  new Promise((resolve) => {
    setTimeout(resolve, 0);
  });

describe('createMenuPartLoader', () => {
  it('imports once however often it is asked, and keeps the part', async () => {
    const importPart = vi.fn(() => Promise.resolve('part'));
    const loader = createMenuPartLoader(importPart);

    loader.load();
    loader.load();
    expect(loader.getSnapshot()).toEqual({ status: 'loading' });
    await settle();
    loader.load();

    expect(loader.getSnapshot()).toEqual({ part: 'part', status: 'loaded' });
    expect(importPart).toHaveBeenCalledOnce();
  });

  it('fails an importer that throws at once like one that rejects, and imports again on the next load', async () => {
    const importPart = vi
      .fn<() => Promise<string>>()
      .mockImplementationOnce(() => {
        throw new Error('Import scheduling failed');
      })
      .mockRejectedValueOnce(new TypeError('Failed to fetch dynamically imported module'))
      .mockResolvedValue('part');
    const loader = createMenuPartLoader(importPart);
    const listener = vi.fn();
    loader.subscribe(listener);

    expect(() => loader.load()).not.toThrow();
    await settle();
    expect(loader.getSnapshot()).toEqual({ status: 'failed' });

    loader.load();
    await settle();
    expect(loader.getSnapshot()).toEqual({ status: 'failed' });

    loader.load();
    await settle();
    expect(loader.getSnapshot()).toEqual({ part: 'part', status: 'loaded' });
    expect(importPart).toHaveBeenCalledTimes(3);
    // loading → failed, twice, then loading → loaded.
    expect(listener).toHaveBeenCalledTimes(6);
  });
});
