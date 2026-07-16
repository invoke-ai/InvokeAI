import { describe, expect, it, vi } from 'vitest';

import type { FontLoadApi } from './fontLoader';

import { createFontLoader } from './fontLoader';

/** A fake fonts api with a controllable `check` and a deferred `load`. */
const createFakeFonts = (initiallyLoaded = false) => {
  let loaded = initiallyLoaded;
  let resolveLoad: (() => void) | null = null;
  const load = vi.fn(
    (_font: string): Promise<unknown> =>
      new Promise<void>((resolve) => {
        resolveLoad = () => {
          loaded = true;
          resolve();
        };
      })
  );
  const api: FontLoadApi = {
    check: () => loaded,
    load,
  };
  return { api, load, settle: () => resolveLoad?.() };
};

const flush = (): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, 0);
  });

describe('createFontLoader', () => {
  it('is a silent no-op with no api (node-safe)', () => {
    const loader = createFontLoader(null);
    const onReady = vi.fn();
    loader.ensure('400 20px Inter', onReady);
    expect(onReady).not.toHaveBeenCalled();
  });

  it('does not load or call onReady when the font is already available', () => {
    const { api, load } = createFakeFonts(true);
    const loader = createFontLoader(api);
    const onReady = vi.fn();
    loader.ensure('400 20px Inter', onReady);
    expect(load).not.toHaveBeenCalled();
    expect(onReady).not.toHaveBeenCalled();
  });

  it('kicks a load for an unavailable font and calls onReady when it resolves', async () => {
    const { api, load, settle } = createFakeFonts(false);
    const loader = createFontLoader(api);
    const onReady = vi.fn();
    loader.ensure('400 20px Inter', onReady);
    expect(load).toHaveBeenCalledTimes(1);
    expect(onReady).not.toHaveBeenCalled();
    settle();
    await flush();
    expect(onReady).toHaveBeenCalledTimes(1);
  });

  it('dedupes concurrent loads of the same font (one load, both onReady fire)', async () => {
    const { api, load, settle } = createFakeFonts(false);
    const loader = createFontLoader(api);
    const onReadyA = vi.fn();
    const onReadyB = vi.fn();
    loader.ensure('400 20px Inter', onReadyA);
    loader.ensure('400 20px Inter', onReadyB);
    expect(load).toHaveBeenCalledTimes(1);
    settle();
    await flush();
    expect(onReadyA).toHaveBeenCalledTimes(1);
    expect(onReadyB).toHaveBeenCalledTimes(1);
  });

  it('swallows a check() that throws (treats the font as unavailable)', () => {
    const api: FontLoadApi = {
      check: () => {
        throw new Error('bad font string');
      },
      load: vi.fn(() => Promise.resolve()),
    };
    const loader = createFontLoader(api);
    expect(() => loader.ensure('garbage', vi.fn())).not.toThrow();
    expect(api.load).toHaveBeenCalledTimes(1);
  });
});
