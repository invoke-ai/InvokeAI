import { describe, expect, it, vi } from 'vitest';

import type { CanvasFontRuntime, FontLoadApi } from './fontLoader';

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

const customText = {
  align: 'left' as const,
  color: '#fff',
  content: 'hello',
  fontFamily: 'Catalog Family',
  fontRef: { contentHash: 'hash-1', family: 'Catalog Family', id: 'font-1', label: 'Catalog Regular' },
  fontSize: 32,
  fontStyle: 'oblique' as const,
  fontVariations: { opsz: 14, wght: 612 },
  fontWeight: 612,
  lineHeight: 1.2,
  type: 'text' as const,
};

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

  it('does not call onReady again after a completed load', async () => {
    const { api, load, settle } = createFakeFonts(false);
    const loader = createFontLoader(api);
    const first = vi.fn();
    const second = vi.fn();
    loader.ensure('400 20px Inter', first);
    settle();
    await flush();
    loader.ensure('400 20px Inter', second);
    await flush();
    expect(load).toHaveBeenCalledTimes(1);
    expect(first).toHaveBeenCalledTimes(1);
    expect(second).not.toHaveBeenCalled();
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

  it('passes the stable reference and exact typography coordinates to the custom runtime', async () => {
    let resolveEnsure!: (family: string) => void;
    const runtime: CanvasFontRuntime = {
      ensure: vi.fn(
        () =>
          new Promise<string>((resolve) => {
            resolveEnsure = resolve;
          })
      ),
      ensureForOutput: vi.fn(() => Promise.resolve('internal-font-family')),
      resolveFamily: vi.fn(() => 'internal-font-family'),
    };
    const loader = createFontLoader(runtime);
    const onReady = vi.fn();

    loader.ensure(customText, onReady);

    expect(loader.resolveFamily(customText)).toBe('internal-font-family');
    expect(runtime.ensure).toHaveBeenCalledWith(
      {
        axes: { opsz: 14, wght: 612 },
        contentHash: 'hash-1',
        family: 'Catalog Family',
        id: 'font-1',
        style: 'oblique',
        weight: 612,
      },
      expect.any(AbortSignal)
    );
    resolveEnsure('internal-font-family');
    await flush();
    expect(onReady).toHaveBeenCalledOnce();
  });

  it('keeps a shared preview alive after one caller cancels until its source is superseded', async () => {
    let resolveEnsure!: (family: string) => void;
    let runtimeSignal!: AbortSignal;
    const runtime: CanvasFontRuntime = {
      ensure: vi.fn((_reference, signal?: AbortSignal) => {
        runtimeSignal = signal!;
        return new Promise<string>((resolve) => {
          resolveEnsure = resolve;
        });
      }),
      ensureForOutput: vi.fn(() => Promise.resolve('validated-family')),
      resolveFamily: () => 'fallback',
    };
    const loader = createFontLoader(runtime);
    loader.setActiveSources([customText]);
    const caller = new AbortController();
    const preview = loader.ensurePreview(customText, caller.signal);

    caller.abort();
    await expect(preview).rejects.toMatchObject({ name: 'AbortError' });
    expect(runtimeSignal.aborted).toBe(false);

    loader.setActiveSources([]);
    expect(runtimeSignal.aborted).toBe(true);
    resolveEnsure('obsolete-family');
    loader.dispose();
  });

  it('cancels an obsolete preview without allowing its late completion to notify', async () => {
    let resolveEnsure!: (family: string) => void;
    let runtimeSignal!: AbortSignal;
    const runtime: CanvasFontRuntime = {
      ensure: vi.fn((_reference, signal?: AbortSignal) => {
        runtimeSignal = signal!;
        return new Promise<string>((resolve) => {
          resolveEnsure = resolve;
        });
      }),
      ensureForOutput: vi.fn(() => Promise.resolve('validated-family')),
      resolveFamily: () => 'fallback',
    };
    const loader = createFontLoader(runtime);
    const onReady = vi.fn();
    loader.setActiveSources([customText]);
    loader.ensure(customText, onReady);
    loader.setActiveSources([]);
    expect(runtimeSignal.aborted).toBe(true);
    resolveEnsure('late-family');
    await flush();
    expect(onReady).not.toHaveBeenCalled();
    loader.dispose();
  });

  it('does not fence a ready result when the runtime announces ordinary loading state', async () => {
    let notifyRuntime!: () => void;
    let resolvedFamily = 'fallback';
    const runtime: CanvasFontRuntime = {
      ensure: vi.fn(() => {
        notifyRuntime();
        resolvedFamily = 'internal-font-family';
        return Promise.resolve('internal-font-family');
      }),
      ensureForOutput: vi.fn(() => Promise.resolve('internal-font-family')),
      getSnapshot: () => ({ generation: 0 }),
      resolveFamily: () => resolvedFamily,
      subscribe: (listener) => {
        notifyRuntime = listener;
        return () => undefined;
      },
    };
    const loader = createFontLoader(runtime);
    const onReady = vi.fn();
    loader.ensure(customText, onReady);
    await flush();
    expect(onReady).toHaveBeenCalledOnce();
    loader.ensure(customText, onReady);
    await flush();
    expect(onReady).toHaveBeenCalledOnce();
  });

  it('reloads a custom preview after the runtime evicts its cached family', async () => {
    let resolvedFamily = 'fallback';
    const runtime: CanvasFontRuntime = {
      ensure: vi.fn(() => {
        resolvedFamily = 'loaded-family';
        return Promise.resolve(resolvedFamily);
      }),
      ensureForOutput: vi.fn(() => Promise.resolve('validated-family')),
      getSnapshot: () => ({ generation: 0 }),
      resolveFamily: () => resolvedFamily,
    };
    const loader = createFontLoader(runtime);
    loader.setActiveSources([customText]);

    await expect(loader.ensurePreview(customText)).resolves.toBe('loaded-family');
    expect(runtime.ensure).toHaveBeenCalledOnce();

    // Runtime LRU eviction removes its browser alias without changing the
    // account generation. A later preview must consult that authoritative
    // resolver instead of trusting the loader's stale ready entry.
    resolvedFamily = 'fallback';
    await expect(loader.ensurePreview(customText)).resolves.toBe('loaded-family');
    expect(runtime.ensure).toHaveBeenCalledTimes(2);
    loader.dispose();
  });

  it('shares custom-face readiness with output gating and subscriptions', async () => {
    let resolveOutput!: (family: string) => void;
    const runtime: CanvasFontRuntime = {
      ensure: vi.fn(() => Promise.resolve('preview-family')),
      ensureForOutput: vi.fn(
        () =>
          new Promise<string>((resolve) => {
            resolveOutput = resolve;
          })
      ),
      resolveFamily: () => 'fallback',
    };
    const loader = createFontLoader(runtime);
    const notified = vi.fn();
    loader.subscribe(notified);

    const ready = loader.waitForReady(customText);
    expect(runtime.ensureForOutput).toHaveBeenCalledOnce();
    resolveOutput('internal-font-family');
    await expect(ready).resolves.toBe('internal-font-family');
    expect(notified).toHaveBeenCalled();
  });

  it('keeps document-active custom faces registered and releases obsolete variations', async () => {
    const release = vi.fn();
    const runtime: CanvasFontRuntime = {
      ensure: vi.fn(() => Promise.resolve('internal-font-family')),
      ensureForOutput: vi.fn(() => Promise.resolve('internal-font-family')),
      retain: vi.fn(() => release),
      resolveFamily: vi.fn(() => 'internal-font-family'),
    };
    const loader = createFontLoader(runtime);
    loader.setActiveSources([customText]);

    await expect(loader.waitForReady(customText)).resolves.toBe('internal-font-family');
    expect(runtime.retain).toHaveBeenCalledWith({
      axes: { opsz: 14, wght: 612 },
      contentHash: 'hash-1',
      family: 'Catalog Family',
      id: 'font-1',
      style: 'oblique',
      weight: 612,
    });
    const nextText = { ...customText, fontVariations: { opsz: 16, wght: 612 } };
    loader.setActiveSources([nextText]);
    expect(release).toHaveBeenCalledOnce();
    loader.dispose();
    expect(release).toHaveBeenCalledOnce();
  });

  it('revalidates a cached custom face for each output readiness check', async () => {
    const runtime: CanvasFontRuntime = {
      ensure: vi
        .fn()
        .mockResolvedValueOnce('internal-font-family')
        .mockRejectedValueOnce(new Error('font was deleted')),
      ensureForOutput: vi
        .fn()
        .mockResolvedValueOnce('internal-font-family')
        .mockRejectedValueOnce(new Error('font was deleted')),
      resolveFamily: () => 'internal-font-family',
    };
    const loader = createFontLoader(runtime);
    loader.setActiveSources([customText]);

    await expect(loader.waitForReady(customText)).resolves.toBe('internal-font-family');
    await expect(loader.waitForReady(customText)).rejects.toThrow('font was deleted');
    expect(runtime.ensureForOutput).toHaveBeenCalledTimes(2);
  });

  it('uses the runtime output check instead of the preview cache', async () => {
    const runtime: CanvasFontRuntime = {
      ensure: vi.fn().mockResolvedValue('cached-family'),
      ensureForOutput: vi.fn().mockRejectedValue(new Error('font metadata changed')),
      resolveFamily: () => 'cached-family',
    };
    const loader = createFontLoader(runtime);

    await expect(loader.waitForReady(customText)).rejects.toThrow('font metadata changed');
    expect(runtime.ensure).not.toHaveBeenCalled();
    expect(runtime.ensureForOutput).toHaveBeenCalledOnce();
  });

  it('does not reuse an in-flight preview request for output readiness', async () => {
    let resolvePreview!: (family: string) => void;
    const runtime: CanvasFontRuntime = {
      ensure: vi.fn(
        () =>
          new Promise<string>((resolve) => {
            resolvePreview = resolve;
          })
      ),
      ensureForOutput: vi.fn().mockResolvedValue('validated-family'),
      resolveFamily: () => 'fallback',
    };
    const loader = createFontLoader(runtime);

    loader.ensure(customText, vi.fn());
    await expect(loader.waitForReady(customText)).resolves.toBe('validated-family');
    expect(runtime.ensureForOutput).toHaveBeenCalledOnce();
    expect(runtime.ensure).toHaveBeenCalledOnce();
    resolvePreview('preview-family');
  });

  it('notifies subscribers when a runtime without its own event stream reports an error', async () => {
    const runtime: CanvasFontRuntime = {
      ensure: vi.fn().mockRejectedValue(new Error('font unavailable')),
      ensureForOutput: vi.fn().mockRejectedValue(new Error('font unavailable')),
      resolveFamily: () => 'fallback',
    };
    const loader = createFontLoader(runtime);
    const notified = vi.fn();
    loader.subscribe(notified);

    await expect(loader.waitForReady(customText)).rejects.toThrow('font unavailable');
    expect(notified).toHaveBeenCalledOnce();
  });

  it('fails closed for a legacy custom runtime without output revalidation', async () => {
    const runtime = {
      ensure: vi.fn().mockResolvedValue('legacy-family'),
      resolveFamily: () => 'legacy-family',
    } as unknown as CanvasFontRuntime;
    const loader = createFontLoader(runtime);

    await expect(loader.waitForReady(customText)).rejects.toThrow('output validation');
    expect(runtime.ensure).not.toHaveBeenCalled();
  });

  it('fails closed for a custom reference when no runtime is available', async () => {
    const loader = createFontLoader(null);
    await expect(loader.waitForReady(customText)).rejects.toThrow('custom Canvas font runtime');
    const onReady = vi.fn();
    loader.ensure(customText, onReady);
    await flush();
    expect(onReady).not.toHaveBeenCalled();
  });

  it('fences a stale custom load after the runtime invalidates its account scope', async () => {
    let invalidate!: () => void;
    const resolves: Array<(family: string) => void> = [];
    const release = vi.fn();
    const runtime: CanvasFontRuntime = {
      ensure: vi.fn(
        () =>
          new Promise<string>((resolve) => {
            resolves.push(resolve);
          })
      ),
      ensureForOutput: vi.fn(
        () =>
          new Promise<string>((resolve) => {
            resolves.push(resolve);
          })
      ),
      retain: vi.fn(() => release),
      resolveFamily: vi.fn(() => 'internal-font-family'),
      subscribe: vi.fn((listener) => {
        invalidate = listener;
        return () => undefined;
      }),
    };
    const loader = createFontLoader(runtime);
    loader.setActiveSources([customText]);
    const first = loader.waitForReady(customText);

    invalidate();
    const second = loader.waitForReady(customText);
    resolves[0]!('stale-family');
    await expect(first).resolves.toBe('stale-family');
    expect(runtime.retain).not.toHaveBeenCalled();

    resolves[1]!('current-family');
    await expect(second).resolves.toBe('current-family');
    expect(runtime.retain).toHaveBeenCalledOnce();
    loader.dispose();
  });
});
