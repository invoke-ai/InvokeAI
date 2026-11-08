import { AccountScopeExpiredError, createAccountLifecycle } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { FontRecord } from './contracts';

import { createFontRuntime, getFontRuntimeKey, type FontRuntimeEnvironment } from './runtime';

const createRuntime = (options: Parameters<typeof createFontRuntime>[0]) =>
  createFontRuntime({
    getFont: (id) =>
      Promise.resolve({
        id,
        contentHash: id.padEnd(64, '0').slice(0, 64),
        style: 'normal',
        weight: 400,
        axes: [],
        instances: [],
        byteSize: 1,
        family: 'Example Sans',
        filename: 'example.ttf',
        label: 'Example Sans',
        scope: 'private',
        source: 'uploaded',
        url: '',
      }),
    ...options,
  });

class FakeFontFace {
  readonly family: string;
  readonly source: ArrayBuffer;
  readonly descriptors: { style?: string; weight?: string } | undefined;

  constructor(family: string, source: ArrayBuffer, descriptors?: { style?: string; weight?: string }) {
    this.family = family;
    this.source = source;
    this.descriptors = descriptors;
  }

  load(): Promise<FakeFontFace> {
    return Promise.resolve(this);
  }
}

const createEnvironment = () => {
  const added: FakeFontFace[] = [];
  const deleted: FakeFontFace[] = [];
  const environment: FontRuntimeEnvironment = {
    FontFace: FakeFontFace,
    fonts: {
      add: (font) => added.push(font as FakeFontFace),
      delete: (font) => {
        deleted.push(font as FakeFontFace);
        return true;
      },
    },
  };
  return { added, deleted, environment };
};

const reference = (id: string, axes?: Readonly<Record<string, number>>) => ({
  axes,
  contentHash: id.padEnd(64, '0').slice(0, 64),
  family: 'Example Sans',
  id,
  style: 'normal',
  weight: 400,
});

describe('font runtime', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('deduplicates equivalent requests and publishes readiness by stable identity', async () => {
    const { added, environment } = createEnvironment();
    const download = vi.fn(() => Promise.resolve(new Uint8Array([1, 2, 3])));
    const runtime = createRuntime({ download, environment, lifecycle: createAccountLifecycle() });
    const first = reference('font-a', { wght: 400, wdth: 90 });
    const second = reference('font-a', { wdth: 90, wght: 400 });

    const [firstFamily, secondFamily] = await Promise.all([runtime.ensure(first), runtime.ensure(second)]);

    expect(firstFamily).toBe(secondFamily);
    expect(download).toHaveBeenCalledTimes(1);
    expect(added).toHaveLength(1);
    expect(runtime.getSnapshot().states.get(getFontRuntimeKey(first))).toBe('ready');
    expect(runtime.resolveFamily(second)).toBe(firstFamily);
    runtime.dispose();
  });

  it('keeps long variation identities distinct inside one session', async () => {
    const { environment } = createEnvironment();
    const runtime = createRuntime({
      download: vi.fn(() => Promise.resolve(new Uint8Array([1]))),
      environment,
      lifecycle: createAccountLifecycle(),
    });
    const axes = Object.fromEntries(Array.from({ length: 14 }, (_, index) => [`axis${index}`, index]));
    const first = reference('font-a', axes);
    const second = reference('font-a', { ...axes, axis13: 999 });

    const [firstFamily, secondFamily] = await Promise.all([runtime.ensure(first), runtime.ensure(second)]);

    expect(firstFamily).not.toBe(secondFamily);
    runtime.dispose();
  });

  it('keeps browser family aliases unique across runtime instances', async () => {
    const first = createEnvironment();
    const second = createEnvironment();
    const firstRuntime = createRuntime({
      download: vi.fn(() => Promise.resolve(new Uint8Array([1]))),
      environment: first.environment,
      lifecycle: createAccountLifecycle(),
    });
    const secondRuntime = createRuntime({
      download: vi.fn(() => Promise.resolve(new Uint8Array([1]))),
      environment: second.environment,
      lifecycle: createAccountLifecycle(),
    });

    const firstFamily = await firstRuntime.ensure(reference('font-a'));
    const secondFamily = await secondRuntime.ensure(reference('font-a'));

    expect(firstFamily).not.toBe(secondFamily);
    firstRuntime.dispose();
    secondRuntime.dispose();
  });

  it('limits simultaneous byte downloads', async () => {
    const { environment } = createEnvironment();
    let active = 0;
    let maximumActive = 0;
    const pending = new Map<string, (bytes: Uint8Array) => void>();
    const download = vi.fn(
      (font: { id: string }) =>
        new Promise<Uint8Array>((resolve) => {
          active += 1;
          maximumActive = Math.max(maximumActive, active);
          pending.set(font.id, (bytes) => {
            active -= 1;
            resolve(bytes);
          });
        })
    );
    const runtime = createRuntime({ download, environment, lifecycle: createAccountLifecycle() });
    const requests = Array.from({ length: 6 }, (_, index) => runtime.ensure(reference(`font-${index}`)));

    await vi.waitFor(() => expect(download).toHaveBeenCalledTimes(4));
    expect(maximumActive).toBe(4);
    for (let index = 0; index < 6; index += 1) {
      await vi.waitFor(() => expect(pending.has(`font-${index}`)).toBe(true));
      pending.get(`font-${index}`)!(new Uint8Array([index]));
    }
    await expect(Promise.all(requests)).resolves.toHaveLength(6);
    runtime.dispose();
  });

  it('removes canceled queued axis previews so the next distinct request starts promptly', async () => {
    const { environment } = createEnvironment();
    const resolvers = new Map<number, (bytes: Uint8Array) => void>();
    const download = vi.fn(
      (font: { axes?: Readonly<Record<string, number>> }, _signal?: AbortSignal) =>
        new Promise<Uint8Array>((resolve) => {
          const axis = font.axes?.wght ?? 0;
          resolvers.set(axis, resolve);
        })
    );
    const runtime = createRuntime({ download, environment, lifecycle: createAccountLifecycle() });
    const requests = Array.from({ length: 4 }, (_, index) =>
      runtime.ensure(reference(`axis-${index}`, { wght: index }))
    );
    await vi.waitFor(() => expect(download).toHaveBeenCalledTimes(4));

    const canceledController = new AbortController();
    const canceled = runtime.ensure(reference('axis-canceled', { wght: 4 }), canceledController.signal);
    const next = runtime.ensure(reference('axis-next', { wght: 5 }));
    canceledController.abort();
    await expect(canceled).rejects.toMatchObject({ name: 'AbortError' });
    expect(download).toHaveBeenCalledTimes(4);

    for (let index = 0; index < 4; index += 1) {
      resolvers.get(index)?.(new Uint8Array([index]));
    }
    await vi.waitFor(() => expect(download).toHaveBeenCalledTimes(5));
    expect(download.mock.calls.some(([font]) => font.axes?.wght === 4)).toBe(false);
    resolvers.get(5)?.(new Uint8Array([5]));
    await expect(Promise.all(requests.concat(next))).resolves.toHaveLength(5);
    runtime.dispose();
  });

  it('isolates caller cancellation from another consumer of the same face', async () => {
    const { environment } = createEnvironment();
    let resolveDownload!: (bytes: Uint8Array) => void;
    let downloadSignal!: AbortSignal;
    const download = vi.fn((_font: { axes?: Readonly<Record<string, number>> }, signal?: AbortSignal) => {
      downloadSignal = signal!;
      return new Promise<Uint8Array>((resolve) => {
        resolveDownload = resolve;
      });
    });
    const runtime = createRuntime({ download, environment, lifecycle: createAccountLifecycle() });
    const font = reference('shared-axis', { wght: 500 });
    const firstController = new AbortController();
    const first = runtime.ensure(font, firstController.signal);
    const second = runtime.ensure(font);
    await vi.waitFor(() => expect(download).toHaveBeenCalledOnce());

    firstController.abort(new Error('obsolete preview'));
    await expect(first).rejects.toMatchObject({ name: 'AbortError' });
    expect(downloadSignal.aborted).toBe(false);
    resolveDownload(new Uint8Array([1]));
    await expect(second).resolves.toMatch(/^__invoke_font_/);
    expect(environment.fonts).toBeTruthy();
    runtime.dispose();
  });

  it('aborts a queued and active preview when the runtime is disposed', async () => {
    const { added, environment } = createEnvironment();
    let resolveDownload!: (bytes: Uint8Array) => void;
    let downloadSignal!: AbortSignal;
    const runtime = createRuntime({
      download: vi.fn((_font: { axes?: Readonly<Record<string, number>> }, signal?: AbortSignal) => {
        downloadSignal = signal!;
        return new Promise<Uint8Array>((resolve) => {
          resolveDownload = resolve;
        });
      }),
      environment,
      lifecycle: createAccountLifecycle(),
    });
    const loading = runtime.ensure(reference('dispose-preview'));
    await vi.waitFor(() => expect(downloadSignal).toBeDefined());

    runtime.dispose();
    expect(downloadSignal.aborted).toBe(true);
    resolveDownload(new Uint8Array([1]));
    await expect(loading).rejects.toBeInstanceOf(AccountScopeExpiredError);
    expect(added).toHaveLength(0);
    expect(runtime.getSnapshot().states.size).toBe(0);
  });

  it('describes pinned variable coordinates while preserving static-face synthesis metadata', async () => {
    const { added, environment } = createEnvironment();
    const download = vi.fn(() => Promise.resolve(new Uint8Array([1])));
    const getFont = (id: string): Promise<FontRecord> =>
      Promise.resolve({
        axes: [
          { default: 400, hidden: false, label: 'Weight', maximum: 900, minimum: 100, tag: 'wght' },
          { default: 0, hidden: false, label: 'Italic', maximum: 1, minimum: 0, tag: 'ital' },
        ],
        byteSize: 1,
        contentHash: id.padEnd(64, '0').slice(0, 64),
        family: 'Example Sans',
        filename: 'example.ttf',
        id,
        instances: [],
        label: 'Example Sans',
        scope: 'private',
        source: 'uploaded',
        style: 'normal',
        url: '',
        weight: 400,
      });
    const runtime = createRuntime({
      download,
      environment,
      getFont,
      lifecycle: createAccountLifecycle(),
    });

    await runtime.ensure(reference('variable-descriptor', { ital: 1, wght: 800 }));
    expect(added[0]?.descriptors).toEqual({ style: 'italic', weight: '800' });

    const staticFont = reference('static-descriptor');
    const staticFamily = await runtime.ensure({ ...staticFont, weight: 700 });
    expect(added[1]?.descriptors).toEqual({ style: 'normal', weight: '400' });
    await expect(runtime.ensure({ ...staticFont, style: 'italic', weight: 900 })).resolves.toBe(staticFamily);
    expect(download).toHaveBeenCalledTimes(2);
    expect(added).toHaveLength(2);
    runtime.dispose();
  });

  it('does not evict a face retained by a live Canvas consumer', async () => {
    const { added, deleted, environment } = createEnvironment();
    const runtime = createRuntime({
      download: vi.fn(() => Promise.resolve(new Uint8Array([1]))),
      environment,
      lifecycle: createAccountLifecycle(),
    });
    const pinned = reference('pinned');
    const pinnedFamily = await runtime.ensure(pinned);
    const release = runtime.retain(pinned);

    for (let index = 0; index < 65; index += 1) {
      await runtime.ensure(reference(`cache-${index}`));
    }

    expect(added.some((face) => face.family === pinnedFamily)).toBe(true);
    expect(deleted.some((face) => face.family === pinnedFamily)).toBe(false);
    release();
    await runtime.ensure(reference('cache-after-release'));
    expect(deleted.some((face) => face.family === pinnedFamily)).toBe(true);
    runtime.dispose();
  });

  it('fails closed when browser font registration is unavailable', async () => {
    const runtime = createRuntime({
      download: vi.fn(() => Promise.resolve(new Uint8Array([1]))),
      environment: { FontFace: null, fonts: null },
      lifecycle: createAccountLifecycle(),
    });

    await expect(runtime.ensure(reference('font-a'))).rejects.toThrow('FontFace API');
    expect(runtime.getSnapshot().states.get(getFontRuntimeKey(reference('font-a')))).toBe('error');
    runtime.dispose();
  });

  it('bounds failed face state while preserving retry behavior', async () => {
    const { environment } = createEnvironment();
    const runtime = createRuntime({
      download: vi.fn(() => {
        throw new Error('font bytes unavailable');
      }),
      environment,
      lifecycle: createAccountLifecycle(),
    });

    for (let index = 0; index < 70; index += 1) {
      await expect(runtime.ensure(reference(`failed-${index}`))).rejects.toThrow('unavailable');
    }

    expect(runtime.getSnapshot().states.size).toBeLessThanOrEqual(64);
    await expect(runtime.ensure(reference('failed-0'))).rejects.toThrow('unavailable');
    runtime.dispose();
  });

  it('refuses new faces at the active-font limit and recovers when a consumer releases', async () => {
    const { added, deleted, environment } = createEnvironment();
    const runtime = createRuntime({
      download: () => Promise.resolve(new Uint8Array([1])),
      environment,
      lifecycle: createAccountLifecycle(),
    });
    const releases = [];
    for (let index = 0; index < 64; index += 1) {
      const font = reference(`active-${index}`);
      releases.push(runtime.retain(font));
      await runtime.ensure(font);
    }
    const extra = reference('extra');
    await expect(runtime.ensure(extra)).rejects.toThrow('budget is full');
    expect(added).toHaveLength(64);
    expect(deleted).toHaveLength(0);
    releases[0]!();
    await expect(runtime.ensure(extra)).resolves.toMatch(/^__invoke_font_/);
    expect(deleted).toHaveLength(1);
    releases.forEach((release) => release());
    runtime.dispose();
  });

  it('reserves bytes during concurrent FontFace loading and releases a failed reservation', async () => {
    const bytes = new Uint8Array(64 * 1024 * 1024);
    const loads: Array<{ resolve: () => void; reject: (error: Error) => void }> = [];
    class DeferredFontFace extends FakeFontFace {
      override load(): Promise<FakeFontFace> {
        return new Promise((resolve, reject) => {
          loads.push({ resolve: () => resolve(this), reject });
        });
      }
    }
    const { added, environment } = createEnvironment();
    environment.FontFace = DeferredFontFace;
    const runtime = createRuntime({
      download: () => Promise.resolve(bytes),
      environment,
      lifecycle: createAccountLifecycle(),
    });
    const first = runtime.ensure(reference('large-a'));
    const second = runtime.ensure(reference('large-b'));
    const failed = expect(second).rejects.toThrow('parse failed');
    await vi.waitFor(() => expect(loads).toHaveLength(2));
    await expect(runtime.ensure(reference('large-c'))).rejects.toThrow('budget is full');
    loads[0]!.resolve();
    loads[1]!.reject(new Error('parse failed'));
    await first;
    await failed;
    const retry = runtime.ensure(reference('large-c'));
    await vi.waitFor(() => expect(loads).toHaveLength(3));
    loads[2]!.resolve();
    await retry;
    expect(added).toHaveLength(2);
    expect(added[0]!.source).toBe(bytes.buffer);
    runtime.dispose();
  });

  it('trims failed state after its retained consumer releases', async () => {
    const { environment } = createEnvironment();
    const runtime = createRuntime({
      download: vi.fn(() => {
        throw new Error('font bytes unavailable');
      }),
      environment,
      lifecycle: createAccountLifecycle(),
    });
    const retained = reference('failed-retained');
    const release = runtime.retain(retained);

    for (let index = 0; index < 64; index += 1) {
      await expect(runtime.ensure(reference(`failed-retained-${index}`))).rejects.toThrow('unavailable');
    }
    await expect(runtime.ensure(retained)).rejects.toThrow('unavailable');
    expect(runtime.getSnapshot().states.size).toBe(65);

    release();
    expect(runtime.getSnapshot().states.size).toBeLessThanOrEqual(64);
    runtime.dispose();
  });

  it('honors an aborted caller before returning a loaded face', async () => {
    const { environment } = createEnvironment();
    const runtime = createRuntime({
      download: vi.fn(() => Promise.resolve(new Uint8Array([1]))),
      environment,
      lifecycle: createAccountLifecycle(),
    });
    const font = reference('font-a');
    await runtime.ensure(font);
    const controller = new AbortController();
    const reason = new Error('cancelled');
    controller.abort(reason);

    await expect(runtime.ensure(font, controller.signal)).rejects.toBe(reason);
    runtime.dispose();
  });

  it('clears an aborted in-flight state instead of retaining an error entry', async () => {
    const { environment } = createEnvironment();
    let resolveDownload: (bytes: Uint8Array) => void = () => undefined;
    const download = vi.fn(
      () =>
        new Promise<Uint8Array>((resolve) => {
          resolveDownload = resolve;
        })
    );
    const runtime = createRuntime({ download, environment, lifecycle: createAccountLifecycle() });
    const font = reference('font-aborted');
    const controller = new AbortController();
    const loading = runtime.ensure(font, controller.signal);

    await vi.waitFor(() => expect(download).toHaveBeenCalledTimes(1));
    controller.abort(new Error('caller cancelled'));
    resolveDownload(new Uint8Array([1]));

    await expect(loading).rejects.toThrow('aborted');
    expect(runtime.getSnapshot().states.has(getFontRuntimeKey(font))).toBe(false);
    runtime.dispose();
  });

  it('revalidates a cached face before output and removes it when the catalog entry changes', async () => {
    const { added, deleted, environment } = createEnvironment();
    const font = reference('font-output');
    let currentHash = font.contentHash;
    const getFont = vi.fn(() =>
      Promise.resolve({
        axes: [],
        byteSize: 1,
        contentHash: currentHash,
        family: 'Example Sans',
        filename: 'example.ttf',
        id: font.id,
        instances: [],
        label: 'Example Sans',
        scope: 'private' as const,
        source: 'uploaded' as const,
        style: 'normal',
        url: '/fonts/example/file',
        weight: 400,
      })
    );
    const runtime = createRuntime({
      download: vi.fn(() => Promise.resolve(new Uint8Array([1]))),
      environment,
      getFont,
      lifecycle: createAccountLifecycle(),
    });
    const loadedFamily = await runtime.ensure(font);
    currentHash = 'f'.repeat(64);
    await expect(runtime.ensureForOutput(font)).rejects.toThrow('changed');

    expect(getFont).toHaveBeenCalledTimes(2);
    expect(added).toHaveLength(1);
    expect(deleted).toHaveLength(1);
    expect(runtime.resolveFamily(font)).toBe('Example Sans');
    expect(loadedFamily).not.toBe(runtime.resolveFamily(font));
    runtime.dispose();
  });

  it('deduplicates concurrent output metadata checks by id and hash', async () => {
    const { environment } = createEnvironment();
    const font = reference('font-output-shared');
    let resolveMetadata: (record: FontRecord) => void = () => undefined;
    const getFont = vi.fn(
      () =>
        new Promise<FontRecord>((resolve) => {
          resolveMetadata = resolve;
        })
    );
    const runtime = createRuntime({
      download: vi.fn(() => Promise.resolve(new Uint8Array([1]))),
      environment,
      getFont,
      lifecycle: createAccountLifecycle(),
    });
    const first = runtime.ensureForOutput(font);
    const second = runtime.ensureForOutput({ ...font, axes: { wght: 600 } });

    await vi.waitFor(() => expect(getFont).toHaveBeenCalledTimes(1));
    resolveMetadata({
      axes: [],
      byteSize: 1,
      contentHash: font.contentHash!,
      family: 'Example Sans',
      filename: 'example.ttf',
      id: font.id,
      instances: [],
      label: 'Example Sans',
      scope: 'private',
      source: 'uploaded',
      style: 'normal',
      url: '/fonts/example/file',
      weight: 400,
    });

    await expect(Promise.all([first, second])).resolves.toHaveLength(2);
    runtime.dispose();
  });

  it('cancels and fences output metadata across dispose and restart', async () => {
    const { environment } = createEnvironment();
    const font = reference('font-output-restart');
    const metadataResolvers: Array<(record: FontRecord) => void> = [];
    const metadataSignals: AbortSignal[] = [];
    const getFont = vi.fn(
      (_id: string, signal?: AbortSignal) =>
        new Promise<FontRecord>((resolve) => {
          metadataSignals.push(signal!);
          metadataResolvers.push(resolve);
        })
    );
    const runtime = createRuntime({
      download: vi.fn(() => Promise.resolve(new Uint8Array([1]))),
      environment,
      getFont,
      lifecycle: createAccountLifecycle(),
    });
    const first = runtime.ensureForOutput(font);

    await vi.waitFor(() => expect(getFont).toHaveBeenCalledTimes(1));
    runtime.dispose();
    expect(metadataSignals[0]?.aborted).toBe(true);
    runtime.start();
    metadataResolvers[0]!({
      axes: [],
      byteSize: 1,
      contentHash: font.contentHash!,
      family: 'Example Sans',
      filename: 'example.ttf',
      id: font.id,
      instances: [],
      label: 'Example Sans',
      scope: 'private',
      source: 'uploaded',
      style: 'normal',
      url: '/fonts/example/file',
      weight: 400,
    });
    await expect(first).rejects.toBeInstanceOf(AccountScopeExpiredError);

    const second = runtime.ensureForOutput(font);
    await vi.waitFor(() => expect(getFont).toHaveBeenCalledTimes(2));
    metadataResolvers[1]!({
      axes: [],
      byteSize: 1,
      contentHash: font.contentHash!,
      family: 'Example Sans',
      filename: 'example.ttf',
      id: font.id,
      instances: [],
      label: 'Example Sans',
      scope: 'private',
      source: 'uploaded',
      style: 'normal',
      url: '/fonts/example/file',
      weight: 400,
    });
    await expect(second).resolves.toMatch(/^__invoke_font_/);
    runtime.dispose();
  });

  it('fences an in-flight load when the account changes before bytes arrive', async () => {
    const { added, environment } = createEnvironment();
    let resolveDownload: (bytes: Uint8Array) => void = () => undefined;
    const download = vi.fn(
      () =>
        new Promise<Uint8Array>((resolve) => {
          resolveDownload = resolve;
        })
    );
    const lifecycle = createAccountLifecycle();
    lifecycle.activate('account-a');
    const runtime = createRuntime({ download, environment, lifecycle });
    const loading = runtime.ensure(reference('font-a'));

    await vi.waitFor(() => expect(download).toHaveBeenCalledTimes(1));
    lifecycle.activate('account-b');
    resolveDownload(new Uint8Array([1, 2, 3]));

    await expect(loading).rejects.toBeInstanceOf(AccountScopeExpiredError);
    expect(added).toHaveLength(0);
    expect(runtime.getSnapshot().states.size).toBe(0);
    runtime.dispose();
  });

  it('clears browser faces when disposed and can be restarted by a StrictMode effect', async () => {
    const { added, deleted, environment } = createEnvironment();
    const runtime = createRuntime({
      download: vi.fn(() => Promise.resolve(new Uint8Array([1]))),
      environment,
      lifecycle: createAccountLifecycle(),
    });
    const firstFamily = await runtime.ensure(reference('font-a'));

    runtime.dispose();
    expect(added).toHaveLength(1);
    expect(deleted).toHaveLength(1);
    expect(runtime.getSnapshot().states.size).toBe(0);

    runtime.start();
    await expect(runtime.ensure(reference('font-a'))).resolves.not.toBe(firstFamily);
    runtime.dispose();
  });
});
