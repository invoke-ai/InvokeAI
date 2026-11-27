import type { CanvasImageRef, CanvasLayerSourceContract } from '@workbench/canvas-engine/contracts';
import type { CanvasImageUploadResult } from '@workbench/canvas-engine/document/imageUpload';
import type { PaintCacheTrim } from '@workbench/canvas-engine/render/paintCacheTrim';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { BitmapStoreTimers } from './bitmapStore';

import { createBitmapStore, DEFAULT_FAILURE_BACKOFF_MS } from './bitmapStore';

const LAYER = 'layer-1';

interface ManualTimers extends BitmapStoreTimers {
  readonly scheduledDelays: number[];
  fireNext(): void;
  pendingCount(): number;
}

const createManualTimers = (): ManualTimers => {
  let nextHandle = 1;
  const jobs = new Map<number, () => void>();
  const scheduledDelays: number[] = [];
  return {
    clearTimeout: (handle) => {
      jobs.delete(handle);
    },
    fireNext: () => {
      const next = jobs.entries().next();
      if (next.done) {
        return;
      }
      const [handle, handler] = next.value;
      jobs.delete(handle);
      handler();
    },
    pendingCount: () => jobs.size,
    scheduledDelays,
    setTimeout: (handler, ms) => {
      const handle = nextHandle++;
      jobs.set(handle, handler);
      scheduledDelays.push(ms);
      return handle;
    },
  };
};

const createDeferred = <T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason: unknown) => void;
} => {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, reject, resolve };
};

const drainUntil = async (predicate: () => boolean, maxTicks = 50): Promise<void> => {
  for (let i = 0; i < maxTicks && !predicate(); i += 1) {
    await Promise.resolve();
  }
};

interface HarnessOptions {
  encodeSurface?: (surface: RasterSurface) => Promise<Blob>;
  getLayerSurface?: (
    surface: RasterSurface,
    offset: { x: number; y: number }
  ) => { surface: RasterSurface; offset: { x: number; y: number } } | 'empty' | null;
  hashBlob?: (blob: Blob) => Promise<string>;
  uploadImage?: (blob: Blob) => Promise<CanvasImageUploadResult>;
  maxUploadAttempts?: number;
  failureBackoffMs?: readonly number[];
  maxConsecutiveFailures?: number;
  onError?: (error: unknown, layerId: string, info: { consecutiveFailures: number; willRetry: boolean }) => void;
  offset?: { x: number; y: number };
  sleep?: (ms: number) => Promise<void>;
  trimLayerPixels?: (layerId: string) => PaintCacheTrim;
  clearBitmap?: (layerId: string) => boolean;
  getAuthoritativeLayerSource?: (layerId: string) => CanvasLayerSourceContract | null;
  timers?: BitmapStoreTimers;
}

const PAINT_SOURCE: CanvasLayerSourceContract = { bitmap: null, type: 'paint' };

const createHarness = (options: HarnessOptions = {}) => {
  const surface: RasterSurface = createTestStubRasterBackend().createSurface(10, 10);
  let encoded = 'pixels-A';
  let uploadSeq = 0;
  let source: CanvasLayerSourceContract | null = PAINT_SOURCE;

  let offset = options.offset ?? { x: 0, y: 0 };

  const encodeSurface = vi.fn(
    options.encodeSurface ?? (() => Promise.resolve(new Blob([encoded], { type: 'image/png' })))
  );
  const uploadImage =
    options.uploadImage ??
    vi.fn((_blob: Blob): Promise<CanvasImageUploadResult> =>
      Promise.resolve({ height: 10, imageName: `img-${uploadSeq++}`, width: 10 })
    );
  const dispatch = vi.fn((_action: CanvasProjectMutation) => true);
  const clearBitmap = options.clearBitmap ? vi.fn(options.clearBitmap) : undefined;
  const trimLayerPixels = options.trimLayerPixels ? vi.fn(options.trimLayerPixels) : undefined;

  const store = createBitmapStore({
    debounceMs: 1500,
    dispatch,
    encodeSurface,
    ...(clearBitmap ? { clearBitmap } : {}),
    ...(trimLayerPixels ? { trimLayerPixels } : {}),
    ...(options.getAuthoritativeLayerSource
      ? { getAuthoritativeLayerSource: options.getAuthoritativeLayerSource }
      : {}),
    failureBackoffMs: options.failureBackoffMs,
    getLayerSource: () => source,
    getLayerSurface: () => (options.getLayerSurface ? options.getLayerSurface(surface, offset) : { offset, surface }),
    hashBlob: options.hashBlob ?? ((blob) => blob.text()),
    maxConsecutiveFailures: options.maxConsecutiveFailures,
    maxUploadAttempts: options.maxUploadAttempts ?? 3,
    onError: options.onError,
    retryDelaysMs: [1],
    sleep: options.sleep ?? (() => Promise.resolve()),
    timers: options.timers,
    uploadImage,
  });

  return {
    clearBitmap,
    dispatch,
    encodeSurface,
    trimLayerPixels,
    setEncoded: (value: string) => {
      encoded = value;
    },
    setOffset: (value: { x: number; y: number }) => {
      offset = value;
    },
    setSource: (value: CanvasLayerSourceContract | null) => {
      source = value;
    },
    store,
    uploadImage: uploadImage as ReturnType<typeof vi.fn>,
  };
};

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('createBitmapStore', () => {
  it('keeps unpersisted pixels pending and fails the barrier when their cache is missing', async () => {
    const onError = vi.fn();
    const h = createHarness({ getLayerSurface: () => null, onError });

    h.store.markLayerDirty(LAYER);

    await expect(h.store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');
    expect(h.encodeSurface).not.toHaveBeenCalled();
    expect(h.uploadImage).not.toHaveBeenCalled();
    expect(h.dispatch).not.toHaveBeenCalled();
    expect(h.store.hasPendingWork(LAYER)).toBe(true);
    expect(onError).toHaveBeenCalledOnce();
    h.store.dispose();
  });

  it('debounces a burst of strokes into a single flush', async () => {
    const h = createHarness();

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(500);
    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(500);
    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1000); // 1000 < 1500 since the last stroke

    expect(h.uploadImage).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();

    expect(h.uploadImage).toHaveBeenCalledTimes(1);
    expect(h.dispatch).toHaveBeenCalledTimes(1);
    h.store.dispose();
  });

  it('dedupes identical pixels: the second flush reuses the image and skips the upload', async () => {
    const h = createHarness();

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();
    expect(h.uploadImage).toHaveBeenCalledTimes(1);
    expect(h.dispatch).toHaveBeenCalledTimes(1);

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();

    expect(h.uploadImage).toHaveBeenCalledTimes(1);
    h.store.dispose();
  });

  it('dispatches the content-rect offset alongside the bitmap ref (paint persistence round-trip)', async () => {
    const h = createHarness({ offset: { x: 40, y: 25 } });

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();

    const dispatched = h.dispatch.mock.calls.at(-1)?.[0];
    expect(dispatched).toMatchObject({
      source: { bitmap: { imageName: 'img-0' }, offset: { x: 40, y: 25 }, type: 'paint' },
      type: 'updateCanvasLayerSource',
    });
    expect(h.encodeSurface).toHaveBeenCalledWith(expect.objectContaining({ height: 10, width: 10 }));
    h.store.dispose();
  });

  it('retains dirty pixels when the intended layer rejects the persisted bitmap', async () => {
    const surface = createTestStubRasterBackend().createSurface(10, 10);
    const dispatchBitmap = vi.fn(() => false);
    const store = createBitmapStore({
      debounceMs: 1500,
      dispatch: vi.fn(),
      dispatchBitmap,
      encodeSurface: () => Promise.resolve(new Blob(['pixels'], { type: 'image/png' })),
      getLayerSource: () => ({ bitmap: null, type: 'paint' }),
      getLayerSurface: () => ({ offset: { x: 0, y: 0 }, surface }),
      hashBlob: (blob) => blob.text(),
      uploadImage: () => Promise.resolve({ height: 10, imageName: 'rejected-img', width: 10 }),
    });

    store.markLayerDirty(LAYER);
    await expect(store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');
    await expect(store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');

    expect(dispatchBitmap).toHaveBeenCalledTimes(2);
    store.dispose();
  });

  it('accepts a thrown dispatch when authoritative state proves the intended bitmap landed', async () => {
    const surface = createTestStubRasterBackend().createSurface(10, 10);
    let source: CanvasLayerSourceContract | null = { bitmap: null, type: 'paint' };
    const dispatchBitmap = vi.fn((_layerId: string, bitmap: CanvasImageRef, offset: { x: number; y: number }) => {
      source = { bitmap, offset, type: 'paint' };
      throw new Error('subscriber failed after commit');
    });
    const onError = vi.fn();
    const store = createBitmapStore({
      debounceMs: 1500,
      dispatch: vi.fn(() => true),
      dispatchBitmap,
      encodeSurface: () => Promise.resolve(new Blob(['pixels'], { type: 'image/png' })),
      getLayerSource: () => source,
      getLayerSurface: () => ({ offset: { x: 7, y: 8 }, surface }),
      hashBlob: (blob) => blob.text(),
      onError,
      uploadImage: () => Promise.resolve({ height: 10, imageName: 'landed-before-throw.png', width: 10 }),
    });

    store.markLayerDirty(LAYER);
    await expect(store.flushPendingUploads()).resolves.toBeUndefined();
    await expect(store.flushPendingUploads()).resolves.toBeUndefined();

    expect(dispatchBitmap).toHaveBeenCalledOnce();
    expect(onError).not.toHaveBeenCalled();
    expect(store.isSelfEcho(LAYER, source)).toBe(true);
    store.dispose();
  });

  it('drops dirty pixels when rejection confirms that the intended layer was deleted', async () => {
    const surface = createTestStubRasterBackend().createSurface(10, 10);
    let source: CanvasLayerSourceContract | null = { bitmap: null, type: 'paint' };
    const dispatchBitmap = vi.fn(() => {
      source = null;
      return false;
    });
    const store = createBitmapStore({
      debounceMs: 1500,
      dispatch: vi.fn(),
      dispatchBitmap,
      encodeSurface: () => Promise.resolve(new Blob(['pixels'], { type: 'image/png' })),
      getLayerSource: () => source,
      getLayerSurface: () => ({ offset: { x: 0, y: 0 }, surface }),
      hashBlob: (blob) => blob.text(),
      uploadImage: () => Promise.resolve({ height: 10, imageName: 'orphan-img', width: 10 }),
    });

    store.markLayerDirty(LAYER);
    await store.flushPendingUploads();
    await store.flushPendingUploads();

    expect(dispatchBitmap).toHaveBeenCalledTimes(1);
    store.dispose();
  });

  it('uploads a new image when the pixels change', async () => {
    const h = createHarness();

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();

    h.setEncoded('pixels-B');
    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();

    expect(h.uploadImage).toHaveBeenCalledTimes(2);
    expect(h.dispatch).toHaveBeenCalledTimes(2);
    h.store.dispose();
  });

  it('undo re-flush reuses the prior image and does not re-upload (history convergence)', async () => {
    const h = createHarness();

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();
    h.setEncoded('pixels-B');
    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();
    expect(h.uploadImage).toHaveBeenCalledTimes(2);

    h.setEncoded('pixels-A');
    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();

    expect(h.uploadImage).toHaveBeenCalledTimes(2);
    const lastDispatch = h.dispatch.mock.calls.at(-1)?.[0];
    expect(lastDispatch).toMatchObject({
      source: { bitmap: { imageName: 'img-0' }, type: 'paint' },
      type: 'updateCanvasLayerSource',
    });
    h.store.dispose();
  });

  it('swaps on success: dispatch fires only after the upload resolves', async () => {
    const deferred = createDeferred<CanvasImageUploadResult>();
    const uploadImage = vi.fn(() => deferred.promise);
    const h = createHarness({ uploadImage });

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);

    expect(uploadImage).toHaveBeenCalledTimes(1);
    expect(h.dispatch).not.toHaveBeenCalled();

    deferred.resolve({ height: 10, imageName: 'img-x', width: 10 });
    await h.store.flushPendingUploads();

    expect(h.dispatch).toHaveBeenCalledTimes(1);
    expect(h.dispatch.mock.calls[0][0]).toMatchObject({
      id: LAYER,
      source: { bitmap: { imageName: 'img-x' }, type: 'paint' },
      type: 'updateCanvasLayerSource',
    });
    h.store.dispose();
  });

  it('on upload failure: no dispatch, layer stays dirty, then recovers on a later success', async () => {
    let shouldFail = true;
    const uploadImage = vi.fn(() => {
      if (shouldFail) {
        return Promise.reject(new Error('upload failed'));
      }
      return Promise.resolve<CanvasImageUploadResult>({ height: 10, imageName: 'img-ok', width: 10 });
    });
    const onError = vi.fn();
    const surface = createTestStubRasterBackend().createSurface(10, 10);
    const dispatch = vi.fn((_action: CanvasProjectMutation) => true);
    const store = createBitmapStore({
      debounceMs: 1500,
      dispatch,
      encodeSurface: () => Promise.resolve(new Blob(['pixels'], { type: 'image/png' })),
      getLayerSource: () => PAINT_SOURCE,
      getLayerSurface: () => ({ offset: { x: 0, y: 0 }, surface }),
      hashBlob: (blob) => blob.text(),
      maxUploadAttempts: 2,
      onError,
      retryDelaysMs: [1],
      sleep: () => Promise.resolve(),
      uploadImage,
    });

    store.markLayerDirty(LAYER);
    await expect(store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');

    expect(uploadImage).toHaveBeenCalledTimes(2);
    expect(dispatch).not.toHaveBeenCalled();
    expect(onError).toHaveBeenCalledTimes(1);

    shouldFail = false;
    store.markLayerDirty(LAYER);
    await store.flushPendingUploads();

    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch.mock.calls[0][0]).toMatchObject({ source: { bitmap: { imageName: 'img-ok' } } });
    store.dispose();
  });

  it('flushPendingUploads is a barrier: it cancels the debounce and resolves only after uploads settle', async () => {
    const deferred = createDeferred<CanvasImageUploadResult>();
    const uploadImage = vi.fn(() => deferred.promise);
    const h = createHarness({ uploadImage });

    h.store.markLayerDirty(LAYER);
    const barrier = h.store.flushPendingUploads();

    let settled = false;
    void barrier.then(() => {
      settled = true;
    });

    for (let i = 0; i < 5; i += 1) {
      await Promise.resolve();
    }
    expect(uploadImage).toHaveBeenCalledTimes(1);
    expect(settled).toBe(false);

    deferred.resolve({ height: 10, imageName: 'img-b', width: 10 });
    await barrier;

    expect(settled).toBe(true);
    expect(h.dispatch).toHaveBeenCalledTimes(1);
    h.store.dispose();
  });

  it('suspends debounced dirty work and resumes it only after release', async () => {
    const h = createHarness();
    h.store.markLayerDirty(LAYER);
    const release = h.store.suspendLayer(LAYER);

    await vi.advanceTimersByTimeAsync(3000);
    expect(h.uploadImage).not.toHaveBeenCalled();
    expect(h.dispatch).not.toHaveBeenCalled();

    release();
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();

    expect(h.uploadImage).toHaveBeenCalledOnce();
    expect(h.dispatch).toHaveBeenCalledOnce();
    h.store.dispose();
  });

  it('invalidates an in-flight result and keeps a barrier pending until suspension releases', async () => {
    const uploads = [createDeferred<CanvasImageUploadResult>(), createDeferred<CanvasImageUploadResult>()];
    let uploadIndex = 0;
    const uploadImage = vi.fn(() => uploads[uploadIndex++]!.promise);
    const h = createHarness({ uploadImage });
    h.store.markLayerDirty(LAYER);
    const barrier = h.store.flushPendingUploads();
    await drainUntil(() => uploadImage.mock.calls.length === 1);

    const release = h.store.suspendLayer(LAYER);
    let settled = false;
    void barrier.then(() => {
      settled = true;
    });
    uploads[0]!.resolve({ height: 10, imageName: 'obsolete', width: 10 });
    await drainUntil(() => settled);

    expect(settled).toBe(false);
    expect(h.dispatch).not.toHaveBeenCalled();
    expect(uploadImage).toHaveBeenCalledOnce();

    release();
    await drainUntil(() => uploadImage.mock.calls.length === 2);
    uploads[1]!.resolve({ height: 10, imageName: 'fresh', width: 10 });
    await barrier;

    expect(h.dispatch).toHaveBeenCalledOnce();
    expect(h.dispatch.mock.calls[0]![0]).toMatchObject({ source: { bitmap: { imageName: 'fresh' } } });
    h.store.dispose();
  });

  it('ignores a stale suspension release after reset reuses the same layer id', async () => {
    const h = createHarness();
    const releaseOldDocument = h.store.suspendLayer(LAYER);

    h.store.reset();

    const releaseNewDocument = h.store.suspendLayer(LAYER);
    h.store.markLayerDirty(LAYER);
    const barrier = h.store.flushPendingUploads();
    let settled = false;
    void barrier.then(() => {
      settled = true;
    });

    releaseOldDocument();
    await vi.advanceTimersByTimeAsync(3000);
    await Promise.resolve();

    expect(settled).toBe(false);
    expect(h.encodeSurface).not.toHaveBeenCalled();

    releaseNewDocument();
    await barrier;

    expect(h.encodeSurface).toHaveBeenCalledOnce();
    expect(h.dispatch).toHaveBeenCalledOnce();
    h.store.dispose();
  });

  it.each(['reset', 'dispose'] as const)('%s settles barriers waiting on suspended dirty work', async (ending) => {
    const h = createHarness();
    h.store.markLayerDirty(LAYER);
    h.store.suspendLayer(LAYER);
    const barrier = h.store.flushPendingUploads();
    let settled = false;
    void barrier.then(() => {
      settled = true;
    });
    await Promise.resolve();
    expect(settled).toBe(false);

    h.store[ending]();
    await barrier;

    expect(settled).toBe(true);
    expect(h.dispatch).not.toHaveBeenCalled();
    if (ending === 'reset') {
      h.store.dispose();
    }
  });

  it('a stroke landing while the barrier awaits an in-flight upload is not dropped: the barrier waits for the follow-up flush', async () => {
    const deferreds = [createDeferred<CanvasImageUploadResult>(), createDeferred<CanvasImageUploadResult>()];
    let call = 0;
    const uploadImage = vi.fn(() => deferreds[call++].promise);
    const h = createHarness({ uploadImage });

    h.store.markLayerDirty(LAYER);
    const barrier = h.store.flushPendingUploads();
    let settled = false;
    void barrier.then(() => {
      settled = true;
    });

    await drainUntil(() => uploadImage.mock.calls.length >= 1);
    expect(uploadImage).toHaveBeenCalledTimes(1);
    expect(settled).toBe(false);

    h.setEncoded('pixels-B');
    h.store.markLayerDirty(LAYER);

    deferreds[0].resolve({ height: 10, imageName: 'img-old', width: 10 });
    await drainUntil(() => uploadImage.mock.calls.length >= 2);
    expect(uploadImage).toHaveBeenCalledTimes(2);
    expect(settled).toBe(false);

    deferreds[1].resolve({ height: 10, imageName: 'img-new', width: 10 });
    await barrier;

    expect(settled).toBe(true);
    expect(h.dispatch).toHaveBeenCalledTimes(2);
    expect(h.dispatch.mock.calls[1][0]).toMatchObject({
      id: LAYER,
      source: { bitmap: { imageName: 'img-new' }, type: 'paint' },
      type: 'updateCanvasLayerSource',
    });
    h.store.dispose();
  });

  it('a persistently failing layer does not spin the barrier: it rejects after one bounded attempt and stays dirty', async () => {
    const uploadImage = vi.fn(() => Promise.reject(new Error('upload failed')));
    const onError = vi.fn();
    const surface = createTestStubRasterBackend().createSurface(10, 10);
    const dispatch = vi.fn((_action: CanvasProjectMutation) => true);
    const store = createBitmapStore({
      debounceMs: 1500,
      dispatch,
      encodeSurface: () => Promise.resolve(new Blob(['pixels'], { type: 'image/png' })),
      failureBackoffMs: [1500],
      getLayerSource: () => PAINT_SOURCE,
      getLayerSurface: () => ({ offset: { x: 0, y: 0 }, surface }),
      hashBlob: (blob) => blob.text(),
      maxUploadAttempts: 2,
      onError,
      retryDelaysMs: [1],
      sleep: () => Promise.resolve(),
      uploadImage,
    });

    store.markLayerDirty(LAYER);
    await expect(store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');

    expect(uploadImage).toHaveBeenCalledTimes(2);
    expect(dispatch).not.toHaveBeenCalled();
    expect(onError).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(1500);
    expect(uploadImage.mock.calls.length).toBeGreaterThan(2);

    store.dispose();
  });

  it('isSelfEcho recognizes the exact ref it just applied and rejects a different one', async () => {
    const h = createHarness();

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();

    const applied = h.dispatch.mock.calls[0][0] as Extract<CanvasProjectMutation, { type: 'updateCanvasLayerSource' }>;
    const appliedSource = applied.source;

    expect(h.store.isSelfEcho(LAYER, appliedSource)).toBe(true);

    const otherPaint: CanvasLayerSourceContract = {
      bitmap: { height: 10, imageName: 'other', width: 10 },
      type: 'paint',
    };
    expect(h.store.isSelfEcho(LAYER, otherPaint)).toBe(false);

    expect(h.store.isSelfEcho('layer-2', appliedSource)).toBe(false);
    expect(h.store.isSelfEcho(LAYER, { image: { height: 1, imageName: 'i', width: 1 }, type: 'image' })).toBe(false);
    expect(h.store.isSelfEcho(LAYER, null)).toBe(false);
    h.store.dispose();
  });

  it('reset() clears the self-echo guard so a reused layer id is not suppressed', async () => {
    const h = createHarness();

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();
    expect(h.dispatch).toHaveBeenCalledTimes(1);
    expect(h.uploadImage).toHaveBeenCalledTimes(1);

    const applied = h.dispatch.mock.calls[0][0] as Extract<CanvasProjectMutation, { type: 'updateCanvasLayerSource' }>;
    expect(h.store.isSelfEcho(LAYER, applied.source)).toBe(true);

    h.store.reset();
    expect(h.store.isSelfEcho(LAYER, applied.source)).toBe(false);

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();
    expect(h.dispatch).toHaveBeenCalledTimes(2);
    expect(h.uploadImage).toHaveBeenCalledTimes(1);

    h.store.dispose();
  });

  it('discardLayer cancels pending and in-flight persistence for one cleared layer', async () => {
    const deferred = createDeferred<CanvasImageUploadResult>();
    const uploadImage = vi.fn(() => deferred.promise);
    const h = createHarness({ uploadImage });

    h.store.markLayerDirty(LAYER);
    const barrier = h.store.flushPendingUploads();
    await drainUntil(() => uploadImage.mock.calls.length >= 1);

    h.store.discardLayer(LAYER);
    deferred.resolve({ height: 10, imageName: 'stale-mask', width: 10 });
    await barrier;
    await vi.advanceTimersByTimeAsync(3000);

    expect(h.dispatch).not.toHaveBeenCalled();
    h.store.dispose();
  });

  it.each(['discard', 'reset'] as const)(
    'does not resurrect dirty persistence when an in-flight upload rejects after %s',
    async (cancellation) => {
      const deferred = createDeferred<CanvasImageUploadResult>();
      const uploadImage = vi.fn(() => deferred.promise);
      const onError = vi.fn();
      const h = createHarness({ maxUploadAttempts: 1, onError, uploadImage });
      h.store.markLayerDirty(LAYER);
      const barrier = h.store.flushPendingUploads();
      await drainUntil(() => uploadImage.mock.calls.length >= 1);

      if (cancellation === 'discard') {
        h.store.discardLayer(LAYER);
      } else {
        h.store.reset();
      }
      deferred.reject(new Error('obsolete upload failed'));
      await barrier;
      await vi.advanceTimersByTimeAsync(3000);

      expect(uploadImage).toHaveBeenCalledOnce();
      expect(h.dispatch).not.toHaveBeenCalled();
      expect(onError).not.toHaveBeenCalled();
      await h.store.flushPendingUploads();
      expect(uploadImage).toHaveBeenCalledOnce();
      h.store.dispose();
    }
  );

  it('does not flush after dispose', async () => {
    const h = createHarness();
    h.store.markLayerDirty(LAYER);
    h.store.dispose();
    await vi.advanceTimersByTimeAsync(3000);
    expect(h.uploadImage).not.toHaveBeenCalled();
  });

  describe('source-type guard (rasterize → undo convergence)', () => {
    it('drops the dirty entry without dispatching when the debounce timer fires after the layer left `paint`', async () => {
      const h = createHarness();

      h.store.markLayerDirty(LAYER);
      h.setSource({
        fill: '#ff0000',
        height: 40,
        kind: 'rect',
        stroke: null,
        strokeWidth: 0,
        type: 'shape',
        width: 60,
      });

      await vi.advanceTimersByTimeAsync(1500);

      expect(h.uploadImage).not.toHaveBeenCalled();
      expect(h.dispatch).not.toHaveBeenCalled();
      h.store.dispose();
    });

    it('closes the race where the source leaves `paint` WHILE an upload is already in flight', async () => {
      const deferred = createDeferred<CanvasImageUploadResult>();
      const uploadImage = vi.fn(() => deferred.promise);
      const h = createHarness({ uploadImage });

      h.store.markLayerDirty(LAYER);
      const barrier = h.store.flushPendingUploads();

      await drainUntil(() => uploadImage.mock.calls.length >= 1);
      expect(uploadImage).toHaveBeenCalledTimes(1);

      h.setSource({ fill: '#000', height: 10, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 10 });

      deferred.resolve({ height: 10, imageName: 'img-x', width: 10 });
      await barrier;

      expect(h.dispatch).not.toHaveBeenCalled();
      h.store.dispose();
    });
  });

  describe('ambient failure handling', () => {
    it('reschedules a failed layer with growing backoff instead of the base debounce', async () => {
      const timers = createManualTimers();
      const uploadImage = vi.fn(() => Promise.reject(new Error('upload failed')));
      const h = createHarness({ maxUploadAttempts: 1, timers, uploadImage });

      h.store.markLayerDirty(LAYER);
      expect(timers.scheduledDelays).toEqual([1500]);

      timers.fireNext();
      await drainUntil(() => timers.scheduledDelays.length === 2);
      expect(timers.scheduledDelays[1]).toBe(DEFAULT_FAILURE_BACKOFF_MS[0]);

      timers.fireNext();
      await drainUntil(() => timers.scheduledDelays.length === 3);
      expect(timers.scheduledDelays[2]).toBe(DEFAULT_FAILURE_BACKOFF_MS[1]);

      timers.fireNext();
      await drainUntil(() => timers.scheduledDelays.length === 4);
      expect(timers.scheduledDelays[3]).toBe(DEFAULT_FAILURE_BACKOFF_MS[2]);

      h.store.dispose();
    });

    it('opens the circuit after maxConsecutiveFailures and stops rescheduling', async () => {
      const timers = createManualTimers();
      const uploadImage = vi.fn(() => Promise.reject(new Error('upload failed')));
      const onError = vi.fn();
      const h = createHarness({ maxConsecutiveFailures: 3, maxUploadAttempts: 1, onError, timers, uploadImage });

      h.store.markLayerDirty(LAYER);
      timers.fireNext();
      await drainUntil(() => timers.scheduledDelays.length === 2);
      timers.fireNext();
      await drainUntil(() => timers.scheduledDelays.length === 3);
      timers.fireNext();
      await drainUntil(() => onError.mock.calls.length === 2);
      await drainUntil(() => false, 10);

      expect(timers.pendingCount()).toBe(0);
      expect(timers.scheduledDelays).toHaveLength(3);

      await expect(h.store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');
      expect(uploadImage).toHaveBeenCalledTimes(4);

      h.store.dispose();
    });

    it('reports the first failure and the final failure only', async () => {
      const timers = createManualTimers();
      const uploadImage = vi.fn(() => Promise.reject(new Error('upload failed')));
      const onError = vi.fn();
      const h = createHarness({ maxConsecutiveFailures: 3, maxUploadAttempts: 1, onError, timers, uploadImage });

      h.store.markLayerDirty(LAYER);
      timers.fireNext();
      await drainUntil(() => timers.scheduledDelays.length === 2);
      timers.fireNext();
      await drainUntil(() => timers.scheduledDelays.length === 3);
      timers.fireNext();
      await drainUntil(() => onError.mock.calls.length === 2);

      expect(onError).toHaveBeenCalledTimes(2);
      expect(onError).toHaveBeenNthCalledWith(1, expect.any(Error), LAYER, {
        consecutiveFailures: 1,
        willRetry: true,
      });
      expect(onError).toHaveBeenNthCalledWith(2, expect.any(Error), LAYER, {
        consecutiveFailures: 3,
        willRetry: false,
      });

      h.store.dispose();
    });

    it('a new stroke closes the circuit and resets the failure count', async () => {
      const timers = createManualTimers();
      let shouldFail = true;
      const uploadImage = vi.fn(() =>
        shouldFail
          ? Promise.reject(new Error('upload failed'))
          : Promise.resolve<CanvasImageUploadResult>({ height: 10, imageName: 'img-recovered', width: 10 })
      );
      const onError = vi.fn();
      const h = createHarness({ maxConsecutiveFailures: 2, maxUploadAttempts: 1, onError, timers, uploadImage });

      h.store.markLayerDirty(LAYER);
      timers.fireNext();
      await drainUntil(() => timers.scheduledDelays.length === 2);
      timers.fireNext();
      await drainUntil(() => onError.mock.calls.length === 2);
      await drainUntil(() => false, 10);

      expect(timers.pendingCount()).toBe(0);

      shouldFail = false;
      h.store.markLayerDirty(LAYER);
      expect(timers.scheduledDelays.at(-1)).toBe(1500);

      timers.fireNext();
      await drainUntil(() => h.dispatch.mock.calls.length === 1);

      expect(h.dispatch).toHaveBeenCalledTimes(1);
      expect(onError).toHaveBeenCalledTimes(2);

      h.store.dispose();
    });
  });
});

describe('truthful extent: trimming and clearing', () => {
  const withBitmap: CanvasLayerSourceContract = {
    bitmap: { contentHash: 'h', height: 10, imageName: 'img-old', width: 10 },
    offset: { x: 0, y: 0 },
    type: 'paint',
  };
  const noSurface = (): 'empty' => 'empty';

  it('clears the ref instead of uploading when the trim finds no visible pixels', async () => {
    const h = createHarness({ getLayerSurface: noSurface, trimLayerPixels: () => 'emptied' });
    h.setSource(withBitmap);

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();

    expect(h.encodeSurface).not.toHaveBeenCalled();
    expect(h.uploadImage).not.toHaveBeenCalled();
    expect(h.dispatch).toHaveBeenCalledTimes(1);
    expect(h.dispatch.mock.calls[0][0]).toEqual({
      id: LAYER,
      source: { bitmap: null, type: 'paint' },
      type: 'updateCanvasLayerSource',
    });
    h.store.dispose();
  });

  it('requeues as a failure when the clear is rejected', async () => {
    const onError = vi.fn();
    const h = createHarness({
      clearBitmap: () => false,
      getLayerSurface: noSurface,
      onError,
      trimLayerPixels: () => 'emptied',
    });
    h.setSource(withBitmap);

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await expect(h.store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');
    expect(h.store.hasPendingClear(LAYER)).toBe(true);
    h.store.dispose();
  });

  it('cancels a pending clear when a new stroke restores visible pixels', async () => {
    let cacheEmpty = false;
    let trimmedOnce = false;
    const h = createHarness({
      clearBitmap: () => false,
      getLayerSurface: (surface, offset) => (cacheEmpty ? 'empty' : { offset, surface }),
      trimLayerPixels: () => {
        if (!trimmedOnce) {
          trimmedOnce = true;
          cacheEmpty = true;
          return 'emptied';
        }
        return 'kept';
      },
    });
    h.setSource(withBitmap);

    h.store.markLayerDirty(LAYER);
    await expect(h.store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');

    cacheEmpty = false;
    h.store.markLayerDirty(LAYER);
    await expect(h.store.flushPendingUploads()).resolves.toBeUndefined();

    expect(h.clearBitmap).toHaveBeenCalledOnce();
    expect(h.uploadImage).toHaveBeenCalledOnce();
    h.store.dispose();
  });

  it('reports a throwing trim as a typed persistence failure and keeps it retryable', async () => {
    const error = new Error('alpha readback failed');
    const onError = vi.fn();
    const h = createHarness({
      onError,
      trimLayerPixels: () => {
        throw error;
      },
    });
    h.setSource(withBitmap);

    h.store.markLayerDirty(LAYER);
    await expect(h.store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');
    await expect(h.store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');

    expect(h.trimLayerPixels).toHaveBeenCalledTimes(2);
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenNthCalledWith(1, error, LAYER, { consecutiveFailures: 1, willRetry: true });
    expect(h.encodeSurface).not.toHaveBeenCalled();
    h.store.dispose();
  });

  it('persists the POST-trim offset, proving the surface is re-read after trimming', async () => {
    const h = createHarness({
      offset: { x: 10, y: 10 },
      trimLayerPixels: () => {
        h.setOffset({ x: 34, y: 41 });
        return 'trimmed';
      },
    });

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();

    expect(h.dispatch.mock.calls[0][0]).toMatchObject({
      source: { offset: { x: 34, y: 41 }, type: 'paint' },
    });
    h.store.dispose();
  });

  it('never trims a layer whose source has converted away from paint', async () => {
    const h = createHarness({ trimLayerPixels: () => 'emptied' });
    h.store.markLayerDirty(LAYER);
    h.setSource({ height: 10, kind: 'rect', fill: '#fff', stroke: null, strokeWidth: 0, type: 'shape', width: 10 });

    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();

    expect(h.trimLayerPixels).not.toHaveBeenCalled();
    expect(h.dispatch).not.toHaveBeenCalled();
    h.store.dispose();
  });
});
