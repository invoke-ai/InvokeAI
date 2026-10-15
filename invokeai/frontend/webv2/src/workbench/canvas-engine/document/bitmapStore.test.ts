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

/**
 * Timers record delays and run only through `fireNext`, allowing exact ambient retry assertions without fake-clock
 * advancement.
 */
interface ManualTimers extends BitmapStoreTimers {
  /** Every delay ever passed to `setTimeout`, in call order. */
  readonly scheduledDelays: number[];
  /** Runs the oldest still-pending job's handler synchronously, if any. */
  fireNext(): void;
  /** Count of jobs queued but not yet fired or cleared. */
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

/** A resolvable deferred, so a test can hold an upload pending on demand. */
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

/** Drains microtasks until `predicate` is true (or `maxTicks` is exhausted). */
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
  /** The layer-local content-rect origin the surface sits at (default 0,0). */
  offset?: { x: number; y: number };
  sleep?: (ms: number) => Promise<void>;
  /** Absent ⇒ the seam is not wired at all, i.e. no trimming (the default). */
  trimLayerPixels?: (layerId: string) => PaintCacheTrim;
  clearBitmap?: (layerId: string) => boolean;
  getAuthoritativeLayerSource?: (layerId: string) => CanvasLayerSourceContract | null;
  /** Overrides the debounce/backoff timer seam (default: real fake timers via `vi`). */
  timers?: BitmapStoreTimers;
}

/** The default source: a plain paint layer, matching every pre-existing test's assumption. */
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
    // Deterministic content hash: the encoded blob's own text.
    hashBlob: options.hashBlob ?? ((blob) => blob.text()),
    maxConsecutiveFailures: options.maxConsecutiveFailures,
    maxUploadAttempts: options.maxUploadAttempts ?? 3,
    onError: options.onError,
    retryDelaysMs: [1],
    // Immediate backoff so retries don't depend on timer advancement.
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

  it('reports whether a layer still has pixels that are not represented by its persisted ref', async () => {
    const h = createHarness();

    expect(h.store.hasPendingWork(LAYER)).toBe(false);
    h.store.markLayerDirty(LAYER);
    expect(h.store.hasPendingWork(LAYER)).toBe(true);

    await h.store.flushPendingUploads();
    expect(h.store.hasPendingWork(LAYER)).toBe(false);
    h.store.dispose();
  });

  it('debounces a burst of strokes into a single flush', async () => {
    const h = createHarness();

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(500);
    h.store.markLayerDirty(LAYER); // resets the timer
    await vi.advanceTimersByTimeAsync(500);
    h.store.markLayerDirty(LAYER); // resets again
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

    // Same encoded pixels → same hash → dedupe hit, no second upload.
    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();

    expect(h.uploadImage).toHaveBeenCalledTimes(1);
    h.store.dispose();
  });

  it('dispatches a same-hash re-flush when the offset changed (pure-translation persistence)', async () => {
    const h = createHarness({ offset: { x: 0, y: 0 } });

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();
    expect(h.uploadImage).toHaveBeenCalledTimes(1);
    expect(h.dispatch).toHaveBeenCalledTimes(1);
    h.setSource({ bitmap: { height: 10, imageName: 'img-0', width: 10 }, offset: { x: 0, y: 0 }, type: 'paint' });

    // Transform-drag by +50px then Apply: the bake produces byte-identical
    // pixels (same hash → dedupe hit, no upload) but the content sits at a new
    // offset. The dispatch that persists the moved offset must still fire.
    h.setOffset({ x: 50, y: 0 });
    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();

    // No new upload (pixels deduped), but a fresh dispatch carrying the offset.
    expect(h.uploadImage).toHaveBeenCalledTimes(1);
    expect(h.dispatch).toHaveBeenCalledTimes(2);
    expect(h.dispatch.mock.calls.at(-1)?.[0]).toMatchObject({
      source: { bitmap: { imageName: 'img-0' }, offset: { x: 50, y: 0 }, type: 'paint' },
      type: 'updateCanvasLayerSource',
    });
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

  it('routes the swap through dispatchBitmap when provided (mask persistence seam), not the default dispatch', async () => {
    const surface = createTestStubRasterBackend().createSurface(10, 10);
    const dispatch = vi.fn((_action: CanvasProjectMutation) => true);
    const dispatchBitmap = vi.fn(
      (_layerId: string, _bitmap: CanvasImageRef, _offset: { x: number; y: number }) => true
    );
    const store = createBitmapStore({
      debounceMs: 1500,
      dispatch,
      dispatchBitmap,
      encodeSurface: () => Promise.resolve(new Blob(['pixels'], { type: 'image/png' })),
      getLayerSource: () => ({ bitmap: null, type: 'paint' }),
      getLayerSurface: () => ({ offset: { x: 7, y: 8 }, surface }),
      hashBlob: (blob) => blob.text(),
      retryDelaysMs: [1],
      sleep: () => Promise.resolve(),
      uploadImage: () => Promise.resolve({ height: 10, imageName: 'mask-img', width: 10 }),
    });

    store.markLayerDirty('mask1');
    await vi.advanceTimersByTimeAsync(1500);
    await store.flushPendingUploads();

    // The engine receives ref and offset and chooses mask actions; default paint dispatch must remain unused.
    expect(dispatchBitmap).toHaveBeenCalledTimes(1);
    expect(dispatchBitmap).toHaveBeenCalledWith('mask1', expect.objectContaining({ imageName: 'mask-img' }), {
      x: 7,
      y: 8,
    });
    expect(dispatch).not.toHaveBeenCalled();
    store.dispose();
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

  it('retains dirty pixels when a legacy dispatcher returns no acceptance signal', async () => {
    const surface = createTestStubRasterBackend().createSurface(10, 10);
    const dispatchBitmap = vi.fn(() => undefined) as unknown as (
      layerId: string,
      bitmap: CanvasImageRef,
      offset: { x: number; y: number }
    ) => boolean;
    const store = createBitmapStore({
      debounceMs: 1500,
      dispatch: vi.fn(() => true),
      dispatchBitmap,
      encodeSurface: () => Promise.resolve(new Blob(['pixels'], { type: 'image/png' })),
      getLayerSource: () => ({ bitmap: null, type: 'paint' }),
      getLayerSurface: () => ({ offset: { x: 0, y: 0 }, surface }),
      hashBlob: (blob) => blob.text(),
      uploadImage: () => Promise.resolve({ height: 10, imageName: 'missing-acceptance-img', width: 10 }),
    });

    store.markLayerDirty(LAYER);
    await expect(store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');
    await expect(store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');

    expect(dispatchBitmap).toHaveBeenCalledTimes(2);
    store.dispose();
  });

  it.each(['dispatch', 'dispatchBitmap'] as const)(
    'retains dirty pixels when %s throws before the intended bitmap lands',
    async (dispatchSeam) => {
      const surface = createTestStubRasterBackend().createSurface(10, 10);
      let shouldThrow = true;
      const throwingDispatch = vi.fn(() => {
        if (shouldThrow) {
          throw new Error('subscriber failed before commit');
        }
        return true;
      });
      const onError = vi.fn();
      const store = createBitmapStore({
        debounceMs: 1500,
        dispatch: dispatchSeam === 'dispatch' ? throwingDispatch : vi.fn(() => true),
        dispatchBitmap: dispatchSeam === 'dispatchBitmap' ? throwingDispatch : undefined,
        encodeSurface: () => Promise.resolve(new Blob(['pixels'], { type: 'image/png' })),
        getLayerSource: () => ({ bitmap: null, type: 'paint' }),
        getLayerSurface: () => ({ offset: { x: 0, y: 0 }, surface }),
        hashBlob: (blob) => blob.text(),
        onError,
        uploadImage: () => Promise.resolve({ height: 10, imageName: 'retry-after-throw.png', width: 10 }),
      });

      store.markLayerDirty(LAYER);
      await expect(store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');

      expect(throwingDispatch).toHaveBeenCalledOnce();
      expect(onError).toHaveBeenCalledOnce();
      expect(
        store.isSelfEcho(LAYER, {
          bitmap: { height: 10, imageName: 'retry-after-throw.png', width: 10 },
          type: 'paint',
        })
      ).toBe(false);

      shouldThrow = false;
      await store.flushPendingUploads();

      expect(throwingDispatch).toHaveBeenCalledTimes(2);
      store.dispose();
    }
  );

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

    // Paint state A → upload img-0.
    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();
    // Paint state B → upload img-1.
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
    // The contract converges back to img-0 (the previously uploaded state-A image).
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

    // Upload is in flight; the contract keeps its old ref (no dispatch yet).
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

    // Retried up to the attempt cap, then gave up without dispatching.
    expect(uploadImage).toHaveBeenCalledTimes(2);
    expect(dispatch).not.toHaveBeenCalled();
    expect(onError).toHaveBeenCalledTimes(1);

    // The layer is still dirty; a subsequent flush succeeds and dispatches.
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
    // Do NOT advance to the debounce window; the barrier must flush immediately.
    const barrier = h.store.flushPendingUploads();

    let settled = false;
    void barrier.then(() => {
      settled = true;
    });

    // Drain encode/hash awaits while leaving the upload unresolved.
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

  it('records dirty work marked during suspension without scheduling until release', async () => {
    const h = createHarness();
    const release = h.store.suspendLayer(LAYER);
    h.store.markLayerDirty(LAYER);

    await vi.advanceTimersByTimeAsync(3000);
    expect(h.encodeSurface).not.toHaveBeenCalled();

    release();
    await h.store.flushPendingUploads();

    expect(h.encodeSurface).toHaveBeenCalledOnce();
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

  it('supports nested suspension leases and idempotent release', async () => {
    const h = createHarness();
    h.store.markLayerDirty(LAYER);
    const releaseOuter = h.store.suspendLayer(LAYER);
    const releaseInner = h.store.suspendLayer(LAYER);

    releaseOuter();
    releaseOuter();
    await vi.advanceTimersByTimeAsync(3000);
    expect(h.uploadImage).not.toHaveBeenCalled();

    releaseInner();
    await h.store.flushPendingUploads();
    expect(h.uploadImage).toHaveBeenCalledOnce();
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

  it('fails fast instead of waiting when the caller cannot wait on an open edit', async () => {
    const h = createHarness();
    h.store.markLayerDirty(LAYER);
    const release = h.store.suspendLayer(LAYER);

    await expect(h.store.flushPendingUploads({ waitForHeldPixels: false })).rejects.toMatchObject({
      layerIds: [LAYER],
      reason: 'held',
    });
    expect(h.store.hasPendingWork(LAYER)).toBe(true);

    release();
    await h.store.flushPendingUploads({ waitForHeldPixels: false });
    expect(h.dispatch).toHaveBeenCalledOnce();
    h.store.dispose();
  });

  it('fails fast on pixels a session defers instead of polling', async () => {
    let busy = true;
    const h = createHarness({ trimLayerPixels: () => (busy ? 'deferred' : 'kept') });
    h.store.markLayerDirty(LAYER);

    await expect(h.store.flushPendingUploads({ waitForHeldPixels: false })).rejects.toMatchObject({ reason: 'held' });
    expect(h.uploadImage).not.toHaveBeenCalled();

    busy = false;
    await h.store.flushPendingUploads({ waitForHeldPixels: false });
    expect(h.uploadImage).toHaveBeenCalledOnce();
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

    // Drain the encode→hash microtask chain: the first (stale) upload is in flight.
    await drainUntil(() => uploadImage.mock.calls.length >= 1);
    expect(uploadImage).toHaveBeenCalledTimes(1);
    expect(settled).toBe(false);

    h.setEncoded('pixels-B');
    h.store.markLayerDirty(LAYER);

    // The barrier must flush the newer stroke before settling, despite completion of the stale upload.
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
      // Fix ambient retries at 1500ms to isolate barrier anti-spin; dedicated tests cover backoff.
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

    // A barrier must not retry a flush already failed in this call beyond `maxUploadAttempts`.
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

    // The dispatch's own round-trip is a self-echo → the engine skips re-raster.
    expect(h.store.isSelfEcho(LAYER, appliedSource)).toBe(true);

    // A different bitmap (undo/import) is NOT an echo → must re-rasterize.
    const otherPaint: CanvasLayerSourceContract = {
      bitmap: { height: 10, imageName: 'other', width: 10 },
      type: 'paint',
    };
    expect(h.store.isSelfEcho(LAYER, otherPaint)).toBe(false);

    // A different layer, and non-paint / null sources, are never echoes.
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

    // Wholesale replacement must discard self-echo state before layer ids are reused.
    h.store.reset();
    expect(h.store.isSelfEcho(LAYER, applied.source)).toBe(false);

    // Hash dedupe reuses the upload, but cleared self-echo state permits the dispatch needed to converge.
    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();
    expect(h.dispatch).toHaveBeenCalledTimes(2);
    expect(h.uploadImage).toHaveBeenCalledTimes(1);

    h.store.dispose();
  });

  it('reset() cancels a pending debounced flush for the outgoing document', async () => {
    const h = createHarness();

    h.store.markLayerDirty(LAYER);
    h.store.reset();
    await vi.advanceTimersByTimeAsync(3000);

    expect(h.uploadImage).not.toHaveBeenCalled();
    expect(h.dispatch).not.toHaveBeenCalled();

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

  it.each(['discard', 'reset'] as const)(
    'does not resurrect dirty persistence when in-flight encoding rejects after %s',
    async (cancellation) => {
      const deferred = createDeferred<Blob>();
      const encodeSurface = vi.fn(() => deferred.promise);
      const onError = vi.fn();
      const h = createHarness({ encodeSurface, onError });
      h.store.markLayerDirty(LAYER);
      const barrier = h.store.flushPendingUploads();
      await drainUntil(() => encodeSurface.mock.calls.length >= 1);

      if (cancellation === 'discard') {
        h.store.discardLayer(LAYER);
      } else {
        h.store.reset();
      }
      deferred.reject(new Error('obsolete encode failed'));
      await barrier;
      await vi.advanceTimersByTimeAsync(3000);

      expect(encodeSurface).toHaveBeenCalledOnce();
      expect(h.uploadImage).not.toHaveBeenCalled();
      expect(h.dispatch).not.toHaveBeenCalled();
      expect(onError).not.toHaveBeenCalled();
      await h.store.flushPendingUploads();
      expect(encodeSurface).toHaveBeenCalledOnce();
      h.store.dispose();
    }
  );

  it.each(['discard', 'reset'] as const)(
    'does not hash or upload when in-flight encoding fulfills after %s',
    async (cancellation) => {
      const deferred = createDeferred<Blob>();
      const encodeSurface = vi.fn(() => deferred.promise);
      const hashBlob = vi.fn((blob: Blob) => blob.text());
      const h = createHarness({ encodeSurface, hashBlob });
      h.store.markLayerDirty(LAYER);
      const barrier = h.store.flushPendingUploads();
      await drainUntil(() => encodeSurface.mock.calls.length >= 1);

      if (cancellation === 'discard') {
        h.store.discardLayer(LAYER);
      } else {
        h.store.reset();
      }
      deferred.resolve(new Blob(['obsolete pixels'], { type: 'image/png' }));
      await barrier;

      expect(hashBlob).not.toHaveBeenCalled();
      expect(h.uploadImage).not.toHaveBeenCalled();
      expect(h.dispatch).not.toHaveBeenCalled();
      h.store.dispose();
    }
  );

  it.each(['discard', 'reset'] as const)(
    'does not upload when in-flight hashing fulfills after %s',
    async (cancellation) => {
      const deferred = createDeferred<string>();
      const hashBlob = vi.fn(() => deferred.promise);
      const h = createHarness({ hashBlob });
      h.store.markLayerDirty(LAYER);
      const barrier = h.store.flushPendingUploads();
      await drainUntil(() => hashBlob.mock.calls.length >= 1);

      if (cancellation === 'discard') {
        h.store.discardLayer(LAYER);
      } else {
        h.store.reset();
      }
      deferred.resolve('obsolete-hash');
      await barrier;

      expect(h.uploadImage).not.toHaveBeenCalled();
      expect(h.dispatch).not.toHaveBeenCalled();
      h.store.dispose();
    }
  );

  it.each(['discard', 'reset'] as const)(
    'stops obsolete upload retries when %s lands during retry backoff',
    async (cancellation) => {
      const backoff = createDeferred<void>();
      const sleep = vi.fn(() => backoff.promise);
      const uploadImage = vi.fn(() => Promise.reject(new Error('retry me')));
      const onError = vi.fn();
      const h = createHarness({ maxUploadAttempts: 3, onError, sleep, uploadImage });
      h.store.markLayerDirty(LAYER);
      const barrier = h.store.flushPendingUploads();
      await drainUntil(() => sleep.mock.calls.length >= 1);
      expect(uploadImage).toHaveBeenCalledOnce();

      if (cancellation === 'discard') {
        h.store.discardLayer(LAYER);
      } else {
        h.store.reset();
      }
      backoff.resolve();
      await barrier;

      expect(uploadImage).toHaveBeenCalledOnce();
      expect(onError).not.toHaveBeenCalled();
      expect(h.dispatch).not.toHaveBeenCalled();
      h.store.dispose();
    }
  );

  it('allows fresh same-id persistence after repeated idle discards', async () => {
    const h = createHarness();
    for (let i = 0; i < 1_000; i += 1) {
      h.store.discardLayer(`removed-${i}`);
    }

    h.store.discardLayer(LAYER);
    h.store.markLayerDirty(LAYER);
    await h.store.flushPendingUploads();

    expect(h.uploadImage).toHaveBeenCalledOnce();
    expect(h.dispatch).toHaveBeenCalledOnce();
    h.store.dispose();
  });

  it('preserves a fresh same-id generation while obsolete encoding settles', async () => {
    const obsoleteEncode = createDeferred<Blob>();
    let encodeCall = 0;
    const encodeSurface = vi.fn(() => {
      encodeCall += 1;
      return encodeCall === 1
        ? obsoleteEncode.promise
        : Promise.resolve(new Blob(['fresh pixels'], { type: 'image/png' }));
    });
    const hashBlob = vi.fn((blob: Blob) => blob.text());
    const h = createHarness({ encodeSurface, hashBlob });
    h.store.markLayerDirty(LAYER);
    const barrier = h.store.flushPendingUploads();
    await drainUntil(() => encodeSurface.mock.calls.length === 1);

    h.store.discardLayer(LAYER);
    h.store.markLayerDirty(LAYER);
    obsoleteEncode.resolve(new Blob(['obsolete pixels'], { type: 'image/png' }));
    await barrier;

    expect(encodeSurface).toHaveBeenCalledTimes(2);
    expect(hashBlob).toHaveBeenCalledOnce();
    expect(await hashBlob.mock.calls[0]![0].text()).toBe('fresh pixels');
    expect(h.uploadImage).toHaveBeenCalledOnce();
    expect(h.dispatch).toHaveBeenCalledOnce();
    h.store.dispose();
  });

  it.each(['discard', 'reset'] as const)(
    'lets an error observer %s failed persistence without a later dirty resurrection',
    async (cancellation) => {
      const uploadImage = vi.fn(() => Promise.reject(new Error('upload failed')));
      let store: ReturnType<typeof createBitmapStore> | null = null;
      const onError = vi.fn(() => {
        if (cancellation === 'discard') {
          store?.discardLayer(LAYER);
        } else {
          store?.reset();
        }
      });
      const h = createHarness({ maxUploadAttempts: 1, onError, uploadImage });
      store = h.store;

      h.store.markLayerDirty(LAYER);
      await h.store.flushPendingUploads();
      await vi.advanceTimersByTimeAsync(3000);

      expect(onError).toHaveBeenCalledOnce();
      expect(uploadImage).toHaveBeenCalledOnce();
      expect(h.dispatch).not.toHaveBeenCalled();
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
      // Source swaps preserve cache surfaces; the guard must stop delayed paint persistence from overwriting a
      // restored parametric source.
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

    it('drops the dirty entry without dispatching when `flushPendingUploads` is awaited after the layer left `paint`', async () => {
      const h = createHarness();

      h.store.markLayerDirty(LAYER);
      h.setSource({ angle: 0, kind: 'linear', stops: [{ color: '#000', offset: 0 }], type: 'gradient' });

      // Do NOT advance timers: exercise the barrier path (e.g. pressing Invoke
      // right after the undo), not just the debounce path.
      await h.store.flushPendingUploads();

      expect(h.uploadImage).not.toHaveBeenCalled();
      expect(h.dispatch).not.toHaveBeenCalled();
      h.store.dispose();
    });

    it('drops the dirty entry without dispatching when the layer no longer exists', async () => {
      const h = createHarness();

      h.store.markLayerDirty(LAYER);
      h.setSource(null);

      await h.store.flushPendingUploads();

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

      // The source changes away from `paint` DURING the in-flight upload —
      // later than the entry-time check, so only the pre-dispatch recheck
      // catches it.
      h.setSource({ fill: '#000', height: 10, kind: 'rect', stroke: null, strokeWidth: 0, type: 'shape', width: 10 });

      deferred.resolve({ height: 10, imageName: 'img-x', width: 10 });
      await barrier;

      expect(h.dispatch).not.toHaveBeenCalled();
      h.store.dispose();
    });

    it('still flushes normally when the layer stays a paint layer (no regression)', async () => {
      const h = createHarness();

      h.store.markLayerDirty(LAYER);
      await vi.advanceTimersByTimeAsync(1500);
      await h.store.flushPendingUploads();

      expect(h.uploadImage).toHaveBeenCalledTimes(1);
      expect(h.dispatch).toHaveBeenCalledTimes(1);
      expect(h.dispatch.mock.calls[0][0]).toMatchObject({
        id: LAYER,
        source: { type: 'paint' },
        type: 'updateCanvasLayerSource',
      });
      h.store.dispose();
    });
  });

  describe('ambient failure handling', () => {
    it('reschedules a failed layer with growing backoff instead of the base debounce', async () => {
      const timers = createManualTimers();
      const uploadImage = vi.fn(() => Promise.reject(new Error('upload failed')));
      const h = createHarness({ maxUploadAttempts: 1, timers, uploadImage });

      h.store.markLayerDirty(LAYER);
      // The very first schedule is a fresh stroke: the base debounce, not backoff.
      expect(timers.scheduledDelays).toEqual([1500]);

      timers.fireNext(); // 1st ambient flush → fails.
      await drainUntil(() => timers.scheduledDelays.length === 2);
      expect(timers.scheduledDelays[1]).toBe(DEFAULT_FAILURE_BACKOFF_MS[0]);

      timers.fireNext(); // 2nd ambient flush → fails again.
      await drainUntil(() => timers.scheduledDelays.length === 3);
      expect(timers.scheduledDelays[2]).toBe(DEFAULT_FAILURE_BACKOFF_MS[1]);

      timers.fireNext(); // 3rd ambient flush → fails again.
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
      timers.fireNext(); // failure 1 of 3 → still retrying.
      await drainUntil(() => timers.scheduledDelays.length === 2);
      timers.fireNext(); // failure 2 of 3 → still retrying.
      await drainUntil(() => timers.scheduledDelays.length === 3);
      timers.fireNext(); // failure 3 of 3 → circuit opens.
      await drainUntil(() => onError.mock.calls.length === 2);
      // Let the settling flush's `finally` run to completion (it schedules
      // nothing on this branch, so there is no state change to await directly).
      await drainUntil(() => false, 10);

      expect(timers.pendingCount()).toBe(0);
      expect(timers.scheduledDelays).toHaveLength(3);

      // The layer remains dirty, not dropped: a fresh barrier call still
      // attempts it (the breaker only gates the AMBIENT reschedule).
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

    it('does not re-report when a barrier retries an already-open circuit', async () => {
      const timers = createManualTimers();
      const uploadImage = vi.fn(() => Promise.reject(new Error('upload failed')));
      const onError = vi.fn();
      const h = createHarness({ maxConsecutiveFailures: 3, maxUploadAttempts: 1, onError, timers, uploadImage });

      h.store.markLayerDirty(LAYER);
      timers.fireNext(); // failure 1 of 3.
      await drainUntil(() => timers.scheduledDelays.length === 2);
      timers.fireNext(); // failure 2 of 3.
      await drainUntil(() => timers.scheduledDelays.length === 3);
      timers.fireNext(); // failure 3 of 3 → circuit opens.
      await drainUntil(() => onError.mock.calls.length === 2);
      await drainUntil(() => false, 10);

      expect(uploadImage).toHaveBeenCalledTimes(3);
      expect(onError).toHaveBeenCalledTimes(2);

      // Explicit persistence barriers bypass the ambient breaker, but repeated failure must not repeatedly report
      // the same streak.
      await expect(h.store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');
      await expect(h.store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');

      expect(uploadImage).toHaveBeenCalledTimes(5);
      expect(onError).toHaveBeenCalledTimes(2);

      h.store.dispose();
    });

    it('reports exactly once when maxConsecutiveFailures is 1 (opens on the very first failure)', async () => {
      const timers = createManualTimers();
      const uploadImage = vi.fn(() => Promise.reject(new Error('upload failed')));
      const onError = vi.fn();
      const h = createHarness({ maxConsecutiveFailures: 1, maxUploadAttempts: 1, onError, timers, uploadImage });

      h.store.markLayerDirty(LAYER);
      timers.fireNext(); // the only ambient attempt: the circuit opens immediately.
      await drainUntil(() => onError.mock.calls.length === 1);
      await drainUntil(() => false, 10);

      expect(timers.pendingCount()).toBe(0);
      expect(onError).toHaveBeenCalledTimes(1);
      expect(onError).toHaveBeenNthCalledWith(1, expect.any(Error), LAYER, {
        consecutiveFailures: 1,
        willRetry: false,
      });

      // Further barrier retries against the already-open circuit stay silent.
      await expect(h.store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');
      await expect(h.store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');

      expect(onError).toHaveBeenCalledTimes(1);

      h.store.dispose();
    });

    it('reports a real failure even when silent declines already advanced the streak', async () => {
      const surface = createTestStubRasterBackend().createSurface(10, 10);
      let content = 'pixels-A';
      let uploadShouldFail = false;
      const encodeSurface = vi.fn(() => Promise.resolve(new Blob([content], { type: 'image/png' })));
      const uploadImage = vi.fn(() => {
        if (uploadShouldFail) {
          return Promise.reject(new Error('network unreachable'));
        }
        return Promise.resolve<CanvasImageUploadResult>({ height: 10, imageName: 'img-decline', width: 10 });
      });
      const dispatchBitmap = vi.fn(() => false);
      const onError = vi.fn();
      const store = createBitmapStore({
        debounceMs: 1500,
        dispatch: vi.fn(() => true),
        dispatchBitmap,
        encodeSurface,
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
      // Two silent declines advance the streak without reporting anything.
      await expect(store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');
      await expect(store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');
      expect(onError).not.toHaveBeenCalled();

      // A real failure midway through a decline streak must still be reported.
      content = 'pixels-B';
      uploadShouldFail = true;
      await expect(store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');

      expect(onError).toHaveBeenCalledOnce();
      expect(onError).toHaveBeenCalledWith(expect.any(Error), LAYER, { consecutiveFailures: 3, willRetry: true });

      store.dispose();
    });

    it('reports when silent declines alone open the circuit', async () => {
      const surface = createTestStubRasterBackend().createSurface(10, 10);
      const dispatchBitmap = vi.fn(() => false);
      const onError = vi.fn();
      const store = createBitmapStore({
        debounceMs: 1500,
        dispatch: vi.fn(() => true),
        dispatchBitmap,
        encodeSurface: () => Promise.resolve(new Blob(['pixels'], { type: 'image/png' })),
        getLayerSource: () => PAINT_SOURCE,
        getLayerSurface: () => ({ offset: { x: 0, y: 0 }, surface }),
        hashBlob: (blob) => blob.text(),
        maxConsecutiveFailures: 3,
        onError,
        uploadImage: () => Promise.resolve({ height: 10, imageName: 'img-decline', width: 10 }),
      });

      store.markLayerDirty(LAYER);
      await expect(store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed'); // decline 1 of 3.
      await expect(store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed'); // decline 2 of 3.
      expect(onError).not.toHaveBeenCalled();

      // decline 3 of 3 opens the circuit — silently opened, but must still be heard.
      await expect(store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');

      expect(onError).toHaveBeenCalledOnce();
      expect(onError).toHaveBeenCalledWith(expect.any(Error), LAYER, { consecutiveFailures: 3, willRetry: false });

      store.dispose();
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
      timers.fireNext(); // failure 1 of 2.
      await drainUntil(() => timers.scheduledDelays.length === 2);
      timers.fireNext(); // failure 2 of 2 → circuit opens.
      await drainUntil(() => onError.mock.calls.length === 2);
      await drainUntil(() => false, 10);

      expect(timers.pendingCount()).toBe(0);

      // A fresh stroke resets to base debounce; successful upload clears the failure streak.
      shouldFail = false;
      h.store.markLayerDirty(LAYER);
      expect(timers.scheduledDelays.at(-1)).toBe(1500);

      timers.fireNext();
      await drainUntil(() => h.dispatch.mock.calls.length === 1);

      expect(h.dispatch).toHaveBeenCalledTimes(1);
      // The recovered flush does not report anything new.
      expect(onError).toHaveBeenCalledTimes(2);

      h.store.dispose();
    });

    it('a successful flush resets the failure count', async () => {
      const timers = createManualTimers();
      let shouldFail = true;
      const uploadImage = vi.fn(() =>
        shouldFail
          ? Promise.reject(new Error('upload failed'))
          : Promise.resolve<CanvasImageUploadResult>({ height: 10, imageName: 'img-ok', width: 10 })
      );
      const onError = vi.fn();
      const h = createHarness({ maxUploadAttempts: 1, onError, timers, uploadImage });

      h.store.markLayerDirty(LAYER);
      timers.fireNext(); // fails once → reported, backoff[0] scheduled.
      await drainUntil(() => timers.scheduledDelays.length === 2);
      expect(timers.scheduledDelays[1]).toBe(DEFAULT_FAILURE_BACKOFF_MS[0]);
      expect(onError).toHaveBeenCalledTimes(1);

      shouldFail = false;
      timers.fireNext(); // succeeds → clears the failure count.
      await drainUntil(() => h.dispatch.mock.calls.length === 1);

      shouldFail = true;
      h.setEncoded('pixels-B');
      h.store.markLayerDirty(LAYER);
      timers.fireNext();
      await drainUntil(() => onError.mock.calls.length === 2);

      expect(onError).toHaveBeenNthCalledWith(2, expect.any(Error), LAYER, {
        consecutiveFailures: 1,
        willRetry: true,
      });
      expect(timers.scheduledDelays.at(-1)).toBe(DEFAULT_FAILURE_BACKOFF_MS[0]);

      h.store.dispose();
    });
  });
});

describe('truthful extent: trimming and clearing', () => {
  /** A paint source already pointing at an uploaded bitmap. */
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

    // A transparent PNG must never be encoded or uploaded.
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

  it('prefers the injected clearBitmap seam over the default dispatch', async () => {
    const h = createHarness({ clearBitmap: () => true, getLayerSurface: noSurface, trimLayerPixels: () => 'emptied' });
    h.setSource(withBitmap);

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();

    expect(h.clearBitmap).toHaveBeenCalledWith(LAYER);
    expect(h.dispatch).not.toHaveBeenCalled();
    h.store.dispose();
  });

  it('does not dispatch when the document already holds no bitmap', async () => {
    const h = createHarness({ getLayerSurface: noSurface, trimLayerPixels: () => 'emptied' });
    // Default source is `{ bitmap: null }` — the clear would be a no-op.

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();

    expect(h.dispatch).not.toHaveBeenCalled();
    h.store.dispose();
  });

  it('drops a stale self-echo entry when the document already holds no bitmap', async () => {
    let erased = false;
    const h = createHarness({
      getLayerSurface: (surface, offset) => (erased ? 'empty' : { offset, surface }),
      trimLayerPixels: () => (erased ? 'emptied' : 'kept'),
    });

    h.store.markLayerDirty(LAYER);
    await h.store.flushPendingUploads();
    const applied = (h.dispatch.mock.calls[0][0] as { source: { bitmap: CanvasImageRef } }).source.bitmap;
    expect(h.store.isSelfEcho(LAYER, { bitmap: applied, type: 'paint' })).toBe(true);

    erased = true;
    h.setSource({ bitmap: null, type: 'paint' });
    h.store.markLayerDirty(LAYER);
    await h.store.flushPendingUploads();

    expect(h.store.isSelfEcho(LAYER, { bitmap: applied, type: 'paint' })).toBe(false);
    expect(h.dispatch).toHaveBeenCalledTimes(1);
    h.store.dispose();
  });

  it('drops the self-echo entry, so a later undo re-dispatching the old name is not mistaken for an echo', async () => {
    let erased = false;
    const h = createHarness({
      getLayerSurface: (surface, offset) => (erased ? 'empty' : { offset, surface }),
      trimLayerPixels: () => (erased ? 'emptied' : 'kept'),
    });

    // First flush establishes `lastApplied` for the uploaded image.
    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();
    const applied = (h.dispatch.mock.calls[0][0] as { source: { bitmap: CanvasImageRef } }).source.bitmap;
    expect(h.store.isSelfEcho(LAYER, { bitmap: applied, type: 'paint' })).toBe(true);

    // Now the layer is erased and the trim empties it.
    erased = true;
    h.setSource({ bitmap: applied, offset: { x: 0, y: 0 }, type: 'paint' });
    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();

    expect(h.store.isSelfEcho(LAYER, { bitmap: applied, type: 'paint' })).toBe(false);
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

  it('retries a rejected clear after the trim has already collapsed the cache', async () => {
    let cacheEmpty = false;
    let clearAttempts = 0;
    const h = createHarness({
      clearBitmap: () => {
        clearAttempts += 1;
        return clearAttempts === 2;
      },
      getLayerSurface: (surface, offset) => (cacheEmpty ? 'empty' : { offset, surface }),
      trimLayerPixels: () => {
        if (cacheEmpty) {
          return 'kept';
        }
        cacheEmpty = true;
        return 'emptied';
      },
    });
    h.setSource(withBitmap);

    h.store.markLayerDirty(LAYER);
    await expect(h.store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');
    await expect(h.store.flushPendingUploads()).resolves.toBeUndefined();

    expect(h.clearBitmap).toHaveBeenCalledTimes(2);
    expect(h.encodeSurface).not.toHaveBeenCalled();
    expect(h.uploadImage).not.toHaveBeenCalled();
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

  it('cancels a pending clear when visible pixels are restored without a new dirty mark', async () => {
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

    // A cache re-rasterization may restore pixels without going through
    // markLayerDirty(). The pending clear must respect the fresh trim/surface
    // verdict instead of deleting those visible pixels from the document.
    cacheEmpty = false;
    await expect(h.store.flushPendingUploads()).resolves.toBeUndefined();

    expect(h.clearBitmap).toHaveBeenCalledOnce();
    expect(h.uploadImage).toHaveBeenCalledOnce();
    h.store.dispose();
  });

  it('lets a surface restored during trim cancel a pending clear in the same flush', async () => {
    let cacheEmpty = true;
    let primedPendingClear = false;
    let restoreDuringTrim = false;
    const h = createHarness({
      clearBitmap: () => false,
      getLayerSurface: (surface, offset) => (cacheEmpty ? 'empty' : { offset, surface }),
      trimLayerPixels: () => {
        if (!primedPendingClear) {
          primedPendingClear = true;
          return 'emptied';
        }
        if (restoreDuringTrim) {
          cacheEmpty = false;
          return 'emptied';
        }
        return 'kept';
      },
    });
    h.setSource(withBitmap);

    h.store.markLayerDirty(LAYER);
    await expect(h.store.flushPendingUploads()).rejects.toThrow('Canvas pixel persistence failed');

    restoreDuringTrim = true;
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
    // Both failures advance the shared breaker; report only the streak's first failure.
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenNthCalledWith(1, error, LAYER, { consecutiveFailures: 1, willRetry: true });
    expect(h.encodeSurface).not.toHaveBeenCalled();
    h.store.dispose();
  });

  it('treats a throwing clear that nevertheless landed as success', async () => {
    let cleared = false;
    const onError = vi.fn();
    const h = createHarness({
      clearBitmap: () => {
        cleared = true;
        throw new Error('mirror notified synchronously and threw');
      },
      getAuthoritativeLayerSource: () => (cleared ? { bitmap: null, type: 'paint' } : withBitmap),
      onError,
      trimLayerPixels: () => 'emptied',
    });
    h.setSource(withBitmap);

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();

    expect(onError).not.toHaveBeenCalled();
    h.store.dispose();
  });

  it('waits through a transient trim deferral without reporting persistence failure', async () => {
    let busy = true;
    const sleep = vi.fn(() => {
      busy = false;
      return Promise.resolve();
    });
    const h = createHarness({ sleep, trimLayerPixels: () => (busy ? 'deferred' : 'kept') });

    h.store.markLayerDirty(LAYER);
    await expect(h.store.flushPendingUploads()).resolves.toBeUndefined();

    expect(sleep).toHaveBeenCalled();
    expect(h.trimLayerPixels).toHaveBeenCalledTimes(2);
    expect(h.uploadImage).toHaveBeenCalledTimes(1);
    expect(h.dispatch).toHaveBeenCalledTimes(1);
    h.store.dispose();
  });

  it('persists the POST-trim offset, proving the surface is re-read after trimming', async () => {
    // The trim moves the cache origin; the flush must carry the new one.
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

    // Trimming a parametric-backed cache would break the compositor's rect invariant.
    expect(h.trimLayerPixels).not.toHaveBeenCalled();
    expect(h.dispatch).not.toHaveBeenCalled();
    h.store.dispose();
  });
});
