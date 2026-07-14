import type { CanvasImageUploadResult } from '@workbench/canvas-engine/document/imageUpload';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { CanvasImageRef, CanvasLayerSourceContract } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { createBitmapStore } from './bitmapStore';

const LAYER = 'layer-1';

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
  hashBlob?: (blob: Blob) => Promise<string>;
  uploadImage?: (blob: Blob) => Promise<CanvasImageUploadResult>;
  maxUploadAttempts?: number;
  onError?: (error: unknown, layerId: string) => void;
  /** The layer-local content-rect origin the surface sits at (default 0,0). */
  offset?: { x: number; y: number };
  sleep?: (ms: number) => Promise<void>;
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
    vi.fn(
      (_blob: Blob): Promise<CanvasImageUploadResult> =>
        Promise.resolve({ height: 10, imageName: `img-${uploadSeq++}`, width: 10 })
    );
  const dispatch = vi.fn<(action: WorkbenchAction) => void>();

  const store = createBitmapStore({
    debounceMs: 1500,
    dispatch,
    encodeSurface,
    getLayerSource: () => source,
    getLayerSurface: () => ({ offset, surface }),
    // Deterministic content hash: the encoded blob's own text.
    hashBlob: options.hashBlob ?? ((blob) => blob.text()),
    maxUploadAttempts: options.maxUploadAttempts ?? 3,
    onError: options.onError,
    retryDelaysMs: [1],
    // Immediate backoff so retries don't depend on timer advancement.
    sleep: options.sleep ?? (() => Promise.resolve()),
    uploadImage,
  });

  return {
    dispatch,
    encodeSurface,
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

    // First flush: paint pixels at the origin → uploads img-0, document now
    // points at { imageName: 'img-0', offset: { x: 0, y: 0 } }.
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
    // The encoded blob covers only the content-sized surface (10×10 stub), so a
    // reload rasterizes those pixels at the persisted offset.
    expect(h.encodeSurface).toHaveBeenCalledWith(expect.objectContaining({ height: 10, width: 10 }));
    h.store.dispose();
  });

  it('routes the swap through dispatchBitmap when provided (mask persistence seam), not the default dispatch', async () => {
    const surface = createTestStubRasterBackend().createSurface(10, 10);
    const dispatch = vi.fn<(action: WorkbenchAction) => void>();
    const dispatchBitmap = vi.fn<(layerId: string, bitmap: CanvasImageRef, offset: { x: number; y: number }) => void>();
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

    // The engine-provided seam receives the ref + offset; the default paint-source
    // dispatch is NOT used (the engine picks updateCanvasLayerConfig for masks).
    expect(dispatchBitmap).toHaveBeenCalledTimes(1);
    expect(dispatchBitmap).toHaveBeenCalledWith('mask1', expect.objectContaining({ imageName: 'mask-img' }), {
      x: 7,
      y: 8,
    });
    expect(dispatch).not.toHaveBeenCalled();
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

    // Undo restores state A's pixels in the cache; the engine re-marks the layer
    // dirty. The re-flush re-hashes to img-0's content and reuses it — NO upload.
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
    const dispatch = vi.fn<(action: WorkbenchAction) => void>();
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
    await store.flushPendingUploads();

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

    // Encode/hash have run and the upload is in flight, but it hasn't resolved.
    // Drain the encode→hash microtask chain (several awaits) without resolving.
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

    // A fresh stroke lands mid-flight: new pixels re-dirty the layer while the
    // stale upload is still pending.
    h.setEncoded('pixels-B');
    h.store.markLayerDirty(LAYER);

    // The stale upload resolves. The barrier must NOT settle yet: the layer
    // was re-dirtied by a newer stroke during the await, so it owes a follow-up
    // flush of the newer pixels before the "latest painted pixels" guarantee holds.
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

  it('a persistently failing layer does not spin the barrier: it resolves after one bounded attempt, and the layer stays dirty for a later retry', async () => {
    const uploadImage = vi.fn(() => Promise.reject(new Error('upload failed')));
    const onError = vi.fn();
    const surface = createTestStubRasterBackend().createSurface(10, 10);
    const dispatch = vi.fn<(action: WorkbenchAction) => void>();
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
    await store.flushPendingUploads();

    // The internal retry cap (maxUploadAttempts) bounds a single flush's own
    // attempts. The barrier must not loop back and re-attempt a layer whose
    // flush already FAILED this call, so the total stays at that cap instead
    // of growing with extra barrier iterations (no spin).
    expect(uploadImage).toHaveBeenCalledTimes(2);
    expect(dispatch).not.toHaveBeenCalled();
    expect(onError).toHaveBeenCalledTimes(1);

    // The layer is still dirty (deferred, not dropped): a later debounce retries it.
    await vi.advanceTimersByTimeAsync(1500);
    expect(uploadImage.mock.calls.length).toBeGreaterThan(2);

    store.dispose();
  });

  it('isSelfEcho recognizes the exact ref it just applied and rejects a different one', async () => {
    const h = createHarness();

    h.store.markLayerDirty(LAYER);
    await vi.advanceTimersByTimeAsync(1500);
    await h.store.flushPendingUploads();

    const applied = h.dispatch.mock.calls[0][0] as Extract<WorkbenchAction, { type: 'updateCanvasLayerSource' }>;
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

    const applied = h.dispatch.mock.calls[0][0] as Extract<WorkbenchAction, { type: 'updateCanvasLayerSource' }>;
    expect(h.store.isSelfEcho(LAYER, applied.source)).toBe(true);

    // A wholesale document replacement drops the outgoing document's self-echo
    // bookkeeping (a reused layer id must not inherit it).
    h.store.reset();
    expect(h.store.isSelfEcho(LAYER, applied.source)).toBe(false);

    // Re-persisting identical pixels now dispatches again: the content-hash
    // dedupe reuses the already-uploaded image (no new upload), but the stale
    // self-echo no longer suppresses the dispatch, so the contract converges.
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
      // The layer converted back to a parametric source (e.g. rasterize → undo)
      // before the debounce window elapsed — the cache surface still resolves
      // (a source swap doesn't clear it), so only this guard prevents a stale
      // paint dispatch.
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

      // Encode/hash/upload have started (passing the entry-time guard while the
      // source was still `paint`); the upload is now in flight.
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
});
