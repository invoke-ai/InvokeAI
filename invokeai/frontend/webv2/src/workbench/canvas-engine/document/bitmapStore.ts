/** Content-hashed paint persistence: the bitmap ref swaps only on success; failures leave pixels dirty.
 Cache surfaces survive source changes, so the source type is rechecked before trim and after every await. */

import type { CanvasImageRef, CanvasLayerSourceContract } from '@workbench/canvas-engine/contracts';
import type { CanvasImageUploadResult } from '@workbench/canvas-engine/document/imageUpload';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { PaintCacheTrim } from '@workbench/canvas-engine/render/paintCacheTrim';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import { sha256Hex } from '@platform/browser/sha256';

export const DEFAULT_DEBOUNCE_MS = 1500;
/** Initial try plus retries. */
export const DEFAULT_MAX_UPLOAD_ATTEMPTS = 3;
export const DEFAULT_RETRY_DELAYS_MS = [250, 1000] as const;
export const DEFAULT_DEDUPE_CAP = 64;
/** Growing re-flush delays after consecutive ambient failures. */
export const DEFAULT_FAILURE_BACKOFF_MS = [2000, 5000, 15000, 30000] as const;
/** Consecutive ambient failures before the circuit opens and auto-retries stop. */
export const DEFAULT_MAX_CONSECUTIVE_FAILURES = 5;
/** Barrier poll while another operation transiently owns pixels. */
export const DEFAULT_DEFERRED_RETRY_MS = 50;

/** `failed` uploads, or an open edit `held` the pixels. */
export class BitmapPersistenceError extends Error {
  readonly layerIds: readonly string[];
  readonly reason: 'failed' | 'held';

  constructor(layerIds: readonly string[], reason: 'failed' | 'held' = 'failed') {
    super(
      reason === 'held'
        ? 'Finish or cancel the current canvas edit first.'
        : `Canvas pixel persistence failed for ${layerIds.length} layer${layerIds.length === 1 ? '' : 's'}.`
    );
    this.name = 'BitmapPersistenceError';
    this.layerIds = [...layerIds];
    this.reason = reason;
  }
}

export interface FlushPendingUploadsOptions {
  /** False fails fast instead of waiting on an open edit's dirty pixels. */
  readonly waitForHeldPixels?: boolean;
  /** Absent means every layer. */
  readonly layerIds?: readonly string[];
  /** Uploads already started still settle. */
  readonly signal?: AbortSignal;
}

export interface BitmapStoreTimers {
  setTimeout(handler: () => void, ms: number): number;
  clearTimeout(handle: number): void;
}

export interface BitmapStoreDeps {
  /** Reads surface and offset atomically; `'empty'` is a cache without visible extent, null no cache at all. */
  getLayerSurface(layerId: string): { surface: RasterSurface; offset: { x: number; y: number } } | 'empty' | null;
  /** Lets an old dirty mark refuse to persist over a layer converted away from paint. */
  getLayerSource(layerId: string): CanvasLayerSourceContract | null;
  /** Reducer-owned state, bypassing mirrors: verifies whether a dispatch that threw after commit landed. */
  getAuthoritativeLayerSource?(layerId: string): CanvasLayerSourceContract | null;
  encodeSurface(surface: RasterSurface): Promise<Blob>;
  uploadImage(blob: Blob): Promise<CanvasImageUploadResult>;
  dispatch(action: CanvasProjectMutation): boolean;
  /** Absent uses the paint-source action. */
  dispatchBitmap?(layerId: string, bitmap: CanvasImageRef, offset: { x: number; y: number }): boolean;
  /** Deferred trims stay dirty; barriers retry until ownership releases. Absent behaves as kept. */
  trimLayerPixels?(layerId: string): PaintCacheTrim;
  /** A separate dispatch preserves mask fill. */
  clearBitmap?(layerId: string): boolean;
  /** Defaults to SHA-256 hex. */
  hashBlob?(blob: Blob): Promise<string>;
  debounceMs?: number;
  maxUploadAttempts?: number;
  retryDelaysMs?: readonly number[];
  dedupeCap?: number;
  failureBackoffMs?: readonly number[];
  maxConsecutiveFailures?: number;
  timers?: BitmapStoreTimers;
  sleep?(ms: number): Promise<void>;
  /** Reports the first failure and the circuit opening in a streak; intermediate retries stay silent. */
  onError?(error: unknown, layerId: string, info: { consecutiveFailures: number; willRetry: boolean }): void;
}

export interface BitmapStore {
  hasPendingClear(layerId: string): boolean;
  hasPendingWork(layerId: string): boolean;
  markLayerDirty(layerId: string): void;
  /** Leases nest; the returned release is idempotent. */
  suspendLayer(layerId: string): () => void;
  discardLayer(layerId: string): void;
  flushPendingUploads(options?: FlushPendingUploadsOptions): Promise<void>;
  /** Null is never an echo: clearing must collapse cache bounds. */
  isSelfEcho(layerId: string, source: CanvasLayerSourceContract | null): boolean;
  /** Keeps content-hash dedupe: identical bytes name immutable images across documents. */
  reset(): void;
  /** In-flight uploads settle, but nothing dispatches after dispose. */
  dispose(): void;
}

const defaultTimers: BitmapStoreTimers = {
  clearTimeout: (handle) => globalThis.clearTimeout(handle),
  setTimeout: (handler, ms) => globalThis.setTimeout(handler, ms),
};

const defaultHashBlob = async (blob: Blob): Promise<string> => sha256Hex(await blob.arrayBuffer());

export const createBitmapStore = (deps: BitmapStoreDeps): BitmapStore => {
  const debounceMs = deps.debounceMs ?? DEFAULT_DEBOUNCE_MS;
  const maxAttempts = Math.max(1, deps.maxUploadAttempts ?? DEFAULT_MAX_UPLOAD_ATTEMPTS);
  const retryDelays = deps.retryDelaysMs ?? DEFAULT_RETRY_DELAYS_MS;
  const dedupeCap = Math.max(1, deps.dedupeCap ?? DEFAULT_DEDUPE_CAP);
  const failureBackoffMs = deps.failureBackoffMs ?? DEFAULT_FAILURE_BACKOFF_MS;
  const maxConsecutiveFailures = Math.max(1, deps.maxConsecutiveFailures ?? DEFAULT_MAX_CONSECUTIVE_FAILURES);
  const timers = deps.timers ?? defaultTimers;
  const hashBlob = deps.hashBlob ?? defaultHashBlob;
  const sleep =
    deps.sleep ??
    ((ms: number): Promise<void> =>
      new Promise((resolve) => {
        timers.setTimeout(resolve, ms);
      }));
  const reportError = (
    error: unknown,
    layerId: string,
    info: { consecutiveFailures: number; willRetry: boolean }
  ): void => deps.onError?.(error, layerId, info);

  const dirty = new Set<string>();
  /** Failures never retry within a barrier, preventing spin; deferred ownership is polled. */
  const dirtyReason = new Map<string, 'deferred' | 'failure' | 'stroke'>();
  const debounceTimers = new Map<string, number>();
  const inFlight = new Map<string, Promise<void>>();
  const failureCounts = new Map<string, number>();
  const reportedStreaks = new Set<string>();
  const hashToImage = new Map<string, CanvasImageUploadResult>();
  const lastApplied = new Map<string, string>();
  const pendingClears = new Set<string>();
  /** Kept only while invalidated work is in flight, so deleted layers leave no permanent tombstones. */
  const layerGenerations = new Map<string, number>();
  const suspensions = new Map<string, { count: number; token: symbol }>();
  const suspensionWaiters = new Set<() => void>();
  let disposed = false;

  const isSuspended = (layerId: string): boolean => (suspensions.get(layerId)?.count ?? 0) > 0;
  const notifySuspensionWaiters = (): void => {
    const waiters = [...suspensionWaiters];
    suspensionWaiters.clear();
    for (const resolve of waiters) {
      resolve();
    }
  };
  const abortError = (): DOMException => new DOMException('Canvas pixel persistence wait aborted.', 'AbortError');
  const waitForSuspensionChange = (signal?: AbortSignal): Promise<void> =>
    new Promise((resolve, reject) => {
      const onAbort = (): void => {
        suspensionWaiters.delete(waiter);
        reject(abortError());
      };
      const waiter = (): void => {
        signal?.removeEventListener('abort', onAbort);
        resolve();
      };
      suspensionWaiters.add(waiter);
      signal?.addEventListener('abort', onAbort, { once: true });
    });

  const clearTimer = (layerId: string): void => {
    const handle = debounceTimers.get(layerId);
    if (handle !== undefined) {
      timers.clearTimeout(handle);
      debounceTimers.delete(layerId);
    }
  };

  const scheduleFlush = (layerId: string, delayMs: number = debounceMs): void => {
    clearTimer(layerId);
    const handle = timers.setTimeout(() => {
      debounceTimers.delete(layerId);
      void runFlush(layerId);
    }, delayMs);
    debounceTimers.set(layerId, handle);
  };

  const rememberDedupe = (hash: string, result: CanvasImageUploadResult): void => {
    hashToImage.delete(hash);
    hashToImage.set(hash, result);
    while (hashToImage.size > dedupeCap) {
      const oldest = hashToImage.keys().next().value;
      if (oldest === undefined) {
        break;
      }
      hashToImage.delete(oldest);
    }
  };

  const touchDedupe = (hash: string, result: CanvasImageUploadResult): void => {
    hashToImage.delete(hash);
    hashToImage.set(hash, result);
  };

  const uploadWithRetry = async (
    blob: Blob,
    isCurrentGeneration: () => boolean
  ): Promise<CanvasImageUploadResult | null> => {
    let lastError: unknown;
    for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
      if (!isCurrentGeneration()) {
        return null;
      }
      if (attempt > 0) {
        const delay = retryDelays[Math.min(attempt - 1, retryDelays.length - 1)] ?? 0;
        if (delay > 0) {
          await sleep(delay);
          if (!isCurrentGeneration()) {
            return null;
          }
        }
      }
      if (!isCurrentGeneration()) {
        return null;
      }
      try {
        const result = await deps.uploadImage(blob);
        if (!isCurrentGeneration()) {
          return null;
        }
        return result;
      } catch (error) {
        if (!isCurrentGeneration()) {
          return null;
        }
        lastError = error;
      }
    }
    throw lastError ?? new Error('Canvas image upload failed');
  };

  /** Report the first non-silent failure and always the circuit opening; bookkeeping commits before observers run. */
  const recordFlushFailure = (layerId: string, error: unknown, options: { silent: boolean }): void => {
    const failures = (failureCounts.get(layerId) ?? 0) + 1;
    failureCounts.set(layerId, failures);
    const willRetry = failures < maxConsecutiveFailures;
    dirty.add(layerId);
    dirtyReason.set(layerId, 'failure');
    const opensCircuit = failures === maxConsecutiveFailures;
    if (!opensCircuit && (options.silent || reportedStreaks.has(layerId))) {
      return;
    }
    reportedStreaks.add(layerId);
    try {
      reportError(error, layerId, { consecutiveFailures: failures, willRetry });
    } catch {
      // Keep the bounded retry state intact when an observer itself fails.
    }
  };

  const clearLayerBitmap = (layerId: string, requeueFailure: (error: unknown) => void): void => {
    // The trim proved the cache no longer matches the last dispatched bitmap.
    lastApplied.delete(layerId);
    // Compare against the current document, not `lastApplied`.
    const sourceNow = deps.getLayerSource(layerId);
    if (!sourceNow || sourceNow.type !== 'paint') {
      pendingClears.delete(layerId);
      return;
    }
    if (!sourceNow.bitmap) {
      pendingClears.delete(layerId);
      return;
    }
    let accepted: boolean;
    try {
      accepted = deps.clearBitmap
        ? deps.clearBitmap(layerId)
        : deps.dispatch({
            id: layerId,
            source: { bitmap: null, type: 'paint' },
            type: 'updateCanvasLayerSource',
          });
    } catch (error) {
      const authoritativeSource = (deps.getAuthoritativeLayerSource ?? deps.getLayerSource)(layerId);
      // A dispatch that threw after the reducer committed still landed; do not requeue it.
      if (authoritativeSource?.type === 'paint' && !authoritativeSource.bitmap) {
        pendingClears.delete(layerId);
        failureCounts.delete(layerId);
        reportedStreaks.delete(layerId);
        return;
      }
      if (authoritativeSource !== null) {
        requeueFailure(error);
      }
      return;
    }
    if (accepted !== true && deps.getLayerSource(layerId) !== null) {
      // Declined clears advance the breaker without a toast and stay pending.
      recordFlushFailure(layerId, new Error('Bitmap clear was not accepted.'), { silent: true });
      return;
    }
    pendingClears.delete(layerId);
    failureCounts.delete(layerId);
    reportedStreaks.delete(layerId);
  };

  const flushLayer = async (layerId: string): Promise<void> => {
    const generationAtEntry = layerGenerations.get(layerId) ?? 0;
    const isCurrentGeneration = (): boolean => !disposed && (layerGenerations.get(layerId) ?? 0) === generationAtEntry;
    const requeueFailure = (error: unknown): void => {
      if (!isCurrentGeneration()) {
        // A discard or reset already invalidated this flush.
        return;
      }
      recordFlushFailure(layerId, error, { silent: false });
    };
    // Cache surfaces survive conversion; trimming one under a non-paint source would corrupt its bounds.
    const sourceAtEntry = deps.getLayerSource(layerId);
    if (!sourceAtEntry || sourceAtEntry.type !== 'paint') {
      pendingClears.delete(layerId);
      dirty.delete(layerId);
      clearTimer(layerId);
      return;
    }
    let trimResult: PaintCacheTrim = 'kept';
    try {
      trimResult = deps.trimLayerPixels?.(layerId) ?? 'kept';
    } catch (error) {
      requeueFailure(error);
      return;
    }
    if (trimResult === 'emptied') {
      pendingClears.add(layerId);
    }
    if (trimResult === 'deferred') {
      dirty.add(layerId);
      dirtyReason.set(layerId, 'deferred');
      clearTimer(layerId);
      return;
    }
    const placed = deps.getLayerSurface(layerId);
    if (placed === 'empty' || (placed === null && pendingClears.has(layerId))) {
      dirty.delete(layerId);
      clearTimer(layerId);
      clearLayerBitmap(layerId, requeueFailure);
      return;
    }
    // A later rasterization can restore pixels without markLayerDirty; the fresh surface verdict wins.
    pendingClears.delete(layerId);
    if (!placed) {
      requeueFailure(new Error('Canvas pixels for this layer are no longer available.'));
      return;
    }
    const { offset, surface } = placed;
    // A stroke landing mid-flush re-marks the layer, so the finally handler reschedules.
    dirty.delete(layerId);
    clearTimer(layerId);

    let hash: string;
    let blob: Blob;
    try {
      blob = await deps.encodeSurface(surface);
      if (!isCurrentGeneration()) {
        return;
      }
      hash = await hashBlob(blob);
      if (!isCurrentGeneration()) {
        return;
      }
    } catch (error) {
      requeueFailure(error);
      return;
    }

    let result = hashToImage.get(hash);
    if (result) {
      touchDedupe(hash, result);
    } else {
      try {
        const uploaded = await uploadWithRetry(blob, isCurrentGeneration);
        if (!uploaded) {
          return;
        }
        result = uploaded;
      } catch (error) {
        requeueFailure(error);
        return;
      }
      rememberDedupe(hash, result);
    }

    if (!isCurrentGeneration()) {
      return;
    }
    // A parametric source may have been restored during the awaits.
    const sourceNow = deps.getLayerSource(layerId);
    if (!sourceNow || sourceNow.type !== 'paint') {
      return;
    }
    // Byte-identical translations still need their new offset persisted.
    const currentOffset = sourceNow.bitmap ? (sourceNow.offset ?? { x: 0, y: 0 }) : null;
    if (
      sourceNow.bitmap?.imageName === result.imageName &&
      currentOffset !== null &&
      currentOffset.x === offset.x &&
      currentOffset.y === offset.y
    ) {
      return;
    }

    const bitmap: CanvasImageRef = {
      contentHash: hash,
      height: result.height,
      imageName: result.imageName,
      width: result.width,
    };
    // Record before dispatching: `dispatch` may notify the mirror synchronously.
    lastApplied.set(layerId, result.imageName);
    let accepted: boolean;
    try {
      accepted = deps.dispatchBitmap
        ? deps.dispatchBitmap(layerId, bitmap, { x: offset.x, y: offset.y })
        : deps.dispatch({
            id: layerId,
            source: { bitmap, offset: { x: offset.x, y: offset.y }, type: 'paint' },
            type: 'updateCanvasLayerSource',
          });
    } catch (error) {
      const authoritativeSource = (deps.getAuthoritativeLayerSource ?? deps.getLayerSource)(layerId);
      const authoritativeOffset =
        authoritativeSource?.type === 'paint' && authoritativeSource.bitmap
          ? (authoritativeSource.offset ?? { x: 0, y: 0 })
          : null;
      const didLand =
        authoritativeSource?.type === 'paint' &&
        authoritativeSource.bitmap?.imageName === bitmap.imageName &&
        authoritativeSource.bitmap.width === bitmap.width &&
        authoritativeSource.bitmap.height === bitmap.height &&
        authoritativeSource.bitmap.contentHash === bitmap.contentHash &&
        authoritativeOffset?.x === offset.x &&
        authoritativeOffset.y === offset.y;
      if (didLand) {
        failureCounts.delete(layerId);
        reportedStreaks.delete(layerId);
        return;
      }
      lastApplied.delete(layerId);
      if (authoritativeSource !== null) {
        requeueFailure(error);
      }
      return;
    }
    if (accepted !== true) {
      lastApplied.delete(layerId);
      if (deps.getLayerSource(layerId) !== null) {
        // A declined acceptance raises no toast but still advances the breaker.
        recordFlushFailure(layerId, new Error('Bitmap update was not accepted.'), { silent: true });
      }
      return;
    }
    failureCounts.delete(layerId);
    reportedStreaks.delete(layerId);
  };

  const runFlush = (layerId: string): Promise<void> => {
    const existing = inFlight.get(layerId);
    if (existing) {
      return existing;
    }
    if (isSuspended(layerId)) {
      return Promise.resolve();
    }
    const op = flushLayer(layerId).finally(() => {
      inFlight.delete(layerId);
      if (dirty.has(layerId) && !disposed && !isSuspended(layerId)) {
        if (dirtyReason.get(layerId) !== 'failure') {
          scheduleFlush(layerId);
        } else {
          const failures = failureCounts.get(layerId) ?? 0;
          if (failures < maxConsecutiveFailures) {
            scheduleFlush(layerId, failureBackoffMs[Math.min(failures - 1, failureBackoffMs.length - 1)] ?? debounceMs);
          }
          // Circuit open: stay dirty with no timer until a new stroke or a barrier.
        }
      } else {
        layerGenerations.delete(layerId);
      }
    });
    inFlight.set(layerId, op);
    return op;
  };

  const markLayerDirty = (layerId: string): void => {
    if (disposed) {
      return;
    }
    failureCounts.delete(layerId);
    reportedStreaks.delete(layerId);
    pendingClears.delete(layerId);
    dirty.add(layerId);
    dirtyReason.set(layerId, 'stroke');
    if (!isSuspended(layerId)) {
      scheduleFlush(layerId);
    }
  };

  const suspendLayer = (layerId: string): (() => void) => {
    if (disposed) {
      return () => undefined;
    }
    const currentSuspension = suspensions.get(layerId);
    const count = currentSuspension?.count ?? 0;
    const token = currentSuspension?.token ?? Symbol(layerId);
    suspensions.set(layerId, { count: count + 1, token });
    if (count === 0) {
      const hadPendingWork = dirty.has(layerId) || debounceTimers.has(layerId) || inFlight.has(layerId);
      clearTimer(layerId);
      if (inFlight.has(layerId)) {
        layerGenerations.set(layerId, (layerGenerations.get(layerId) ?? 0) + 1);
      }
      if (hadPendingWork) {
        dirty.add(layerId);
        dirtyReason.set(layerId, 'stroke');
      }
    }

    let released = false;
    return () => {
      if (released) {
        return;
      }
      released = true;
      const current = suspensions.get(layerId);
      if (!current || current.token !== token) {
        return;
      }
      if (current.count <= 1) {
        suspensions.delete(layerId);
        if (dirty.has(layerId) && !disposed) {
          scheduleFlush(layerId);
        }
      } else {
        suspensions.set(layerId, { count: current.count - 1, token });
      }
      notifySuspensionWaiters();
    };
  };

  const discardLayer = (layerId: string): void => {
    if (inFlight.has(layerId)) {
      // Same-id work arriving before the old work settles inherits this generation.
      layerGenerations.set(layerId, (layerGenerations.get(layerId) ?? 0) + 1);
    } else {
      layerGenerations.delete(layerId);
    }
    dirty.delete(layerId);
    dirtyReason.delete(layerId);
    lastApplied.delete(layerId);
    failureCounts.delete(layerId);
    reportedStreaks.delete(layerId);
    pendingClears.delete(layerId);
    clearTimer(layerId);
  };

  const MAX_BARRIER_ITERATIONS = 10_000;

  const untilAborted = <T>(promise: Promise<T>, signal: AbortSignal | undefined): Promise<T> => {
    if (!signal) {
      return promise;
    }
    return new Promise<T>((resolve, reject) => {
      const onAbort = (): void => reject(abortError());
      if (signal.aborted) {
        onAbort();
        return;
      }
      signal.addEventListener('abort', onAbort, { once: true });
      promise.then(resolve, reject).finally(() => signal.removeEventListener('abort', onAbort));
    });
  };

  const flushPendingUploads = async ({
    layerIds,
    signal,
    waitForHeldPixels = true,
  }: FlushPendingUploadsOptions = {}): Promise<void> => {
    // Failures do not retry within this barrier; deferred ownership is polled unless the caller cannot wait.
    const scope = layerIds ? new Set(layerIds) : null;
    const inScope = (layerId: string): boolean => scope === null || scope.has(layerId);
    const blockedThisBarrier = new Set<string>();
    for (let iteration = 0; iteration < MAX_BARRIER_ITERATIONS; iteration += 1) {
      if (signal?.aborted) {
        throw abortError();
      }
      const scopedDirty = Array.from(dirty).filter(inScope);
      const toFlush = scopedDirty.filter((layerId) => !blockedThisBarrier.has(layerId) && !isSuspended(layerId));
      for (const layerId of toFlush) {
        clearTimer(layerId);
        void runFlush(layerId);
      }
      const ops = [...inFlight].filter(([layerId]) => inScope(layerId)).map(([, op]) => op);
      if (ops.length === 0) {
        const suspendedLayerIds = scopedDirty.filter((layerId) => isSuspended(layerId));
        if (suspendedLayerIds.length > 0) {
          if (!waitForHeldPixels) {
            throw new BitmapPersistenceError(suspendedLayerIds, 'held');
          }
          await waitForSuspensionChange(signal);
          continue;
        }
        const unpersistedLayerIds = Array.from(blockedThisBarrier).filter(
          (layerId) => dirty.has(layerId) && dirtyReason.get(layerId) !== 'stroke'
        );
        if (unpersistedLayerIds.length > 0) {
          throw new BitmapPersistenceError(unpersistedLayerIds);
        }
        return;
      }
      await untilAborted(Promise.all(ops), signal);
      const deferredLayerIds: string[] = [];
      for (const layerId of toFlush) {
        if (dirty.has(layerId) && dirtyReason.get(layerId) === 'failure') {
          blockedThisBarrier.add(layerId);
        } else if (dirty.has(layerId) && dirtyReason.get(layerId) === 'deferred') {
          deferredLayerIds.push(layerId);
        }
      }
      if (deferredLayerIds.length > 0) {
        if (!waitForHeldPixels) {
          throw new BitmapPersistenceError(deferredLayerIds, 'held');
        }
        await untilAborted(sleep(DEFAULT_DEFERRED_RETRY_MS), signal);
      }
    }
    throw new Error('Canvas pixel persistence barrier exceeded its iteration limit.');
  };

  const isSelfEcho = (layerId: string, source: CanvasLayerSourceContract | null): boolean => {
    if (!source || source.type !== 'paint') {
      return false;
    }
    const imageName = source.bitmap?.imageName;
    return imageName !== undefined && lastApplied.get(layerId) === imageName;
  };

  const reset = (): void => {
    for (const layerId of inFlight.keys()) {
      layerGenerations.set(layerId, (layerGenerations.get(layerId) ?? 0) + 1);
    }
    for (const layerId of layerGenerations.keys()) {
      if (!inFlight.has(layerId)) {
        layerGenerations.delete(layerId);
      }
    }
    for (const handle of debounceTimers.values()) {
      timers.clearTimeout(handle);
    }
    debounceTimers.clear();
    dirty.clear();
    dirtyReason.clear();
    pendingClears.clear();
    suspensions.clear();
    notifySuspensionWaiters();
    // Retain content-addressed `hashToImage`.
    lastApplied.clear();
    // A reused layer id must start with a closed circuit.
    failureCounts.clear();
    reportedStreaks.clear();
  };

  const dispose = (): void => {
    disposed = true;
    for (const handle of debounceTimers.values()) {
      timers.clearTimeout(handle);
    }
    debounceTimers.clear();
    dirty.clear();
    dirtyReason.clear();
    pendingClears.clear();
    suspensions.clear();
    notifySuspensionWaiters();
    inFlight.clear();
    hashToImage.clear();
    lastApplied.clear();
    layerGenerations.clear();
    failureCounts.clear();
    reportedStreaks.clear();
  };

  const hasPendingWork = (layerId: string): boolean =>
    dirty.has(layerId) ||
    pendingClears.has(layerId) ||
    debounceTimers.has(layerId) ||
    inFlight.has(layerId) ||
    isSuspended(layerId);

  const hasPendingClear = (layerId: string): boolean => pendingClears.has(layerId);

  return {
    discardLayer,
    dispose,
    flushPendingUploads,
    hasPendingClear,
    hasPendingWork,
    isSelfEcho,
    markLayerDirty,
    reset,
    suspendLayer,
  };
};
