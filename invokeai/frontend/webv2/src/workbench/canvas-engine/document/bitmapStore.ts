/**
 * Content-hashed paint persistence: debounce dirty layers, encode PNG, dedupe uploads, and swap the bitmap ref
 * only on success. Failures leave pixels dirty. Self-echo detection avoids rerasterizing accepted pixels.
 *
 * Recheck source type before trim and after awaits because cache surfaces survive source changes. Compare dispatch
 * dedupe against the current document, not remembered refs. Trim to visible pixels before encoding; clear empty
 * bitmaps so content bounds remain truthful.
 */

import type { CanvasImageRef, CanvasLayerSourceContract } from '@workbench/canvas-engine/contracts';
import type { CanvasImageUploadResult } from '@workbench/canvas-engine/document/imageUpload';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { PaintCacheTrim } from '@workbench/canvas-engine/render/paintCacheTrim';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import { sha256Hex } from '@platform/browser/sha256';

/** Default idle window before a dirty layer is flushed. */
export const DEFAULT_DEBOUNCE_MS = 1500;
/** Default upload attempts per flush (initial try + retries). */
export const DEFAULT_MAX_UPLOAD_ATTEMPTS = 3;
/** Default backoff delays (ms) between upload retries. */
export const DEFAULT_RETRY_DELAYS_MS = [250, 1000] as const;
/** Default cap on the hash→image dedupe map. */
export const DEFAULT_DEDUPE_CAP = 64;
/** Growing re-flush delays after consecutive ambient failures. */
export const DEFAULT_FAILURE_BACKOFF_MS = [2000, 5000, 15000, 30000] as const;
/** Consecutive ambient failures before the circuit opens (no more auto-retries). */
export const DEFAULT_MAX_CONSECUTIVE_FAILURES = 5;
/** Short barrier poll while another canvas operation transiently owns pixels. */
export const DEFAULT_DEFERRED_RETRY_MS = 50;

/** Unsaved pixels the barrier could not persist: uploads `failed`, or an open edit `held` them. */
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
  /**
   * Whether to wait while an open edit (a session, gesture or suspension) holds dirty pixels. Document pushes
   * that the user cannot see through pass false and fail fast instead.
   */
  readonly waitForHeldPixels?: boolean;
}

/** Injectable timer seam (defaults to the global timers). */
export interface BitmapStoreTimers {
  setTimeout(handler: () => void, ms: number): number;
  clearTimeout(handle: number): void;
}

/** Dependencies for {@link createBitmapStore}. */
export interface BitmapStoreDeps {
  /**
   * Atomically reads the content-sized cache surface and layer-local offset. Encoding and persisted placement must
   * describe the same pixels. `'empty'` is a cache without visible extent; null means no cache exists at all.
   */
  getLayerSurface(layerId: string): { surface: RasterSurface; offset: { x: number; y: number } } | 'empty' | null;
  /**
   * Reads the current source to prevent an old dirty mark from persisting surviving cache pixels over a layer
   * converted away from paint.
   */
  getLayerSource(layerId: string): CanvasLayerSourceContract | null;
  /**
   * Reads a layer source from reducer-owned project state, bypassing subscriber-
   * refreshed mirrors. Used only to verify whether a dispatch that threw after
   * reducer commit nevertheless landed exactly as intended.
   */
  getAuthoritativeLayerSource?(layerId: string): CanvasLayerSourceContract | null;
  /** Encodes a surface to an image `Blob` (PNG). Usually `backend.encodeSurface`. */
  encodeSurface(surface: RasterSurface): Promise<Blob>;
  /** Uploads a bitmap blob, resolving to its server image name and dimensions. */
  uploadImage(blob: Blob): Promise<CanvasImageUploadResult>;
  /** Dispatches to the reducer (the single swap-on-success `updateCanvasLayerSource`). */
  dispatch(action: CanvasProjectMutation): boolean;
  /**
   * Swap-on-success bitmap/offset dispatch. The engine chooses paint-source versus mask-config actions; absence
   * uses the paint-source default.
   */
  dispatchBitmap?(layerId: string, bitmap: CanvasImageRef, offset: { x: number; y: number }): boolean;
  /**
   * Trim visible bounds before reading the surface. Deferred trims stay dirty; barriers retry until ownership
   * releases. Absence behaves as kept.
   */
  trimLayerPixels?(layerId: string): PaintCacheTrim;
  /**
   * Clears empty bitmap refs and reports acceptance like {@link dispatchBitmap}. Separate dispatch preserves mask
   * fill and needs no offset.
   */
  clearBitmap?(layerId: string): boolean;
  /** Content-hashes a blob (defaults to SHA-256 hex via `@platform/browser/sha256`). */
  hashBlob?(blob: Blob): Promise<string>;
  /** Idle debounce window in ms (default {@link DEFAULT_DEBOUNCE_MS}). */
  debounceMs?: number;
  /** Upload attempts per flush (default {@link DEFAULT_MAX_UPLOAD_ATTEMPTS}). */
  maxUploadAttempts?: number;
  /** Backoff delays between retries (default {@link DEFAULT_RETRY_DELAYS_MS}). */
  retryDelaysMs?: readonly number[];
  /** Cap on the dedupe map (default {@link DEFAULT_DEDUPE_CAP}). */
  dedupeCap?: number;
  /** Growing re-flush delays after consecutive ambient failures (default {@link DEFAULT_FAILURE_BACKOFF_MS}). */
  failureBackoffMs?: readonly number[];
  /** Consecutive ambient failures before the circuit opens (default {@link DEFAULT_MAX_CONSECUTIVE_FAILURES}). */
  maxConsecutiveFailures?: number;
  /** Injectable timers (default: global). */
  timers?: BitmapStoreTimers;
  /** Injectable delay used for retry backoff (default: `timers.setTimeout`). */
  sleep?(ms: number): Promise<void>;
  /**
   * Reports the first failure and circuit opening in a streak; intermediate retries stay silent. Optional
   * callback.
   */
  onError?(error: unknown, layerId: string, info: { consecutiveFailures: number; willRetry: boolean }): void;
}

/** The imperative bitmap-store handle. */
export interface BitmapStore {
  /** Whether the live cache is empty but clearing the durable bitmap is still pending. */
  hasPendingClear(layerId: string): boolean;
  /** Whether this layer has pixels that are not yet represented by its persisted bitmap ref. */
  hasPendingWork(layerId: string): boolean;
  /** Marks a layer dirty and (re)arms its debounce timer. Called on each committed stroke. */
  markLayerDirty(layerId: string): void;
  /**
   * Temporarily prevents persistence from reading or dispatching `layerId` while
   * preserving dirty work. Returns an idempotent release; leases may be nested.
   */
  suspendLayer(layerId: string): () => void;
  /** Cancels pending persistence and invalidates an in-flight result for one layer. */
  discardLayer(layerId: string): void;
  /** Flushes every dirty layer immediately and resolves once all in-flight uploads settle. */
  flushPendingUploads(options?: FlushPendingUploadsOptions): Promise<void>;
  /**
   * Detects the most recently applied paint ref so the engine skips self-echo rasterization. Different refs
   * rerasterize. Null is never an echo: clearing must collapse cache bounds.
   */
  isSelfEcho(layerId: string, source: CanvasLayerSourceContract | null): boolean;
  /**
   * Drops outgoing-document dirty work and self-echo state so reused ids cannot suppress dispatches. Retains
   * content-hash dedupe because identical bytes name immutable images across documents.
   */
  reset(): void;
  /** Cancels all timers; in-flight uploads are left to settle (no dispatch after dispose). */
  dispose(): void;
}

const defaultTimers: BitmapStoreTimers = {
  clearTimeout: (handle) => globalThis.clearTimeout(handle),
  setTimeout: (handler, ms) => globalThis.setTimeout(handler, ms),
};

/** SHA-256 hex of a blob's bytes, via `@platform/browser/sha256`. */
const defaultHashBlob = async (blob: Blob): Promise<string> => sha256Hex(await blob.arrayBuffer());

/** Creates a bitmap store wired to the given seams. */
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

  /** Layers awaiting a flush (either debounced or re-dirtied during a flush). */
  const dirty = new Set<string>();
  /**
   * Dirty reasons: new strokes may retry within a barrier; failures may not, preventing spin; deferred pixel
   * ownership is polled until released. A new stroke resets either reason.
   */
  const dirtyReason = new Map<string, 'deferred' | 'failure' | 'stroke'>();
  /** Active debounce timers, keyed by layer id. */
  const debounceTimers = new Map<string, number>();
  /** The in-flight flush op per layer (at most one), used by the barrier and to serialize. */
  const inFlight = new Map<string, Promise<void>>();
  /** Consecutive ambient flush failures per layer; cleared on success or a fresh stroke. */
  const failureCounts = new Map<string, number>();
  /** Tracks reported failure streaks; clearing with `failureCounts` lets the next streak report again. */
  const reportedStreaks = new Set<string>();
  /** Content-hash → uploaded image, an LRU-ish dedupe cache (bounded). */
  const hashToImage = new Map<string, CanvasImageUploadResult>();
  /** Layer id → the image name most recently dispatched by this store (self-echo guard). */
  const lastApplied = new Map<string, string>();
  /** Empty-cache clears that have not yet been accepted by the document. */
  const pendingClears = new Set<string>();
  /**
   * Per-layer generation used only while invalidated work is still in flight.
   * Idle ids are removed so ordinary layer deletion cannot accumulate permanent
   * tombstones for the lifetime of the engine.
   */
  const layerGenerations = new Map<string, number>();
  /** Active nested persistence-suspension generation per layer. */
  const suspensions = new Map<string, { count: number; token: symbol }>();
  /** Barriers waiting for a suspended dirty layer to resume or be reset/disposed. */
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
  const waitForSuspensionChange = (): Promise<void> =>
    new Promise((resolve) => {
      suspensionWaiters.add(resolve);
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
    // Move to the most-recently-used end.
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

  /**
   * Throws and declined dispatches share backoff and circuit accounting. Report the first non-silent failure and
   * always report circuit opening, including silent declines. Commit bookkeeping before notifying observers, whose
   * reentrant changes must win.
   */
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

  /** Clears empty bitmap refs synchronously, requiring no generation recheck. */
  const clearLayerBitmap = (layerId: string, requeueFailure: (error: unknown) => void): void => {
    // Drop the self-echo entry on every path: the trim established that the live
    // cache no longer matches the last bitmap this store dispatched, including
    // when the document has already independently reached `bitmap: null`.
    lastApplied.delete(layerId);
    // Redundant-dispatch skip against GROUND TRUTH, not `lastApplied` — see the header.
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
      // As on the upload path: a dispatch that threw after the reducer committed
      // still landed, so it must not be requeued — and it closes the breaker.
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
      // Declined clears advance the shared breaker without a network-error toast. Keep the clear pending even
      // after trim collapses the cache.
      recordFlushFailure(layerId, new Error('Bitmap clear was not accepted.'), { silent: true });
      return;
    }
    pendingClears.delete(layerId);
    failureCounts.delete(layerId);
    reportedStreaks.delete(layerId);
  };

  /** Encodes → hashes → dedupes/uploads → swaps the layer's ref, once. */
  const flushLayer = async (layerId: string): Promise<void> => {
    const generationAtEntry = layerGenerations.get(layerId) ?? 0;
    const isCurrentGeneration = (): boolean => !disposed && (layerGenerations.get(layerId) ?? 0) === generationAtEntry;
    const requeueFailure = (error: unknown): void => {
      if (!isCurrentGeneration()) {
        // A discard/reset already invalidated this flush; its own bookkeeping
        // stands, and this stale failure has nothing left to report.
        return;
      }
      recordFlushFailure(layerId, error, { silent: false });
    };
    // Reject obsolete dirty work before trim if the source is no longer paint. Cache surfaces survive conversion;
    // trimming or persisting them would corrupt parametric content bounds or source state.
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
    // A failed clear may outlive the empty cache that requested it. A later
    // rasterization can restore visible pixels without calling markLayerDirty,
    // so the fresh surface verdict wins over the stale clear intent.
    pendingClears.delete(layerId);
    if (!placed) {
      // Unpersisted pixels without a cache cannot be saved; reporting success would upload nothing.
      requeueFailure(new Error('Canvas pixels for this layer are no longer available.'));
      return;
    }
    // Capture surface and offset atomically for encoding. Growth during awaits marks dirty again, so a follow-up
    // flush converges placement.
    const { offset, surface } = placed;
    // Consume the dirty flag up front; a failure re-adds it below. A stroke that
    // lands mid-flush re-marks the layer, so the finally handler re-schedules.
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
      // Dedupe hit: identical pixels already uploaded — reuse the name, no upload.
      touchDedupe(hash, result);
    } else {
      try {
        const uploaded = await uploadWithRetry(blob, isCurrentGeneration);
        if (!uploaded) {
          return;
        }
        result = uploaded;
      } catch (error) {
        // Dispatch only on success; failed uploads retain the old ref and dirty state.
        requeueFailure(error);
        return;
      }
      rememberDedupe(hash, result);
    }

    if (!isCurrentGeneration()) {
      return;
    }
    // Recheck source type after encode/hash/upload awaits before dispatching, preventing overwrite of a newly
    // restored parametric source.
    const sourceNow = deps.getLayerSource(layerId);
    if (!sourceNow || sourceNow.type !== 'paint') {
      return;
    }
    // Skip only when the current document matches both bitmap and offset. Remembered refs survive source round
    // trips, and byte-identical translations still need their new offset persisted.
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
    // Record BEFORE dispatching: `dispatch` may notify the mirror synchronously,
    // so `isSelfEcho` must already see the applied name when the engine reacts.
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
        // The bitmap landed despite an observer throw; treat it as accepted and close the breaker.
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
        // A declined acceptance isn't a network error worth a toast of its own,
        // but it still advances (and can open) the shared breaker.
        recordFlushFailure(layerId, new Error('Bitmap update was not accepted.'), { silent: true });
      }
      return;
    }
    // Successful persistence closes the breaker so later failures start a new reported streak.
    failureCounts.delete(layerId);
    reportedStreaks.delete(layerId);
  };

  /** Runs (or joins) a flush for a layer, serializing to one in-flight op per layer. */
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
          // Strokes and deferred ownership retry at ordinary debounce; a deferral only checks trim ownership.
          scheduleFlush(layerId);
        } else {
          const failures = failureCounts.get(layerId) ?? 0;
          if (failures < maxConsecutiveFailures) {
            scheduleFlush(layerId, failureBackoffMs[Math.min(failures - 1, failureBackoffMs.length - 1)] ?? debounceMs);
          }
          // failures >= max: the circuit is open — stay dirty, no timer. A new
          // stroke (markLayerDirty) or a flushPendingUploads barrier call is
          // the only way back in.
        }
      } else {
        // Any invalidated operation for this id has now settled and there is no
        // successor waiting to inherit its generation. Retire the tombstone.
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
    // A fresh stroke closes the circuit and retries the new pixels.
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
      // Keep an invalidating generation only while obsolete async work can
      // still settle. Same-id work arriving before it settles inherits this
      // generation and is scheduled after the old operation completes.
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

  /** Safety net against a genuine infinite loop; real barrier calls settle in a handful of rounds. */
  const MAX_BARRIER_ITERATIONS = 10_000;

  const flushPendingUploads = async ({ waitForHeldPixels = true }: FlushPendingUploadsOptions = {}): Promise<void> => {
    // Flush dirty layers and await in-flight work until newer strokes also persist. Do not retry failures within
    // this barrier; poll deferred ownership until pixels are released, unless the caller cannot wait.
    const blockedThisBarrier = new Set<string>();
    for (let iteration = 0; iteration < MAX_BARRIER_ITERATIONS; iteration += 1) {
      const toFlush = Array.from(dirty).filter((layerId) => !blockedThisBarrier.has(layerId) && !isSuspended(layerId));
      for (const layerId of toFlush) {
        clearTimer(layerId);
        void runFlush(layerId);
      }
      const ops = [...inFlight.values()];
      if (ops.length === 0) {
        const suspendedLayerIds = Array.from(dirty).filter((layerId) => isSuspended(layerId));
        if (suspendedLayerIds.length > 0) {
          if (!waitForHeldPixels) {
            throw new BitmapPersistenceError(suspendedLayerIds, 'held');
          }
          await waitForSuspensionChange();
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
      await Promise.all(ops);
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
        await sleep(DEFAULT_DEFERRED_RETRY_MS);
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
    // Cancel pending debounced flushes and drop dirty state for the OLD document.
    for (const handle of debounceTimers.values()) {
      timers.clearTimeout(handle);
    }
    debounceTimers.clear();
    dirty.clear();
    dirtyReason.clear();
    pendingClears.clear();
    suspensions.clear();
    notifySuspensionWaiters();
    // Clear per-document self-echo state before ids are reused; retain content-addressed `hashToImage`.
    lastApplied.clear();
    // Same reasoning as `lastApplied`: a reused layer id in the new document
    // must start with a closed circuit, not inherit the old document's streak.
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
