/**
 * Paint persistence via content-hashed server images.
 *
 * Strokes bake into a layer's raster cache surface but nothing persists. On each
 * committed stroke this marks the layer dirty; after an idle window it encodes
 * the cache surface to PNG, SHA-256s it, dedupes, uploads, and only on success
 * dispatches `updateCanvasLayerSource`. The reducer stays pixel-free — only a
 * `CanvasImageRef` crosses the boundary.
 *
 * Invariants:
 * - **Swap-on-success**: a failed upload never dispatches, so the layer stays
 *   dirty and a reload still shows the last persisted pixels.
 * - **Debounce per layer** (~1.5 s): a stroke burst uploads once.
 * - **Content-hash dedupe**: identical pixels skip the upload, which is what
 *   makes undo cheap.
 * - **Self-echo guard**: the dispatch round-trips back as a source change;
 *   {@link BitmapStore.isSelfEcho} lets the engine skip re-rasterizing it.
 * - **Source-type guard**: a cache surface survives a source-type change, so
 *   every flush re-checks `getLayerSource` both at entry and again before
 *   dispatching (encode/hash/upload all await). Otherwise a stale flush would
 *   convert a parametric layer back to `paint` with wrong-extent pixels.
 * - **Redundant-dispatch skip reads the document, not `lastApplied`**: a round
 *   trip through a non-`paint` source (rasterize → undo → redo) leaves
 *   `lastApplied` naming an image the document no longer references, and
 *   comparing against it would suppress the re-dispatch forever.
 * - **Truthful extent**: the persisted bitmap's dimensions become the layer's
 *   content rect, which draws the move outline, frames the transform tool and
 *   drives fit-to-content. The cache only ever grows, so each flush first trims
 *   it to its visible pixels ({@link BitmapStoreDeps.trimLayerPixels}); a layer
 *   left with none is cleared to `{ bitmap: null }` rather than uploading a
 *   transparent PNG whose dimensions would keep reporting a phantom rect.
 *
 * Every side effect is injectable, so this runs in node tests. Zero React.
 */

import type { CanvasImageRef, CanvasLayerSourceContract } from '@workbench/canvas-engine/contracts';
import type { CanvasImageUploadResult } from '@workbench/canvas-engine/document/imageUpload';
import type { CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';
import type { PaintCacheTrim } from '@workbench/canvas-engine/render/paintCacheTrim';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

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

export class BitmapPersistenceError extends Error {
  readonly layerIds: readonly string[];

  constructor(layerIds: readonly string[]) {
    super(`Canvas pixel persistence failed for ${layerIds.length} layer${layerIds.length === 1 ? '' : 's'}.`);
    this.name = 'BitmapPersistenceError';
    this.layerIds = [...layerIds];
  }
}

/** Injectable timer seam (defaults to the global timers). */
export interface BitmapStoreTimers {
  setTimeout(handler: () => void, ms: number): number;
  clearTimeout(handle: number): void;
}

/** Dependencies for {@link createBitmapStore}. */
export interface BitmapStoreDeps {
  /**
   * Returns a layer's live cache surface (its painted pixels) plus the layer-local
   * `offset` its top-left pixel sits at (its content rect origin), or `null` when
   * the cache is gone/empty. The surface is CONTENT-SIZED, so the encoded PNG
   * covers only the painted region and the dispatched paint source carries the
   * offset (loading rasterizes at it). Read atomically here so the encoded pixels
   * and the offset always agree.
   */
  getLayerSurface(layerId: string): { surface: RasterSurface; offset: { x: number; y: number } } | null;
  /**
   * Returns a layer's CURRENT document source, or `null` if the layer no
   * longer exists. Used to guard a flush against a source-type change that
   * happened AFTER the dirty mark was recorded — e.g. rasterize (paint) →
   * undo (back to shape/gradient) — where the layer's cache surface still
   * resolves (it isn't cleared by the source swap) but persisting it would
   * dispatch stale paint pixels over a now-parametric layer.
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
   * Applies the persisted bitmap ref + offset to the layer's document contract,
   * as the single swap-on-success dispatch. Lets the engine pick the right
   * action per layer type — `updateCanvasLayerSource` (paint source) for raster/
   * control layers, `updateCanvasLayerConfig` (mask) for inpaint/regional masks —
   * while the store stays type-agnostic. Absent ⇒ the default paint-source
   * dispatch (used by the store's own tests, which only exercise paint layers).
   */
  dispatchBitmap?(layerId: string, bitmap: CanvasImageRef, offset: { x: number; y: number }): boolean;
  /**
   * Trims a layer's cache to its visible pixels (see **Truthful extent** above).
   * Called after the source-type guard and BEFORE `getLayerSurface`, so this flush
   * reads the trimmed surface and offset. A `'deferred'` result leaves the layer
   * dirty without encoding; a barrier waits and retries until ownership is released.
   * Absent ⇒ `'kept'` ⇒ no trimming.
   */
  trimLayerPixels?(layerId: string): PaintCacheTrim;
  /**
   * Clears a layer's bitmap ref, for a layer the trim found empty. Returns whether
   * the layer accepted it, like {@link dispatchBitmap}. Kept separate from that
   * rather than widening it to a nullable ref: there is no offset to carry, and a
   * mask must clear `mask.bitmap` while preserving its `fill`.
   */
  clearBitmap?(layerId: string): boolean;
  /** Content-hashes a blob (defaults to SHA-256 hex via `crypto.subtle`). */
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
   * Reports a persistent flush/upload failure. Called on the FIRST failure of a
   * streak and again when the circuit opens; intermediate retries are silent, so
   * a persistently failing layer surfaces one report, not one every retry cycle.
   * Omitted callbacks leave the failure unreported.
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
  flushPendingUploads(): Promise<void>;
  /**
   * True when `source` is exactly the paint bitmap ref this store most recently
   * applied to `layerId` — i.e. the engine is seeing its own dispatch round-trip
   * and must NOT re-rasterize/invalidate the cache (the pixels already match).
   * A different bitmap (undo/import) returns `false` and re-rasterizes as usual.
   *
   * A clear (`bitmap: null`) is deliberately never an echo: re-rasterizing a
   * bitmap-less paint source collapses the cache to a zero rect, which is exactly
   * the reconciliation a cleared layer needs.
   */
  isSelfEcho(layerId: string, source: CanvasLayerSourceContract | null): boolean;
  /**
   * Drops the persistence bookkeeping that describes the OUTGOING document, for
   * use on a wholesale document replacement. Clears the `lastApplied` self-echo
   * map (a reused layer id in the new document could otherwise have a legit
   * persistence dispatch suppressed forever) and any pending dirty/debounced
   * work for the old document. The content-hash dedupe cache is intentionally
   * kept — it is a pure content-addressed mapping (identical PNG bytes → the
   * same immutable uploaded image) and so is never stale across documents.
   */
  reset(): void;
  /** Cancels all timers; in-flight uploads are left to settle (no dispatch after dispose). */
  dispose(): void;
}

const defaultTimers: BitmapStoreTimers = {
  clearTimeout: (handle) => globalThis.clearTimeout(handle),
  setTimeout: (handler, ms) => globalThis.setTimeout(handler, ms),
};

/** SHA-256 hex of a blob's bytes, via the Web Crypto API (Node ≥ 20 exposes `crypto.subtle`). */
const defaultHashBlob = async (blob: Blob): Promise<string> => {
  const buffer = await blob.arrayBuffer();
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  const bytes = new Uint8Array(digest);
  let hex = '';
  for (const byte of bytes) {
    hex += byte.toString(16).padStart(2, '0');
  }
  return hex;
};

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
   * Why a layer is currently in `dirty`: `'stroke'` means a new paint stroke
   * (re)marked it — worth retrying inside a barrier call; `'failure'` means
   * its last flush attempt failed, and `'deferred'` means another operation still
   * owns the pixels. Failures are not retried again within the same
   * {@link flushPendingUploads} call (anti-spin); deferrals are polled until the
   * owner releases the pixels. A new stroke flips either back to `'stroke'`.
   */
  const dirtyReason = new Map<string, 'deferred' | 'failure' | 'stroke'>();
  /** Active debounce timers, keyed by layer id. */
  const debounceTimers = new Map<string, number>();
  /** The in-flight flush op per layer (at most one), used by the barrier and to serialize. */
  const inFlight = new Map<string, Promise<void>>();
  /** Consecutive ambient flush failures per layer; cleared on success or a fresh stroke. */
  const failureCounts = new Map<string, number>();
  /**
   * Layer ids whose CURRENT failure streak has already produced one report.
   * Cleared everywhere `failureCounts` is cleared, so a fresh streak (a new
   * stroke, or a streak that closed via a successful flush) reports its own
   * first failure again.
   */
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
   * Shared bookkeeping for both ways a flush fails: a throw, and a decline
   * (`accepted !== true` — not an error, but bounded by the same breaker).
   * Advances `failureCounts` and re-dirties the layer, so `runFlush` reschedules
   * with backoff until `maxConsecutiveFailures`, after which only a fresh stroke
   * gets back in.
   *
   * Reports the first non-silent failure of a streak, plus — unconditionally —
   * the failure that opens the circuit, even a silent decline: once open,
   * strokes stop persisting entirely and that has to be heard. Everything else
   * stays quiet, or an unreachable server would toast once per retry forever,
   * including the barrier retries every Generate/export/blur attempts.
   *
   * Bookkeeping commits BEFORE `reportError`, because an observer may call back
   * into this store and its outcome must be the final word.
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

  /**
   * Clears a layer's bitmap ref, for a layer the trim found empty. Synchronous
   * throughout, so no generation re-check is needed before the dispatch.
   */
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
      // As with a declined bitmap dispatch: not a network error worth a toast of
      // its own, but it still advances (and can open) the shared breaker. The
      // clear stays pending, so the next flush re-attempts it even though the
      // trim has already collapsed the cache.
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
    // Source-type guard (see `getLayerSource` doc): the dirty mark may predate
    // a conversion away from `paint` (rasterize → undo is the motivating case,
    // but any convert-back qualifies). The cache surface survives a source swap,
    // so without this check we'd encode and dispatch a `paint` source over a layer
    // that is no longer paint at all. Drop the pending flush entirely: nothing
    // about this dirty mark is still valid, and a future genuine paint stroke will
    // re-mark it if the layer ever becomes a paint layer again.
    //
    // Also before the TRIM: shrinking a cache that now backs a parametric render
    // would break the compositor's cache-rect-equals-content-rect invariant.
    const sourceAtEntry = deps.getLayerSource(layerId);
    if (!sourceAtEntry || sourceAtEntry.type !== 'paint') {
      pendingClears.delete(layerId);
      dirty.delete(layerId);
      clearTimer(layerId);
      return;
    }
    // Truthful extent (see the header). `getLayerSurface` below re-reads the cache,
    // so it picks up the trimmed surface and origin with no further work.
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
    if (pendingClears.has(layerId) && !placed) {
      dirty.delete(layerId);
      clearTimer(layerId);
      clearLayerBitmap(layerId, requeueFailure);
      return;
    }
    if (pendingClears.has(layerId)) {
      // A failed clear may outlive the empty cache that requested it. A later
      // rasterization can restore visible pixels without calling markLayerDirty,
      // so the fresh surface verdict wins over the stale clear intent.
      pendingClears.delete(layerId);
    }
    if (!placed) {
      // Layer or its cache is gone (or empty); nothing to persist.
      dirty.delete(layerId);
      clearTimer(layerId);
      return;
    }
    // Capture the surface AND its offset together at encode time so they agree:
    // encode reads these pixels, and the dispatch below carries this offset. A
    // growth during the async encode window re-marks the layer (its stroke marks
    // it dirty), so a follow-up flush re-converges with the current rect + offset.
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
        // Swap-on-success: never dispatch on failure. The old ref stays valid
        // and the layer stays dirty for a later retry.
        requeueFailure(error);
        return;
      }
      rememberDedupe(hash, result);
    }

    if (!isCurrentGeneration()) {
      return;
    }
    // Re-check the source right before dispatching: `encodeSurface`/`hashBlob`/
    // `uploadImage` above all awaited, so a source-type change (rasterize →
    // undo) landing DURING that window would slip past the entry-time
    // `sourceAtEntry` check otherwise. This is the final gate before the
    // side-effecting dispatch.
    const sourceNow = deps.getLayerSource(layerId);
    if (!sourceNow || sourceNow.type !== 'paint') {
      return;
    }
    // The document already points at this image, so skip the dispatch and its
    // self-echo round-trip.
    //
    // Compared against `sourceNow`, not `lastApplied`: a round trip away from
    // `paint` and back (rasterize → undo → redo) lands the document on
    // `{ bitmap: null }` while `lastApplied` still names the pre-undo image, so
    // comparing against memory would suppress the dispatch forever.
    //
    // The offset must match too. A pure translation bakes byte-identical pixels
    // that dedupe to the same image, so comparing `imageName` alone would skip
    // the dispatch that persists the new offset and lose the move on reload.
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
        // The dispatch's THROW was ancillary (e.g. a subscriber failing after
        // commit): the bitmap itself landed, so this attempt succeeded — close
        // the ambient breaker the same as a clean accept.
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
    // The upload+dispatch that may have been failing repeatedly just
    // succeeded: close the ambient breaker so a later failure is reported
    // (and backed off) as a fresh streak, not a silent continuation of one
    // already surfaced.
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
      // Re-dirtied during the flush (new stroke), deferred by an operation that
      // still owns the pixels, or a failure re-queued it.
      if (dirty.has(layerId) && !disposed && !isSuspended(layerId)) {
        if (dirtyReason.get(layerId) !== 'failure') {
          // `'stroke'` and `'deferred'` both re-poll on the ordinary debounce;
          // a deferral is transient and costs only the trim's busy check.
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
    // A fresh stroke closes the circuit: whatever was failing about the old
    // pixels no longer applies to the ones about to be persisted.
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

  const flushPendingUploads = async (): Promise<void> => {
    // Immediately flush every currently-dirty layer (cancelling its debounce),
    // then await the in-flight ops — looping so a layer re-dirtied by a NEW
    // stroke that lands while its upload is in flight gets a follow-up flush
    // before the barrier resolves (the "document points at the latest painted
    // pixels" guarantee). A layer whose flush FAILED within this barrier call is
    // not retried again this call. A transient DEFERRED layer is different: the
    // barrier polls until the operation owning its pixels releases them, matching
    // suspension semantics instead of surfacing a false persistence failure.
    const blockedThisBarrier = new Set<string>();
    for (let iteration = 0; iteration < MAX_BARRIER_ITERATIONS; iteration += 1) {
      const toFlush = Array.from(dirty).filter((layerId) => !blockedThisBarrier.has(layerId) && !isSuspended(layerId));
      for (const layerId of toFlush) {
        clearTimer(layerId);
        void runFlush(layerId);
      }
      const ops = [...inFlight.values()];
      if (ops.length === 0) {
        if (Array.from(dirty).some((layerId) => isSuspended(layerId))) {
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
      let deferredThisRound = false;
      for (const layerId of toFlush) {
        if (dirty.has(layerId) && dirtyReason.get(layerId) === 'failure') {
          blockedThisBarrier.add(layerId);
        } else if (dirty.has(layerId) && dirtyReason.get(layerId) === 'deferred') {
          deferredThisRound = true;
        }
      }
      if (deferredThisRound) {
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
    // The self-echo map is per-(old)document; a reused layer id in the new
    // document must not inherit it. `hashToImage` is content-addressed and kept.
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
