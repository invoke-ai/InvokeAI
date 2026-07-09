/**
 * The bitmap store: paint persistence via content-hashed server images.
 *
 * Painting bakes strokes straight into a layer's raster cache surface (see
 * `strokeSession`/`paintTool`), but nothing persists — a reload would lose the
 * strokes. This store closes that loop: on each committed stroke it marks the
 * layer dirty, and after a short idle window it encodes the layer's full cache
 * surface to a PNG, content-hashes it (SHA-256), dedupes against already
 * uploaded bitmaps, uploads new ones, and — only once the upload succeeds —
 * dispatches `updateCanvasLayerSource` so the reducer-owned document points its
 * paint layer at the persisted image name. The reducer stays pixel-free: only a
 * `CanvasImageRef` (name + dims + hash) ever crosses the boundary.
 *
 * Key invariants:
 * - **Swap-on-success**: the contract keeps its previous ref until the upload
 *   resolves. A failed upload never dispatches; the layer stays dirty and the
 *   old ref stays valid, so a reload still shows the last persisted pixels.
 * - **Debounce per layer** (~1.5 s idle): a new stroke resets the timer, so a
 *   burst of strokes uploads once.
 * - **Content-hash dedupe**: identical pixels reuse the previously uploaded
 *   image name and skip the upload entirely. This is what makes undo cheap
 *   (restoring old pixels re-hashes to the already-uploaded image).
 * - **Self-echo guard**: the dispatch round-trips back through the document
 *   mirror as a source change for the layer. {@link BitmapStore.isSelfEcho}
 *   lets the engine skip re-rasterizing/invalidating the cache for the exact
 *   bitmap ref this store just applied (the pixels already match).
 * - **Source-type guard**: a layer's cache surface survives a source-type
 *   change (e.g. rasterize's paint bake, then an undo back to shape/gradient)
 *   — only the document's source pointer changes, not the cache. So every
 *   flush re-checks `getLayerSource` (at entry AND again right before the
 *   dispatch, since encode/hash/upload all await) and drops the pending work
 *   without dispatching if the layer is no longer `paint` (or no longer
 *   exists). Otherwise a stale debounced flush would silently convert a
 *   parametric layer back to `paint` with wrong-extent pixels.
 * - **Redundant-dispatch skip is ground-truth, not memory**: right before
 *   dispatching, a flush skips if the DOCUMENT's current bitmap ref (via
 *   `getLayerSource`) already equals the resolved image name — not if
 *   `lastApplied` (this store's memory of what it last dispatched) does. A
 *   round trip through a non-`paint` source and back (rasterize → undo →
 *   redo) leaves `lastApplied` pointing at a name the document no longer
 *   references (the redo lands on `{ bitmap: null }`); comparing against that
 *   stale memory would suppress the re-dispatch forever. Comparing against
 *   the document itself self-heals regardless of how it drifted.
 *
 * Every side-effecting dependency (encode, upload, hash, dispatch, timers) is
 * injectable, so this runs in node tests with fakes. Zero React.
 */

import type { CanvasImageUploadResult } from '@workbench/canvas-engine/backend/canvasImages';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { CanvasImageRef, CanvasLayerSourceContract } from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';

/** Default idle window before a dirty layer is flushed. */
export const DEFAULT_DEBOUNCE_MS = 1500;
/** Default upload attempts per flush (initial try + retries). */
export const DEFAULT_MAX_UPLOAD_ATTEMPTS = 3;
/** Default backoff delays (ms) between upload retries. */
export const DEFAULT_RETRY_DELAYS_MS = [250, 1000] as const;
/** Default cap on the hash→image dedupe map. */
export const DEFAULT_DEDUPE_CAP = 64;

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
  /** Encodes a surface to an image `Blob` (PNG). Usually `backend.encodeSurface`. */
  encodeSurface(surface: RasterSurface): Promise<Blob>;
  /** Uploads a bitmap blob, resolving to its server image name and dimensions. */
  uploadImage(blob: Blob): Promise<CanvasImageUploadResult>;
  /** Dispatches to the reducer (the single swap-on-success `updateCanvasLayerSource`). */
  dispatch(action: WorkbenchAction): void;
  /**
   * Applies the persisted bitmap ref + offset to the layer's document contract,
   * as the single swap-on-success dispatch. Lets the engine pick the right
   * action per layer type — `updateCanvasLayerSource` (paint source) for raster/
   * control layers, `updateCanvasLayerConfig` (mask) for inpaint/regional masks —
   * while the store stays type-agnostic. Absent ⇒ the default paint-source
   * dispatch (used by the store's own tests, which only exercise paint layers).
   */
  dispatchBitmap?(layerId: string, bitmap: CanvasImageRef, offset: { x: number; y: number }): void;
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
  /** Injectable timers (default: global). */
  timers?: BitmapStoreTimers;
  /** Injectable delay used for retry backoff (default: `timers.setTimeout`). */
  sleep?(ms: number): Promise<void>;
  /** Reports a persistent flush/upload failure (default: `console.warn`). */
  onError?(error: unknown, layerId: string): void;
}

/** The imperative bitmap-store handle. */
export interface BitmapStore {
  /** Marks a layer dirty and (re)arms its debounce timer. Called on each committed stroke. */
  markLayerDirty(layerId: string): void;
  /** Cancels pending persistence and invalidates an in-flight result for one layer. */
  discardLayer(layerId: string): void;
  /** Flushes every dirty layer immediately and resolves once all in-flight uploads settle. */
  flushPendingUploads(): Promise<void>;
  /**
   * True when `source` is exactly the paint bitmap ref this store most recently
   * applied to `layerId` — i.e. the engine is seeing its own dispatch round-trip
   * and must NOT re-rasterize/invalidate the cache (the pixels already match).
   * A different bitmap (undo/import) returns `false` and re-rasterizes as usual.
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
  const timers = deps.timers ?? defaultTimers;
  const hashBlob = deps.hashBlob ?? defaultHashBlob;
  const sleep =
    deps.sleep ??
    ((ms: number): Promise<void> =>
      new Promise((resolve) => {
        timers.setTimeout(resolve, ms);
      }));
  const reportError = (error: unknown, layerId: string): void => {
    if (deps.onError) {
      deps.onError(error, layerId);
    } else {
      // eslint-disable-next-line no-console
      console.warn(`bitmapStore: failed to persist layer "${layerId}"`, error);
    }
  };

  /** Layers awaiting a flush (either debounced or re-dirtied during a flush). */
  const dirty = new Set<string>();
  /**
   * Why a layer is currently in `dirty`: `'stroke'` means a new paint stroke
   * (re)marked it — worth retrying inside a barrier call; `'failure'` means
   * its last flush attempt exhausted upload retries — the barrier must not
   * retry it again within the same {@link flushPendingUploads} call (anti-spin).
   * A stroke landing after a failure flips this back to `'stroke'`.
   */
  const dirtyReason = new Map<string, 'failure' | 'stroke'>();
  /** Active debounce timers, keyed by layer id. */
  const debounceTimers = new Map<string, number>();
  /** The in-flight flush op per layer (at most one), used by the barrier and to serialize. */
  const inFlight = new Map<string, Promise<void>>();
  /** Content-hash → uploaded image, an LRU-ish dedupe cache (bounded). */
  const hashToImage = new Map<string, CanvasImageUploadResult>();
  /** Layer id → the image name most recently dispatched by this store (self-echo guard). */
  const lastApplied = new Map<string, string>();
  /** Per-layer generation used to invalidate an upload already in flight when its pixels are discarded. */
  const layerGenerations = new Map<string, number>();
  let disposed = false;

  const clearTimer = (layerId: string): void => {
    const handle = debounceTimers.get(layerId);
    if (handle !== undefined) {
      timers.clearTimeout(handle);
      debounceTimers.delete(layerId);
    }
  };

  const scheduleFlush = (layerId: string): void => {
    clearTimer(layerId);
    const handle = timers.setTimeout(() => {
      debounceTimers.delete(layerId);
      void runFlush(layerId);
    }, debounceMs);
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

  const uploadWithRetry = async (blob: Blob): Promise<CanvasImageUploadResult> => {
    let lastError: unknown;
    for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
      if (attempt > 0) {
        const delay = retryDelays[Math.min(attempt - 1, retryDelays.length - 1)] ?? 0;
        if (delay > 0) {
          await sleep(delay);
        }
      }
      try {
        return await deps.uploadImage(blob);
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError ?? new Error('Canvas image upload failed');
  };

  /** Encodes → hashes → dedupes/uploads → swaps the layer's ref, once. */
  const flushLayer = async (layerId: string): Promise<void> => {
    const generationAtEntry = layerGenerations.get(layerId) ?? 0;
    const placed = deps.getLayerSurface(layerId);
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
    // Source-type guard (see `getLayerSource` doc): the dirty mark may predate
    // a conversion away from `paint` (rasterize → undo is the motivating case,
    // but any convert-back qualifies). The cache surface still resolves above
    // — a source swap doesn't clear it — so without this check we'd encode and
    // dispatch a `paint` source over a layer that is no longer paint at all.
    // Drop the pending flush entirely: nothing about this dirty mark is still
    // valid, and a future genuine paint stroke will re-mark it if the layer
    // ever becomes a paint layer again.
    const sourceAtEntry = deps.getLayerSource(layerId);
    if (!sourceAtEntry || sourceAtEntry.type !== 'paint') {
      dirty.delete(layerId);
      clearTimer(layerId);
      return;
    }
    // Consume the dirty flag up front; a failure re-adds it below. A stroke that
    // lands mid-flush re-marks the layer, so the finally handler re-schedules.
    dirty.delete(layerId);
    clearTimer(layerId);

    let hash: string;
    let blob: Blob;
    try {
      blob = await deps.encodeSurface(surface);
      hash = await hashBlob(blob);
    } catch (error) {
      reportError(error, layerId);
      dirty.add(layerId);
      dirtyReason.set(layerId, 'failure');
      return;
    }

    let result = hashToImage.get(hash);
    if (result) {
      // Dedupe hit: identical pixels already uploaded — reuse the name, no upload.
      touchDedupe(hash, result);
    } else {
      try {
        result = await uploadWithRetry(blob);
      } catch (error) {
        // Swap-on-success: never dispatch on failure. The old ref stays valid
        // and the layer stays dirty for a later retry.
        reportError(error, layerId);
        dirty.add(layerId);
        dirtyReason.set(layerId, 'failure');
        return;
      }
      rememberDedupe(hash, result);
    }

    if (disposed) {
      return;
    }
    if ((layerGenerations.get(layerId) ?? 0) !== generationAtEntry) {
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
    // The DOCUMENT already points at this exact image (e.g. a re-flush of
    // identical pixels that hash-deduped to an already-applied ref): skip a
    // redundant dispatch and its self-echo round-trip.
    //
    // This is checked against `sourceNow` (the document's CURRENT bitmap ref),
    // not `lastApplied` (this store's memory of what it last dispatched) — the
    // two can diverge. `lastApplied` is never cleared when a layer's source is
    // converted away from `paint` and back (e.g. `rasterizeLayer` → undo →
    // redo: the document round-trips through a parametric source and back to
    // `paint`, landing on `{ bitmap: null }`, while `lastApplied` still holds
    // the previously-uploaded name from before the undo). Comparing against
    // `lastApplied` in that case would suppress the dispatch forever, leaving
    // the document permanently pointed at `bitmap: null` even though the
    // dedupe correctly resolved the identical baked pixels back to the prior
    // image. Comparing against the document's actual current ref is the
    // ground truth for "is this dispatch a no-op" and self-heals regardless
    // of how the document's source got out of sync with this store's memory.
    //
    // The OFFSET must match too: a pure-translation transform (drag + Apply)
    // bakes byte-identical pixels → same hash → dedupe resolves to the
    // already-referenced image, but the paint offset moved. Comparing only
    // `imageName` would skip the dispatch that persists the new offset, so the
    // translation would be silently lost on reload (and every re-flush would
    // re-skip it forever). Compare the captured `offset` (which travels with
    // this dispatch) against the document's current offset — absent offsets are
    // the legacy origin `{ x: 0, y: 0 }`.
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
    if (deps.dispatchBitmap) {
      deps.dispatchBitmap(layerId, bitmap, { x: offset.x, y: offset.y });
    } else {
      deps.dispatch({
        id: layerId,
        source: { bitmap, offset: { x: offset.x, y: offset.y }, type: 'paint' },
        type: 'updateCanvasLayerSource',
      });
    }
  };

  /** Runs (or joins) a flush for a layer, serializing to one in-flight op per layer. */
  const runFlush = (layerId: string): Promise<void> => {
    const existing = inFlight.get(layerId);
    if (existing) {
      return existing;
    }
    const op = flushLayer(layerId).finally(() => {
      inFlight.delete(layerId);
      // Re-dirtied during the flush (new stroke) or a failure re-queued it.
      if (dirty.has(layerId) && !disposed) {
        scheduleFlush(layerId);
      }
    });
    inFlight.set(layerId, op);
    return op;
  };

  const markLayerDirty = (layerId: string): void => {
    if (disposed) {
      return;
    }
    dirty.add(layerId);
    dirtyReason.set(layerId, 'stroke');
    scheduleFlush(layerId);
  };

  const discardLayer = (layerId: string): void => {
    layerGenerations.set(layerId, (layerGenerations.get(layerId) ?? 0) + 1);
    dirty.delete(layerId);
    dirtyReason.delete(layerId);
    lastApplied.delete(layerId);
    clearTimer(layerId);
  };

  /** Safety net against a genuine infinite loop; real barrier calls settle in a handful of rounds. */
  const MAX_BARRIER_ITERATIONS = 10_000;

  const flushPendingUploads = async (): Promise<void> => {
    // Immediately flush every currently-dirty layer (cancelling its debounce),
    // then await the in-flight ops — looping so a layer re-dirtied by a NEW
    // stroke that lands while its upload is in flight gets a follow-up flush
    // before the barrier resolves (the "document points at the latest painted
    // pixels" guarantee). A layer whose flush FAILED within this barrier call
    // is not retried again this call — only a fresh stroke re-enters the loop
    // for it — so a persistently failing upload still can't spin the barrier
    // forever.
    const failedThisBarrier = new Set<string>();
    for (let iteration = 0; iteration < MAX_BARRIER_ITERATIONS; iteration += 1) {
      const toFlush = Array.from(dirty).filter((layerId) => !failedThisBarrier.has(layerId));
      for (const layerId of toFlush) {
        clearTimer(layerId);
        void runFlush(layerId);
      }
      const ops = [...inFlight.values()];
      if (ops.length === 0) {
        return;
      }
      await Promise.all(ops);
      for (const layerId of toFlush) {
        if (dirty.has(layerId) && dirtyReason.get(layerId) === 'failure') {
          failedThisBarrier.add(layerId);
        }
      }
    }
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
    // Cancel pending debounced flushes and drop dirty state for the OLD document.
    for (const handle of debounceTimers.values()) {
      timers.clearTimeout(handle);
    }
    debounceTimers.clear();
    dirty.clear();
    dirtyReason.clear();
    // The self-echo map is per-(old)document; a reused layer id in the new
    // document must not inherit it. `hashToImage` is content-addressed and kept.
    lastApplied.clear();
  };

  const dispose = (): void => {
    disposed = true;
    for (const handle of debounceTimers.values()) {
      timers.clearTimeout(handle);
    }
    debounceTimers.clear();
    dirty.clear();
    dirtyReason.clear();
    inFlight.clear();
    hashToImage.clear();
    lastApplied.clear();
    layerGenerations.clear();
  };

  return { discardLayer, dispose, flushPendingUploads, isSelfEcho, markLayerDirty, reset };
};
