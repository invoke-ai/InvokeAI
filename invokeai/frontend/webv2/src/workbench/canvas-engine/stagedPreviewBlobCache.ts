import type { ImageResolver } from '@workbench/canvas-engine/render/rasterizers';

const DEFAULT_MAX_BYTES = 64 * 1024 * 1024;
const DEFAULT_MAX_ENTRIES = 24;
const DEFAULT_MAX_CONCURRENT_REQUESTS = 2;

interface CacheEntry {
  active: boolean;
  bytes: number | null;
  readonly controller: AbortController;
  demandPromise: Promise<Blob> | null;
  readonly promise: Promise<Blob>;
  priority: 'demand' | 'prefetch';
}

export interface StagedPreviewBlobCache {
  /** Aborts outstanding work and releases every retained Blob without disposing the reusable cache. */
  clear(): void;
  /** Aborts outstanding work, releases retained Blobs, and permanently closes the cache. */
  dispose(): void;
  /** Resolves from cache or promotes the latest selection ahead of stale/background work. */
  get(imageName: string): Promise<Blob>;
  /** Queues a bounded background fetch without decoding the image. */
  preload(imageName: string): void;
}

export interface StagedPreviewBlobCacheOptions {
  readonly maxBytes?: number;
  readonly maxConcurrentRequests?: number;
  readonly maxEntries?: number;
}

/**
 * Keeps compressed staged-result bytes warm without retaining decoded full-size
 * surfaces for every candidate. Background requests are bounded, while the
 * latest explicit `get()` preempts stale demand and promotes queued work.
 */
export const createStagedPreviewBlobCache = (
  resolveImage: ImageResolver,
  options: StagedPreviewBlobCacheOptions = {}
): StagedPreviewBlobCache => {
  const maxBytes = Math.max(0, options.maxBytes ?? DEFAULT_MAX_BYTES);
  const maxConcurrentRequests = Math.max(1, options.maxConcurrentRequests ?? DEFAULT_MAX_CONCURRENT_REQUESTS);
  const maxEntries = Math.max(1, options.maxEntries ?? DEFAULT_MAX_ENTRIES);
  const entries = new Map<string, CacheEntry>();
  const queued = new Set<string>();
  let retainedBytes = 0;
  let activeRequests = 0;
  let disposed = false;
  let latestDemand: string | null = null;

  const touch = (imageName: string, entry: CacheEntry): void => {
    if (entries.get(imageName) !== entry) {
      return;
    }
    entries.delete(imageName);
    entries.set(imageName, entry);
  };

  const releaseActive = (entry: CacheEntry): void => {
    if (!entry.active) {
      return;
    }
    entry.active = false;
    activeRequests = Math.max(0, activeRequests - 1);
  };

  const removeEntry = (imageName: string, entry: CacheEntry): void => {
    if (entries.get(imageName) !== entry) {
      return;
    }
    entries.delete(imageName);
    if (entry.bytes !== null) {
      retainedBytes = Math.max(0, retainedBytes - entry.bytes);
    }
  };

  const trim = (): void => {
    for (const [imageName, entry] of entries) {
      if (entries.size <= maxEntries && retainedBytes <= maxBytes) {
        return;
      }
      // In-flight requests are bounded separately and become eligible as soon
      // as they settle, so trimming never breaks request coalescing.
      if (entry.active) {
        continue;
      }
      removeEntry(imageName, entry);
    }
  };

  let drainPrefetchQueue = (): void => undefined;

  const start = (imageName: string, priority: CacheEntry['priority']): CacheEntry => {
    const controller = new AbortController();
    let entry!: CacheEntry;
    activeRequests += 1;
    let load: Promise<Blob>;
    try {
      load = resolveImage(imageName, controller.signal);
    } catch (error: unknown) {
      load = Promise.reject(error);
    }
    const promise = load
      .then((blob) => {
        if (disposed || entries.get(imageName) !== entry) {
          throw new Error('Staged preview cache discarded the resolved image.');
        }
        entry.bytes = blob.size;
        retainedBytes += blob.size;
        touch(imageName, entry);
        return blob;
      })
      .catch((error: unknown) => {
        removeEntry(imageName, entry);
        throw error;
      })
      .finally(() => {
        releaseActive(entry);
        trim();
        // A demanded request may need its single foreground retry. Its wrapper
        // owns queue draining so a background item cannot take the freed slot
        // only to be aborted immediately by that retry.
        if (!entry.demandPromise) {
          drainPrefetchQueue();
        }
      });
    entry = {
      active: true,
      bytes: null,
      controller,
      demandPromise: null,
      priority,
      promise,
    };
    entries.set(imageName, entry);
    return entry;
  };

  const abortEntry = (imageName: string, entry: CacheEntry, requeue: boolean): void => {
    removeEntry(imageName, entry);
    releaseActive(entry);
    entry.controller.abort();
    if (requeue && !disposed) {
      queued.add(imageName);
    }
  };

  drainPrefetchQueue = (): void => {
    if (disposed) {
      queued.clear();
      return;
    }
    const availableSlots = maxConcurrentRequests - activeRequests;
    for (let slot = 0; slot < availableSlots; slot += 1) {
      const imageName = queued.values().next().value as string | undefined;
      if (imageName === undefined) {
        return;
      }
      queued.delete(imageName);
      void start(imageName, 'prefetch').promise.catch(() => undefined);
    }
  };

  const makeRoomForLatestDemand = (imageName: string): void => {
    if (activeRequests < maxConcurrentRequests) {
      return;
    }
    for (const [candidateName, entry] of entries) {
      if (candidateName !== imageName && entry.active && entry.priority === 'demand') {
        abortEntry(candidateName, entry, true);
        return;
      }
    }
    for (const [candidateName, entry] of entries) {
      if (candidateName !== imageName && entry.active) {
        abortEntry(candidateName, entry, true);
        return;
      }
    }
  };

  const retryLatestDemand = (imageName: string, error: unknown): Promise<Blob> => {
    if (disposed || latestDemand !== imageName) {
      return Promise.reject(error);
    }
    const replacement = entries.get(imageName);
    if (replacement) {
      return replacement.demandPromise ?? replacement.promise;
    }
    queued.delete(imageName);
    makeRoomForLatestDemand(imageName);
    const demanded = start(imageName, 'demand');
    // Every explicit demand gets at most one retry. A caller that arrives while
    // this retry is active shares the raw retry promise instead of wrapping it
    // in another retry cycle.
    demanded.demandPromise = demanded.promise;
    drainPrefetchQueue();
    return demanded.promise;
  };

  const get = (imageName: string): Promise<Blob> => {
    if (disposed) {
      return Promise.reject(new Error('Staged preview cache is disposed.'));
    }
    latestDemand = imageName;
    const existing = entries.get(imageName);
    if (existing) {
      touch(imageName, existing);
      if (!existing.active) {
        return existing.promise;
      }
      existing.priority = 'demand';
      makeRoomForLatestDemand(imageName);
      drainPrefetchQueue();
      existing.demandPromise ??= existing.promise
        .catch((error: unknown) => retryLatestDemand(imageName, error))
        .finally(() => drainPrefetchQueue());
      return existing.demandPromise;
    }
    queued.delete(imageName);
    makeRoomForLatestDemand(imageName);
    const demanded = start(imageName, 'demand');
    demanded.demandPromise = demanded.promise
      .catch((error: unknown) => retryLatestDemand(imageName, error))
      .finally(() => drainPrefetchQueue());
    drainPrefetchQueue();
    return demanded.demandPromise;
  };

  const preload = (imageName: string): void => {
    if (disposed || entries.has(imageName) || queued.has(imageName)) {
      return;
    }
    queued.add(imageName);
    drainPrefetchQueue();
  };

  const clear = (): void => {
    latestDemand = null;
    queued.clear();
    for (const [imageName, entry] of entries) {
      abortEntry(imageName, entry, false);
    }
    entries.clear();
    retainedBytes = 0;
  };

  const dispose = (): void => {
    if (disposed) {
      return;
    }
    disposed = true;
    clear();
  };

  return { clear, dispose, get, preload };
};
