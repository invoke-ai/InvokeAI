import { AccountScopeExpiredError, accountLifecycle, type AccountLifecycle } from '@platform/state/accountLifecycle';

import type { FontDownloadReference, FontLoadState, FontRecord, FontReference } from './contracts';

const MAX_CONCURRENT_DOWNLOADS = 4;
const MAX_REGISTERED_FONTS = 64;
const MAX_REGISTERED_FONT_BYTES = 128 * 1024 * 1024;
const MAX_FAILED_FONTS = 64;
let nextRuntimeId = 1;

interface FontFaceLike {
  load(): Promise<FontFaceLike>;
}

interface FontFaceSetLike {
  add(font: FontFaceLike): void;
  delete(font: FontFaceLike): boolean;
}

interface FontFaceConstructorLike {
  new (family: string, source: ArrayBuffer, descriptors?: { style?: string; weight?: string }): FontFaceLike;
}

export interface FontRuntimeEnvironment {
  FontFace: FontFaceConstructorLike | null;
  fonts: FontFaceSetLike | null;
}

export interface FontRuntimeSnapshot {
  generation: number;
  states: ReadonlyMap<string, FontLoadState>;
}

export interface FontRuntime {
  /** Starts account cleanup registration. Safe to call more than once. */
  start(): void;
  /** Ensures the requested face/instance is registered and returns its CSS family name. */
  ensure(reference: FontDownloadReference, signal?: AbortSignal): Promise<string>;
  /** Revalidates the persisted face before an operation whose pixels leave the editor. */
  ensureForOutput(reference: FontDownloadReference, signal?: AbortSignal): Promise<string>;
  /** Keeps a loaded face registered while a long-lived Canvas consumer uses it. */
  retain(reference: FontReference): () => void;
  /** Returns the registered family if ready, otherwise the readable fallback family. */
  resolveFamily(reference: FontReference): string;
  getSnapshot(): FontRuntimeSnapshot;
  subscribe(listener: () => void): () => void;
  dispose(): void;
}

export interface CreateFontRuntimeOptions {
  download?: (reference: FontDownloadReference, signal?: AbortSignal) => Promise<Uint8Array>;
  getFont?: (id: string, signal?: AbortSignal) => Promise<FontRecord>;
  environment?: FontRuntimeEnvironment;
  lifecycle?: AccountLifecycle;
}

const downloadDefaultFont = async (reference: FontDownloadReference, signal?: AbortSignal): Promise<Uint8Array> => {
  // Load catalog transport only when a custom face is requested.
  const { downloadFont } = await import('./data/api');
  return downloadFont(reference, signal);
};

const getDefaultFont = async (id: string, signal?: AbortSignal): Promise<FontRecord> => {
  const { getFont } = await import('./data/api');
  return getFont(id, signal);
};

const getBrowserEnvironment = (): FontRuntimeEnvironment => {
  if (typeof document === 'undefined' || typeof globalThis.FontFace === 'undefined' || !document.fonts) {
    return { FontFace: null, fonts: null };
  }

  return {
    FontFace: globalThis.FontFace as unknown as FontFaceConstructorLike,
    fonts: document.fonts as unknown as FontFaceSetLike,
  };
};

const normalizeAxes = (axes: FontReference['axes']): string =>
  Object.entries(axes ?? {})
    .filter(([, value]) => Number.isFinite(value))
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([tag, value]) => `${tag}:${value}`)
    .join(',');

export const getFontRuntimeKey = (reference: FontReference): string =>
  JSON.stringify([reference.id, reference.contentHash ?? null, normalizeAxes(reference.axes)]);

const createAbortError = (): Error => {
  if (typeof DOMException !== 'undefined') {
    return new DOMException('The font load was aborted.', 'AbortError');
  }
  const error = new Error('The font load was aborted.');
  error.name = 'AbortError';
  return error;
};

const withAbortSignal = <T>(promise: Promise<T>, signal?: AbortSignal, abortReason?: unknown): Promise<T> => {
  if (!signal) {
    return promise;
  }
  const getReason = (): unknown => abortReason ?? signal.reason ?? createAbortError();
  if (signal.aborted) {
    return Promise.reject(getReason());
  }

  return new Promise<T>((resolve, reject) => {
    const onAbort = (): void => {
      signal.removeEventListener('abort', onAbort);
      reject(getReason());
    };
    signal.addEventListener('abort', onAbort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener('abort', onAbort);
        resolve(value);
      },
      (error: unknown) => {
        signal.removeEventListener('abort', onAbort);
        reject(error);
      }
    );
  });
};

const isAbortError = (error: unknown): boolean =>
  typeof DOMException !== 'undefined' && error instanceof DOMException
    ? error.name === 'AbortError'
    : error instanceof Error && (error.name === 'AbortError' || error.message.includes('aborted'));

const combineSignals = (
  first: AbortSignal | undefined,
  second: AbortSignal
): { signal: AbortSignal; dispose: () => void } => {
  const controller = new AbortController();
  const abort = (): void => controller.abort();

  if (first?.aborted || second.aborted) {
    controller.abort();
  }
  first?.addEventListener('abort', abort, { once: true });
  second.addEventListener('abort', abort, { once: true });

  return {
    dispose: () => {
      first?.removeEventListener('abort', abort);
      second.removeEventListener('abort', abort);
    },
    signal: controller.signal,
  };
};

interface QueueTask<T> {
  run: () => Promise<T>;
  reject: (error: unknown) => void;
  resolve: (value: T) => void;
  signal?: AbortSignal;
  started: boolean;
  settled: boolean;
  cleanup?: () => void;
}

const createLimiter = (limit: number) => {
  let active = 0;
  const queue: QueueTask<unknown>[] = [];

  const pump = (): void => {
    while (active < limit && queue.length > 0) {
      const task = queue.shift()!;
      if (task.settled) {
        continue;
      }
      if (task.signal?.aborted) {
        task.settled = true;
        task.reject(task.signal.reason ?? createAbortError());
        task.cleanup?.();
        continue;
      }
      task.started = true;
      active += 1;
      void Promise.resolve()
        .then(task.run)
        .then(task.resolve, task.reject)
        .finally(() => {
          active -= 1;
          task.settled = true;
          task.cleanup?.();
          pump();
        });
    }
  };

  return <T>(run: () => Promise<T>, signal?: AbortSignal): Promise<T> =>
    new Promise<T>((resolve, reject) => {
      const task: QueueTask<unknown> = {
        reject,
        resolve: (value) => resolve(value as T),
        run: run as () => Promise<unknown>,
        signal,
        settled: false,
        started: false,
      };
      const onAbort = (): void => {
        if (task.started || task.settled) {
          return;
        }
        const index = queue.indexOf(task);
        if (index >= 0) {
          queue.splice(index, 1);
        }
        task.settled = true;
        signal?.removeEventListener('abort', onAbort);
        reject(signal?.reason ?? createAbortError());
        pump();
      };
      task.cleanup = () => signal?.removeEventListener('abort', onAbort);
      signal?.addEventListener('abort', onAbort, { once: true });
      queue.push(task);
      pump();
    });
};

interface InflightRequest {
  readonly key: string;
  readonly family: string;
  readonly requestController: AbortController;
  readonly combined: { signal: AbortSignal; dispose: () => void };
  readonly consumers: Set<symbol>;
  readonly promise: Promise<string>;
}

const fontFaceDescriptors = (face: FontRecord, reference: FontDownloadReference): { style: string; weight: string } => {
  let style = face.style ?? 'normal';
  let weight = face.weight ?? 400;
  const axes = new Set(face.axes.map((axis) => axis.tag));
  const requestedAxes = reference.axes ?? {};

  // Pinned axes must set the CSS descriptor to avoid synthesizing bold over an already-bold instance. Otherwise
  // retain catalog descriptors for browser synthesis.
  if (axes.has('wght') && Number.isFinite(requestedAxes.wght)) {
    weight = requestedAxes.wght!;
  }
  const requestedItalic = axes.has('ital') && Number.isFinite(requestedAxes.ital);
  const requestedSlant = axes.has('slnt') && Number.isFinite(requestedAxes.slnt);
  if (requestedItalic && requestedAxes.ital! >= 0.5) {
    style = 'italic';
  } else if (requestedSlant && requestedAxes.slnt !== 0) {
    style = 'oblique';
  } else if (requestedItalic || requestedSlant) {
    style = 'normal';
  }

  return { style, weight: String(weight) };
};

const createSnapshot = (generation: number, states: ReadonlyMap<string, FontLoadState>): FontRuntimeSnapshot =>
  Object.freeze({ generation, states });

/**
 * Registration identity includes resource and normalized axes, but excludes requested CSS style/weight: catalog
 * descriptors own static-face synthesis.
 */
export const createFontRuntime = (options: CreateFontRuntimeOptions = {}): FontRuntime => {
  const runtimeId = nextRuntimeId++;
  const environment = options.environment ?? getBrowserEnvironment();
  const lifecycle = options.lifecycle ?? accountLifecycle;
  const fetchFont = options.download ?? downloadDefaultFont;
  const fetchFontRecord = options.getFont ?? getDefaultFont;
  const limit = createLimiter(MAX_CONCURRENT_DOWNLOADS);
  const listeners = new Set<() => void>();
  const inflight = new Map<string, InflightRequest>();
  const outputInflight = new Map<string, Promise<FontRecord>>();
  const registered = new Map<
    string,
    { face: FontFaceLike; family: string; retained: number; touched: number; byteSize: number }
  >();
  const reservations = new Map<string, number>();
  const retainCounts = new Map<string, number>();
  const familyNames = new Map<string, string>();
  const abortControllers = new Set<AbortController>();
  const stateTouches = new Map<string, number>();
  let states = new Map<string, FontLoadState>();
  let snapshot = createSnapshot(0, states);
  let generation = 0;
  let touch = 0;
  let nextFamilyId = 1;
  let disposed = false;
  let unregister: (() => void) | undefined;
  let started = false;

  const publish = (key: string, state: FontLoadState): void => {
    if (disposed) {
      return;
    }
    stateTouches.set(key, ++touch);
    if (states.get(key) === state) {
      return;
    }
    states = new Map(states).set(key, state);
    snapshot = createSnapshot(generation, states);
    for (const listener of listeners) {
      listener();
    }
  };

  const removeState = (key: string): void => {
    if (disposed) {
      return;
    }
    stateTouches.delete(key);
    if (!states.has(key)) {
      if (!registered.has(key) && !inflight.has(key)) {
        familyNames.delete(key);
      }
      return;
    }
    states = new Map(states);
    states.delete(key);
    if (!registered.has(key) && !inflight.has(key)) {
      familyNames.delete(key);
    }
    snapshot = createSnapshot(generation, states);
    for (const listener of listeners) {
      listener();
    }
  };

  const trimFailed = (): void => {
    const failed = [...states.entries()]
      .filter(
        ([key, state]) => state === 'error' && !registered.has(key) && !inflight.has(key) && !retainCounts.has(key)
      )
      .sort(([left], [right]) => (stateTouches.get(left) ?? 0) - (stateTouches.get(right) ?? 0));

    while (failed.length > MAX_FAILED_FONTS) {
      const oldest = failed.shift();
      if (oldest) {
        removeState(oldest[0]);
      }
    }
  };

  const reserveFont = (key: string, byteSize: number): void => {
    let usedBytes =
      [...registered.values()].reduce((sum, entry) => sum + entry.byteSize, 0) +
      [...reservations.values()].reduce((sum, size) => sum + size, 0);
    const evictable = [...registered.entries()]
      .filter(([, entry]) => entry.retained === 0)
      .sort(([, left], [, right]) => left.touched - right.touched);
    while (
      registered.size + reservations.size >= MAX_REGISTERED_FONTS ||
      usedBytes + byteSize > MAX_REGISTERED_FONT_BYTES
    ) {
      const oldest = evictable.shift();
      if (!oldest) {
        throw new Error(
          'The custom font budget is full (64 faces / 128 MiB). Replace unused fonts or close other documents, then retry.'
        );
      }
      environment.fonts?.delete(oldest[1].face);
      registered.delete(oldest[0]);
      usedBytes -= oldest[1].byteSize;
      removeState(oldest[0]);
    }
    reservations.set(key, byteSize);
    trimFailed();
  };

  const clear = (): void => {
    generation += 1;
    for (const controller of abortControllers) {
      controller.abort();
    }
    abortControllers.clear();
    for (const entry of registered.values()) {
      environment.fonts?.delete(entry.face);
    }
    registered.clear();
    reservations.clear();
    retainCounts.clear();
    familyNames.clear();
    inflight.clear();
    outputInflight.clear();
    stateTouches.clear();
    states = new Map();
    snapshot = createSnapshot(generation, states);
    for (const listener of listeners) {
      listener();
    }
  };

  const start = (): void => {
    if (started && !disposed) {
      return;
    }
    disposed = false;
    started = true;
    unregister = lifecycle.register({ clear, name: `fonts-runtime-${runtimeId}` });
  };

  const attachConsumer = (request: InflightRequest, signal?: AbortSignal): Promise<string> => {
    if (signal?.aborted) {
      return Promise.reject(signal.reason ?? createAbortError());
    }
    const token = Symbol('font-consumer');
    request.consumers.add(token);
    let released = false;
    const release = (): void => {
      if (released) {
        return;
      }
      released = true;
      request.consumers.delete(token);
      // Abort shared transport only after its last consumer releases it.
      if (request.consumers.size === 0 && inflight.get(request.key) === request) {
        request.requestController.abort(createAbortError());
      }
    };
    return withAbortSignal(request.promise, signal, createAbortError()).finally(release);
  };

  const ensure = (reference: FontDownloadReference, signal?: AbortSignal, knownFace?: FontRecord): Promise<string> => {
    if (disposed) {
      return Promise.reject(new Error('The font runtime has been disposed.'));
    }
    start();
    const owner = lifecycle.capture();
    if (signal?.aborted) {
      return Promise.reject(signal.reason ?? createAbortError());
    }
    if (!lifecycle.isCurrent(owner)) {
      return Promise.reject(new AccountScopeExpiredError());
    }
    const key = getFontRuntimeKey(reference);
    const loaded = registered.get(key);
    if (loaded) {
      loaded.touched = ++touch;
      return withAbortSignal(Promise.resolve(loaded.family), signal);
    }
    const pending = inflight.get(key);
    if (pending) {
      return attachConsumer(pending, signal);
    }

    const requestController = new AbortController();
    abortControllers.add(requestController);
    const combined = combineSignals(owner.signal, requestController.signal);
    const requestGeneration = generation;
    const family = familyNames.get(key) ?? `__invoke_font_${runtimeId}_${nextFamilyId++}`;
    familyNames.set(key, family);

    const task = limit(async () => {
      if (disposed || generation !== requestGeneration || !lifecycle.isCurrent(owner) || combined.signal.aborted) {
        throw new AccountScopeExpiredError();
      }
      if (!environment.FontFace || !environment.fonts) {
        throw new Error('The browser FontFace API is unavailable for custom font output.');
      }
      const [bytes, face] = await Promise.all([
        fetchFont(reference, combined.signal),
        knownFace ?? fetchFontRecord(reference.id, combined.signal),
      ]);
      if (disposed || generation !== requestGeneration || !lifecycle.isCurrent(owner) || combined.signal.aborted) {
        throw new AccountScopeExpiredError();
      }
      if (face.id !== reference.id || (reference.contentHash && face.contentHash !== reference.contentHash)) {
        throw new Error('The font face does not match the requested reference.');
      }
      reserveFont(key, bytes.byteLength);
      try {
        const descriptors = fontFaceDescriptors(face, reference);
        const source =
          bytes.buffer instanceof ArrayBuffer && bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
            ? bytes.buffer
            : (bytes.slice().buffer as ArrayBuffer);
        const fontFace = new environment.FontFace(family, source, {
          style: descriptors.style,
          weight: descriptors.weight,
        });
        const loadedFace = await fontFace.load();
        if (disposed || generation !== requestGeneration || !lifecycle.isCurrent(owner) || combined.signal.aborted) {
          throw new AccountScopeExpiredError();
        }
        environment.fonts.add(loadedFace);
        registered.set(key, {
          face: loadedFace,
          family,
          retained: retainCounts.get(key) ?? 0,
          touched: ++touch,
          byteSize: bytes.byteLength,
        });
        return family;
      } finally {
        if (generation === requestGeneration) {
          reservations.delete(key);
        }
      }
    }, combined.signal);

    let request!: InflightRequest;

    const result = task
      .then((resolvedFamily) => {
        if (lifecycle.isCurrent(owner) && !disposed && generation === requestGeneration) {
          publish(key, 'ready');
        }
        return resolvedFamily;
      })
      .catch((error: unknown) => {
        if (lifecycle.isCurrent(owner) && !disposed && generation === requestGeneration) {
          if (combined.signal.aborted || isAbortError(error) || error instanceof AccountScopeExpiredError) {
            removeState(key);
          } else {
            publish(key, 'error');
            trimFailed();
          }
        }
        throw error;
      })
      .finally(() => {
        combined.dispose();
        abortControllers.delete(requestController);
        if (inflight.get(key) === request) {
          inflight.delete(key);
        }
        if (!registered.has(key) && !states.has(key)) {
          familyNames.delete(key);
        }
        trimFailed();
      });

    request = {
      combined,
      consumers: new Set(),
      family,
      key,
      promise: result,
      requestController,
    };
    inflight.set(key, request);
    publish(key, 'loading');
    return attachConsumer(request, signal);
  };

  const clearLoaded = (reference: FontReference): void => {
    const key = getFontRuntimeKey(reference);
    const loaded = registered.get(key);
    if (loaded) {
      environment.fonts?.delete(loaded.face);
      registered.delete(key);
    }
    removeState(key);
  };

  const ensureForOutput = (reference: FontDownloadReference, signal?: AbortSignal): Promise<string> => {
    if (disposed) {
      return Promise.reject(new Error('The font runtime has been disposed.'));
    }
    start();
    const owner = lifecycle.capture();
    if (signal?.aborted) {
      return Promise.reject(signal.reason ?? createAbortError());
    }
    if (!lifecycle.isCurrent(owner)) {
      return Promise.reject(new AccountScopeExpiredError());
    }

    const metadataKey = JSON.stringify([reference.id, reference.contentHash ?? null]);
    let metadata = outputInflight.get(metadataKey);
    if (!metadata) {
      const metadataGeneration = generation;
      const metadataRequestController = new AbortController();
      abortControllers.add(metadataRequestController);
      const ownerAndRuntime = combineSignals(owner.signal, metadataRequestController.signal);
      const metadataRequest = fetchFontRecord(reference.id, ownerAndRuntime.signal).then((record) => {
        if (
          disposed ||
          generation !== metadataGeneration ||
          ownerAndRuntime.signal.aborted ||
          !lifecycle.isCurrent(owner)
        ) {
          throw new AccountScopeExpiredError();
        }
        if (record.id !== reference.id) {
          throw new Error(`Font metadata did not match requested font "${reference.id}".`);
        }
        if (!record.contentHash) {
          throw new Error(`Font "${reference.id}" has no content hash.`);
        }
        if (reference.contentHash && record.contentHash !== reference.contentHash) {
          throw new Error(`Font "${reference.id}" changed since this document was created.`);
        }
        return record;
      });
      metadata = metadataRequest
        .catch((error: unknown) => {
          if (lifecycle.isCurrent(owner) && !(error instanceof AccountScopeExpiredError)) {
            clearLoaded(reference);
          }
          throw error;
        })
        .finally(() => {
          if (outputInflight.get(metadataKey) === metadata) {
            outputInflight.delete(metadataKey);
          }
          ownerAndRuntime.dispose();
          abortControllers.delete(metadataRequestController);
        });
      outputInflight.set(metadataKey, metadata);
    }

    return withAbortSignal(metadata, signal).then((record) => {
      if (!lifecycle.isCurrent(owner)) {
        throw new AccountScopeExpiredError();
      }
      const ownerAndCaller = combineSignals(signal, owner.signal);
      return ensure(
        {
          ...reference,
          contentHash: record.contentHash,
        },
        ownerAndCaller.signal,
        record
      ).finally(ownerAndCaller.dispose);
    });
  };

  const runtime: FontRuntime = {
    dispose: () => {
      if (disposed) {
        return;
      }
      unregister?.();
      unregister = undefined;
      disposed = true;
      started = false;
      clear();
      listeners.clear();
    },
    ensure,
    ensureForOutput,
    retain: (reference) => {
      const key = getFontRuntimeKey(reference);
      retainCounts.set(key, (retainCounts.get(key) ?? 0) + 1);
      const entry = registered.get(key);
      if (entry) {
        entry.retained += 1;
      }
      let released = false;
      return () => {
        if (released) {
          return;
        }
        released = true;
        const count = retainCounts.get(key) ?? 0;
        if (count <= 1) {
          retainCounts.delete(key);
        } else {
          retainCounts.set(key, count - 1);
        }
        const current = registered.get(key);
        if (current) {
          current.retained = Math.max(0, current.retained - 1);
        }
        trimFailed();
      };
    },
    getSnapshot: () => snapshot,
    resolveFamily: (reference) => {
      const key = getFontRuntimeKey(reference);
      const loaded = registered.get(key);
      if (loaded) {
        loaded.touched = ++touch;
        return loaded.family;
      }
      return reference.family ?? 'sans-serif';
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    start,
  };

  return runtime;
};
