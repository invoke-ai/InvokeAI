/**
 * Canvas font boundary: built-ins use FontFaceSet; injected account-scoped runtimes resolve custom refs with exact
 * style, weight and axes while owning browser-only aliases.
 */

import type { CanvasLayerSourceContract } from '@workbench/canvas-engine/contracts';

/** A CSS `font` shorthand readiness API, normally backed by `document.fonts`. */
export interface FontLoadApi {
  /** Whether every face for the CSS `font` shorthand is loaded and usable. */
  check(font: string): boolean;
  /** Loads the faces for the CSS `font` shorthand; resolves when they're ready. */
  load(font: string): Promise<unknown>;
}

/** The optional fields accepted by the feature font runtime. */
export interface CanvasFontReference {
  id: string;
  contentHash?: string | null;
  family?: string;
  style?: string;
  weight?: number;
  axes?: Readonly<Record<string, number>>;
}

/** Structural custom-font runtime keeps Canvas independent of feature composition and usable in node tests. */
export interface CanvasFontRuntime {
  /** Ensures the exact static face/instance is loaded and returns its browser family alias. */
  ensure(reference: CanvasFontReference, signal?: AbortSignal): Promise<string>;
  /**
   * Revalidate exact faces against current account/catalog state for outputs; cached preview aliases must not
   * allow deleted or changed resources.
   */
  ensureForOutput(reference: CanvasFontReference, signal?: AbortSignal): Promise<string>;
  /** Keeps the loaded face registered while this engine has a live text consumer. */
  retain?(reference: CanvasFontReference): () => void;
  /** Optional generation that changes when account scope cleanup removes registered faces. */
  getSnapshot?(): { generation: number };
  /** Returns the already-registered browser family alias, or the readable fallback family. */
  resolveFamily(reference: CanvasFontReference): string;
  /** Optional readiness/account-lifecycle notifications for editor overlays. */
  subscribe?(listener: () => void): () => void;
}

/** A text source, kept local to avoid importing the rasterizer into this seam. */
export type CanvasTextSource = Extract<CanvasLayerSourceContract, { type: 'text' }>;

/** The per-engine adapter used by layer rasterization and editor overlays. */
export interface FontLoader {
  /** Ensures a legacy CSS shorthand or source-aware custom face is ready. */
  ensure(font: string, onReady: () => void, signal?: AbortSignal): void;
  ensure(source: CanvasTextSource, onReady: () => void, signal?: AbortSignal): void;
  /** Ensures an active editor/raster preview while keeping the shared request alive. */
  ensurePreview(source: CanvasTextSource, signal?: AbortSignal): Promise<string>;
  /** Resolves the family passed to both CanvasRenderingContext2D and CSS. */
  resolveFamily(source: CanvasTextSource): string;
  /** Updates the long-lived document/session sources that must keep custom faces registered. */
  setActiveSources(sources: readonly CanvasTextSource[]): void;
  /** Waits for a source before operations whose output must use final font metrics. */
  waitForReady(source: CanvasTextSource, signal?: AbortSignal): Promise<string>;
  subscribe(listener: () => void): () => void;
  /** Stops callbacks owned by this engine; the application runtime remains app-owned. */
  dispose(): void;
}

/** `check` can throw on an unparseable font string; treat a throw as "not available". */
const safeCheck = (api: FontLoadApi, font: string): boolean => {
  try {
    return api.check(font);
  } catch {
    return false;
  }
};

const createAbortError = (): Error => {
  if (typeof DOMException !== 'undefined') {
    return new DOMException('The Canvas font preview was aborted.', 'AbortError');
  }
  const error = new Error('The Canvas font preview was aborted.');
  error.name = 'AbortError';
  return error;
};

/** Gives one caller cancellation without aborting the shared request. */
const withAbortSignal = <T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> => {
  if (!signal) {
    return promise;
  }
  const getReason = (): unknown => signal.reason ?? createAbortError();
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

const isCanvasFontRuntime = (value: FontLoadApi | CanvasFontRuntime): value is CanvasFontRuntime =>
  typeof (value as CanvasFontRuntime).ensure === 'function' &&
  typeof (value as CanvasFontRuntime).resolveFamily === 'function' &&
  typeof (value as FontLoadApi).check !== 'function';

const sourceReference = (source: CanvasTextSource): CanvasFontReference | null => {
  if (!source.fontRef) {
    return null;
  }
  return {
    axes: source.fontVariations,
    contentHash: source.fontRef.contentHash,
    family: source.fontRef.family || source.fontFamily,
    id: source.fontRef.id,
    style: source.fontStyle ?? 'normal',
    weight: source.fontWeight,
  };
};

const variationKey = (axes: CanvasTextSource['fontVariations']): string =>
  Object.entries(axes ?? {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([tag, value]) => `${tag}:${value}`)
    .join(',');

const sourceKey = (source: CanvasTextSource): string => {
  const ref = sourceReference(source);
  if (!ref) {
    return `css:${source.fontStyle ?? 'normal'} ${source.fontWeight} ${source.fontSize}px ${source.fontFamily}`;
  }
  return [
    ref.id,
    ref.contentHash ?? '',
    ref.family ?? '',
    ref.style ?? '',
    ref.weight ?? '',
    variationKey(ref.axes),
  ].join('|');
};

/** Creates a font loader bound to a legacy API or an application custom-font runtime. */
export const createFontLoader = (api: FontLoadApi | CanvasFontRuntime | null): FontLoader => {
  const cssApi = api && isCanvasFontRuntime(api) ? domFontLoadApi() : api;
  const pending = new Map<string, Promise<string>>();
  const previewControllers = new Map<string, AbortController>();
  const ready = new Map<string, string>();
  const leases = new Map<string, () => void>();
  const activeReferences = new Map<string, CanvasFontReference>();
  const listeners = new Set<() => void>();
  let disposed = false;
  let generation = 0;
  let runtimeSnapshotGeneration = api && isCanvasFontRuntime(api) ? api.getSnapshot?.().generation : undefined;

  const notify = (): void => {
    if (disposed) {
      return;
    }
    for (const listener of listeners) {
      listener();
    }
  };
  const cancelPendingPreviews = (): void => {
    for (const controller of previewControllers.values()) {
      controller.abort(createAbortError());
    }
    previewControllers.clear();
  };

  const releaseLease = (key: string): void => {
    const release = leases.get(key);
    if (!release) {
      return;
    }
    release();
    leases.delete(key);
  };

  const invalidateEvictedReady = (): void => {
    if (!api || !isCanvasFontRuntime(api)) {
      return;
    }
    for (const [key, reference] of activeReferences) {
      const cached = ready.get(key);
      if (cached !== undefined && api.resolveFamily(reference) !== cached) {
        ready.delete(key);
        releaseLease(key);
      }
    }
  };
  const runtimeUnsubscribe =
    api && isCanvasFontRuntime(api)
      ? api.subscribe?.(() => {
          const nextGeneration = api.getSnapshot?.().generation;
          const generationChanged = nextGeneration === undefined || nextGeneration !== runtimeSnapshotGeneration;
          runtimeSnapshotGeneration = nextGeneration ?? runtimeSnapshotGeneration;
          // Runtime generations distinguish account resets from ordinary font notifications. Without generation
          // support, treat every notification as a coarse lifecycle boundary.
          if (generationChanged) {
            generation += 1;
            cancelPendingPreviews();
            for (const release of leases.values()) {
              release();
            }
            leases.clear();
            ready.clear();
          } else {
            invalidateEvictedReady();
          }
          notify();
        })
      : undefined;

  const retainActive = (key: string, reference: CanvasFontReference): void => {
    if (!activeReferences.has(key) || leases.has(key) || !api || !isCanvasFontRuntime(api) || !api.retain) {
      return;
    }
    leases.set(key, api.retain(reference));
  };

  const setActiveSources = (sources: readonly CanvasTextSource[]): void => {
    const next = new Map<string, CanvasFontReference>();
    for (const source of sources) {
      const reference = sourceReference(source);
      if (reference) {
        next.set(`runtime:${sourceKey(source)}`, reference);
      }
    }
    for (const [key, controller] of previewControllers) {
      if (!next.has(key)) {
        controller.abort(createAbortError());
        previewControllers.delete(key);
        pending.delete(key);
      }
    }
    for (const key of leases.keys()) {
      if (!next.has(key)) {
        releaseLease(key);
      }
    }
    for (const key of ready.keys()) {
      if (key.startsWith('runtime:') && !next.has(key)) {
        ready.delete(key);
      }
    }
    activeReferences.clear();
    for (const [key, reference] of next) {
      activeReferences.set(key, reference);
      if (ready.has(key)) {
        retainActive(key, reference);
      }
    }
  };

  const track = (
    key: string,
    load: () => Promise<string>,
    notifyOnResolve = true,
    reference?: CanvasFontReference,
    bypassReady = false,
    controller?: AbortController
  ): { pending: boolean; promise: Promise<string> } => {
    const resolved = ready.get(key);
    if (resolved !== undefined && !bypassReady) {
      return { pending: false, promise: Promise.resolve(resolved) };
    }
    // An output check must not inherit a preview request that started before
    // the operation. That request may have used a cached alias without the
    // fresh metadata/authentication validation required at the output boundary.
    const existing = bypassReady ? undefined : pending.get(key);
    if (existing) {
      return { pending: true, promise: existing };
    }
    let loaded: Promise<string>;
    try {
      loaded = load();
    } catch (error) {
      loaded = Promise.reject(error);
    }
    // A runtime may publish its ordinary `loading` state synchronously from
    // inside ensure(). Capture the fence after that call; a later lifecycle
    // notification still invalidates this request before it can be retained.
    const requestGeneration = generation;
    let promise!: Promise<string>;
    promise = loaded
      .then((value) => {
        if (disposed || requestGeneration !== generation || (!bypassReady && pending.get(key) !== promise)) {
          return value;
        }
        ready.set(key, value);
        if (reference) {
          retainActive(key, reference);
        }
        if (notifyOnResolve) {
          notify();
        }
        return value;
      })
      .catch((error: unknown) => {
        if (
          !disposed &&
          requestGeneration === generation &&
          notifyOnResolve &&
          (bypassReady || pending.get(key) === promise)
        ) {
          if (reference) {
            ready.delete(key);
            releaseLease(key);
          }
          notify();
        }
        throw error;
      })
      .finally(() => {
        if (pending.get(key) === promise) {
          pending.delete(key);
        }
        if (controller && previewControllers.get(key) === controller) {
          previewControllers.delete(key);
        }
      });
    if (!bypassReady) {
      pending.set(key, promise);
    }
    return { pending: true, promise };
  };

  const ensureCss = (font: string): { pending: boolean; promise: Promise<string> } => {
    const key = `css:${font}`;
    if (!cssApi || safeCheck(cssApi, font)) {
      ready.set(key, font);
      return { pending: false, promise: Promise.resolve(font) };
    }
    return track(key, async () => {
      await cssApi.load(font);
      return font;
    });
  };

  const ensureSource = (
    source: CanvasTextSource,
    signal?: AbortSignal,
    preview = true
  ): { pending: boolean; promise: Promise<string> } => {
    const reference = sourceReference(source);
    if (reference && api && isCanvasFontRuntime(api)) {
      const key = `runtime:${sourceKey(source)}`;
      if (preview) {
        const cached = ready.get(key);
        if (cached !== undefined && api.resolveFamily(reference) !== cached) {
          ready.delete(key);
          releaseLease(key);
        }
      }
      const existing = !preview ? undefined : pending.get(key);
      if (existing) {
        const controller = previewControllers.get(key);
        const shared = controller ? withAbortSignal(existing, controller.signal) : existing;
        return { pending: true, promise: withAbortSignal(shared, signal) };
      }
      const controller = preview ? new AbortController() : undefined;
      const result = track(
        key,
        () => (preview ? api.ensure(reference, controller!.signal) : api.ensure(reference, signal)),
        !api.subscribe,
        reference,
        !preview,
        controller
      );
      if (preview && result.pending && controller) {
        previewControllers.set(key, controller);
      }
      if (!preview || !controller) {
        return result;
      }
      const shared = withAbortSignal(result.promise, controller.signal);
      return { ...result, promise: withAbortSignal(shared, signal) };
    }
    if (reference) {
      // A custom reference cannot be made trustworthy by loading the readable
      // family name through document.fonts. Failing closed keeps export and
      // generation output from silently baking fallback glyphs.
      return {
        pending: true,
        promise: Promise.reject(new Error('A custom Canvas font runtime is required for this text source.')),
      };
    }
    const result = ensureCss(textFontShorthand(source));
    return { ...result, promise: withAbortSignal(result.promise, signal) };
  };

  const ensurePreview = (source: CanvasTextSource, signal?: AbortSignal): Promise<string> =>
    ensureSource(source, signal, true).promise;

  return {
    dispose: () => {
      disposed = true;
      generation += 1;
      runtimeUnsubscribe?.();
      cancelPendingPreviews();
      for (const release of leases.values()) {
        release();
      }
      leases.clear();
      activeReferences.clear();
      listeners.clear();
      ready.clear();
      pending.clear();
    },
    ensure: (value: string | CanvasTextSource, onReady: () => void, signal?: AbortSignal) => {
      if (disposed) {
        return;
      }
      const result =
        typeof value === 'string'
          ? (() => {
              const css = ensureCss(value);
              return { ...css, promise: withAbortSignal(css.promise, signal) };
            })()
          : ensureSource(value, signal, true);
      if (!result.pending) {
        return;
      }
      result.promise.then(
        () => {
          if (!disposed && !signal?.aborted) {
            onReady();
          }
        },
        () => {
          // Keep fallback pixels on load failure without unhandled raster-job rejection; the UI may retry.
        }
      );
    },
    ensurePreview,
    resolveFamily: (source) => {
      const reference = sourceReference(source);
      return reference && api && isCanvasFontRuntime(api) ? api.resolveFamily(reference) : source.fontFamily;
    },
    setActiveSources,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    waitForReady: (source, signal) => {
      if (disposed) {
        return Promise.reject(new Error('The Canvas font loader has been disposed.'));
      }
      const reference = sourceReference(source);
      if (!reference || !api || !isCanvasFontRuntime(api)) {
        return ensureSource(source, signal).promise;
      }
      if (!api.ensureForOutput) {
        return Promise.reject(new Error('A custom Canvas font runtime needs output validation before export.'));
      }
      const key = `runtime:${sourceKey(source)}`;
      const result = track(key, () => api.ensureForOutput(reference, signal), !api.subscribe, reference, true);
      return result.promise.catch((error: unknown) => {
        // A readiness operation is allowed to discover that an account-scoped
        // face disappeared after rasterization cached its previous alias.
        ready.delete(key);
        releaseLease(key);
        throw error;
      });
    },
  };
};

/** The CSS shorthand used for legacy sources and for the FontFaceSet fallback path. */
export const textFontShorthand = (source: CanvasTextSource): string => {
  const style = source.fontStyle && source.fontStyle !== 'normal' ? `${source.fontStyle} ` : '';
  return `${style}${source.fontWeight} ${source.fontSize}px ${source.fontFamily}`;
};

/** Resolves the browser's `document.fonts` api, or `null` where it is unavailable. */
export const domFontLoadApi = (): FontLoadApi | null => {
  if (typeof document === 'undefined') {
    return null;
  }
  const fonts = (document as Document & { fonts?: FontLoadApi }).fonts;
  return fonts && typeof fonts.check === 'function' && typeof fonts.load === 'function' ? fonts : null;
};
