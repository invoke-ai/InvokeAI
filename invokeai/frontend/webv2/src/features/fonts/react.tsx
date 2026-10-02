import type { ReactNode } from 'react';

import { useMountEffect } from '@platform/react/useMountEffect';
import { createContext, use, useState, useSyncExternalStore } from 'react';

import type { FontDownloadReference, FontReference } from './contracts';
import type { FontRuntime, FontRuntimeSnapshot } from './runtime';

const DISPOSED_MESSAGE = 'The font runtime has been disposed.';

interface PendingRetain {
  reference: FontReference;
  release?: () => void;
  released: boolean;
}

interface DeferredFontRuntime extends FontRuntime {
  attach(runtime: FontRuntime): void;
  fail(error: unknown): void;
}

const createAbortError = (): Error => {
  if (typeof DOMException !== 'undefined') {
    return new DOMException('The font load was aborted.', 'AbortError');
  }
  const error = new Error('The font load was aborted.');
  error.name = 'AbortError';
  return error;
};

/** Loads the browser registry after mount; callers may retain faces and queue ensures during initialization. */
const createDeferredFontRuntime = (): DeferredFontRuntime => {
  const listeners = new Set<() => void>();
  const pendingRetains = new Set<PendingRetain>();
  let delegate: FontRuntime | undefined;
  let delegateUnsubscribe: (() => void) | undefined;
  let delegatePromise: Promise<FontRuntime> | undefined;
  let resolveDelegate: ((runtime: FontRuntime) => void) | undefined;
  let rejectDelegate: ((error: unknown) => void) | undefined;
  let snapshot: FontRuntimeSnapshot = Object.freeze({ generation: 0, states: new Map() });
  let startupError: unknown;
  let disposed = false;
  let started = false;

  const notify = (): void => {
    if (disposed) {
      return;
    }
    for (const listener of listeners) {
      listener();
    }
  };

  const createDelegatePromise = (): Promise<FontRuntime> => {
    if (delegatePromise) {
      return delegatePromise;
    }

    delegatePromise = new Promise<FontRuntime>((resolve, reject) => {
      resolveDelegate = resolve;
      rejectDelegate = reject;
    });
    // Handle failures before consumers attach, while preserving rejection for waiting callers.
    void delegatePromise.catch(() => undefined);
    return delegatePromise;
  };

  const waitForDelegate = (signal?: AbortSignal): Promise<FontRuntime> => {
    const pending = createDelegatePromise();
    if (!signal) {
      return pending;
    }
    if (signal.aborted) {
      return Promise.reject(signal.reason ?? createAbortError());
    }

    return new Promise<FontRuntime>((resolve, reject) => {
      const onAbort = (): void => {
        signal.removeEventListener('abort', onAbort);
        reject(signal.reason ?? createAbortError());
      };
      signal.addEventListener('abort', onAbort, { once: true });
      pending.then(
        (runtime) => {
          signal.removeEventListener('abort', onAbort);
          resolve(runtime);
        },
        (error: unknown) => {
          signal.removeEventListener('abort', onAbort);
          reject(error);
        }
      );
    });
  };

  const start = (): void => {
    if (started && !disposed) {
      return;
    }
    disposed = false;
    started = true;
    startupError = undefined;
    createDelegatePromise();
  };

  const runtime: DeferredFontRuntime = {
    attach: (next) => {
      if (disposed || !started) {
        next.dispose();
        return;
      }
      delegateUnsubscribe?.();
      delegate?.dispose();
      delegate = next;
      startupError = undefined;
      next.start();
      snapshot = next.getSnapshot();
      delegateUnsubscribe = next.subscribe(() => {
        snapshot = next.getSnapshot();
        notify();
      });
      for (const pending of pendingRetains) {
        if (!pending.released) {
          pending.release = next.retain(pending.reference);
        }
      }
      resolveDelegate?.(next);
      resolveDelegate = undefined;
      rejectDelegate = undefined;
      notify();
    },
    dispose: () => {
      if (disposed) {
        return;
      }
      disposed = true;
      started = false;
      delegateUnsubscribe?.();
      delegateUnsubscribe = undefined;
      delegate?.dispose();
      delegate = undefined;
      rejectDelegate?.(new Error(DISPOSED_MESSAGE));
      resolveDelegate = undefined;
      rejectDelegate = undefined;
      delegatePromise = undefined;
      startupError = undefined;
      for (const pending of pendingRetains) {
        pending.release?.();
      }
      pendingRetains.clear();
      snapshot = Object.freeze({ generation: snapshot.generation + 1, states: new Map() });
      for (const listener of listeners) {
        listener();
      }
    },
    ensure: (reference: FontDownloadReference, signal?: AbortSignal) => {
      if (disposed) {
        return Promise.reject(new Error(DISPOSED_MESSAGE));
      }
      start();
      if (startupError !== undefined) {
        return Promise.reject(startupError);
      }
      const wait = delegate ? Promise.resolve(delegate) : waitForDelegate(signal);
      return wait.then((next) => {
        if (disposed) {
          throw new Error(DISPOSED_MESSAGE);
        }
        return next.ensure(reference, signal);
      });
    },
    ensureForOutput: (reference: FontDownloadReference, signal?: AbortSignal) => {
      if (disposed) {
        return Promise.reject(new Error(DISPOSED_MESSAGE));
      }
      start();
      if (startupError !== undefined) {
        return Promise.reject(startupError);
      }
      const wait = delegate ? Promise.resolve(delegate) : waitForDelegate(signal);
      return wait.then((next) => {
        if (disposed) {
          throw new Error(DISPOSED_MESSAGE);
        }
        return next.ensureForOutput(reference, signal);
      });
    },
    fail: (error) => {
      if (disposed) {
        return;
      }
      startupError = error;
      rejectDelegate?.(error);
      resolveDelegate = undefined;
      rejectDelegate = undefined;
      delegatePromise = undefined;
      notify();
    },
    getSnapshot: () => snapshot,
    resolveFamily: (reference) => delegate?.resolveFamily(reference) ?? reference.family ?? 'sans-serif',
    retain: (reference) => {
      const pending: PendingRetain = { reference, released: false };
      pendingRetains.add(pending);
      if (delegate) {
        pending.release = delegate.retain(reference);
      }
      return () => {
        if (pending.released) {
          return;
        }
        pending.released = true;
        pending.release?.();
        pendingRetains.delete(pending);
      };
    },
    start,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };

  return runtime;
};

const FontRuntimeContext = createContext<FontRuntime | null>(null);

export const FontsRuntimeProvider = ({ children, runtime }: { children: ReactNode; runtime?: FontRuntime }) => {
  const [providedRuntime] = useState(() => runtime);
  const [ownedRuntime] = useState<FontRuntime>(() => runtime ?? createDeferredFontRuntime());
  if (providedRuntime !== runtime) {
    throw new Error('FontsRuntimeProvider runtime must remain stable for its mounted lifetime.');
  }

  useMountEffect(() => {
    ownedRuntime.start();
    if (runtime) {
      return;
    }

    const deferredRuntime = ownedRuntime as DeferredFontRuntime;
    let cancelled = false;
    void import('./runtime')
      .then(({ createFontRuntime }) => {
        const implementation = createFontRuntime();
        if (cancelled) {
          implementation.dispose();
          return;
        }
        deferredRuntime.attach(implementation);
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          deferredRuntime.fail(error);
        }
      });

    return () => {
      cancelled = true;
      deferredRuntime.dispose();
    };
  });

  return <FontRuntimeContext value={ownedRuntime}>{children}</FontRuntimeContext>;
};

export const useFontRuntime = (): FontRuntime => {
  const runtime = use(FontRuntimeContext);

  if (!runtime) {
    throw new Error('Font runtime requires an App-composed FontsRuntimeProvider.');
  }

  return runtime;
};

export const useFontRuntimeSnapshot = () => {
  const runtime = useFontRuntime();
  return useSyncExternalStore(runtime.subscribe, runtime.getSnapshot, runtime.getSnapshot);
};
