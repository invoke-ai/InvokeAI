/**
 * Set React use() settlement fields when loading completes so the first read can return synchronously. Native
 * promises otherwise suspend on first read, potentially triggering fallback throttling after the chunk is already
 * available.
 */
interface TrackedThenable<T> extends Promise<T> {
  reason?: unknown;
  status?: 'fulfilled' | 'pending' | 'rejected';
  value?: T;
}

export type DeferredResourceStatus = 'idle' | 'loading' | 'loaded' | 'failed';

/** Registry-owned, single-flight resource for a deferred module. */
export interface DeferredResource<T> {
  getStatus: () => DeferredResourceStatus;
  load: () => Promise<T>;
  preload: () => void;
  retry: () => Promise<T>;
}

type ResourceState<T> =
  | { status: 'idle' }
  | { promise: Promise<T>; status: 'loading' }
  | { promise: Promise<T>; status: 'loaded'; value: T }
  | { error: unknown; promise: Promise<T>; status: 'failed' };

export const createDeferredResource = <T>(
  loader: () => Promise<T>,
  validate?: (value: T) => T
): DeferredResource<T> => {
  let state: ResourceState<T> = { status: 'idle' };

  const start = (): Promise<T> => {
    const promise: TrackedThenable<T> = Promise.resolve()
      .then(loader)
      .then((value) => (validate ? validate(value) : value));
    promise.status = 'pending';
    state = { promise, status: 'loading' };
    void promise.then(
      (value) => {
        promise.status = 'fulfilled';
        promise.value = value;
        state = { promise, status: 'loaded', value };
      },
      (error: unknown) => {
        promise.status = 'rejected';
        promise.reason = error;
        state = { error, promise, status: 'failed' };
      }
    );
    return promise;
  };

  const load = (): Promise<T> => {
    if (state.status === 'loading' || state.status === 'loaded' || state.status === 'failed') {
      return state.promise;
    }

    return start();
  };

  return {
    getStatus: () => state.status,
    load,
    preload: () => {
      void load().catch(() => undefined);
    },
    retry: () => {
      if (state.status === 'failed') {
        state = { status: 'idle' };
      }

      return load();
    },
  };
};
