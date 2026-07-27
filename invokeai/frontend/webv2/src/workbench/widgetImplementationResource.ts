import type {
  WidgetImplementation,
  WidgetImplementationLoadStatus,
  WidgetImplementationResource,
  WidgetTypeId,
} from './widgetContracts';

type ResourceState =
  | { status: 'idle' }
  | { promise: Promise<WidgetImplementation>; status: 'loading' }
  | { promise: Promise<WidgetImplementation>; status: 'loaded'; value: WidgetImplementation }
  | { error: unknown; promise: Promise<WidgetImplementation>; status: 'failed' };

/**
 * The shape React's `use()` inspects before deciding to suspend. A bare promise
 * — even an already-resolved one — always costs a suspension on first read,
 * because `use()` cannot read a native promise synchronously; it has to attach a
 * callback, throw, and retry on the ping. Settling these fields as soon as the
 * load completes lets `use()` return the value on the very first render instead.
 *
 * That distinction is worth real time. Suspending shows a fallback, and once a
 * fallback has been shown React withholds the resolved tree for
 * `FALLBACK_THROTTLE_MS` (300ms) to avoid a flash — which measured as the whole
 * cost of switching layout, long after the chunk had finished downloading.
 */
interface TrackedThenable<T> extends Promise<T> {
  reason?: unknown;
  status?: 'fulfilled' | 'pending' | 'rejected';
  value?: T;
}

const validateImplementation = (widgetId: WidgetTypeId, value: WidgetImplementation): WidgetImplementation => {
  if (!value || typeof value !== 'object' || typeof value.view !== 'function') {
    throw new TypeError(`Widget ${widgetId} implementation must provide a view component.`);
  }

  return value;
};

export const createWidgetImplementationResource = (
  widgetId: WidgetTypeId,
  loader: () => Promise<WidgetImplementation>
): WidgetImplementationResource => {
  let state: ResourceState = { status: 'idle' };

  const start = (): Promise<WidgetImplementation> => {
    const promise: TrackedThenable<WidgetImplementation> = Promise.resolve()
      .then(loader)
      .then((implementation) => validateImplementation(widgetId, implementation));
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

  const load = (): Promise<WidgetImplementation> => {
    if (state.status === 'loading' || state.status === 'loaded' || state.status === 'failed') {
      return state.promise;
    }

    return start();
  };

  return {
    getStatus: (): WidgetImplementationLoadStatus => state.status,
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
