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
    const promise = Promise.resolve()
      .then(loader)
      .then((implementation) => validateImplementation(widgetId, implementation));
    state = { promise, status: 'loading' };
    void promise.then(
      (value) => {
        state = { promise, status: 'loaded', value };
      },
      (error: unknown) => {
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
