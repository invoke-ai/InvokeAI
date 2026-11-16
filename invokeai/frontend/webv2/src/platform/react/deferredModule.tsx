import type { ReactNode } from 'react';

import { createLogger } from '@platform/logging/logger';
import { lazy, Suspense, useState } from 'react';

const logger = createLogger({ area: 'deferred-module', namespace: 'app' });

export interface DeferredModule<Module> {
  /** Starts the load if none is running or done; resolves to the module. A failed load is tried again on the next call. */
  load: () => Promise<Module>;
  /** Starts the load ahead of use, as on hover or focus, ignoring a failure. */
  preload: () => void;
  /**
   * Renders `render(module)`. Once the module has loaded it renders directly. Before that, each mount owns its own
   * lazy load, so a load that failed is tried again on the next mount, where a module-level `lazy` would keep its first
   * failure for the page's lifetime. While loading it shows `fallback`; a failed load shows `renderError(error)`
   * when provided, else `fallback`. An error the loaded
   * module throws while rendering reaches the host's error boundary instead.
   */
  Deferred: (props: {
    fallback: ReactNode;
    renderError?: (error: unknown) => ReactNode;
    render: (module: Module) => ReactNode;
  }) => ReactNode;
}

interface DeferredSource<Module> {
  getLoaded: () => Module | null;
  load: () => Promise<Module>;
}

type RenderModule<Module> = (module: Module) => ReactNode;

interface LazyViewProps<Module> {
  fallback: ReactNode;
  renderError?: (error: unknown) => ReactNode;
  renderModule: RenderModule<Module>;
}

// A top-level component, so it closes over nothing: the module's state is read through `source` on each mount.
const DeferredView = <Module,>({
  fallback,
  renderError,
  render,
  source,
}: {
  fallback: ReactNode;
  renderError?: (error: unknown) => ReactNode;
  render: RenderModule<Module>;
  source: DeferredSource<Module>;
}) => {
  // A failed load resolves to a view of the fallback rather than rejecting, so it never reaches an error boundary;
  // errors the loaded module throws while rendering still do.
  const [LazyView] = useState(() =>
    source.getLoaded() === null
      ? lazy(() =>
          source.load().then(
            (module) => ({ default: ({ renderModule }: LazyViewProps<Module>) => renderModule(module) }),
            (error: unknown) => {
              logger.warn({
                error,
                message: 'A view loaded on demand failed to load; showing its fallback',
                name: 'deferred-module.load-failed',
              });
              return {
                default: ({ fallback: failed, renderError }: LazyViewProps<Module>) =>
                  renderError ? renderError(error) : failed,
              };
            }
          )
        )
      : null
  );
  return LazyView ? (
    <Suspense fallback={fallback}>
      <LazyView fallback={fallback} renderError={renderError} renderModule={render} />
    </Suspense>
  ) : (
    // Mounted without a lazy view only once the module had loaded, and a loaded module stays loaded.
    render(source.getLoaded()!)
  );
};

/** A code-split module for views that open on demand, such as panels and menus. */
export const createDeferredModule = <Module,>(importModule: () => Promise<Module>): DeferredModule<Module> => {
  let loaded: Module | null = null;
  let pending: Promise<Module> | null = null;
  // Made once per deferred module, so `Deferred` passes the same object on every render.
  // eslint-disable-next-line react-perf/jsx-no-new-object-as-prop
  const source: DeferredSource<Module> = {
    getLoaded: () => loaded,
    load: () =>
      (pending ??= importModule().then(
        (module) => (loaded = module),
        (error: unknown) => {
          pending = null;
          throw error;
        }
      )),
  };

  return {
    Deferred: ({ fallback, renderError, render }) => (
      <DeferredView fallback={fallback} render={render} renderError={renderError} source={source} />
    ),
    load: source.load,
    preload: () => void source.load().catch(() => {}),
  };
};
