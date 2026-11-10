import type { ReactNode } from 'react';

import { Component, lazy, Suspense, useState } from 'react';

/** A deferred view that failed to load shows its fallback rather than taking its host down. */
class DeferredModuleBoundary extends Component<{ children: ReactNode; fallback: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

export interface DeferredModule<Module> {
  /** Starts the load if none is running or done; resolves to the module. A failed load is tried again on the next call. */
  load: () => Promise<Module>;
  /** Starts the load ahead of use, as on hover or focus, ignoring a failure. */
  preload: () => void;
  /**
   * Renders `render(module)`. Once the module has loaded it renders directly. Before that, each mount owns its own
   * lazy load, so a load that failed is tried again on the next mount, where a module-level `lazy` would keep its first
   * failure for the page's lifetime. While loading, and if the load fails, it shows `fallback`.
   */
  Deferred: (props: { fallback: ReactNode; render: (module: Module) => ReactNode }) => ReactNode;
}

interface DeferredSource<Module> {
  getLoaded: () => Module | null;
  load: () => Promise<Module>;
}

type RenderModule<Module> = (module: Module) => ReactNode;

// A top-level component, so it closes over nothing: the module's state is read through `source` on each mount.
const DeferredView = <Module,>({
  fallback,
  render,
  source,
}: {
  fallback: ReactNode;
  render: RenderModule<Module>;
  source: DeferredSource<Module>;
}) => {
  const [LazyView] = useState(() =>
    source.getLoaded() === null
      ? lazy(() =>
          source.load().then((module) => ({
            default: ({ renderModule }: { renderModule: RenderModule<Module> }) => renderModule(module),
          }))
        )
      : null
  );
  return (
    <DeferredModuleBoundary fallback={fallback}>
      {LazyView ? (
        <Suspense fallback={fallback}>
          <LazyView renderModule={render} />
        </Suspense>
      ) : (
        // Mounted without a lazy view only once the module had loaded, and a loaded module stays loaded.
        render(source.getLoaded()!)
      )}
    </DeferredModuleBoundary>
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
    Deferred: ({ fallback, render }) => <DeferredView fallback={fallback} render={render} source={source} />,
    load: source.load,
    preload: () => void source.load().catch(() => {}),
  };
};
