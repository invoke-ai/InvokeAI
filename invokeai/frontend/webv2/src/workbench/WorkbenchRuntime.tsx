import type { ReactNode } from 'react';

import { useMountEffect } from '@platform/react/useMountEffect';
import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { useState } from 'react';

import { createCanvasDimsSync } from './canvasDimsSync';
import { createWorkbenchFocusController, FocusRegionProvider } from './focusRegions';
import { useWorkbenchInternalStore, useWorkbenchQueries, useWorkbenchSubscription } from './WorkbenchContext';

/** Workbench-owned lifecycle adapter for aggregate-local synchronization. Cross-module adapters are constructed by App. */
export const WorkbenchRuntime = () => {
  const store = useWorkbenchInternalStore();

  useMountEffect(() => {
    const canvasDimsSync = createCanvasDimsSync(store);

    return () => {
      canvasDimsSync.dispose();
    };
  });

  return null;
};

/**
 * Owns workbench focus for everything below it — the shell and the hotkey runtime read the same target. Focus is
 * transient: it is forgotten, along with any focus move still in flight, when the project on screen changes, the
 * account changes, or the workbench unmounts.
 *
 * It lives in this module, beside the other workbench lifecycle adapter, because the editor's startup module set
 * is pinned by the architecture performance gate; a module of its own would have to be added to that baseline.
 */
export const WorkbenchFocusProvider = ({ children }: { children: ReactNode }) => {
  const { getSnapshot } = useWorkbenchQueries();
  const subscribe = useWorkbenchSubscription();
  const [controller] = useState(() =>
    createWorkbenchFocusController({
      getProjectId: () => getSnapshot().activeProject.id,
      isFloating: (instanceId) => getSnapshot().activeProject.floatingWidgets?.[instanceId] !== undefined,
    })
  );

  useMountEffect(() => {
    let projectId = getSnapshot().activeProject.id;
    const unsubscribe = subscribe(() => {
      const nextProjectId = getSnapshot().activeProject.id;

      if (nextProjectId !== projectId) {
        projectId = nextProjectId;
        controller.clear();
      }
    });
    const unregister = registerAccountOwnedResource({ clear: controller.clear, name: 'workbench-focus' });

    return () => {
      unsubscribe();
      unregister();
      controller.clear();
    };
  });

  return <FocusRegionProvider controller={controller}>{children}</FocusRegionProvider>;
};
