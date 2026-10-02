import type { ReactNode } from 'react';

import { useMountEffect } from '@platform/react/useMountEffect';
import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { useState } from 'react';

import { createCanvasDimsSync } from './canvasDimsSync';
import { createWorkbenchFocusController, FocusRegionProvider } from './focusRegions';
import { useWorkbenchInternalStore } from './WorkbenchContext';

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
 */
export const WorkbenchFocusProvider = ({ children }: { children: ReactNode }) => {
  const store = useWorkbenchInternalStore();
  const [controller] = useState(() => createWorkbenchFocusController(() => store.getSnapshot().activeProject.id));

  useMountEffect(() => {
    let projectId = store.getSnapshot().activeProject.id;
    const unsubscribe = store.subscribe(() => {
      const nextProjectId = store.getSnapshot().activeProject.id;

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
