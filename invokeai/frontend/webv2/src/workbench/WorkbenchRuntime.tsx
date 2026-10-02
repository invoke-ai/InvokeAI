import { useMountEffect } from '@platform/react/useMountEffect';

import { createCanvasDimsSync } from './canvasDimsSync';
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
