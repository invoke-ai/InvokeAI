import { useMountEffect } from '@platform/react/useMountEffect';

import { createCanvasDimsSync } from './canvasDimsSync';
import { configureDiagnostics } from './diagnostics/logger';
import { subscribeWorkbenchPreferences } from './settings/store';
import { useWorkbenchInternalStore } from './WorkbenchContext';

/**
 * Workbench-owned lifecycle adapter for aggregate-local synchronization and
 * diagnostics. Cross-module adapters are constructed by App.
 */
export const WorkbenchRuntime = () => {
  const store = useWorkbenchInternalStore();

  useMountEffect(() => {
    const canvasDimsSync = createCanvasDimsSync(store);
    const unsubscribeDiagnostics = subscribeWorkbenchPreferences((preferences) => {
      configureDiagnostics({
        enabled: preferences.developerLogEnabled,
        level: preferences.developerLogLevel,
        namespaces: preferences.developerLogNamespaces,
        performanceTimingsEnabled: preferences.developerPerformanceTimingsEnabled,
      });
    });

    return () => {
      unsubscribeDiagnostics();
      canvasDimsSync.dispose();
    };
  });

  return null;
};
