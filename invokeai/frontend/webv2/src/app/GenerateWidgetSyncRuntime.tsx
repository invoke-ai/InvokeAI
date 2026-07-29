import { ensureModelsLoaded, getModelsSnapshot, subscribeModels } from '@features/models';
import { useMountEffect } from '@platform/react/useMountEffect';
import { createProjectedExternalStore } from '@platform/state/projectedExternalStore';
import { useQueryClient } from '@tanstack/react-query';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useWorkbenchInternalStore } from '@workbench/WorkbenchContext';

import { createGenerateWidgetSyncRuntime, type GenerateWidgetSyncProjectSnapshot } from './generateWidgetSyncRuntime';

/** App-owned production composition of Generation, Models, Query, and Workbench. */
export const GenerateWidgetSyncRuntime = () => {
  const store = useWorkbenchInternalStore();
  const queryClient = useQueryClient();

  useMountEffect(() => {
    const project = createProjectedExternalStore({
      isEqual: (left: GenerateWidgetSyncProjectSnapshot, right: GenerateWidgetSyncProjectSnapshot) =>
        left.id === right.id && left.values === right.values,
      select: (snapshot) => ({
        id: snapshot.activeProject.id,
        values: getProjectWidgetValues(snapshot.activeProject, 'generate'),
      }),
      source: { getSnapshot: store.getSnapshot, subscribe: store.subscribe },
    });
    const models = createProjectedExternalStore({
      select: (snapshot) => snapshot.models,
      source: { getSnapshot: getModelsSnapshot, subscribe: subscribeModels },
    });
    const runtime = createGenerateWidgetSyncRuntime({
      models,
      patchValues: store.commands.generation.patchSettings,
      project,
      queryClient,
    });

    void ensureModelsLoaded();
    return () => runtime.dispose();
  });

  return null;
};
