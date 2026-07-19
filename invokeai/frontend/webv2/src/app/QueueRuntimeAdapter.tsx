import { nodeExecutionStore } from '@features/nodes';
import { createProductionQueueRuntime } from '@features/queue';
import { ensureInvocationTemplatesLoaded } from '@features/workflow/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useWorkbenchCommands, useWorkbenchQueries, useWorkbenchSubscription } from '@workbench/WorkbenchContext';

/** App-owned production composition of Queue, Workbench, Gallery, and Workflow adapters. */
export const QueueRuntimeAdapter = () => {
  const { notifications, queue } = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const subscribe = useWorkbenchSubscription();

  useMountEffect(() => {
    const runtime = createProductionQueueRuntime({
      destinations: {
        addImagesToGalleryBoard: async (boardId, imageNames) => {
          const { galleryOrganization } = await import('@features/gallery');

          await galleryOrganization.addToBoard(boardId, imageNames);
        },
      },
      ensureTemplatesLoaded: ensureInvocationTemplatesLoaded,
      history: {
        commands: { ...queue, recordError: notifications.reportError },
        getSnapshot: () => {
          const snapshot = queries.getSnapshot();

          return {
            connectionStatus: snapshot.backendConnection.status,
            isHydrated: snapshot.hasHydrated,
            projects: snapshot.projects,
          };
        },
        subscribe,
      },
      nodeExecution: nodeExecutionStore,
    });

    runtime.start();
    return () => runtime.dispose();
  });

  return null;
};
