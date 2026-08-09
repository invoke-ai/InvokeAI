import { invalidateGallery } from '@features/gallery/queries';
import { modelLoadActivitySink } from '@features/models';
import { nodeExecutionStore } from '@features/nodes';
import { createProductionQueueRuntime } from '@features/queue';
import { ensureInvocationTemplatesLoaded } from '@features/workflow/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { useQueryClient } from '@tanstack/react-query';
import { useWorkbenchCommands, useWorkbenchQueries, useWorkbenchSubscription } from '@workbench/WorkbenchContext';

/** App-owned production composition of Queue, Workbench, Gallery, Workflow, Nodes, and Models adapters. */
export const QueueRuntimeAdapter = () => {
  const { notifications, queue } = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const queryClient = useQueryClient();
  const subscribe = useWorkbenchSubscription();

  useMountEffect(() => {
    const owner = captureAccountScope();
    const runtime = createProductionQueueRuntime({
      destinations: {
        addImagesToGalleryBoard: async (boardId, imageNames) => {
          assertAccountScopeCurrent(owner);
          const { galleryOrganization } = await import('@features/gallery');

          assertAccountScopeCurrent(owner);
          await galleryOrganization.addToBoard(boardId, imageNames, owner.signal);
          assertAccountScopeCurrent(owner);
        },
        addVideosToGalleryBoard: async (boardId, videoNames) => {
          assertAccountScopeCurrent(owner);
          const { galleryItemOrganization } = await import('@features/gallery');

          assertAccountScopeCurrent(owner);
          await galleryItemOrganization.moveToBoard(
            videoNames.map((name) => ({ kind: 'video' as const, name })),
            boardId,
            owner.signal
          );
          assertAccountScopeCurrent(owner);
        },
      },
      ensureTemplatesLoaded: ensureInvocationTemplatesLoaded,
      history: {
        commands: {
          ...queue,
          recordError: notifications.reportError,
          refreshBackendData: () => void invalidateGallery(queryClient),
        },
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
      modelLoads: modelLoadActivitySink,
      nodeExecution: nodeExecutionStore,
    });

    runtime.start();
    return () => runtime.dispose();
  });

  return null;
};
