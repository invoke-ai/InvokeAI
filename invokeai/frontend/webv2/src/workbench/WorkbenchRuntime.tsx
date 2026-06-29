import { useEffect, useRef, useState, type Dispatch } from 'react';

import type { Project, QueueItem } from './types';
import type { WorkbenchAction } from './workbenchState';

import { getConnectionStatus, subscribeConnection } from './backend/connectionStore';
import { getApiErrorMessage } from './backend/http';
import {
  createQueueCoordinator,
  QueueItemCancelledError,
  type QueueCoordinator,
  type ReconcileInput,
} from './backend/queueCoordinator';
import { socketHub } from './backend/socketHub';
import { configureDiagnostics } from './diagnostics/logger';
import { addImagesToGalleryBoard } from './gallery/api';
import { getQueueItemResultImages, resumeQueueProcessor } from './generation/api';
import { sanitizeBatchCount } from './generation/batch';
import { normalizeGenerateSettings } from './generation/settings';
import { useWorkbenchPreferenceSelector } from './settings/store';
import {
  useWorkbenchDispatch,
  useWorkbenchHasHydrated,
  useWorkbenchSelector,
  useWorkbenchStore,
} from './WorkbenchContext';
import { createStableSelector } from './workbenchSelectors';
import { ensureInvocationTemplatesLoaded } from './workflows/templates';

const getSnapshotGalleryBoardId = (queueItem: QueueItem): string | null => {
  const selectedBoardId = queueItem.snapshot.widgetStates.gallery.values.selectedBoardId;

  return typeof selectedBoardId === 'string' ? selectedBoardId : null;
};

const toErrorMessage = (error: unknown): string =>
  getApiErrorMessage(error, error instanceof Error ? error.message : String(error));

const getSnapshotBatchCount = (queueItem: QueueItem): number => {
  const batchCount = queueItem.snapshot.widgetStates.generate?.values.batchCount;

  return sanitizeBatchCount(batchCount);
};

const selectQueueRevision = createStableSelector((snapshot: { state: { projects: Project[] } }) =>
  snapshot.state.projects
    .map((project) =>
      [
        project.id,
        project.queue.items
          .map((item) =>
            [
              item.id,
              item.status,
              item.backendBatchId ?? '',
              item.backendItemIds?.join('.') ?? '',
              item.cancelledBackendItemIds?.join('.') ?? '',
              item.completedBackendItemIds?.join('.') ?? '',
            ].join(':')
          )
          .join(','),
      ].join('=')
    )
    .join('|')
);

/**
 * Await a tracked run's terminal outcome, route its images to the queue item's
 * destination, and reflect the outcome in workbench state.
 */
const routeRunResults = async (
  coordinator: QueueCoordinator,
  projectId: string,
  queueItem: QueueItem,
  dispatch: Dispatch<WorkbenchAction>
): Promise<void> => {
  try {
    const allImages = await coordinator.waitForResults(queueItem.id, queueItem.snapshot.submittedAt);
    // A workflow session reports every image its nodes produced; only the
    // non-intermediate outputs are user-facing results.
    const images =
      queueItem.snapshot.sourceId === 'workflow' ? allImages.filter((image) => !image.isIntermediate) : allImages;

    if (queueItem.snapshot.destination === 'gallery') {
      const selectedBoardId = getSnapshotGalleryBoardId(queueItem);

      if (selectedBoardId && selectedBoardId !== 'none') {
        await addImagesToGalleryBoard(
          selectedBoardId,
          images.map((image) => image.imageName)
        );
      }
    }

    dispatch({ images, projectId, queueItemId: queueItem.id, type: 'routeQueueItemResults' });
    if (queueItem.snapshot.destination === 'gallery') {
      dispatch({ type: 'refreshBackendData' });
    }
  } catch (error) {
    if (error instanceof QueueItemCancelledError) {
      dispatch({ projectId, queueItemId: queueItem.id, status: 'cancelled', type: 'setQueueItemStatus' });
      return;
    }

    dispatch({
      error: toErrorMessage(error),
      projectId,
      queueItemId: queueItem.id,
      status: 'failed',
      type: 'setQueueItemStatus',
    });
  }
};

/** Route one completed backend item from a still-running local batch into Gallery. */
const routeBackendItemResults = async (
  projectId: string,
  queueItem: QueueItem,
  backendItemId: number,
  dispatch: Dispatch<WorkbenchAction>
): Promise<void> => {
  if (queueItem.snapshot.destination !== 'gallery') {
    return;
  }

  try {
    const allImages = await getQueueItemResultImages(backendItemId, queueItem.id, queueItem.snapshot.submittedAt);
    const images =
      queueItem.snapshot.sourceId === 'workflow' ? allImages.filter((image) => !image.isIntermediate) : allImages;
    const selectedBoardId = getSnapshotGalleryBoardId(queueItem);

    if (selectedBoardId && selectedBoardId !== 'none') {
      await addImagesToGalleryBoard(
        selectedBoardId,
        images.map((image) => image.imageName)
      );
    }

    dispatch({ backendItemId, images, projectId, queueItemId: queueItem.id, type: 'routeQueueItemPartialResults' });
    dispatch({ type: 'refreshBackendData' });
  } catch (error) {
    dispatch({
      area: 'queue-results',
      message: toErrorMessage(error),
      namespace: 'queue',
      projectId,
      type: 'recordError',
    });
  }
};

const submitQueueItem = (
  coordinator: QueueCoordinator,
  project: Project,
  queueItem: QueueItem,
  dispatch: Dispatch<WorkbenchAction>
): void => {
  const graph = queueItem.snapshot.graph.backendGraph;
  const generateValues =
    queueItem.snapshot.sourceId === 'generate'
      ? normalizeGenerateSettings(queueItem.snapshot.widgetStates.generate.values)
      : null;

  if (!graph || (queueItem.snapshot.sourceId === 'generate' && !generateValues)) {
    dispatch({
      error: `${queueItem.snapshot.sourceId} queue item is missing a compiled backend graph.`,
      projectId: project.id,
      queueItemId: queueItem.id,
      status: 'failed',
      type: 'setQueueItemStatus',
    });
    return;
  }

  const submission = generateValues
    ? coordinator.submitGenerate(queueItem.id, {
        batchCount: generateValues.batchCount,
        destination: queueItem.snapshot.destination,
        graph,
        negativePrompt: generateValues.negativePromptEnabled ? generateValues.negativePrompt : '',
        negativePromptNodeId: 'negative_prompt',
        positivePrompt: generateValues.positivePrompt,
        positivePromptNodeId: 'positive_prompt',
        projectId: project.id,
        seed: generateValues.seed,
        seedNodeId: 'seed',
        shouldRandomizeSeed: generateValues.shouldRandomizeSeed,
        sourceQueueItemId: queueItem.id,
      })
    : coordinator.submitWorkflow(queueItem.id, {
        batchCount: getSnapshotBatchCount(queueItem),
        destination: queueItem.snapshot.destination,
        graph,
        projectId: project.id,
        sourceQueueItemId: queueItem.id,
      });

  submission
    .then(({ batchId, itemIds }) => {
      dispatch({
        backendBatchId: batchId,
        backendItemIds: itemIds,
        projectId: project.id,
        queueItemId: queueItem.id,
        type: 'markQueueItemBackendSubmitted',
      });

      // Enqueuing implies intent to run: a paused processor would otherwise leave
      // the batch stuck "waiting for progress" forever. Mirrors the legacy app's
      // `anyEnqueued` listener; `resume` is a no-op when already running.
      void resumeQueueProcessor().catch(() => undefined);

      return routeRunResults(coordinator, project.id, queueItem, dispatch);
    })
    .catch((error: unknown) => {
      dispatch({
        error: toErrorMessage(error),
        projectId: project.id,
        queueItemId: queueItem.id,
        status: 'failed',
        type: 'setQueueItemStatus',
      });
    });
};

/**
 * Bridges workbench state to the backend queue coordinator: it submits pending
 * queue items, forwards cancellations, and reflects connection status, while
 * the coordinator owns the socket and event-driven settlement. Mounted once
 * inside the WorkbenchProvider; renders nothing.
 */
export const WorkbenchRuntime = () => {
  const store = useWorkbenchStore();
  const dispatch = useWorkbenchDispatch();
  const hasHydrated = useWorkbenchHasHydrated();
  const backendConnectionStatus = useWorkbenchSelector((snapshot) => snapshot.state.backendConnection.status);
  const diagnosticsPreferences = useWorkbenchPreferenceSelector((preferences) => ({
    enabled: preferences.developerLogEnabled,
    level: preferences.developerLogLevel,
    namespaces: preferences.developerLogNamespaces,
    performanceTimingsEnabled: preferences.developerPerformanceTimingsEnabled,
  }));
  const queueRevision = useWorkbenchSelector(selectQueueRevision);
  const coordinatorRef = useRef<QueueCoordinator | null>(null);
  const stateRef = useRef(store.getState());
  const startedQueueItemIdsRef = useRef(new Set<string>());
  const cancelledQueueItemIdsRef = useRef(new Set<string>());
  const reconcileStateRef = useRef<'idle' | 'running' | 'done'>('idle');
  const [isReconciled, setIsReconciled] = useState(false);

  useEffect(
    () =>
      store.subscribe(() => {
        stateRef.current = store.getState();
      }),
    [store]
  );

  useEffect(() => {
    configureDiagnostics(diagnosticsPreferences);
  }, [diagnosticsPreferences]);

  // Mirror the shared socket's connection status (owned by the hub, above the
  // workbench) into workbench state so the editor's many `backendConnection`
  // consumers and the reconcile gate keep working unchanged. Seeds on mount so
  // opening the editor over an already-live socket reconciles immediately.
  useEffect(() => {
    const syncConnection = () => {
      const { error, status } = getConnectionStatus();

      dispatch({ error, status, type: 'setBackendConnectionStatus' });
    };

    syncConnection();

    return subscribeConnection(syncConnection);
  }, [dispatch]);

  useEffect(() => {
    const coordinator = createQueueCoordinator(
      {
        onBackendItemComplete: (localQueueItemId, backendItemId) => {
          const routeTarget = stateRef.current.projects
            .map((project) => ({
              project,
              queueItem: project.queue.items.find((item) => item.id === localQueueItemId),
            }))
            .find((target) => target.queueItem !== undefined);

          if (!routeTarget?.queueItem || routeTarget.queueItem.completedBackendItemIds?.includes(backendItemId)) {
            return;
          }

          return routeBackendItemResults(routeTarget.project.id, routeTarget.queueItem, backendItemId, dispatch);
        },
        onBackendItemCancelled: (localQueueItemId, backendItemId) => {
          const routeTarget = stateRef.current.projects
            .map((project) => ({
              project,
              queueItem: project.queue.items.find((item) => item.id === localQueueItemId),
            }))
            .find((target) => target.queueItem !== undefined);

          if (!routeTarget?.queueItem || routeTarget.queueItem.cancelledBackendItemIds?.includes(backendItemId)) {
            return;
          }

          dispatch({
            backendItemId,
            projectId: routeTarget.project.id,
            queueItemId: localQueueItemId,
            type: 'markQueueItemBackendCancelled',
          });
        },
        onGalleryRefresh: () => {
          dispatch({ type: 'refreshBackendData' });
        },
      },
      { hub: socketHub }
    );

    coordinatorRef.current = coordinator;
    coordinator.connect();
    // Node definitions back workflow route validation, which can be the
    // persisted invocation source before any workflow surface has mounted.
    ensureInvocationTemplatesLoaded();

    return () => {
      coordinatorRef.current = null;
      coordinator.dispose();
    };
  }, [dispatch]);

  // Reconcile persisted queue items against the live backend queue exactly
  // once, after hydration and first connect. Until this finishes, submission
  // is held back so an item the backend already accepted is not enqueued twice.
  useEffect(() => {
    const coordinator = coordinatorRef.current;

    if (
      !coordinator ||
      !hasHydrated ||
      backendConnectionStatus !== 'connected' ||
      reconcileStateRef.current !== 'idle'
    ) {
      return;
    }

    reconcileStateRef.current = 'running';

    const openItems = stateRef.current.projects.flatMap((project) =>
      project.queue.items
        .filter((queueItem) => queueItem.status === 'pending' || queueItem.status === 'running')
        .map((queueItem) => ({ project, queueItem }))
    );

    for (const { queueItem } of openItems) {
      startedQueueItemIdsRef.current.add(queueItem.id);
    }

    const inputs: ReconcileInput[] = openItems.map(({ queueItem }) => ({
      backendBatchId: queueItem.backendBatchId,
      backendItemIds: queueItem.backendItemIds,
      id: queueItem.id,
      status: queueItem.status === 'running' ? 'running' : 'pending',
    }));

    coordinator
      .reconcile(inputs)
      .then((outcomes) => {
        for (const { project, queueItem } of openItems) {
          const outcome = outcomes.get(queueItem.id);

          switch (outcome?.kind) {
            case 'enqueue': {
              startedQueueItemIdsRef.current.delete(queueItem.id);
              break;
            }
            case 'adopted': {
              dispatch({
                backendBatchId: outcome.backendBatchId,
                backendItemIds: outcome.backendItemIds,
                projectId: project.id,
                queueItemId: queueItem.id,
                type: 'markQueueItemBackendSubmitted',
              });
              void routeRunResults(coordinator, project.id, queueItem, dispatch);
              break;
            }
            case 'resumed': {
              void routeRunResults(coordinator, project.id, queueItem, dispatch);
              break;
            }
            case 'missing': {
              dispatch({
                error: 'This run is no longer on the backend queue (it may have been cleared).',
                notify: false,
                projectId: project.id,
                queueItemId: queueItem.id,
                status: 'failed',
                type: 'setQueueItemStatus',
              });
              break;
            }
          }
        }
      })
      .catch((error: unknown) => {
        // Release pending items to the normal submission path; running items
        // cannot be re-attached without backend state, so fail them visibly
        // rather than leaving them stuck as running forever.
        dispatch({
          area: 'queue-reconciliation',
          message: `Queue reconciliation failed: ${toErrorMessage(error)}`,
          namespace: 'queue',
          type: 'recordError',
        });

        for (const { project, queueItem } of openItems) {
          if (queueItem.status === 'pending') {
            startedQueueItemIdsRef.current.delete(queueItem.id);
            continue;
          }

          dispatch({
            error: 'Could not reconcile this run with the backend queue after reload.',
            projectId: project.id,
            queueItemId: queueItem.id,
            status: 'failed',
            type: 'setQueueItemStatus',
          });
        }
      })
      .finally(() => {
        reconcileStateRef.current = 'done';
        setIsReconciled(true);
      });
  }, [backendConnectionStatus, dispatch, hasHydrated, queueRevision]);

  useEffect(() => {
    const coordinator = coordinatorRef.current;

    if (!coordinator || !isReconciled) {
      return;
    }

    for (const project of stateRef.current.projects) {
      for (const queueItem of project.queue.items) {
        if (
          queueItem.status === 'pending' &&
          (queueItem.snapshot.sourceId === 'generate' || queueItem.snapshot.sourceId === 'workflow') &&
          !startedQueueItemIdsRef.current.has(queueItem.id)
        ) {
          startedQueueItemIdsRef.current.add(queueItem.id);
          submitQueueItem(coordinator, project, queueItem, dispatch);
        }

        if (
          queueItem.status === 'cancelled' &&
          (queueItem.backendBatchId || queueItem.backendItemIds?.length) &&
          !cancelledQueueItemIdsRef.current.has(queueItem.id)
        ) {
          cancelledQueueItemIdsRef.current.add(queueItem.id);
          coordinator
            .cancelRun({ backendBatchId: queueItem.backendBatchId, backendItemIds: queueItem.backendItemIds })
            .catch((error: unknown) => {
              dispatch({
                area: 'queue-cancel',
                message: toErrorMessage(error),
                namespace: 'queue',
                type: 'recordError',
              });
            });
        }
      }
    }
  }, [dispatch, isReconciled, queueRevision]);

  return null;
};
