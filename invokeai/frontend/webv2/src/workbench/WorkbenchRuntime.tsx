import { useEffect, useRef, useState, type Dispatch } from 'react';

import {
  createQueueCoordinator,
  QueueItemCancelledError,
  type QueueCoordinator,
  type ReconcileInput,
} from './backend/queueCoordinator';
import { normalizeGenerateSettings } from './generation/settings';
import { addImagesToGalleryBoard } from './gallery/api';
import { ensureInvocationTemplatesLoaded } from './workflows/templates';
import type { Project, QueueItem } from './types';
import { useWorkbenchDispatch, useWorkbenchHasHydrated, useWorkbenchSelector } from './WorkbenchContext';
import type { WorkbenchAction } from './workbenchState';

const getSnapshotGalleryBoardId = (queueItem: QueueItem): string | null => {
  const selectedBoardId = queueItem.snapshot.widgetStates.gallery.values.selectedBoardId;

  return typeof selectedBoardId === 'string' ? selectedBoardId : null;
};

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

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
      queueItem.snapshot.sourceId === 'project-graph' ? allImages.filter((image) => !image.isIntermediate) : allImages;

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
        negativePrompt: generateValues.negativePrompt,
        negativePromptNodeId: 'negative_prompt',
        positivePrompt: generateValues.positivePrompt,
        positivePromptNodeId: 'positive_prompt',
        seed: generateValues.seed,
        seedNodeId: 'seed',
        shouldRandomizeSeed: generateValues.shouldRandomizeSeed,
        sourceQueueItemId: queueItem.id,
      })
    : coordinator.submitWorkflow(queueItem.id, {
        destination: queueItem.snapshot.destination,
        graph,
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
  const state = useWorkbenchSelector((snapshot) => snapshot.state);
  const dispatch = useWorkbenchDispatch();
  const hasHydrated = useWorkbenchHasHydrated();
  const coordinatorRef = useRef<QueueCoordinator | null>(null);
  const startedQueueItemIdsRef = useRef(new Set<string>());
  const cancelledQueueItemIdsRef = useRef(new Set<string>());
  const reconcileStateRef = useRef<'idle' | 'running' | 'done'>('idle');
  const [isReconciled, setIsReconciled] = useState(false);

  useEffect(() => {
    const coordinator = createQueueCoordinator({
      onConnectionChange: (status, error) => {
        dispatch({ error, status, type: 'setBackendConnectionStatus' });
      },
      onGalleryRefresh: () => {
        dispatch({ type: 'refreshBackendData' });
      },
    });

    coordinatorRef.current = coordinator;
    coordinator.connect();
    // Node definitions back project-graph route validation, which can be the
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
      state.backendConnection.status !== 'connected' ||
      reconcileStateRef.current !== 'idle'
    ) {
      return;
    }

    reconcileStateRef.current = 'running';

    const openItems = state.projects.flatMap((project) =>
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
        dispatch({ message: `Queue reconciliation failed: ${toErrorMessage(error)}`, type: 'recordError' });

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
  }, [dispatch, hasHydrated, state.backendConnection.status, state.projects]);

  useEffect(() => {
    const coordinator = coordinatorRef.current;

    if (!coordinator || !isReconciled) {
      return;
    }

    for (const project of state.projects) {
      for (const queueItem of project.queue.items) {
        if (
          queueItem.status === 'pending' &&
          (queueItem.snapshot.sourceId === 'generate' || queueItem.snapshot.sourceId === 'project-graph') &&
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
              dispatch({ message: toErrorMessage(error), type: 'recordError' });
            });
        }
      }
    }
  }, [dispatch, isReconciled, state.projects]);

  return null;
};
