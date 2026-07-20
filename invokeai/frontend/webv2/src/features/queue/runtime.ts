import type { QueueItem } from '@features/queue/core/historyTypes';
import type {
  QueueBackendPort,
  QueueEnqueueGenerateRequest,
  QueueEnqueueWorkflowRequest,
  QueueResultImage,
  QueueResultImageOptions,
} from '@features/queue/core/types';
import type { BackendConnectionStatus } from '@platform/transport/types';

import { shouldSubmitPendingQueueItem } from '@features/queue/core/submissionRules';
import {
  createQueueCoordinator,
  QueueItemCancelledError,
  type QueueCoordinator,
  type QueueModelLoadPort,
  type QueueNodeExecutionPort,
  type ReconcileInput,
} from '@features/queue/runtime/coordinator';
import { getApiErrorMessage } from '@platform/transport/http';

export interface QueueResultDestinationPort {
  addImagesToGalleryBoard(boardId: string, imageNames: string[]): Promise<void>;
}

export interface QueueRuntime {
  dispose(): void;
  start(): void;
}

export interface QueueHistoryProject {
  id: string;
  queue: { items: QueueItem[] };
}

export interface QueueHistoryCommands {
  markBackendCancelled(payload: { backendItemId: number; projectId: string; queueItemId: string }): void;
  markBackendSubmitted(payload: {
    backendBatchId?: string;
    backendItemIds: number[];
    projectId: string;
    queueItemId: string;
  }): void;
  recordError(payload: { area: string; message: string; namespace: 'queue'; projectId?: string }): void;
  refreshBackendData(): void;
  routePartialResults(payload: {
    backendItemId: number;
    images: QueueResultImage[];
    projectId: string;
    queueItemId: string;
  }): void;
  routeResults(payload: { images: QueueResultImage[]; projectId: string; queueItemId: string }): void;
  setConnectionStatus(payload: { error?: string; status: BackendConnectionStatus }): void;
  setStatus(payload: {
    error?: string;
    notify?: boolean;
    projectId: string;
    queueItemId: string;
    status: QueueItem['status'];
  }): void;
}

type QueueItemBackendSubmission =
  | { kind: 'generate'; request: QueueEnqueueGenerateRequest }
  | { kind: 'workflow'; request: QueueEnqueueWorkflowRequest }
  | { error: string; kind: 'invalid' };

const toErrorMessage = (error: unknown): string =>
  getApiErrorMessage(error, error instanceof Error ? error.message : String(error));

export const getQueueItemResultImageOptions = (queueItem: QueueItem): QueueResultImageOptions | undefined => {
  return queueItem.snapshot.resultNodeIds ? { resultNodeIds: queueItem.snapshot.resultNodeIds } : undefined;
};

export const createQueueItemBackendSubmission = (
  project: Pick<QueueHistoryProject, 'id'>,
  queueItem: QueueItem
): QueueItemBackendSubmission => {
  const submission = (queueItem.snapshot as Partial<QueueItem['snapshot']>).backendSubmission;

  if (!submission || typeof submission !== 'object' || !('kind' in submission)) {
    return { error: 'Queue item is missing a compiled backend submission.', kind: 'invalid' };
  }

  if (submission.kind !== 'invalid' && submission.kind !== 'generate' && submission.kind !== 'workflow') {
    return { error: 'Queue item has an unsupported compiled backend submission.', kind: 'invalid' };
  }

  if (submission.kind === 'invalid') {
    return typeof submission.error === 'string'
      ? submission
      : { error: 'Queue item has an invalid compiled backend submission.', kind: 'invalid' };
  }

  if (!submission.graph || typeof submission.graph !== 'object') {
    return { error: 'Queue item backend submission is missing its compiled graph.', kind: 'invalid' };
  }

  if (
    typeof submission.batchCount !== 'number' ||
    !Number.isFinite(submission.batchCount) ||
    submission.batchCount < 1
  ) {
    return { error: 'Queue item backend submission has an invalid batch count.', kind: 'invalid' };
  }

  if (submission.kind === 'generate') {
    if (
      typeof submission.negativePrompt !== 'string' ||
      typeof submission.negativePromptNodeId !== 'string' ||
      typeof submission.positivePrompt !== 'string' ||
      typeof submission.positivePromptNodeId !== 'string' ||
      typeof submission.seed !== 'number' ||
      !Number.isFinite(submission.seed) ||
      typeof submission.seedNodeId !== 'string' ||
      typeof submission.shouldRandomizeSeed !== 'boolean'
    ) {
      return { error: 'Queue item has malformed generate submission metadata.', kind: 'invalid' };
    }
    const { kind: _, ...compiled } = submission;
    return {
      kind: 'generate',
      request: {
        ...compiled,
        destination: queueItem.snapshot.destination,
        projectId: project.id,
        sourceQueueItemId: queueItem.id,
      },
    };
  }

  const { kind: _, ...compiled } = submission;
  return {
    kind: 'workflow',
    request: {
      ...compiled,
      destination: queueItem.snapshot.destination,
      projectId: project.id,
      sourceQueueItemId: queueItem.id,
    },
  };
};

export interface QueueHistoryPort {
  commands: QueueHistoryCommands;
  getSnapshot(): {
    connectionStatus: BackendConnectionStatus;
    isHydrated: boolean;
    projects: QueueHistoryProject[];
  };
  subscribe(listener: () => void): () => void;
}

const getRouteTarget = (history: QueueHistoryPort, localQueueItemId: string) =>
  history
    .getSnapshot()
    .projects.map((project) => ({
      project,
      queueItem: project.queue.items.find((item) => item.id === localQueueItemId),
    }))
    .find((target) => target.queueItem !== undefined);

export const createQueueRuntime = ({
  backend,
  destinations,
  ensureTemplatesLoaded,
  history,
  modelLoads,
  nodeExecution,
}: {
  backend: QueueBackendPort;
  destinations: QueueResultDestinationPort;
  ensureTemplatesLoaded: () => void;
  history: QueueHistoryPort;
  modelLoads: QueueModelLoadPort;
  nodeExecution: QueueNodeExecutionPort;
}): QueueRuntime => {
  const commands = history.commands;
  const startedQueueItemIds = new Set<string>();
  const cancelledQueueItemIds = new Set<string>();
  const detachers: Array<() => void> = [];
  let reconcileState: 'idle' | 'running' | 'done' = 'idle';
  let isDisposed = false;
  let isStarted = false;

  const addImagesToDestination = async (queueItem: QueueItem, imageNames: string[]): Promise<void> => {
    if (queueItem.snapshot.destination !== 'gallery') {
      return;
    }

    const boardId = queueItem.snapshot.galleryBoardId;

    if (boardId && boardId !== 'none') {
      await destinations.addImagesToGalleryBoard(boardId, imageNames);
    }
  };

  /** Drop intermediates when the item asks for it, then land what remains on its destination. */
  const deliverVisibleImages = async (
    queueItem: QueueItem,
    allImages: QueueResultImage[]
  ): Promise<QueueResultImage[]> => {
    const images = queueItem.snapshot.filterIntermediateResults
      ? allImages.filter((image) => !image.isIntermediate)
      : allImages;

    await addImagesToDestination(
      queueItem,
      images.map((image) => image.imageName)
    );

    return images;
  };

  const routeRunResults = async (
    coordinator: QueueCoordinator,
    projectId: string,
    queueItem: QueueItem
  ): Promise<void> => {
    try {
      const allImages = await coordinator.waitForResults(
        queueItem.id,
        queueItem.snapshot.submittedAt,
        getQueueItemResultImageOptions(queueItem)
      );
      const images = await deliverVisibleImages(queueItem, allImages);

      if (isDisposed) {
        return;
      }

      commands.routeResults({ images, projectId, queueItemId: queueItem.id });
      if (queueItem.snapshot.destination === 'gallery') {
        commands.refreshBackendData();
      }
    } catch (error) {
      if (isDisposed) {
        return;
      }

      if (error instanceof QueueItemCancelledError) {
        commands.setStatus({ projectId, queueItemId: queueItem.id, status: 'cancelled' });
        return;
      }

      commands.setStatus({
        error: toErrorMessage(error),
        projectId,
        queueItemId: queueItem.id,
        status: 'failed',
      });
    }
  };

  const routeBackendItemResults = async (
    projectId: string,
    queueItem: QueueItem,
    backendItemId: number
  ): Promise<void> => {
    try {
      const images = await backend.getResultImages(
        backendItemId,
        queueItem.id,
        queueItem.snapshot.submittedAt,
        getQueueItemResultImageOptions(queueItem)
      );
      const visibleImages = await deliverVisibleImages(queueItem, images);

      if (isDisposed) {
        return;
      }

      commands.routePartialResults({
        backendItemId,
        images: visibleImages,
        projectId,
        queueItemId: queueItem.id,
      });
      if (queueItem.snapshot.destination === 'gallery') {
        commands.refreshBackendData();
      }
    } catch (error) {
      if (!isDisposed) {
        commands.recordError({
          area: 'queue-results',
          message: toErrorMessage(error),
          namespace: 'queue',
          projectId,
        });
      }
    }
  };

  const coordinator = createQueueCoordinator(
    {
      onBackendItemCancelled: (localQueueItemId, backendItemId) => {
        const target = getRouteTarget(history, localQueueItemId);

        if (!target?.queueItem || target.queueItem.cancelledBackendItemIds?.includes(backendItemId)) {
          return;
        }

        commands.markBackendCancelled({
          backendItemId,
          projectId: target.project.id,
          queueItemId: localQueueItemId,
        });
      },
      onBackendItemComplete: (localQueueItemId, backendItemId) => {
        const target = getRouteTarget(history, localQueueItemId);

        if (!target?.queueItem || target.queueItem.completedBackendItemIds?.includes(backendItemId)) {
          return;
        }

        return routeBackendItemResults(target.project.id, target.queueItem, backendItemId);
      },
      onGalleryRefresh: commands.refreshBackendData,
    },
    { backend, modelLoads, nodeExecution }
  );

  const submitQueueItem = (project: QueueHistoryProject, queueItem: QueueItem): void => {
    const submission = createQueueItemBackendSubmission(project, queueItem);

    if (submission.kind === 'invalid') {
      commands.setStatus({
        error: submission.error,
        projectId: project.id,
        queueItemId: queueItem.id,
        status: 'failed',
      });
      return;
    }

    const request =
      submission.kind === 'generate'
        ? coordinator.submitGenerate(queueItem.id, submission.request)
        : coordinator.submitWorkflow(queueItem.id, submission.request);

    request
      .then(({ batchId, itemIds }) => {
        if (isDisposed) {
          return;
        }

        commands.markBackendSubmitted({
          backendBatchId: batchId,
          backendItemIds: itemIds,
          projectId: project.id,
          queueItemId: queueItem.id,
        });
        void backend.resumeProcessor().catch(() => undefined);

        return routeRunResults(coordinator, project.id, queueItem);
      })
      .catch((error: unknown) => {
        if (!isDisposed) {
          commands.setStatus({
            error: toErrorMessage(error),
            projectId: project.id,
            queueItemId: queueItem.id,
            status: 'failed',
          });
        }
      });
  };

  const processQueueItems = (): void => {
    if (isDisposed || reconcileState !== 'done') {
      return;
    }

    for (const project of history.getSnapshot().projects) {
      for (const queueItem of project.queue.items) {
        if (shouldSubmitPendingQueueItem(queueItem) && !startedQueueItemIds.has(queueItem.id)) {
          startedQueueItemIds.add(queueItem.id);
          submitQueueItem(project, queueItem);
        }

        if (
          queueItem.status === 'cancelled' &&
          (queueItem.backendBatchId || queueItem.backendItemIds?.length) &&
          !cancelledQueueItemIds.has(queueItem.id)
        ) {
          cancelledQueueItemIds.add(queueItem.id);
          coordinator
            .cancelRun({ backendBatchId: queueItem.backendBatchId, backendItemIds: queueItem.backendItemIds })
            .catch((error: unknown) => {
              if (!isDisposed) {
                commands.recordError({
                  area: 'queue-cancel',
                  message: toErrorMessage(error),
                  namespace: 'queue',
                });
              }
            });
        }
      }
    }
  };

  const reconcile = (): void => {
    if (
      isDisposed ||
      reconcileState !== 'idle' ||
      !history.getSnapshot().isHydrated ||
      history.getSnapshot().connectionStatus !== 'connected'
    ) {
      return;
    }

    reconcileState = 'running';
    const openItems = history
      .getSnapshot()
      .projects.flatMap((project) =>
        project.queue.items
          .filter((queueItem) => queueItem.status === 'pending' || queueItem.status === 'running')
          .map((queueItem) => ({ project, queueItem }))
      );

    for (const { queueItem } of openItems) {
      startedQueueItemIds.add(queueItem.id);
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
        if (isDisposed) {
          return;
        }

        for (const { project, queueItem } of openItems) {
          const outcome = outcomes.get(queueItem.id);

          switch (outcome?.kind) {
            case 'enqueue':
              startedQueueItemIds.delete(queueItem.id);
              break;
            case 'adopted':
              commands.markBackendSubmitted({
                backendBatchId: outcome.backendBatchId,
                backendItemIds: outcome.backendItemIds,
                projectId: project.id,
                queueItemId: queueItem.id,
              });
              void routeRunResults(coordinator, project.id, queueItem);
              break;
            case 'resumed':
              void routeRunResults(coordinator, project.id, queueItem);
              break;
            case 'missing':
              commands.setStatus({
                error: 'This run is no longer on the backend queue (it may have been cleared).',
                notify: false,
                projectId: project.id,
                queueItemId: queueItem.id,
                status: 'failed',
              });
              break;
          }
        }
      })
      .catch((error: unknown) => {
        if (isDisposed) {
          return;
        }

        commands.recordError({
          area: 'queue-reconciliation',
          message: `Queue reconciliation failed: ${toErrorMessage(error)}`,
          namespace: 'queue',
        });

        for (const { project, queueItem } of openItems) {
          if (queueItem.status === 'pending') {
            startedQueueItemIds.delete(queueItem.id);
          } else {
            commands.setStatus({
              error: 'Could not reconcile this run with the backend queue after reload.',
              projectId: project.id,
              queueItemId: queueItem.id,
              status: 'failed',
            });
          }
        }
      })
      .finally(() => {
        if (!isDisposed) {
          reconcileState = 'done';
          processQueueItems();
        }
      });
  };

  const synchronize = (): void => {
    reconcile();
    processQueueItems();
  };

  const start = (): void => {
    if (isStarted || isDisposed) {
      return;
    }

    isStarted = true;
    coordinator.connect();
    ensureTemplatesLoaded();
    detachers.push(
      history.subscribe(synchronize),
      backend.onConnectionChange((status, error) => {
        commands.setConnectionStatus({ error, status });
        synchronize();
      })
    );
    synchronize();
  };

  const dispose = (): void => {
    isDisposed = true;

    for (const detach of detachers.splice(0)) {
      detach();
    }

    coordinator.dispose();
  };

  return { dispose, start };
};
