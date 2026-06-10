import { useEffect, useRef } from 'react';

import { cancelQueueItems, enqueueGenerateGraph, waitForQueueItemImages } from './generation/api';
import type { GenerateSettings } from './generation/types';
import { addImagesToGalleryBoard } from './gallery/api';
import type { QueueItem } from './types';
import { useWorkbench } from './WorkbenchContext';

const isGenerateSettings = (value: unknown): value is GenerateSettings => {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const record = value as Record<string, unknown>;

  return (
    typeof record.modelKey === 'string' &&
    typeof record.positivePrompt === 'string' &&
    typeof record.negativePrompt === 'string' &&
    typeof record.seed === 'number' &&
    Number.isFinite(record.seed)
  );
};

const getPendingGenerateQueueItems = (items: QueueItem[]): QueueItem[] =>
  items.filter((item) => item.status === 'pending' && item.snapshot.sourceId === 'generate');

const getCancelledQueueItems = (items: QueueItem[]): QueueItem[] => items.filter((item) => item.status === 'cancelled');

const getCancelledBackendQueueItems = (items: QueueItem[]): QueueItem[] =>
  items.filter((item) => item.status === 'cancelled' && item.backendItemIds?.length);

const getSnapshotGalleryBoardId = (queueItem: QueueItem): string | null => {
  const selectedBoardId = queueItem.snapshot.widgetStates.gallery.values.selectedBoardId;

  return typeof selectedBoardId === 'string' ? selectedBoardId : null;
};

export const WorkbenchRuntime = () => {
  const { state, dispatch } = useWorkbench();
  const startedQueueItemIdsRef = useRef(new Set<string>());
  const cancelledQueueItemIdsRef = useRef(new Set<string>());
  const backendCancellationQueueItemIdsRef = useRef(new Set<string>());

  useEffect(() => {
    for (const project of state.projects) {
      for (const queueItem of getCancelledQueueItems(project.queue.items)) {
        cancelledQueueItemIdsRef.current.add(queueItem.id);
      }

      for (const queueItem of getPendingGenerateQueueItems(project.queue.items)) {
        if (startedQueueItemIdsRef.current.has(queueItem.id)) {
          continue;
        }

        startedQueueItemIdsRef.current.add(queueItem.id);

        const graph = queueItem.snapshot.graph.backendGraph;
        const generateValues = queueItem.snapshot.widgetStates.generate.values;

        if (!graph || !isGenerateSettings(generateValues)) {
          dispatch({
            error: 'Generate queue item is missing a compiled backend graph.',
            projectId: project.id,
            queueItemId: queueItem.id,
            status: 'failed',
            type: 'setQueueItemStatus',
          });
          continue;
        }

        const queuedAt = queueItem.snapshot.submittedAt;

        enqueueGenerateGraph({
          batchCount: generateValues.batchCount,
          destination: queueItem.snapshot.destination,
          graph,
          negativePrompt: generateValues.negativePrompt,
          negativePromptNodeId: 'negative_prompt',
          positivePrompt: generateValues.positivePrompt,
          positivePromptNodeId: 'positive_prompt',
          seed: generateValues.seed,
          seedNodeId: 'seed',
          sourceQueueItemId: queueItem.id,
        })
          .then(async ({ itemIds }) => {
            dispatch({
              backendItemIds: itemIds,
              projectId: project.id,
              queueItemId: queueItem.id,
              type: 'markQueueItemBackendSubmitted',
            });

            if (cancelledQueueItemIdsRef.current.has(queueItem.id)) {
              return;
            }

            const images = (
              await Promise.all(itemIds.map((itemId) => waitForQueueItemImages(itemId, queueItem.id, queuedAt)))
            ).flat();

            if (cancelledQueueItemIdsRef.current.has(queueItem.id)) {
              return;
            }

            if (queueItem.snapshot.destination === 'gallery') {
              const selectedBoardId = getSnapshotGalleryBoardId(queueItem);

              if (selectedBoardId && selectedBoardId !== 'none') {
                await addImagesToGalleryBoard(
                  selectedBoardId,
                  images.map((image) => image.imageName)
                );
              }
            }

            dispatch({ images, projectId: project.id, queueItemId: queueItem.id, type: 'routeQueueItemResults' });
          })
          .catch((error: unknown) => {
            dispatch({
              error: error instanceof Error ? error.message : String(error),
              projectId: project.id,
              queueItemId: queueItem.id,
              status: 'failed',
              type: 'setQueueItemStatus',
            });
          });
      }

      for (const queueItem of getCancelledBackendQueueItems(project.queue.items)) {
        if (backendCancellationQueueItemIdsRef.current.has(queueItem.id) || !queueItem.backendItemIds?.length) {
          continue;
        }

        backendCancellationQueueItemIdsRef.current.add(queueItem.id);

        cancelQueueItems(queueItem.backendItemIds).catch((error: unknown) => {
          dispatch({ message: error instanceof Error ? error.message : String(error), type: 'recordError' });
        });
      }
    }
  }, [dispatch, state.projects]);

  return null;
};
