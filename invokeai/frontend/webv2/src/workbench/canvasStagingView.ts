import type { CanvasStagingCandidateContract, CanvasStateContractV2, QueueItem } from './types';

import { sanitizeBatchCount } from './generation/batch';

export interface CanvasQueuePlaceholderSlot {
  height: number;
  id: string;
  itemIndex: number;
  kind: 'placeholder';
  queueItemId: string;
  width: number;
}

export interface CanvasCandidateSlot {
  candidate: CanvasStagingCandidateContract;
  height: number;
  id: string;
  imageName: string;
  itemIndex?: number;
  kind: 'candidate';
  queueItemId: string;
  width: number;
}

export type CanvasStagingSlot = CanvasCandidateSlot | CanvasQueuePlaceholderSlot;

const isActiveCanvasQueueItemForDocument = (canvas: CanvasStateContractV2, item: QueueItem): boolean =>
  item.snapshot.destination === 'canvas' &&
  (item.status === 'pending' || item.status === 'running') &&
  item.snapshot.canvas.documentRevision === canvas.documentRevision;

const getFiniteNumber = (value: unknown, fallback: number): number =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback;

const getSubmittedBboxSize = (item: QueueItem): { height: number; width: number } => {
  const { bbox } = item.snapshot.canvas.document;

  return {
    height: Math.max(1, Math.round(getFiniteNumber(bbox.height, item.snapshot.canvas.document.height))),
    width: Math.max(1, Math.round(getFiniteNumber(bbox.width, item.snapshot.canvas.document.width))),
  };
};

const getExpectedImageCount = (item: QueueItem): number => {
  if (item.backendItemIds?.length) {
    return item.backendItemIds.length;
  }

  return sanitizeBatchCount(
    item.snapshot.generate?.values.batchCount ?? item.snapshot.widgetStates.generate?.values.batchCount
  );
};

const compareQueueItemsByExecutionOrder = (left: QueueItem, right: QueueItem): number => {
  const leftBackendItemId = left.backendItemIds?.[0];
  const rightBackendItemId = right.backendItemIds?.[0];

  if (leftBackendItemId !== undefined && rightBackendItemId !== undefined && leftBackendItemId !== rightBackendItemId) {
    return leftBackendItemId - rightBackendItemId;
  }

  const leftSubmittedAt = Date.parse(left.snapshot.submittedAt);
  const rightSubmittedAt = Date.parse(right.snapshot.submittedAt);

  if (Number.isFinite(leftSubmittedAt) && Number.isFinite(rightSubmittedAt) && leftSubmittedAt !== rightSubmittedAt) {
    return leftSubmittedAt - rightSubmittedAt;
  }

  return 0;
};

const getQueueItemsInExecutionOrder = (queueItems: readonly QueueItem[]): QueueItem[] =>
  queueItems
    .map((item, index) => ({ index, item }))
    .sort((left, right) => compareQueueItemsByExecutionOrder(left.item, right.item) || right.index - left.index)
    .map(({ item }) => item);

const createCandidateSlot = (candidate: CanvasStagingCandidateContract, itemIndex?: number): CanvasCandidateSlot => ({
  candidate,
  height: candidate.height,
  id: `candidate:${candidate.sourceQueueItemId}:${candidate.imageName}`,
  imageName: candidate.imageName,
  ...(itemIndex === undefined ? {} : { itemIndex }),
  kind: 'candidate',
  queueItemId: candidate.sourceQueueItemId,
  width: candidate.width,
});

const createPlaceholderSlot = (item: QueueItem, itemIndex: number): CanvasQueuePlaceholderSlot => {
  const { height, width } = getSubmittedBboxSize(item);

  return {
    height,
    id: `${item.id}:${itemIndex - 1}`,
    itemIndex,
    kind: 'placeholder',
    queueItemId: item.id,
    width,
  };
};

export const getCanvasQueuePlaceholderSlots = (
  canvas: CanvasStateContractV2,
  queueItems: readonly QueueItem[]
): CanvasQueuePlaceholderSlot[] => {
  const placeholders: CanvasQueuePlaceholderSlot[] = [];

  for (const item of getQueueItemsInExecutionOrder(queueItems)) {
    if (!isActiveCanvasQueueItemForDocument(canvas, item)) {
      continue;
    }

    const completedBackendItemIds = new Set(item.completedBackendItemIds ?? []);
    const cancelledBackendItemIds = new Set(item.cancelledBackendItemIds ?? []);

    if (item.backendItemIds?.length) {
      for (let index = 0; index < item.backendItemIds.length; index += 1) {
        const backendItemId = item.backendItemIds[index];

        if (
          backendItemId === undefined ||
          completedBackendItemIds.has(backendItemId) ||
          cancelledBackendItemIds.has(backendItemId)
        ) {
          continue;
        }

        placeholders.push(createPlaceholderSlot(item, index + 1));
      }

      continue;
    }

    for (let index = 0; index < getExpectedImageCount(item); index += 1) {
      placeholders.push(createPlaceholderSlot(item, index + 1));
    }
  }

  return placeholders;
};

const getCandidatesByQueueItemId = (
  candidates: readonly CanvasStagingCandidateContract[]
): Map<string, CanvasStagingCandidateContract[]> => {
  const candidatesByQueueItemId = new Map<string, CanvasStagingCandidateContract[]>();

  for (const candidate of candidates) {
    const candidatesForQueueItem = candidatesByQueueItemId.get(candidate.sourceQueueItemId) ?? [];

    candidatesForQueueItem.push(candidate);
    candidatesByQueueItemId.set(candidate.sourceQueueItemId, candidatesForQueueItem);
  }

  return candidatesByQueueItemId;
};

const getCanvasStagingSlotsForQueueItem = (
  canvas: CanvasStateContractV2,
  item: QueueItem,
  candidates: readonly CanvasStagingCandidateContract[]
): CanvasStagingSlot[] => {
  const slots: CanvasStagingSlot[] = [];
  const remainingCandidates = [...candidates];
  const isActive = isActiveCanvasQueueItemForDocument(canvas, item);
  const completedBackendItemIds = new Set(item.completedBackendItemIds ?? []);
  const cancelledBackendItemIds = new Set(item.cancelledBackendItemIds ?? []);

  if (item.backendItemIds?.length) {
    for (let index = 0; index < item.backendItemIds.length; index += 1) {
      const backendItemId = item.backendItemIds[index];
      const itemIndex = index + 1;

      if (backendItemId === undefined) {
        continue;
      }

      const candidateIndex = remainingCandidates.findIndex(
        (candidate) => candidate.sourceBackendItemId === backendItemId
      );

      if (candidateIndex !== -1) {
        const [candidate] = remainingCandidates.splice(candidateIndex, 1);

        if (candidate) {
          slots.push(createCandidateSlot(candidate, itemIndex));
        }

        continue;
      }

      if (isActive && !completedBackendItemIds.has(backendItemId) && !cancelledBackendItemIds.has(backendItemId)) {
        slots.push(createPlaceholderSlot(item, itemIndex));
      }
    }

    return [...slots, ...remainingCandidates.map((candidate) => createCandidateSlot(candidate))];
  }

  slots.push(...remainingCandidates.map((candidate, index) => createCandidateSlot(candidate, index + 1)));

  if (!isActive) {
    return slots;
  }

  for (let index = remainingCandidates.length; index < getExpectedImageCount(item); index += 1) {
    slots.push(createPlaceholderSlot(item, index + 1));
  }

  return slots;
};

export const getCanvasStagingSlots = (
  canvas: CanvasStateContractV2,
  queueItems: readonly QueueItem[]
): CanvasStagingSlot[] => {
  const candidatesByQueueItemId = getCandidatesByQueueItemId(canvas.stagingArea.pendingImages);
  const slots: CanvasStagingSlot[] = [];

  for (const item of getQueueItemsInExecutionOrder(queueItems)) {
    const candidates = candidatesByQueueItemId.get(item.id) ?? [];

    if (!isActiveCanvasQueueItemForDocument(canvas, item) && candidates.length === 0) {
      continue;
    }

    slots.push(...getCanvasStagingSlotsForQueueItem(canvas, item, candidates));
    candidatesByQueueItemId.delete(item.id);
  }

  const orphanCandidateSlots = [...candidatesByQueueItemId.values()].flatMap((candidates) =>
    candidates.map((candidate) => createCandidateSlot(candidate))
  );

  return [...orphanCandidateSlots, ...slots];
};

export const getCanvasStagingSlotCount = (canvas: CanvasStateContractV2, queueItems: readonly QueueItem[]): number =>
  getCanvasStagingSlots(canvas, queueItems).length;

export const getFirstCanvasPlaceholderSlotIndex = (
  canvas: CanvasStateContractV2,
  queueItems: readonly QueueItem[]
): number => getCanvasStagingSlots(canvas, queueItems).findIndex((slot) => slot.kind === 'placeholder');
