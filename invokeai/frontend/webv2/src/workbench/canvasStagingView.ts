import type { CanvasStagingCandidateContract, CanvasStateContractV3, Rect } from '@workbench/canvas-engine/api';
import type { WorkbenchQueueItem as QueueItem } from '@workbench/queueHistoryContracts';

import { getQueueItemSnapshotBatchCount, getQueueItemSnapshotDimensions } from '@features/queue/contracts';

/** A result still generating. `x`/`y`/`width`/`height` are where it lands: the rect its run was submitted with. */
export interface CanvasQueuePlaceholderSlot {
  height: number;
  id: string;
  itemIndex: number;
  kind: 'placeholder';
  queueItemId: string;
  width: number;
  x: number;
  y: number;
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

const isActiveCanvasQueueItemForDocument = (canvas: CanvasStateContractV3, item: QueueItem): boolean =>
  item.snapshot.destination === 'canvas' &&
  (item.status === 'pending' || item.status === 'running') &&
  item.snapshot.canvas.documentRevision === canvas.documentRevision;

const getFiniteNumber = (value: unknown, fallback: number): number =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback;

/** The submitted bbox's origin and the result's size there (an upscale's output is larger than its bbox). */
const getSubmittedRect = (item: QueueItem): Rect => {
  const { bbox } = item.snapshot.canvas.document;
  const bboxSize = {
    height: Math.max(1, Math.round(getFiniteNumber(bbox.height, item.snapshot.canvas.document.height))),
    width: Math.max(1, Math.round(getFiniteNumber(bbox.width, item.snapshot.canvas.document.width))),
  };
  const { height, width } =
    item.snapshot.sourceId === 'upscale' ? getQueueItemSnapshotDimensions(item, bboxSize) : bboxSize;

  return { height, width, x: getFiniteNumber(bbox.x, 0), y: getFiniteNumber(bbox.y, 0) };
};

const getExpectedImageCount = (item: QueueItem): number => {
  if (item.backendItemIds?.length) {
    return item.backendItemIds.length;
  }

  return getQueueItemSnapshotBatchCount(item);
};

/** Collision-safe candidate value identity used with the selected slot index for guarded acceptance. */
export const getCanvasStagingCandidateFingerprint = (candidate: CanvasStagingCandidateContract): string =>
  JSON.stringify([
    candidate.sourceQueueItemId,
    candidate.sourceBackendItemId ?? null,
    candidate.imageName,
    candidate.width,
    candidate.height,
    candidate.placement.x,
    candidate.placement.y,
    candidate.placement.width,
    candidate.placement.height,
    candidate.placement.opacity,
  ]);

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
  id: `candidate:${getCanvasStagingCandidateFingerprint(candidate)}`,
  imageName: candidate.imageName,
  ...(itemIndex === undefined ? {} : { itemIndex }),
  kind: 'candidate',
  queueItemId: candidate.sourceQueueItemId,
  width: candidate.width,
});

const createPlaceholderSlot = (item: QueueItem, itemIndex: number): CanvasQueuePlaceholderSlot => ({
  ...getSubmittedRect(item),
  id: `${item.id}:${itemIndex - 1}`,
  itemIndex,
  kind: 'placeholder',
  queueItemId: item.id,
});

/** Where a slot's result lands in document space: a candidate's placement, a placeholder's submitted rect. */
export const getCanvasStagingSlotRect = (slot: CanvasStagingSlot): Rect =>
  slot.kind === 'candidate'
    ? {
        height: slot.candidate.placement.height,
        width: slot.candidate.placement.width,
        x: slot.candidate.placement.x,
        y: slot.candidate.placement.y,
      }
    : { height: slot.height, width: slot.width, x: slot.x, y: slot.y };

export const getCanvasQueuePlaceholderSlots = (
  canvas: CanvasStateContractV3,
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
  canvas: CanvasStateContractV3,
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
  canvas: CanvasStateContractV3,
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

export const getCanvasStagingSlotCount = (canvas: CanvasStateContractV3, queueItems: readonly QueueItem[]): number =>
  getCanvasStagingSlots(canvas, queueItems).length;

/**
 * Whether this canvas session has results to lose: a staged candidate, or a generation still queued or running for
 * the current document revision (even one whose placeholders are all spent). A document swap throws them away.
 */
export const hasPendingCanvasResults = (
  canvas: Pick<CanvasStateContractV3, 'documentRevision' | 'stagingArea'>,
  queueItems: readonly QueueItem[]
): boolean =>
  canvas.stagingArea.pendingImages.length > 0 ||
  queueItems.some(
    (item) =>
      item.snapshot.destination === 'canvas' &&
      item.snapshot.canvas.documentRevision === canvas.documentRevision &&
      (item.status === 'pending' || item.status === 'running')
  );

export const getFirstCanvasPlaceholderSlotIndex = (
  canvas: CanvasStateContractV3,
  queueItems: readonly QueueItem[]
): number => getCanvasStagingSlots(canvas, queueItems).findIndex((slot) => slot.kind === 'placeholder');

/** Whether the queue can still cancel `item`: the backend supports it and its work is queued or running. */
export const isCancellableQueueItem = (item: QueueItem): boolean =>
  item.cancellable && (item.status === 'pending' || item.status === 'running');

/**
 * A run this tab queued that the backend has not accepted: never sent, or its enqueue still in flight. A backend
 * cancel cannot reach it; only marking it canceled here stops it (an enqueue that lands then is canceled at once).
 */
export const isUnsubmittedQueueItem = (item: QueueItem): boolean =>
  isCancellableQueueItem(item) && !item.backendItemIds?.length;

export const getCancelableCanvasStagingQueueItemId = (slot: CanvasStagingSlot | undefined): string | null =>
  slot?.kind === 'placeholder' ? slot.queueItemId : null;
