import type { WorkbenchQueueItem } from '@workbench/queueHistoryContracts';

import { cloneGenerateWidgetValues, normalizeGenerateWidgetValues } from '@features/generation/settings';
import { MAX_QUEUE_BATCH_ITEMS } from '@features/queue';
import { cloneVideoWidgetValues, normalizeVideoWidgetValues } from '@features/video';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const isBackendItemIdList = (value: unknown): value is number[] =>
  Array.isArray(value) &&
  value.length <= MAX_QUEUE_BATCH_ITEMS &&
  new Set(value).size === value.length &&
  value.every((item) => Number.isSafeInteger(item) && item > 0);

export const normalizeRestoredQueueItem = (value: unknown): WorkbenchQueueItem | null => {
  if (
    !isRecord(value) ||
    (value.status !== 'pending' &&
      value.status !== 'running' &&
      !(value.status === 'cancelled' && value.cancellationPending === true))
  ) {
    return null;
  }
  const snapshot = value.snapshot;
  if (
    typeof value.id !== 'string' ||
    value.id.length === 0 ||
    typeof value.cancellable !== 'boolean' ||
    !isRecord(snapshot) ||
    !isRecord(snapshot.backendSubmission) ||
    (snapshot.backendSubmission.kind !== 'generate' &&
      snapshot.backendSubmission.kind !== 'workflow' &&
      snapshot.backendSubmission.kind !== 'invalid') ||
    (snapshot.sourceId !== 'canvas' &&
      snapshot.sourceId !== 'generate' &&
      snapshot.sourceId !== 'upscale' &&
      snapshot.sourceId !== 'video' &&
      snapshot.sourceId !== 'workflow') ||
    (snapshot.destination !== 'canvas' && snapshot.destination !== 'gallery') ||
    !isRecord(snapshot.graph) ||
    typeof snapshot.graph.id !== 'string' ||
    typeof snapshot.graph.label !== 'string' ||
    (snapshot.galleryBoardId !== null && typeof snapshot.galleryBoardId !== 'string') ||
    typeof snapshot.filterIntermediateResults !== 'boolean' ||
    (snapshot.resultNodeIds !== undefined &&
      (!Array.isArray(snapshot.resultNodeIds) || snapshot.resultNodeIds.some((id) => typeof id !== 'string'))) ||
    !isRecord(snapshot.presentation) ||
    !isFiniteNumber(snapshot.presentation.batchCount) ||
    !Number.isSafeInteger(snapshot.presentation.batchCount) ||
    snapshot.presentation.batchCount < 1 ||
    snapshot.presentation.batchCount > MAX_QUEUE_BATCH_ITEMS ||
    !isFiniteNumber(snapshot.presentation.height) ||
    snapshot.presentation.height <= 0 ||
    !isFiniteNumber(snapshot.presentation.width) ||
    snapshot.presentation.width <= 0 ||
    (snapshot.presentation.positivePrompt !== undefined && typeof snapshot.presentation.positivePrompt !== 'string') ||
    typeof snapshot.submittedAt !== 'string' ||
    !isRecord(snapshot.canvas) ||
    !isFiniteNumber(snapshot.canvas.documentRevision) ||
    !Number.isSafeInteger(snapshot.canvas.documentRevision) ||
    snapshot.canvas.documentRevision < 0 ||
    !isRecord(snapshot.canvas.document) ||
    !isRecord(snapshot.canvas.document.bbox)
  ) {
    return null;
  }
  if (
    (value.backendItemIds !== undefined && !isBackendItemIdList(value.backendItemIds)) ||
    (value.completedBackendItemIds !== undefined && !isBackendItemIdList(value.completedBackendItemIds)) ||
    (value.cancelledBackendItemIds !== undefined && !isBackendItemIdList(value.cancelledBackendItemIds))
  ) {
    return null;
  }
  const backendItemIds = value.backendItemIds;
  const completedBackendItemIds = value.completedBackendItemIds;
  const cancelledBackendItemIds = value.cancelledBackendItemIds;
  const backendIds = new Set(backendItemIds ?? []);
  const cancelledIds = new Set(cancelledBackendItemIds ?? []);
  if (
    [...(completedBackendItemIds ?? []), ...(cancelledBackendItemIds ?? [])].some((id) => !backendIds.has(id)) ||
    completedBackendItemIds?.some((id) => cancelledIds.has(id))
  ) {
    return null;
  }
  const { bbox } = snapshot.canvas.document;
  if (
    !isFiniteNumber(bbox.x) ||
    !isFiniteNumber(bbox.y) ||
    !isFiniteNumber(bbox.width) ||
    bbox.width <= 0 ||
    !isFiniteNumber(bbox.height) ||
    bbox.height <= 0 ||
    !isFiniteNumber(snapshot.canvas.document.width) ||
    snapshot.canvas.document.width <= 0 ||
    !isFiniteNumber(snapshot.canvas.document.height) ||
    snapshot.canvas.document.height <= 0 ||
    (value.backendBatchId !== undefined &&
      (typeof value.backendBatchId !== 'string' || value.backendBatchId.length === 0))
  ) {
    return null;
  }

  let recall: WorkbenchQueueItem['snapshot']['recall'];
  try {
    const rawRecall = isRecord(snapshot.recall) ? snapshot.recall : null;
    if ((snapshot.sourceId === 'canvas' || snapshot.sourceId === 'generate') && isRecord(rawRecall?.generateValues)) {
      const values = normalizeGenerateWidgetValues(rawRecall.generateValues);
      if (values) {
        recall = { generateValues: cloneGenerateWidgetValues(values) };
      }
    } else if (snapshot.sourceId === 'video' && isRecord(rawRecall?.videoValues)) {
      const values = normalizeVideoWidgetValues(rawRecall.videoValues);
      if (values) {
        recall = { videoValues: cloneVideoWidgetValues(values) };
      }
    }
  } catch {
    recall = undefined;
  }

  return {
    ...(value.backendBatchId === undefined ? {} : { backendBatchId: value.backendBatchId }),
    ...(backendItemIds === undefined ? {} : { backendItemIds: [...backendItemIds] }),
    ...(cancelledBackendItemIds === undefined ? {} : { cancelledBackendItemIds: [...cancelledBackendItemIds] }),
    ...(value.cancellationPending === true ? { cancellationPending: true } : {}),
    cancellable: value.cancellable,
    ...(completedBackendItemIds === undefined ? {} : { completedBackendItemIds: [...completedBackendItemIds] }),
    id: value.id,
    // Keeps a restored run from reselecting videos it already showed before the reload.
    ...(Array.isArray(value.resultVideoNames) && value.resultVideoNames.every((name) => typeof name === 'string')
      ? { resultVideoNames: [...value.resultVideoNames] }
      : {}),
    snapshot: {
      backendSubmission: snapshot.backendSubmission as WorkbenchQueueItem['snapshot']['backendSubmission'],
      canvas: {
        document: {
          bbox: { height: bbox.height, width: bbox.width, x: bbox.x, y: bbox.y },
          height: snapshot.canvas.document.height,
          width: snapshot.canvas.document.width,
        },
        documentRevision: snapshot.canvas.documentRevision,
      },
      destination: snapshot.destination,
      filterIntermediateResults: snapshot.filterIntermediateResults,
      galleryBoardId: snapshot.galleryBoardId,
      graph: { id: snapshot.graph.id, label: snapshot.graph.label },
      presentation: {
        batchCount: snapshot.presentation.batchCount,
        height: snapshot.presentation.height,
        ...(snapshot.presentation.positivePrompt === undefined
          ? {}
          : { positivePrompt: snapshot.presentation.positivePrompt }),
        width: snapshot.presentation.width,
      },
      ...(recall ? { recall } : {}),
      ...(snapshot.resultNodeIds === undefined ? {} : { resultNodeIds: [...snapshot.resultNodeIds] }),
      sourceId: snapshot.sourceId,
      submittedAt: snapshot.submittedAt,
    },
    status: value.status,
  };
};
