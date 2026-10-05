import type { CanvasStateContractV3, ToolId } from '@workbench/canvas-engine/api';
import type { WorkbenchQueueItem as QueueItem } from '@workbench/queueHistoryContracts';

import { getCanvasStagingSlots } from '@workbench/canvasStagingView';

export const isCanvasInteractionLocked = (canvas: CanvasStateContractV3, queueItems: readonly QueueItem[]): boolean =>
  getCanvasStagingSlots(canvas, queueItems).length > 0 ||
  queueItems.some(
    (item) =>
      item.snapshot.destination === 'canvas' &&
      item.snapshot.canvas.documentRevision === canvas.documentRevision &&
      (item.status === 'pending' || item.status === 'running')
  );

export const isCanvasStagingActive = ({
  hasStagedCandidates,
  isCanvasGenerationInFlight,
}: {
  hasStagedCandidates: boolean;
  isCanvasGenerationInFlight: boolean;
}): boolean => hasStagedCandidates || isCanvasGenerationInFlight;

export interface CanvasInteractionCapabilities {
  areOperationActionsEnabled: boolean;
  canAcceptStagedImage: boolean;
  isDocumentEditingLocked: boolean;
  isSurfaceInteractionLocked: boolean;
}

export const getCanvasInteractionCapabilities = ({
  hasCanvasEngine,
  hasSelectedCandidate,
  hasStagingSlots,
  isCanvasGenerationInFlight,
  operationKind,
}: {
  hasCanvasEngine: boolean;
  hasSelectedCandidate: boolean;
  hasStagingSlots: boolean;
  isCanvasGenerationInFlight: boolean;
  operationKind: 'filter' | 'select-object' | null;
}): CanvasInteractionCapabilities => {
  const isSurfaceInteractionLocked = isCanvasStagingActive({
    hasStagedCandidates: hasStagingSlots,
    isCanvasGenerationInFlight,
  });
  const isDocumentEditingLocked = operationKind !== null;
  return {
    areOperationActionsEnabled: isDocumentEditingLocked && !isSurfaceInteractionLocked,
    canAcceptStagedImage: hasCanvasEngine && hasSelectedCandidate && !isDocumentEditingLocked,
    isDocumentEditingLocked,
    isSurfaceInteractionLocked,
  };
};

export const isCanvasToolEnabled = (toolId: ToolId, isInteractionLocked: boolean): boolean =>
  !isInteractionLocked || toolId === 'view';
