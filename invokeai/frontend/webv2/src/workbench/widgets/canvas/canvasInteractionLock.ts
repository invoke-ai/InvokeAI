import type { ToolId } from '@workbench/canvas-engine/types';
import type { CanvasStateContractV2, QueueItem } from '@workbench/types';

import { getCanvasStagingSlots } from '@workbench/canvasStagingView';

export const isCanvasInteractionLocked = (canvas: CanvasStateContractV2, queueItems: readonly QueueItem[]): boolean =>
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
  isOperationChromeVisible: boolean;
  isRegularToolOptionsVisible: boolean;
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
    isOperationChromeVisible: isDocumentEditingLocked,
    isRegularToolOptionsVisible: !isDocumentEditingLocked && !isSurfaceInteractionLocked,
    isSurfaceInteractionLocked,
  };
};

export const isCanvasToolEnabled = (toolId: ToolId, isInteractionLocked: boolean): boolean =>
  !isInteractionLocked || toolId === 'view';
