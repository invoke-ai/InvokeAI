import type { CanvasStateContractV3, ToolId } from '@workbench/canvas-engine/api';
import type { WorkbenchQueueItem as QueueItem } from '@workbench/queueHistoryContracts';

export const isCanvasInteractionLocked = (canvas: CanvasStateContractV3, queueItems: readonly QueueItem[]): boolean =>
  // Every staged candidate produces a slot; active queue work locks even when its placeholders are exhausted.
  canvas.stagingArea.pendingImages.length > 0 ||
  queueItems.some(
    (item) =>
      item.snapshot.destination === 'canvas' &&
      item.snapshot.canvas.documentRevision === canvas.documentRevision &&
      (item.status === 'pending' || item.status === 'running')
  );

/** Aggregate notifications can change prompts/layout without changing lock inputs. Reuse unchanged reads. */
export const createCanvasInteractionLockReader = (
  getProject: () => { canvas: CanvasStateContractV3; queue: { items: readonly QueueItem[] } } | null | undefined
): (() => boolean) => {
  let revision: number | undefined;
  let candidates: CanvasStateContractV3['stagingArea']['pendingImages'] | undefined;
  let items: readonly QueueItem[] | undefined;
  let locked = true;
  return () => {
    const project = getProject();
    if (!project) {
      revision = undefined;
      candidates = undefined;
      items = undefined;
      return true;
    }
    const { canvas, queue } = project;
    if (
      revision !== canvas.documentRevision ||
      candidates !== canvas.stagingArea.pendingImages ||
      items !== queue.items
    ) {
      revision = canvas.documentRevision;
      candidates = canvas.stagingArea.pendingImages;
      items = queue.items;
      locked = isCanvasInteractionLocked(canvas, items);
    }
    return locked;
  };
};

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
