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

export const isCanvasToolEnabled = (toolId: ToolId, isInteractionLocked: boolean): boolean =>
  !isInteractionLocked || toolId === 'view';
