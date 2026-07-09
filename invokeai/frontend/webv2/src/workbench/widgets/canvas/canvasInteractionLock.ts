import type { ToolId } from '@workbench/canvas-engine/types';

export const isCanvasStagingActive = ({
  hasStagedCandidates,
  isCanvasGenerationInFlight,
}: {
  hasStagedCandidates: boolean;
  isCanvasGenerationInFlight: boolean;
}): boolean => hasStagedCandidates || isCanvasGenerationInFlight;

export const isCanvasToolEnabled = (toolId: ToolId, isInteractionLocked: boolean): boolean =>
  !isInteractionLocked || toolId === 'view';
