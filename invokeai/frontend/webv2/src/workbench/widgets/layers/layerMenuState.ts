import type { CanvasLayerContextMenuTarget } from './LayerContextMenu';

/** Retain a captured layer target while sibling dialogs outlive the menu's cleared live target. */
export type LayerMenuDialogKind = 'rename' | 'run-workflow';

export interface LayerMenuDialogState {
  kind: LayerMenuDialogKind;
  target: CanvasLayerContextMenuTarget;
}

export const resolveMenuTargetForRender = (
  liveTarget: CanvasLayerContextMenuTarget | null,
  dialogState: LayerMenuDialogState | null
): CanvasLayerContextMenuTarget | null => liveTarget ?? dialogState?.target ?? null;

export interface LayerContextMenuEvent {
  clientX: number;
  clientY: number;
  preventDefault: () => void;
  stopPropagation: () => void;
}

export const createLayerMenuTargetFromContextEvent = (
  layerId: string,
  event: LayerContextMenuEvent
): CanvasLayerContextMenuTarget => {
  event.preventDefault();
  event.stopPropagation();

  return { layerId, x: event.clientX, y: event.clientY };
};
