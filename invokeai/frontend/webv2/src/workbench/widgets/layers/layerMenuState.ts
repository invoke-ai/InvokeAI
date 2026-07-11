import type { CanvasLayerContextMenuTarget } from './LayerContextMenu';

/**
 * Which layer the canvas right-click menu should render for.
 *
 * Opening a sibling dialog closes the menu, which nulls the live `target`. The
 * wrapper keeps a captured dialog target until that dialog closes so the shared
 * subtree remains mounted for rename and workflow actions.
 */
export type LayerMenuDialogKind = 'rename';

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
