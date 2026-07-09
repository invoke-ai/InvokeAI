import type { CanvasLayerContextMenuTarget } from './LayerContextMenu';

/**
 * Which layer the canvas right-click menu should render for.
 *
 * Choosing "Rename" closes the menu, which nulls the live `target`. But the rename
 * dialog is a sibling of the menu inside the same subtree — if the wrapper stopped
 * rendering the moment `target` went null, the dialog would unmount before it ever
 * painted (the F1 bug). So while a rename is in flight the wrapper keeps a captured
 * `renameTarget` (set when the dialog opens, cleared when it closes) and falls back
 * to it, keeping the subtree — and thus the dialog — mounted until the rename ends.
 */
export const resolveMenuTargetForRender = (
  liveTarget: CanvasLayerContextMenuTarget | null,
  renameTarget: CanvasLayerContextMenuTarget | null
): CanvasLayerContextMenuTarget | null => liveTarget ?? renameTarget;
