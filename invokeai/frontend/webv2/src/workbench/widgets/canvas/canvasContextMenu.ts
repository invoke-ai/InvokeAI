export interface CanvasContextMenuTarget {
  x: number;
  y: number;
  layerId: string | null;
}

interface ResolveCanvasContextMenuOptions {
  clientX: number;
  clientY: number;
  /**
   * The document's selected layer — the menu's target. The canvas deliberately
   * does NOT hit-test the pixel under the pointer: the layers panel is the sole
   * authority on which layer is active, so right-clicking a layer's pixels must
   * not silently re-target the menu (or the selection) to it.
   */
  selectedLayerId?: string | null;
  isInlineEditor: boolean;
  isInteractionLocked: boolean;
}

interface CanvasContextMenuResolution {
  preventDefault: boolean;
  target: CanvasContextMenuTarget | null;
}

export type CanvasContextMenuBranch = 'global' | 'layer' | null;

export const resolveCanvasContextMenuBranch = (
  target: CanvasContextMenuTarget | null,
  hasEngine: boolean
): CanvasContextMenuBranch => {
  if (!target) {
    return null;
  }

  return target.layerId !== null && hasEngine ? 'layer' : 'global';
};

export const resolveCanvasContextMenu = ({
  clientX,
  clientY,
  isInlineEditor,
  isInteractionLocked,
  selectedLayerId,
}: ResolveCanvasContextMenuOptions): CanvasContextMenuResolution => {
  if (isInlineEditor) {
    return { preventDefault: false, target: null };
  }

  // A locked surface offers only the global menu — there is nothing to act on.
  const layerId = isInteractionLocked ? null : (selectedLayerId ?? null);

  return {
    preventDefault: true,
    target: { layerId, x: clientX, y: clientY },
  };
};
