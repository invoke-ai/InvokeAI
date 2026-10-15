export interface CanvasContextMenuTarget {
  x: number;
  y: number;
  layerId: string | null;
}

interface ResolveCanvasContextMenuOptions {
  clientX: number;
  clientY: number;
  /** Target the selected layer without hit-testing right-click pixels; the layers panel owns selection. */
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
