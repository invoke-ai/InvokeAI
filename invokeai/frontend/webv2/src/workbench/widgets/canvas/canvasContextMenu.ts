export interface CanvasContextMenuTarget {
  x: number;
  y: number;
  layerId: string | null;
}

interface ResolveCanvasContextMenuOptions {
  clientX: number;
  clientY: number;
  hitTest?: (point: { x: number; y: number }) => string | null;
  isInlineEditor: boolean;
  isInteractionLocked: boolean;
  surfaceLeft: number;
  surfaceTop: number;
}

interface CanvasContextMenuResolution {
  preventDefault: boolean;
  target: CanvasContextMenuTarget | null;
}

export const resolveCanvasContextMenu = ({
  clientX,
  clientY,
  hitTest,
  isInlineEditor,
  isInteractionLocked,
  surfaceLeft,
  surfaceTop,
}: ResolveCanvasContextMenuOptions): CanvasContextMenuResolution => {
  if (isInlineEditor) {
    return { preventDefault: false, target: null };
  }

  const layerId = isInteractionLocked
    ? null
    : (hitTest?.({ x: clientX - surfaceLeft, y: clientY - surfaceTop }) ?? null);

  return {
    preventDefault: true,
    target: { layerId, x: clientX, y: clientY },
  };
};
