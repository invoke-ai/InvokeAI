import type { Tool } from 'features/controlLayers/store/types';

type ShapeType = 'rect' | 'oval' | 'polygon' | 'freehand';

export const shouldPreserveSuspendableShapesSession = (
  tool: Tool,
  toolBuffer: Tool | null,
  hasSuspendableShapeSession: boolean
): boolean => {
  if (!hasSuspendableShapeSession || toolBuffer !== 'rect') {
    return false;
  }

  return tool === 'view' || tool === 'colorPicker' || tool === 'rect';
};

export const shouldQuickSwitchToColorPickerOnAlt = (
  tool: Tool,
  shapeType: ShapeType,
  hasActiveShapeDragSession: boolean
): boolean => {
  if (tool !== 'rect') {
    return true;
  }

  if (shapeType === 'polygon') {
    return true;
  }

  return !hasActiveShapeDragSession;
};

export const shouldTranslateShapeDragOnSpace = (
  tool: Tool,
  shapeType: ShapeType,
  hasActiveShapeDragSession: boolean,
  isPrimaryPointerDown: boolean
): boolean => {
  if (tool !== 'rect' || !hasActiveShapeDragSession || !isPrimaryPointerDown) {
    return false;
  }

  return shapeType === 'rect' || shapeType === 'oval';
};
