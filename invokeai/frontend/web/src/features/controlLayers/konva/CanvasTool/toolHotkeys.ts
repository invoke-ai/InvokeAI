import type { Tool } from 'features/controlLayers/store/types';

type ShapeType = 'rect' | 'oval' | 'polygon' | 'freehand';

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
