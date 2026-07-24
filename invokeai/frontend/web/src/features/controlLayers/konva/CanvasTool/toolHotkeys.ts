import type { Tool } from 'features/controlLayers/store/types';

type ShapeType = 'rect' | 'oval' | 'polygon' | 'freehand';

export const shouldPreserveSuspendableShapesSession = (
  tool: Tool,
  baseTool: Tool | null,
  hasSuspendableShapeSession: boolean
): boolean => {
  if (!hasSuspendableShapeSession || baseTool !== 'rect') {
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

export const getToolToCancelOnEscape = (
  tool: Tool,
  baseTool: Tool | null,
  hasActiveLassoSession: boolean,
  hasSuspendableShapeSession: boolean
): Tool | null => {
  if (tool === 'rect' || tool === 'lasso') {
    return tool;
  }

  if (tool === 'view' && baseTool === 'lasso' && hasActiveLassoSession) {
    return 'lasso';
  }

  if ((tool === 'view' || tool === 'colorPicker') && baseTool === 'rect' && hasSuspendableShapeSession) {
    return 'rect';
  }

  return null;
};
