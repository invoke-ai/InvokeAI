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

export const getToolToCancelOnEscape = (
  tool: Tool,
  toolBuffer: Tool | null,
  hasActiveLassoSession: boolean,
  hasSuspendableShapeSession: boolean
): Tool | null => {
  if (tool === 'rect' || tool === 'lasso') {
    return tool;
  }

  if (tool === 'view' && toolBuffer === 'lasso' && hasActiveLassoSession) {
    return 'lasso';
  }

  if ((tool === 'view' || tool === 'colorPicker') && toolBuffer === 'rect' && hasSuspendableShapeSession) {
    return 'rect';
  }

  return null;
};
