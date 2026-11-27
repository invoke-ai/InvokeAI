import type { Tool } from 'features/controlLayers/store/types';

export const TEMPORARY_BBOX_TOOL_HOLD_THRESHOLD_MS = 250;

type ShapeType = 'rect' | 'oval' | 'polygon' | 'freehand';

export type BboxToolHotkeyPressedState = {
  bindingId: string;
  pressedAt: number;
};

export type CanvasToolHotkeyState = {
  baseTool: Tool;
  isSpacePressed: boolean;
  isAltPressed: boolean;
  bboxToolHotkeyPressedState: BboxToolHotkeyPressedState | null;
};

export const getActiveToolFromState = (state: CanvasToolHotkeyState): Tool => {
  if (state.isSpacePressed) {
    return 'view';
  }

  if (state.isAltPressed) {
    return 'colorPicker';
  }

  if (state.bboxToolHotkeyPressedState) {
    return 'bbox';
  }

  return state.baseTool;
};

export const setBaseToolInState = (state: CanvasToolHotkeyState, baseTool: Tool): CanvasToolHotkeyState => {
  if (state.baseTool === baseTool) {
    return state;
  }

  return { ...state, baseTool };
};

export const pressSpaceInState = (state: CanvasToolHotkeyState): CanvasToolHotkeyState => {
  if (state.isSpacePressed) {
    return state;
  }

  return { ...state, isSpacePressed: true };
};

export const releaseSpaceInState = (state: CanvasToolHotkeyState): CanvasToolHotkeyState => {
  if (!state.isSpacePressed) {
    return state;
  }

  return { ...state, isSpacePressed: false };
};

export const pressAltInState = (state: CanvasToolHotkeyState): CanvasToolHotkeyState => {
  if (state.isAltPressed) {
    return state;
  }

  return { ...state, isAltPressed: true };
};

export const releaseAltInState = (state: CanvasToolHotkeyState): CanvasToolHotkeyState => {
  if (!state.isAltPressed) {
    return state;
  }

  return { ...state, isAltPressed: false };
};

export const clearTemporaryToolHotkeysInState = (state: CanvasToolHotkeyState): CanvasToolHotkeyState => {
  if (!state.isSpacePressed && !state.isAltPressed && !state.bboxToolHotkeyPressedState) {
    return state;
  }

  return {
    ...state,
    isSpacePressed: false,
    isAltPressed: false,
    bboxToolHotkeyPressedState: null,
  };
};

export const beginBboxToolHotkeyPress = (
  state: CanvasToolHotkeyState,
  payload: {
    bindingId: string;
    pressedAt: number;
  }
): CanvasToolHotkeyState => {
  if (state.baseTool === 'bbox' || state.bboxToolHotkeyPressedState) {
    return state;
  }

  return {
    ...state,
    bboxToolHotkeyPressedState: {
      bindingId: payload.bindingId,
      pressedAt: payload.pressedAt,
    },
  };
};

export const endBboxToolHotkeyPress = ({
  state,
  bindingId,
  releasedAt,
  holdThresholdMs = TEMPORARY_BBOX_TOOL_HOLD_THRESHOLD_MS,
}: {
  state: CanvasToolHotkeyState;
  bindingId: string;
  releasedAt: number;
  holdThresholdMs?: number;
}): CanvasToolHotkeyState => {
  if (!state.bboxToolHotkeyPressedState || state.bboxToolHotkeyPressedState.bindingId !== bindingId) {
    return state;
  }

  const wasHeldLongEnough = releasedAt - state.bboxToolHotkeyPressedState.pressedAt >= holdThresholdMs;
  const nextState = {
    ...state,
    bboxToolHotkeyPressedState: null,
  };

  if (!wasHeldLongEnough) {
    return {
      ...nextState,
      baseTool: 'bbox',
    };
  }

  return nextState;
};

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
