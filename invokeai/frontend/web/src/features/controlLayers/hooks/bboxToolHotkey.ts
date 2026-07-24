import type { Tool } from 'features/controlLayers/store/types';

export const TEMPORARY_BBOX_TOOL_HOLD_THRESHOLD_MS = 250;

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
  return { ...state, baseTool };
};

export const pressSpaceInState = (state: CanvasToolHotkeyState): CanvasToolHotkeyState => {
  return { ...state, isSpacePressed: true };
};

export const releaseSpaceInState = (state: CanvasToolHotkeyState): CanvasToolHotkeyState => {
  return { ...state, isSpacePressed: false };
};

export const pressAltInState = (state: CanvasToolHotkeyState): CanvasToolHotkeyState => {
  return { ...state, isAltPressed: true };
};

export const releaseAltInState = (state: CanvasToolHotkeyState): CanvasToolHotkeyState => {
  return { ...state, isAltPressed: false };
};

export const clearTemporaryToolHotkeysInState = (state: CanvasToolHotkeyState): CanvasToolHotkeyState => {
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
