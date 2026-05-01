import type { Tool } from 'features/controlLayers/store/types';

export const TEMPORARY_BBOX_TOOL_HOLD_THRESHOLD_MS = 250;

export type BboxToolHotkeyPressedState = {
  previousTool: Tool;
  pressedAt: number;
};

export const beginBboxToolHotkeyPress = (
  currentTool: Tool,
  pressedAt: number
): { nextTool: Tool | null; pressedState: BboxToolHotkeyPressedState | null } => {
  if (currentTool === 'bbox') {
    return { nextTool: null, pressedState: null };
  }

  return {
    nextTool: 'bbox',
    pressedState: {
      previousTool: currentTool,
      pressedAt,
    },
  };
};

export const endBboxToolHotkeyPress = ({
  currentTool,
  pressedState,
  releasedAt,
  holdThresholdMs = TEMPORARY_BBOX_TOOL_HOLD_THRESHOLD_MS,
}: {
  currentTool: Tool;
  pressedState: BboxToolHotkeyPressedState | null;
  releasedAt: number;
  holdThresholdMs?: number;
}): { revertToTool: Tool | null } => {
  if (!pressedState || currentTool !== 'bbox') {
    return { revertToTool: null };
  }

  const wasHeldLongEnough = releasedAt - pressedState.pressedAt >= holdThresholdMs;

  if (!wasHeldLongEnough) {
    return { revertToTool: null };
  }

  return { revertToTool: pressedState.previousTool };
};
