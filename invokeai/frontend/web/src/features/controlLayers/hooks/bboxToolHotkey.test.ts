import { describe, expect, it } from 'vitest';

import {
  beginBboxToolHotkeyPress,
  endBboxToolHotkeyPress,
  TEMPORARY_BBOX_TOOL_HOLD_THRESHOLD_MS,
} from './bboxToolHotkey';

describe('bboxToolHotkey', () => {
  it('keeps the bbox tool selected after a short press', () => {
    const started = beginBboxToolHotkeyPress('move', 1_000);

    expect(started.nextTool).toBe('bbox');
    expect(started.pressedState).toEqual({ previousTool: 'move', pressedAt: 1_000 });

    const ended = endBboxToolHotkeyPress({
      currentTool: 'bbox',
      pressedState: started.pressedState,
      releasedAt: 1_000 + TEMPORARY_BBOX_TOOL_HOLD_THRESHOLD_MS - 1,
    });

    expect(ended.revertToTool).toBeNull();
  });

  it('reverts to the previous tool after a long hold', () => {
    const started = beginBboxToolHotkeyPress('brush', 1_000);

    const ended = endBboxToolHotkeyPress({
      currentTool: 'bbox',
      pressedState: started.pressedState,
      releasedAt: 1_000 + TEMPORARY_BBOX_TOOL_HOLD_THRESHOLD_MS,
    });

    expect(ended.revertToTool).toBe('brush');
  });

  it('does nothing when the bbox tool is already selected', () => {
    const started = beginBboxToolHotkeyPress('bbox', 1_000);

    expect(started.nextTool).toBeNull();
    expect(started.pressedState).toBeNull();
  });

  it('does not override a tool change that happened while the key was held', () => {
    const started = beginBboxToolHotkeyPress('move', 1_000);

    const ended = endBboxToolHotkeyPress({
      currentTool: 'brush',
      pressedState: started.pressedState,
      releasedAt: 1_000 + TEMPORARY_BBOX_TOOL_HOLD_THRESHOLD_MS + 10,
    });

    expect(ended.revertToTool).toBeNull();
  });
});
