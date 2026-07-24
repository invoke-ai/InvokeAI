import type { Tool } from 'features/controlLayers/store/types';
import { describe, expect, it } from 'vitest';

import {
  beginBboxToolHotkeyPress,
  clearTemporaryToolHotkeysInState,
  endBboxToolHotkeyPress,
  getActiveToolFromState,
  pressAltInState,
  pressSpaceInState,
  releaseAltInState,
  releaseSpaceInState,
  TEMPORARY_BBOX_TOOL_HOLD_THRESHOLD_MS,
} from './bboxToolHotkey';

const buildState = (baseTool: Tool = 'move') => ({
  baseTool,
  isSpacePressed: false,
  isAltPressed: false,
  bboxToolHotkeyPressedState: null,
});

describe('bboxToolHotkey', () => {
  it('keeps the bbox tool selected after a short press', () => {
    const started = beginBboxToolHotkeyPress(buildState(), { bindingId: 'KeyC', pressedAt: 1_000 });

    expect(getActiveToolFromState(started)).toBe('bbox');
    expect(started.bboxToolHotkeyPressedState).toEqual({ bindingId: 'KeyC', pressedAt: 1_000 });

    const ended = endBboxToolHotkeyPress({
      state: started,
      bindingId: 'KeyC',
      releasedAt: 1_000 + TEMPORARY_BBOX_TOOL_HOLD_THRESHOLD_MS - 1,
    });

    expect(ended.baseTool).toBe('bbox');
    expect(getActiveToolFromState(ended)).toBe('bbox');
  });

  it('reverts to the base tool after a long hold', () => {
    const started = beginBboxToolHotkeyPress(buildState('brush'), { bindingId: 'KeyC', pressedAt: 1_000 });

    const ended = endBboxToolHotkeyPress({
      state: started,
      bindingId: 'KeyC',
      releasedAt: 1_000 + TEMPORARY_BBOX_TOOL_HOLD_THRESHOLD_MS,
    });

    expect(ended.baseTool).toBe('brush');
    expect(getActiveToolFromState(ended)).toBe('brush');
  });

  it('restores the base tool after releasing Space over a long bbox hold', () => {
    const started = beginBboxToolHotkeyPress(buildState('brush'), { bindingId: 'KeyC', pressedAt: 1_000 });
    const spacePressed = pressSpaceInState(started);
    const bboxReleased = endBboxToolHotkeyPress({
      state: spacePressed,
      bindingId: 'KeyC',
      releasedAt: 1_000 + TEMPORARY_BBOX_TOOL_HOLD_THRESHOLD_MS,
    });
    const spaceReleased = releaseSpaceInState(bboxReleased);

    expect(getActiveToolFromState(spacePressed)).toBe('view');
    expect(getActiveToolFromState(bboxReleased)).toBe('view');
    expect(getActiveToolFromState(spaceReleased)).toBe('brush');
  });

  it('restores the base tool after releasing Alt over a long bbox hold', () => {
    const started = beginBboxToolHotkeyPress(buildState('brush'), { bindingId: 'KeyC', pressedAt: 1_000 });
    const altPressed = pressAltInState(started);
    const bboxReleased = endBboxToolHotkeyPress({
      state: altPressed,
      bindingId: 'KeyC',
      releasedAt: 1_000 + TEMPORARY_BBOX_TOOL_HOLD_THRESHOLD_MS,
    });
    const altReleased = releaseAltInState(bboxReleased);

    expect(getActiveToolFromState(altPressed)).toBe('colorPicker');
    expect(getActiveToolFromState(bboxReleased)).toBe('colorPicker');
    expect(getActiveToolFromState(altReleased)).toBe('brush');
  });

  it('ignores a second binding while the first bbox hold is active', () => {
    const started = beginBboxToolHotkeyPress(buildState('move'), { bindingId: 'KeyC', pressedAt: 1_000 });
    const secondBindingPressed = beginBboxToolHotkeyPress(started, { bindingId: 'KeyB', pressedAt: 1_050 });
    const wrongBindingReleased = endBboxToolHotkeyPress({
      state: secondBindingPressed,
      bindingId: 'KeyB',
      releasedAt: 1_400,
    });
    const ownerReleased = endBboxToolHotkeyPress({
      state: wrongBindingReleased,
      bindingId: 'KeyC',
      releasedAt: 1_400,
    });

    expect(secondBindingPressed.bboxToolHotkeyPressedState).toEqual({ bindingId: 'KeyC', pressedAt: 1_000 });
    expect(getActiveToolFromState(wrongBindingReleased)).toBe('bbox');
    expect(getActiveToolFromState(ownerReleased)).toBe('move');
  });

  it('clears all temporary hotkeys on blur-like reset', () => {
    const state = clearTemporaryToolHotkeysInState(
      pressAltInState(
        pressSpaceInState(beginBboxToolHotkeyPress(buildState('lasso'), { bindingId: 'KeyC', pressedAt: 1_000 }))
      )
    );

    expect(getActiveToolFromState(state)).toBe('lasso');
    expect(state.isSpacePressed).toBe(false);
    expect(state.isAltPressed).toBe(false);
    expect(state.bboxToolHotkeyPressedState).toBeNull();
  });
});
