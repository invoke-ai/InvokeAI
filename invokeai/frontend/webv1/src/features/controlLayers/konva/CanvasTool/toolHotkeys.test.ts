import type { Tool } from 'features/controlLayers/store/types';
import { describe, expect, it } from 'vitest';

import {
  beginBboxToolHotkeyPress,
  clearTemporaryToolHotkeysInState,
  endBboxToolHotkeyPress,
  getActiveToolFromState,
  getToolToCancelOnEscape,
  pressAltInState,
  pressSpaceInState,
  releaseAltInState,
  releaseSpaceInState,
  setBaseToolInState,
  shouldPreserveSuspendableShapesSession,
  shouldQuickSwitchToColorPickerOnAlt,
  shouldTranslateShapeDragOnSpace,
  TEMPORARY_BBOX_TOOL_HOLD_THRESHOLD_MS,
} from './toolHotkeys';

const buildCanvasToolHotkeyState = (baseTool: Tool = 'move') => ({
  baseTool,
  isSpacePressed: false,
  isAltPressed: false,
  bboxToolHotkeyPressedState: null,
});

describe('tool hotkeys', () => {
  it('keeps the color-picker quick-switch available before starting rect and oval drags', () => {
    expect(shouldQuickSwitchToColorPickerOnAlt('rect', 'rect', false)).toBe(true);
    expect(shouldQuickSwitchToColorPickerOnAlt('rect', 'oval', false)).toBe(true);
  });

  it('blocks the color-picker quick-switch while a rect, oval, or freehand drag is active', () => {
    expect(shouldQuickSwitchToColorPickerOnAlt('rect', 'rect', true)).toBe(false);
    expect(shouldQuickSwitchToColorPickerOnAlt('rect', 'oval', true)).toBe(false);
    expect(shouldQuickSwitchToColorPickerOnAlt('rect', 'freehand', true)).toBe(false);
  });

  it('keeps the color-picker quick-switch for polygon mode and non-shape tools', () => {
    expect(shouldQuickSwitchToColorPickerOnAlt('rect', 'polygon', false)).toBe(true);
    expect(shouldQuickSwitchToColorPickerOnAlt('rect', 'polygon', true)).toBe(true);
    expect(shouldQuickSwitchToColorPickerOnAlt('brush', 'rect', true)).toBe(true);
    expect(shouldQuickSwitchToColorPickerOnAlt('lasso', 'polygon', false)).toBe(true);
  });

  it('uses Space to translate active rect and oval drags instead of switching to view', () => {
    expect(shouldTranslateShapeDragOnSpace('rect', 'rect', true, true)).toBe(true);
    expect(shouldTranslateShapeDragOnSpace('rect', 'oval', true, true)).toBe(true);
  });

  it('does not use Space translation outside active rect and oval drags', () => {
    expect(shouldTranslateShapeDragOnSpace('rect', 'rect', false, true)).toBe(false);
    expect(shouldTranslateShapeDragOnSpace('rect', 'rect', true, false)).toBe(false);
    expect(shouldTranslateShapeDragOnSpace('rect', 'polygon', true, true)).toBe(false);
    expect(shouldTranslateShapeDragOnSpace('rect', 'freehand', true, true)).toBe(false);
    expect(shouldTranslateShapeDragOnSpace('brush', 'rect', true, true)).toBe(false);
  });

  it('preserves suspendable shapes sessions across temporary view and color-picker switches', () => {
    expect(shouldPreserveSuspendableShapesSession('view', 'rect', true)).toBe(true);
    expect(shouldPreserveSuspendableShapesSession('colorPicker', 'rect', true)).toBe(true);
    expect(shouldPreserveSuspendableShapesSession('rect', 'rect', true)).toBe(true);
  });

  it('does not preserve suspendable shapes sessions for unrelated tool switches', () => {
    expect(shouldPreserveSuspendableShapesSession('brush', 'rect', true)).toBe(false);
    expect(shouldPreserveSuspendableShapesSession('view', null, true)).toBe(false);
    expect(shouldPreserveSuspendableShapesSession('colorPicker', 'rect', false)).toBe(false);
  });

  it('cancels the active drawing tool directly on escape', () => {
    expect(getToolToCancelOnEscape('rect', null, false, false)).toBe('rect');
    expect(getToolToCancelOnEscape('lasso', null, false, false)).toBe('lasso');
  });

  it('cancels preserved drawing sessions while temporarily switched away', () => {
    expect(getToolToCancelOnEscape('view', 'lasso', true, false)).toBe('lasso');
    expect(getToolToCancelOnEscape('view', 'rect', false, true)).toBe('rect');
    expect(getToolToCancelOnEscape('colorPicker', 'rect', false, true)).toBe('rect');
  });

  it('does not cancel unrelated buffered tools on escape', () => {
    expect(getToolToCancelOnEscape('view', 'lasso', false, false)).toBeNull();
    expect(getToolToCancelOnEscape('colorPicker', 'lasso', true, false)).toBeNull();
    expect(getToolToCancelOnEscape('view', 'brush', false, true)).toBeNull();
  });
});

describe('canvas tool hotkey state', () => {
  it('keeps the bbox tool selected after a short press', () => {
    const started = beginBboxToolHotkeyPress(buildCanvasToolHotkeyState(), { bindingId: 'KeyC', pressedAt: 1_000 });

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
    const started = beginBboxToolHotkeyPress(buildCanvasToolHotkeyState('brush'), {
      bindingId: 'KeyC',
      pressedAt: 1_000,
    });

    const ended = endBboxToolHotkeyPress({
      state: started,
      bindingId: 'KeyC',
      releasedAt: 1_000 + TEMPORARY_BBOX_TOOL_HOLD_THRESHOLD_MS,
    });

    expect(ended.baseTool).toBe('brush');
    expect(getActiveToolFromState(ended)).toBe('brush');
  });

  it('restores the base tool after releasing Space over a long bbox hold', () => {
    const started = beginBboxToolHotkeyPress(buildCanvasToolHotkeyState('brush'), {
      bindingId: 'KeyC',
      pressedAt: 1_000,
    });
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
    const started = beginBboxToolHotkeyPress(buildCanvasToolHotkeyState('brush'), {
      bindingId: 'KeyC',
      pressedAt: 1_000,
    });
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
    const started = beginBboxToolHotkeyPress(buildCanvasToolHotkeyState('move'), {
      bindingId: 'KeyC',
      pressedAt: 1_000,
    });
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

  it('preserves the new base tool while a bbox hold is active', () => {
    const started = beginBboxToolHotkeyPress(buildCanvasToolHotkeyState('move'), {
      bindingId: 'KeyC',
      pressedAt: 1_000,
    });
    const baseToolChanged = setBaseToolInState(started, 'brush');
    const released = endBboxToolHotkeyPress({
      state: baseToolChanged,
      bindingId: 'KeyC',
      releasedAt: 1_000 + TEMPORARY_BBOX_TOOL_HOLD_THRESHOLD_MS,
    });

    expect(getActiveToolFromState(baseToolChanged)).toBe('bbox');
    expect(baseToolChanged.baseTool).toBe('brush');
    expect(getActiveToolFromState(released)).toBe('brush');
  });

  it('clears all temporary hotkeys on blur-like reset', () => {
    const state = clearTemporaryToolHotkeysInState(
      pressAltInState(
        pressSpaceInState(
          beginBboxToolHotkeyPress(buildCanvasToolHotkeyState('lasso'), {
            bindingId: 'KeyC',
            pressedAt: 1_000,
          })
        )
      )
    );

    expect(getActiveToolFromState(state)).toBe('lasso');
    expect(state.isSpacePressed).toBe(false);
    expect(state.isAltPressed).toBe(false);
    expect(state.bboxToolHotkeyPressedState).toBeNull();
  });
});
