import { describe, expect, it } from 'vitest';

import {
  getToolToCancelOnEscape,
  shouldPreserveSuspendableShapesSession,
  shouldQuickSwitchToColorPickerOnAlt,
  shouldTranslateShapeDragOnSpace,
} from './toolHotkeys';

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
