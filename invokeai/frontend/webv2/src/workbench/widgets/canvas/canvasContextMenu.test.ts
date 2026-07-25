import { describe, expect, it } from 'vitest';

import { resolveCanvasContextMenu, resolveCanvasContextMenuBranch } from './canvasContextMenu';

const baseOptions = {
  clientX: 240,
  clientY: 160,
  isInlineEditor: false,
  isInteractionLocked: false,
};

describe('resolveCanvasContextMenu', () => {
  it('leaves inline editors to the native context menu', () => {
    expect(resolveCanvasContextMenu({ ...baseOptions, isInlineEditor: true, selectedLayerId: 'layer-1' })).toEqual({
      preventDefault: false,
      target: null,
    });
  });

  it('opens the global menu while interaction is locked, ignoring the selected layer', () => {
    expect(resolveCanvasContextMenu({ ...baseOptions, isInteractionLocked: true, selectedLayerId: 'layer-1' })).toEqual(
      {
        preventDefault: true,
        target: { layerId: null, x: 240, y: 160 },
      }
    );
  });

  it('opens the layer menu at the pointer for the SELECTED layer', () => {
    expect(resolveCanvasContextMenu({ ...baseOptions, selectedLayerId: 'layer-1' })).toEqual({
      preventDefault: true,
      target: { layerId: 'layer-1', x: 240, y: 160 },
    });
  });

  it('targets the selected layer regardless of where the pointer is', () => {
    // The canvas never hit-tests for the menu: the layers panel is the sole
    // authority on which layer is active, so the pointer position is irrelevant.
    const overOneCorner = resolveCanvasContextMenu({ ...baseOptions, selectedLayerId: 'layer-7' });
    const overAnother = resolveCanvasContextMenu({
      ...baseOptions,
      clientX: 900,
      clientY: 900,
      selectedLayerId: 'layer-7',
    });

    expect(overOneCorner.target?.layerId).toBe('layer-7');
    expect(overAnother.target?.layerId).toBe('layer-7');
  });

  it('opens the global menu when no layer is selected', () => {
    expect(resolveCanvasContextMenu({ ...baseOptions, selectedLayerId: null })).toEqual({
      preventDefault: true,
      target: { layerId: null, x: 240, y: 160 },
    });
  });

  it('opens the global menu without a target when the engine is unavailable', () => {
    const resolution = resolveCanvasContextMenu(baseOptions);

    expect(resolution).toEqual({
      preventDefault: true,
      target: { layerId: null, x: 240, y: 160 },
    });
    expect(resolveCanvasContextMenuBranch(resolution.target, false)).toBe('global');
  });
});
