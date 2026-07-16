import { describe, expect, it, vi } from 'vitest';

import { resolveCanvasContextMenu, resolveCanvasContextMenuBranch } from './canvasContextMenu';

const baseOptions = {
  clientX: 240,
  clientY: 160,
  isInlineEditor: false,
  isInteractionLocked: false,
  surfaceLeft: 40,
  surfaceTop: 20,
};

describe('resolveCanvasContextMenu', () => {
  it('leaves inline editors to the native context menu', () => {
    const hitTest = vi.fn(() => 'layer-1');

    expect(resolveCanvasContextMenu({ ...baseOptions, hitTest, isInlineEditor: true })).toEqual({
      preventDefault: false,
      target: null,
    });
    expect(hitTest).not.toHaveBeenCalled();
  });

  it('opens the global menu without hit-testing while interaction is locked', () => {
    const hitTest = vi.fn(() => 'layer-1');

    expect(resolveCanvasContextMenu({ ...baseOptions, hitTest, isInteractionLocked: true })).toEqual({
      preventDefault: true,
      target: { layerId: null, x: 240, y: 160 },
    });
    expect(hitTest).not.toHaveBeenCalled();
  });

  it('opens the layer menu at the pointer for a hit layer', () => {
    const hitTest = vi.fn(() => 'layer-1');

    expect(resolveCanvasContextMenu({ ...baseOptions, hitTest })).toEqual({
      preventDefault: true,
      target: { layerId: 'layer-1', x: 240, y: 160 },
    });
    expect(hitTest).toHaveBeenCalledWith({ x: 200, y: 140 });
  });

  it('opens the global menu when the pointer is over empty canvas', () => {
    expect(resolveCanvasContextMenu({ ...baseOptions, hitTest: () => null })).toEqual({
      preventDefault: true,
      target: { layerId: null, x: 240, y: 160 },
    });
  });

  it('opens the global menu without a hit-test when the engine is unavailable', () => {
    const resolution = resolveCanvasContextMenu(baseOptions);

    expect(resolution).toEqual({
      preventDefault: true,
      target: { layerId: null, x: 240, y: 160 },
    });
    expect(resolveCanvasContextMenuBranch(resolution.target, false)).toBe('global');
  });
});
