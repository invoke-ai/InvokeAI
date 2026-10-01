import { describe, expect, it } from 'vitest';

import { getCanvasDeleteTarget } from './useCanvasDeleteLayerHotkey';

describe('canvas Delete hotkey routing', () => {
  it('never targets a path with gallery focus', () => {
    expect(getCanvasDeleteTarget('gallery', false, true)).toBeNull();
    expect(getCanvasDeleteTarget(null, false, true)).toBeNull();
  });

  it.each(['canvas', 'layers'] as const)('targets selected geometry with %s focus', (region) => {
    expect(getCanvasDeleteTarget(region, false, true)).toBe('path');
  });

  it.each(['canvas', 'layers'] as const)('never deletes with %s focus while busy', (region) => {
    expect(getCanvasDeleteTarget(region, true, true)).toBeNull();
    expect(getCanvasDeleteTarget(region, true, false)).toBeNull();
  });

  it('targets the layer only with layers focus outside Edit mode', () => {
    expect(getCanvasDeleteTarget('canvas', false, false)).toBeNull();
    expect(getCanvasDeleteTarget('gallery', false, false)).toBeNull();
    expect(getCanvasDeleteTarget('layers', false, false)).toBe('layer');
  });
});
