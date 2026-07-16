import { describe, expect, it } from 'vitest';

import { createEmptyPaintLayer } from './layerOps';
import { isLayerEditingDisabled, isSameSelection } from './LayersPanelHeader';

describe('LayersPanelHeader selection equality', () => {
  it('invalidates the selected-layer view when blend mode changes', () => {
    const normal = createEmptyPaintLayer('Layer', 'layer');
    const multiply = { ...normal, blendMode: 'multiply' as const };

    expect(isSameSelection(normal, multiply)).toBe(false);
  });
});

describe('LayersPanelHeader editing state', () => {
  it('disables layer controls without a selection or while engine editing is locked', () => {
    const layer = createEmptyPaintLayer('Layer', 'layer');
    expect(isLayerEditingDisabled(null, false)).toBe(true);
    expect(isLayerEditingDisabled(layer, true)).toBe(true);
    expect(isLayerEditingDisabled(layer, false)).toBe(false);
  });
});
