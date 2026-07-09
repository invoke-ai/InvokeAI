import { describe, expect, it } from 'vitest';

import { createEmptyPaintLayer } from './layerOps';
import { isSameSelection } from './LayersPanelHeader';

describe('LayersPanelHeader selection equality', () => {
  it('invalidates the selected-layer view when blend mode changes', () => {
    const normal = createEmptyPaintLayer('Layer', 'layer');
    const multiply = { ...normal, blendMode: 'multiply' as const };

    expect(isSameSelection(normal, multiply)).toBe(false);
  });
});
