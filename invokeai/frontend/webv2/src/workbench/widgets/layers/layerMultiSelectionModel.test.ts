import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';

import { describe, expect, it } from 'vitest';

import { deleteLayersActions } from './layerMultiSelectionModel';

const layer = (id: string): CanvasLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name: id,
  opacity: 1,
  source: { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

describe('deleteLayersActions', () => {
  it('restores non-contiguous layers, order, and primary selection on undo', () => {
    const layers = [layer('a'), layer('b'), layer('c'), layer('d')];
    expect(deleteLayersActions(layers, ['a', 'c'], 'c')).toEqual({
      forward: { ids: ['a', 'c'], type: 'removeCanvasLayers' },
      inverse: {
        add: { index: 0, layers: [layers[0], layers[2]] },
        enabledUpdates: [],
        orderedIds: ['a', 'b', 'c', 'd'],
        selectedLayerId: 'c',
        type: 'applyCanvasLayerStackMutation',
      },
    });
  });

  it('ignores absent ids and returns null when none exist', () => {
    expect(deleteLayersActions([layer('a')], ['missing'], 'a')).toBeNull();
  });

  it('refuses to delete a selection containing a locked layer', () => {
    expect(deleteLayersActions([layer('a'), { ...layer('b'), isLocked: true }], ['a', 'b'], 'a')).toBeNull();
  });
});
