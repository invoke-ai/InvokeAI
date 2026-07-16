import { describe, expect, it } from 'vitest';

import type { CanvasLayerContract } from './types';

import {
  deleteLayerActions,
  duplicateLayerActions,
  reorderIdsForHotkey,
  reorderLayerActions,
  reorderTargetIndex,
} from './canvasLayerOps';

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

describe('duplicateLayerActions', () => {
  it('duplicates forward and removes the duplicate on undo', () => {
    expect(duplicateLayerActions('a', 'a-copy')).toEqual({
      forward: { newId: 'a-copy', sourceId: 'a', type: 'duplicateCanvasLayer' },
      inverse: { ids: ['a-copy'], type: 'removeCanvasLayers' },
    });
  });
});

describe('deleteLayerActions', () => {
  it('removes forward and re-adds at the original index on undo', () => {
    const l = layer('a');
    expect(deleteLayerActions(l, 2)).toEqual({
      forward: { ids: ['a'], type: 'removeCanvasLayers' },
      inverse: { index: 2, layer: l, type: 'addCanvasLayer' },
    });
  });
});

describe('reorderLayerActions', () => {
  it('reorders forward and restores the prior order on undo (copies arrays)', () => {
    const current = ['a', 'b', 'c'];
    const next = ['b', 'a', 'c'];
    const actions = reorderLayerActions(current, next);
    expect(actions).toEqual({
      forward: { orderedIds: ['b', 'a', 'c'], type: 'reorderCanvasLayers' },
      inverse: { orderedIds: ['a', 'b', 'c'], type: 'reorderCanvasLayers' },
    });
    // Snapshotted, not aliased.
    if (actions.forward.type === 'reorderCanvasLayers') {
      expect(actions.forward.orderedIds).not.toBe(next);
    }
  });
});

describe('reorderTargetIndex (index 0 = top-most)', () => {
  it('moves toward/away and clamps at boundaries', () => {
    expect(reorderTargetIndex(2, 5, 'forward')).toBe(1);
    expect(reorderTargetIndex(0, 5, 'forward')).toBe(0);
    expect(reorderTargetIndex(2, 5, 'backward')).toBe(3);
    expect(reorderTargetIndex(4, 5, 'backward')).toBe(4);
    expect(reorderTargetIndex(3, 5, 'front')).toBe(0);
    expect(reorderTargetIndex(1, 5, 'back')).toBe(4);
  });
});

describe('reorderIdsForHotkey', () => {
  it('moves one step toward the front', () => {
    expect(reorderIdsForHotkey(['a', 'b', 'c'], 2, 'forward')).toEqual(['a', 'c', 'b']);
  });

  it('moves one step toward the back', () => {
    expect(reorderIdsForHotkey(['a', 'b', 'c'], 0, 'backward')).toEqual(['b', 'a', 'c']);
  });

  it('moves to front and to back', () => {
    expect(reorderIdsForHotkey(['a', 'b', 'c'], 2, 'front')).toEqual(['c', 'a', 'b']);
    expect(reorderIdsForHotkey(['a', 'b', 'c'], 0, 'back')).toEqual(['b', 'c', 'a']);
  });

  it('returns null when nothing would move (already at the boundary)', () => {
    expect(reorderIdsForHotkey(['a', 'b', 'c'], 0, 'forward')).toBeNull();
    expect(reorderIdsForHotkey(['a', 'b', 'c'], 2, 'backward')).toBeNull();
    expect(reorderIdsForHotkey(['a', 'b', 'c'], 0, 'front')).toBeNull();
    expect(reorderIdsForHotkey(['a', 'b', 'c'], 2, 'back')).toBeNull();
  });

  it('returns null for an out-of-range index', () => {
    expect(reorderIdsForHotkey(['a', 'b'], 5, 'forward')).toBeNull();
    expect(reorderIdsForHotkey(['a', 'b'], -1, 'forward')).toBeNull();
  });
});
