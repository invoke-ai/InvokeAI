import type { CanvasLayerContract } from '@workbench/canvas-engine/api';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

export interface MultiLayerStructuralActions {
  forward: CanvasProjectMutation;
  inverse: CanvasProjectMutation;
}

/** Builds one reversible edit for a non-contiguous, unlocked multi-layer delete. */
export const deleteLayersActions = (
  layers: readonly CanvasLayerContract[],
  selectedIds: readonly string[],
  selectedLayerId: string | null
): MultiLayerStructuralActions | null => {
  const selected = new Set(selectedIds);
  const removed = layers.filter((layer) => selected.has(layer.id));
  if (removed.length === 0 || removed.some((layer) => layer.isLocked)) {
    return null;
  }
  return {
    forward: { ids: removed.map((layer) => layer.id), type: 'removeCanvasLayers' },
    inverse: {
      add: { index: 0, layers: removed },
      enabledUpdates: [],
      orderedIds: layers.map((layer) => layer.id),
      selectedLayerId,
      type: 'applyCanvasLayerStackMutation',
    },
  };
};
