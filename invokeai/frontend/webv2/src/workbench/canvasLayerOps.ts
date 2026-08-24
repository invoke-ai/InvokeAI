/**
 * Pure builders for the structural document edits shared by the layers panel and
 * the canvas widget's layer hotkeys: each returns the forward + inverse reducer
 * action pair for `engine.layers.commitStructural` / `applyStructural`, so the
 * inverse-construction logic lives in exactly one place.
 *
 * Index convention matches the contract and the layers panel: index 0 is the
 * top-most layer, so "up"/"forward" moves toward index 0.
 *
 * Zero React, zero import-time side effects.
 */

import type { CanvasLayerContract } from '@workbench/canvas-engine/api';

import type { CanvasProjectMutation } from './canvasProjectMutations';

/** A forward/inverse reducer-action pair for one reversible structural edit. */
export interface StructuralActions {
  forward: CanvasProjectMutation;
  inverse: CanvasProjectMutation;
}

/** Mints a fresh layer id (matches the engine's / layers panel's id shape). */
export const createLayerId = (): string => `layer-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

/** Duplicate a layer (forward), removing the duplicate on undo (inverse). */
export const duplicateLayerActions = (sourceId: string, newId: string): StructuralActions => ({
  forward: { newId, sourceId, type: 'duplicateCanvasLayer' },
  inverse: { ids: [newId], type: 'removeCanvasLayers' },
});

/** Delete a layer (forward), re-adding it at its original index on undo (inverse). */
export const deleteLayerActions = (layer: CanvasLayerContract, index: number): StructuralActions => ({
  forward: { ids: [layer.id], type: 'removeCanvasLayers' },
  inverse: { index, layer, type: 'addCanvasLayer' },
});

/** Reorder to `nextIds` (forward), restoring `currentIds` on undo (inverse). */
export const reorderLayerActions = (currentIds: readonly string[], nextIds: readonly string[]): StructuralActions => ({
  forward: { orderedIds: [...nextIds], type: 'reorderCanvasLayers' },
  inverse: { orderedIds: [...currentIds], type: 'reorderCanvasLayers' },
});

/** A z-reorder direction for the layer hotkeys. */
export type LayerReorderKind = 'forward' | 'backward' | 'front' | 'back';

const LAYER_TYPE_ORDER: readonly CanvasLayerContract['type'][] = [
  'inpaint_mask',
  'regional_guidance',
  'control',
  'raster',
];

const reorderSelectedIds = (
  groupIds: readonly string[],
  selected: ReadonlySet<string>,
  kind: LayerReorderKind
): string[] => {
  if (kind === 'front' || kind === 'back') {
    const moving = groupIds.filter((id) => selected.has(id));
    const remaining = groupIds.filter((id) => !selected.has(id));
    return kind === 'front' ? [...moving, ...remaining] : [...remaining, ...moving];
  }
  const next = [...groupIds];
  if (kind === 'forward') {
    for (let index = 1; index < next.length; index += 1) {
      if (selected.has(next[index]!) && !selected.has(next[index - 1]!)) {
        [next[index - 1], next[index]] = [next[index]!, next[index - 1]!];
      }
    }
  } else {
    for (let index = next.length - 2; index >= 0; index -= 1) {
      if (selected.has(next[index]!) && !selected.has(next[index + 1]!)) {
        [next[index], next[index + 1]] = [next[index + 1]!, next[index]!];
      }
    }
  }
  return next;
};

/** Moves every selected layer within its own type group, preserving group boundaries. */
export const reorderSelectionWithinGroupsByKind = (
  layers: readonly CanvasLayerContract[],
  selectedIds: readonly string[],
  kind: LayerReorderKind
): string[] | null => {
  const selected = new Set(selectedIds);
  const nextIds = layers.map((layer) => layer.id);
  let changed = false;
  for (const type of LAYER_TYPE_ORDER) {
    const slots: number[] = [];
    const groupIds: string[] = [];
    layers.forEach((layer, index) => {
      if (layer.type === type) {
        slots.push(index);
        groupIds.push(layer.id);
      }
    });
    const reordered = reorderSelectedIds(groupIds, selected, kind);
    if (reordered.some((id, index) => id !== groupIds[index])) {
      reordered.forEach((id, index) => {
        nextIds[slots[index]!] = id;
      });
      changed = true;
    }
  }
  return changed ? nextIds : null;
};

/** The destination index for a z-reorder of the layer at `index` in a stack of `count`. */
export const reorderTargetIndex = (index: number, count: number, kind: LayerReorderKind): number => {
  switch (kind) {
    case 'forward':
      return Math.max(0, index - 1);
    case 'backward':
      return Math.min(count - 1, index + 1);
    case 'front':
      return 0;
    case 'back':
      return count - 1;
  }
};

/** Moves `items[from]` to `to`, shifting the rest. Returns a new array. */
const moveItem = <T>(items: readonly T[], from: number, to: number): T[] => {
  const next = [...items];
  const [moved] = next.splice(from, 1);
  if (moved === undefined) {
    return next;
  }
  next.splice(to, 0, moved);
  return next;
};

/**
 * The reordered id list for a z-reorder hotkey, or `null` when nothing would move
 * (already at the boundary, or the index is out of range) so callers can no-op.
 */
export const reorderIdsForHotkey = (
  currentIds: readonly string[],
  index: number,
  kind: LayerReorderKind
): string[] | null => {
  if (index < 0 || index >= currentIds.length) {
    return null;
  }
  const target = reorderTargetIndex(index, currentIds.length, kind);
  if (target === index) {
    return null;
  }
  return moveItem(currentIds, index, target);
};
