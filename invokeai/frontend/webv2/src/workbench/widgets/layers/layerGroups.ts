import type { CanvasLayerContract } from '@workbench/canvas-engine/api';

import { moveItem } from './layersDnd';

/** The four layer-type groups. Keys equal the contract's `layer.type`. */
export type LayerGroupKey = 'inpaint_mask' | 'regional_guidance' | 'control' | 'raster';

/**
 * Top-to-bottom display order of the type groups, matching legacy's
 * `CanvasEntityList` (InpaintMask → RegionalGuidance → ControlLayer →
 * RasterLayer). This is a *display* order only; it does not touch global z.
 */
export const LAYER_GROUP_ORDER: readonly LayerGroupKey[] = ['inpaint_mask', 'regional_guidance', 'control', 'raster'];

/** The group a layer belongs to — its contract type maps 1:1 to a group key. */
export const getLayerGroupKey = (layer: CanvasLayerContract): LayerGroupKey => layer.type;

/** A non-empty type group: its key plus its members in global relative order. */
export interface LayerGroup {
  key: LayerGroupKey;
  layers: CanvasLayerContract[];
}

/**
 * Partitions layers into the non-empty type groups, in display order. Each
 * group's members keep their global relative order; empty groups are dropped.
 */
export const groupLayers = (layers: readonly CanvasLayerContract[]): LayerGroup[] =>
  LAYER_GROUP_ORDER.map((key) => ({
    key,
    layers: layers.filter((layer) => getLayerGroupKey(layer) === key),
  })).filter((group) => group.layers.length > 0);

/** A layer's position within its own group (index 0 = top of the group). */
export interface GroupPosition {
  index: number;
  count: number;
}

/** Where `layerId` sits inside its type group, or null when it is absent. */
export const getGroupPosition = (layers: readonly CanvasLayerContract[], layerId: string): GroupPosition | null => {
  const layer = layers.find((entry) => entry.id === layerId);
  if (!layer) {
    return null;
  }
  const key = getLayerGroupKey(layer);
  const index = layers.filter((entry) => getLayerGroupKey(entry) === key).findIndex((entry) => entry.id === layerId);
  const count = layers.filter((entry) => getLayerGroupKey(entry) === key).length;
  return { count, index };
};

const sameOrder = (a: readonly string[], b: readonly string[]): boolean =>
  a.length === b.length && a.every((id, index) => id === b[index]);

/**
 * Reorders one type group in place inside the global id order: `reorderGroup`
 * receives the group's ids (top-to-bottom) and returns their new order (or null
 * for a no-op). The returned order is written back into exactly the slots the
 * group occupied, so every other layer keeps its global position. Returns the
 * full new global id list, or null when nothing moved.
 */
const remapGroupOrder = (
  layers: readonly CanvasLayerContract[],
  key: LayerGroupKey,
  reorderGroup: (groupIds: string[]) => string[] | null
): string[] | null => {
  const slots: number[] = [];
  const groupIds: string[] = [];
  layers.forEach((layer, index) => {
    if (getLayerGroupKey(layer) === key) {
      slots.push(index);
      groupIds.push(layer.id);
    }
  });
  const reordered = reorderGroup(groupIds);
  if (!reordered || sameOrder(reordered, groupIds)) {
    return null;
  }
  const next = layers.map((layer) => layer.id);
  reordered.forEach((id, i) => {
    const slot = slots[i];
    if (slot !== undefined) {
      next[slot] = id;
    }
  });
  return next;
};

/**
 * Maps a drag-to-reorder (drop `activeId` onto same-group `overId`) to the new
 * global id order. Returns null — a no-op — when the ids are equal, either is
 * absent, they live in different groups (cross-group drop), or nothing moved.
 */
export const reorderWithinGroup = (
  layers: readonly CanvasLayerContract[],
  activeId: string,
  overId: string
): string[] | null => {
  if (activeId === overId) {
    return null;
  }
  const active = layers.find((layer) => layer.id === activeId);
  const over = layers.find((layer) => layer.id === overId);
  if (!active || !over || getLayerGroupKey(active) !== getLayerGroupKey(over)) {
    return null;
  }
  return remapGroupOrder(layers, getLayerGroupKey(active), (groupIds) => {
    const from = groupIds.indexOf(activeId);
    const to = groupIds.indexOf(overId);
    return moveItem(groupIds, from, to);
  });
};

/** A z-reorder direction for the context menu. Index 0 = front/top. */
export type LayerMoveKind = 'front' | 'forward' | 'backward' | 'back';

const moveTargetIndex = (index: number, count: number, kind: LayerMoveKind): number => {
  switch (kind) {
    case 'front':
      return 0;
    case 'forward':
      return Math.max(0, index - 1);
    case 'backward':
      return Math.min(count - 1, index + 1);
    case 'back':
      return count - 1;
  }
};

/**
 * Maps a "move to front / forward / backward / to back" command to the new
 * global id order, moving `layerId` within its own group only. Returns null
 * when the layer is absent or already at the group boundary for that direction.
 */
export const reorderWithinGroupByKind = (
  layers: readonly CanvasLayerContract[],
  layerId: string,
  kind: LayerMoveKind
): string[] | null => {
  const layer = layers.find((entry) => entry.id === layerId);
  if (!layer) {
    return null;
  }
  return remapGroupOrder(layers, getLayerGroupKey(layer), (groupIds) => {
    const index = groupIds.indexOf(layerId);
    const target = moveTargetIndex(index, groupIds.length, kind);
    if (target === index) {
      return null;
    }
    return moveItem(groupIds, index, target);
  });
};
