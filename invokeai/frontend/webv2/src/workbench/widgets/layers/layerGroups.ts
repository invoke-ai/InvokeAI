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

/** Transient Layers-panel selection; only `primaryId` is persisted in the canvas document. */
export interface LayerPanelSelection {
  anchorId: string | null;
  primaryId: string | null;
  projectId: string;
  selectedIds: readonly string[];
}

export interface LayerSelectionModifiers {
  additive: boolean;
  range: boolean;
}

export const createLayerPanelSelection = (projectId: string, primaryId: string | null): LayerPanelSelection => ({
  anchorId: primaryId,
  primaryId,
  projectId,
  selectedIds: primaryId ? [primaryId] : [],
});

/** Reconciles transient selection after an external primary change, project switch, or layer removal. */
export const reconcileLayerPanelSelection = (
  selection: LayerPanelSelection,
  projectId: string,
  orderedIds: readonly string[],
  primaryId: string | null
): LayerPanelSelection => {
  const existing = new Set(orderedIds);
  const validPrimaryId = primaryId && existing.has(primaryId) ? primaryId : null;
  if (selection.projectId !== projectId || selection.primaryId !== validPrimaryId) {
    return createLayerPanelSelection(projectId, validPrimaryId);
  }
  const selected = new Set(selection.selectedIds.filter((id) => existing.has(id)));
  if (validPrimaryId) {
    selected.add(validPrimaryId);
  }
  const selectedIds = orderedIds.filter((id) => selected.has(id));
  const anchorId = selection.anchorId && existing.has(selection.anchorId) ? selection.anchorId : validPrimaryId;
  if (
    anchorId === selection.anchorId &&
    selectedIds.length === selection.selectedIds.length &&
    selectedIds.every((id, index) => id === selection.selectedIds[index])
  ) {
    return selection;
  }
  return { ...selection, anchorId, selectedIds };
};

/** Applies plain, Ctrl/Cmd-toggle, and Shift-range row selection semantics. */
export const selectLayerInPanel = (
  selection: LayerPanelSelection,
  layerId: string,
  orderedIds: readonly string[],
  modifiers: LayerSelectionModifiers
): LayerPanelSelection => {
  if (!orderedIds.includes(layerId)) {
    return selection;
  }
  if (modifiers.range) {
    const anchorId = selection.anchorId && orderedIds.includes(selection.anchorId) ? selection.anchorId : layerId;
    const start = orderedIds.indexOf(anchorId);
    const end = orderedIds.indexOf(layerId);
    const rangeIds = orderedIds.slice(Math.min(start, end), Math.max(start, end) + 1);
    const selected = modifiers.additive ? new Set(selection.selectedIds) : new Set<string>();
    rangeIds.forEach((id) => selected.add(id));
    return { ...selection, anchorId, primaryId: layerId, selectedIds: orderedIds.filter((id) => selected.has(id)) };
  }
  if (modifiers.additive) {
    const selected = new Set(selection.selectedIds);
    const wasSelected = selected.has(layerId);
    if (wasSelected) {
      selected.delete(layerId);
    } else {
      selected.add(layerId);
    }
    const selectedIds = orderedIds.filter((id) => selected.has(id));
    const primaryId = wasSelected
      ? selection.primaryId === layerId || !selection.primaryId || !selected.has(selection.primaryId)
        ? (selectedIds[0] ?? null)
        : selection.primaryId
      : layerId;
    return { ...selection, anchorId: layerId, primaryId, selectedIds };
  }
  return { ...selection, anchorId: layerId, primaryId: layerId, selectedIds: [layerId] };
};

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

/**
 * Drag-reorders the selected members of the active layer's group as one block.
 * Selections in other type groups stay put. Dragging an unselected row retains
 * the normal single-row behaviour.
 */
export const reorderSelectionWithinGroup = (
  layers: readonly CanvasLayerContract[],
  activeId: string,
  overId: string,
  selectedIds: readonly string[]
): string[] | null => {
  const active = layers.find((layer) => layer.id === activeId);
  const over = layers.find((layer) => layer.id === overId);
  if (!active || !over || getLayerGroupKey(active) !== getLayerGroupKey(over)) {
    return null;
  }
  const selected = new Set(selectedIds);
  if (!selected.has(activeId)) {
    return reorderWithinGroup(layers, activeId, overId);
  }
  return remapGroupOrder(layers, getLayerGroupKey(active), (groupIds) => {
    const moving = groupIds.filter((id) => selected.has(id));
    if (moving.length < 2) {
      return moveItem(groupIds, groupIds.indexOf(activeId), groupIds.indexOf(overId));
    }
    if (selected.has(overId)) {
      return null;
    }
    const activeIndex = groupIds.indexOf(activeId);
    const overIndex = groupIds.indexOf(overId);
    const remaining = groupIds.filter((id) => !selected.has(id));
    const remainingOverIndex = remaining.indexOf(overId);
    const insertAt = activeIndex < overIndex ? remainingOverIndex + 1 : remainingOverIndex;
    return [...remaining.slice(0, insertAt), ...moving, ...remaining.slice(insertAt)];
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
