/**
 * Pure logic for the layers-panel type-group header actions (round-3 restructure).
 *
 * Each type group renders a right-aligned action cluster. The set of actions per
 * group is data (`getGroupActions`), so a new action — e.g. Task 44's PSD export —
 * is added by extending one array, not by threading a new prop through the header.
 * The visibility-toggle planning lives here as pure functions so it is
 * unit-testable in node (no React, no engine); merge-visible planning lives in
 * the engine's `document/mergeVisible` (shared with the engine op).
 */

import type { CanvasLayerContract } from '@workbench/types';

import type { LayerGroupKey } from './layerGroups';

/** A group-header action id. Extend this + `getGroupActions` to add a new action. */
export type GroupActionId = 'mergeVisible' | 'exportPsd' | 'toggleVisibility' | 'new';

/**
 * The right-aligned actions for a group, in left-to-right render order (the "New"
 * action sits rightmost, nearest the panel's own add-layer menu). Only the raster
 * group offers "merge visible" + "export to PSD"; every group offers
 * hide/show-all + new.
 */
export const getGroupActions = (groupKey: LayerGroupKey): GroupActionId[] => {
  const actions: GroupActionId[] = [];
  if (groupKey === 'raster') {
    actions.push('mergeVisible', 'exportPsd');
  }
  actions.push('toggleVisibility', 'new');
  return actions;
};

/**
 * True when a raster layer carries content that a PSD export could contain: an
 * image/paint/gradient/text source, or a non-`polygon` shape (`polygon` has no
 * rasterizer). A brand-new paint layer with unflushed live strokes is still
 * counted (its `source.bitmap` is null but its live cache holds pixels the
 * export bakes) — enablement never under-counts; a genuinely empty layer is
 * handled gracefully at export time (the planner returns `empty`).
 */
const hasExportableRasterContent = (layer: CanvasLayerContract): boolean => {
  if (layer.type !== 'raster') {
    return false;
  }
  switch (layer.source.type) {
    case 'image':
    case 'paint':
    case 'gradient':
    case 'text':
      return true;
    case 'shape':
      return layer.source.kind !== 'polygon';
    default:
      return false;
  }
};

/** Whether the raster group's "export to PSD" action has anything to export. */
export const canExportRasterPsd = (layers: readonly CanvasLayerContract[]): boolean =>
  layers.some(hasExportableRasterContent);

/** A single layer's target visibility, for the bulk `setCanvasLayersEnabled` action. */
export interface LayerVisibilityUpdate {
  id: string;
  isEnabled: boolean;
}

/** True when every layer in the group is currently visible (enabled). Empty ⇒ true. */
export const isGroupAllVisible = (groupLayers: readonly CanvasLayerContract[]): boolean =>
  groupLayers.every((layer) => layer.isEnabled);

/**
 * Plans a group hide/show-all toggle as ONE reversible bulk action: if every layer
 * is visible, hide them all; otherwise show them all (legacy `CanvasEntityType`
 * toggle semantics). `forward` sets the shared target; `inverse` restores each
 * layer's prior visibility verbatim, so undo is a single history entry.
 */
export const planGroupVisibilityToggle = (
  groupLayers: readonly CanvasLayerContract[]
): { forward: LayerVisibilityUpdate[]; inverse: LayerVisibilityUpdate[]; nextVisible: boolean } => {
  const nextVisible = !isGroupAllVisible(groupLayers);
  return {
    forward: groupLayers.map((layer) => ({ id: layer.id, isEnabled: nextVisible })),
    inverse: groupLayers.map((layer) => ({ id: layer.id, isEnabled: layer.isEnabled })),
    nextVisible,
  };
};

// Merge-visible contributor selection lives in the engine's document layer
// (`@workbench/canvas-engine/document/mergeVisible`) so button enablement and
// execution use the same legacy-parity eligibility rules.
