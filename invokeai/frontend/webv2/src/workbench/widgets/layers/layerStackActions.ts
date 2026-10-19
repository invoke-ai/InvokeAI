import type { CanvasLayerContract, LayerStackKind, SemanticNode } from '@workbench/canvas-engine/api';

import { isExportableRasterLayer, isNodeHidden } from '@workbench/canvas-engine/api';

/** A stack-header action id. Extend this + `getStackActions` to add a new action. */
export type StackActionId = 'mergeVisible' | 'exportPsd' | 'toggleVisibility' | 'new';

/**
 * Offer merge/export only for raster stacks and hide/show only for overlays, where visibility must not disable
 * image content. Place New nearest the panel add menu.
 */
export const getStackActions = (stack: LayerStackKind): StackActionId[] => {
  if (stack === 'raster') {
    return ['mergeVisible', 'exportPsd', 'new'];
  }
  return ['toggleVisibility', 'new'];
};

/** Whether the raster stack's "export to PSD" action has anything to export. */
export const canExportRasterPsd = (leaves: readonly CanvasLayerContract[]): boolean =>
  leaves.some(isExportableRasterLayer);

export type StackVisibilityAxis = 'enabled' | 'hidden';

/**
 * Which axis a stack's show/hide-all button drives. The three overlay stacks drive the DISPLAY
 * axis: their layers are drawn to show where an effect applies, so getting them out of the way
 * must not change the image. The raster stack IS the generation input, so it drives enablement.
 */
export const stackVisibilityAxis = (stack: LayerStackKind): StackVisibilityAxis =>
  stack === 'raster' ? 'enabled' : 'hidden';

const isVisible = (node: SemanticNode, axis: StackVisibilityAxis): boolean =>
  axis === 'hidden' ? !node.documentHidden : node.contributionEnabled;

const isOwnVisible = (node: SemanticNode, axis: StackVisibilityAxis): boolean =>
  axis === 'hidden' ? !isNodeHidden(node.node) : node.node.isEnabled;

/** True when every leaf of the stack is effectively visible on `axis`. Empty ⇒ true. */
export const isStackAllVisible = (nodes: readonly SemanticNode[], axis: StackVisibilityAxis): boolean =>
  nodes.every((node) => node.kind === 'group' || isVisible(node, axis));

/**
 * Plans a stack show/hide-all toggle as ONE reversible bulk edit. When every leaf is visible the
 * roots alone are turned off, so descendants keep their own flags; otherwise every node that is
 * off in its own right is turned on, so nothing stays gated behind an ancestor.
 */
export const planStackVisibilityToggle = (
  nodes: readonly SemanticNode[],
  axis: StackVisibilityAxis
): { ids: string[]; nextVisible: boolean } => {
  const nextVisible = !isStackAllVisible(nodes, axis);
  const targets = nextVisible
    ? nodes.filter((node) => !isOwnVisible(node, axis))
    : nodes.filter((node) => node.parentId === null);
  return { ids: targets.map((node) => node.id), nextVisible };
};
