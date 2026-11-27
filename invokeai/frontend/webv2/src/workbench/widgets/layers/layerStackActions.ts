import type { CanvasLayerContract, LayerStackKind, SemanticNode } from '@workbench/canvas-engine/api';

import { isExportableRasterLayer, isNodeHidden } from '@workbench/canvas-engine/api';

/** A stack-header action id. Extend this + `getStackActions` to add a new action. */
export type StackActionId = 'mergeVisible' | 'cancelExportPsd' | 'exportPsd' | 'toggleVisibility' | 'new';

/** Merge/export on raster stacks only; hide/show on overlays only, so visibility never disables image content. */
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

/** Overlay stacks drive display, so hiding never changes the image; the raster stack drives enablement. */
export const stackVisibilityAxis = (stack: LayerStackKind): StackVisibilityAxis =>
  stack === 'raster' ? 'enabled' : 'hidden';

const isVisible = (node: SemanticNode, axis: StackVisibilityAxis): boolean =>
  axis === 'hidden' ? !node.documentHidden : node.contributionEnabled;

const isOwnVisible = (node: SemanticNode, axis: StackVisibilityAxis): boolean =>
  axis === 'hidden' ? !isNodeHidden(node.node) : node.node.isEnabled;

/** True when every leaf of the stack is effectively visible on `axis`. Empty ⇒ true. */
export const isStackAllVisible = (nodes: readonly SemanticNode[], axis: StackVisibilityAxis): boolean =>
  nodes.every((node) => node.kind === 'group' || isVisible(node, axis));

/** One reversible bulk edit: all visible turns roots off; otherwise everything off in its own right turns on. */
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
