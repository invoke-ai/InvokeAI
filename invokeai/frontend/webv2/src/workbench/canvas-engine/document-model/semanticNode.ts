import type { CanvasNodeContract } from '@workbench/canvas-engine/contracts';
import type { CanvasNodeEntry } from '@workbench/canvas-engine/document/documentIndex';
import type { LayerStackKind } from '@workbench/canvas-engine/document/layerStacks';

import { collectSubtreeLeaves, isGroupNode, subtreeDepth } from '@workbench/canvas-engine/document/documentTree';
import { isNodeHidden } from '@workbench/canvas-engine/document/layerEligibility';

/**
 * Node facts with ancestor state applied, shared by rows, tools and planners without screen or session
 * dependencies.
 */
export interface SemanticNode {
  readonly id: string;
  readonly kind: 'group' | 'leaf';
  readonly stack: LayerStackKind;
  readonly node: CanvasNodeContract;
  readonly parentId: string | null;
  /** Ancestor group ids, root first. */
  readonly parentIds: readonly string[];
  readonly depth: number;
  readonly siblingIndex: number;
  /** The node and every ancestor are enabled. */
  readonly contributionEnabled: boolean;
  /** The node or an ancestor is display-hidden. */
  readonly documentHidden: boolean;
  /** The node or an ancestor is locked; edits refuse on this. */
  readonly effectiveLocked: boolean;
  readonly ancestorsEnabled: boolean;
  readonly ancestorsHidden: boolean;
  readonly ancestorsLocked: boolean;
  readonly childCount: number;
  readonly leafCount: number;
  /** Levels the node's child lists reach below it; a group always opens one. */
  readonly subtreeDepth: number;
}

/**
 * Compiles the node for `entry`. `previous` is the node compiled for the same id before a value
 * edit: structure is unchanged then, so its counts and depth carry over instead of a subtree walk.
 */
export const compileSemanticNode = (entry: CanvasNodeEntry, previous?: SemanticNode): SemanticNode => {
  const { node } = entry;
  const group = isGroupNode(node);
  const reuse = previous && previous.id === node.id && (previous.kind === 'group') === group ? previous : null;
  return {
    ancestorsEnabled: entry.ancestorsEnabled,
    ancestorsHidden: entry.ancestorsHidden,
    ancestorsLocked: entry.ancestorsLocked,
    childCount: group ? node.children.length : 0,
    contributionEnabled: entry.ancestorsEnabled && node.isEnabled,
    depth: entry.path.length,
    documentHidden: entry.ancestorsHidden || isNodeHidden(node),
    effectiveLocked: entry.ancestorsLocked || node.isLocked,
    id: node.id,
    kind: group ? 'group' : 'leaf',
    leafCount: reuse ? reuse.leafCount : group ? collectSubtreeLeaves(node).length : 1,
    node,
    parentId: entry.parentId,
    parentIds: entry.path,
    siblingIndex: entry.siblingIndex,
    stack: entry.stack,
    subtreeDepth: reuse ? reuse.subtreeDepth : group ? Math.max(1, subtreeDepth(node)) : 0,
  };
};

/** Whether a cached node still describes `entry`'s node in the place `entry` gives it. */
export const isSemanticNodeCurrent = (semantic: SemanticNode, entry: CanvasNodeEntry): boolean =>
  semantic.node === entry.node &&
  semantic.parentId === entry.parentId &&
  semantic.siblingIndex === entry.siblingIndex &&
  semantic.stack === entry.stack &&
  semantic.ancestorsEnabled === entry.ancestorsEnabled &&
  semantic.ancestorsHidden === entry.ancestorsHidden &&
  semantic.ancestorsLocked === entry.ancestorsLocked &&
  semantic.parentIds.length === entry.path.length &&
  semantic.parentIds.every((id, index) => id === entry.path[index]);
