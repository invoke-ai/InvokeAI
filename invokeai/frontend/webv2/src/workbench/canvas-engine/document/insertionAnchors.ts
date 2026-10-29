import type { CanvasNodeContract, CanvasStackForests } from '@workbench/canvas-engine/contracts';

import type { LayerStackKind } from './layerStacks';

import { indexStacks, type CanvasDocumentIndex } from './documentIndex';
import { insertNodes, isGroupNode } from './documentTree';

/**
 * Where new nodes land in their stack, captured once against the document the caller saw.
 * Resolution is exact and in this order: before a surviving `beforeId` under its current parent,
 * after a surviving `afterId` under its current parent, at the top of the surviving captured
 * parent, at the top of the nearest surviving group on the captured parent path, at the top of
 * the stack. Ids that no longer belong to `stack` count as gone. The reducer refuses an anchor
 * from another project; a structural commit refuses one captured at an older edit revision.
 */
export interface CanvasNodeInsertionAnchor {
  readonly projectId: string;
  readonly stack: LayerStackKind;
  /** Ancestor ids of the target parent, root first; empty for the stack root. */
  readonly parentPath: readonly string[];
  readonly beforeId: string | null;
  readonly afterId: string | null;
  readonly capturedEditRevision: number;
}

export interface CanvasNodeInsertion {
  readonly anchor: CanvasNodeInsertionAnchor;
  readonly nodes: readonly CanvasNodeContract[];
}

/** Subtrees to detach and reinsert at `anchor`, keeping their relative order. */
export interface CanvasNodeMove {
  readonly ids: readonly string[];
  readonly anchor: CanvasNodeInsertionAnchor;
}

export interface InsertionAnchorCapture {
  readonly projectId: string;
  readonly stack: LayerStackKind;
  readonly editRevision: number;
  /** Land directly above this node when it belongs to `stack`; otherwise at the stack top. */
  readonly aboveId?: string | null;
  /** Land at the top of this group when it belongs to `stack`; takes precedence over `aboveId`. */
  readonly insideId?: string | null;
}

export interface ResolvedInsertionTarget {
  readonly parentId: string | null;
  readonly index: number;
}

export const captureInsertionAnchor = (
  stacks: CanvasStackForests,
  capture: InsertionAnchorCapture
): CanvasNodeInsertionAnchor => {
  const { editRevision, projectId, stack } = capture;
  const index = indexStacks(stacks);
  const inside = capture.insideId ? index.byId.get(capture.insideId) : undefined;
  if (inside && inside.stack === stack && isGroupNode(inside.node)) {
    return {
      afterId: null,
      beforeId: inside.node.children[0]?.id ?? null,
      capturedEditRevision: editRevision,
      parentPath: [...inside.path, inside.node.id],
      projectId,
      stack,
    };
  }
  const above = capture.aboveId ? index.byId.get(capture.aboveId) : undefined;
  if (above && above.stack === stack) {
    const siblings =
      above.parentId === null
        ? stacks[stack]
        : (index.byId.get(above.parentId)!.node as { children: CanvasNodeContract[] }).children;
    return {
      afterId: siblings[above.siblingIndex - 1]?.id ?? null,
      beforeId: above.node.id,
      capturedEditRevision: editRevision,
      parentPath: above.path,
      projectId,
      stack,
    };
  }
  return {
    afterId: null,
    beforeId: stacks[stack][0]?.id ?? null,
    capturedEditRevision: editRevision,
    parentPath: [],
    projectId,
    stack,
  };
};

/** The anchor that puts `nodeId` back between its current siblings once removed. */
export const captureRestoreAnchor = (
  stacks: CanvasStackForests,
  nodeId: string,
  projectId: string,
  editRevision: number
): CanvasNodeInsertionAnchor | null => {
  const index = indexStacks(stacks);
  const entry = index.byId.get(nodeId);
  if (!entry) {
    return null;
  }
  const siblings =
    entry.parentId === null
      ? stacks[entry.stack]
      : (index.byId.get(entry.parentId)!.node as { children: CanvasNodeContract[] }).children;
  return {
    afterId: siblings[entry.siblingIndex - 1]?.id ?? null,
    beforeId: siblings[entry.siblingIndex + 1]?.id ?? null,
    capturedEditRevision: editRevision,
    parentPath: entry.path,
    projectId,
    stack: entry.stack,
  };
};

const survivingGroup = (index: CanvasDocumentIndex, stack: LayerStackKind, id: string): boolean => {
  const entry = index.byId.get(id);
  return !!entry && entry.stack === stack && isGroupNode(entry.node);
};

export const resolveInsertionTarget = (
  stacks: CanvasStackForests,
  anchor: CanvasNodeInsertionAnchor
): ResolvedInsertionTarget => {
  const index = indexStacks(stacks);
  const before = anchor.beforeId ? index.byId.get(anchor.beforeId) : undefined;
  if (before && before.stack === anchor.stack) {
    return { index: before.siblingIndex, parentId: before.parentId };
  }
  const after = anchor.afterId ? index.byId.get(anchor.afterId) : undefined;
  if (after && after.stack === anchor.stack) {
    return { index: after.siblingIndex + 1, parentId: after.parentId };
  }
  for (let position = anchor.parentPath.length - 1; position >= 0; position -= 1) {
    const id = anchor.parentPath[position]!;
    if (survivingGroup(index, anchor.stack, id)) {
      return { index: 0, parentId: id };
    }
  }
  return { index: 0, parentId: null };
};

export const insertNodesAtAnchor = (
  stacks: CanvasStackForests,
  anchor: CanvasNodeInsertionAnchor,
  nodes: readonly CanvasNodeContract[]
): CanvasStackForests => {
  const target = resolveInsertionTarget(stacks, anchor);
  return insertNodes(stacks, anchor.stack, target.parentId, target.index, nodes) ?? stacks;
};
