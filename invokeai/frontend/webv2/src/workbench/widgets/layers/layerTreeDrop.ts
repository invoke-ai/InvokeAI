import type { LayerStackKind } from '@workbench/canvas-engine/api';

import { CANVAS_MAX_NODE_DEPTH, isGroupNode } from '@workbench/canvas-engine/api';

import type { LayerTreeRow } from './layerTreeRows';

/** Where a dragged block lands, expressed as the model's `reparent` target. */
export interface LayerDropTarget {
  readonly stack: LayerStackKind;
  readonly parentId: string | null;
  /** The sibling the block lands above, or `null` for the bottom of `parentId`. */
  readonly beforeId: string | null;
  /** The depth the preview draws the block at. */
  readonly depth: number;
  /** Ids that move, outermost only, in document order. */
  readonly ids: readonly string[];
  /** The rendered row the block lands above, or `null` for the end of the list. */
  readonly beforeRowId: string | null;
}

export interface LayerDropInput {
  /** One stack's rendered rows, top first. */
  readonly rows: readonly LayerTreeRow[];
  /** Every selected id that drags along; descendants of another dragged id are folded in. */
  readonly activeIds: readonly string[];
  readonly overId: string;
  /** Where the pointer sits on the row it is over; `inside` (group rows only) nests into it. */
  readonly edge: 'above' | 'inside' | 'below';
  /** How many indent steps the pointer has moved horizontally since the drag began. */
  readonly depthOffset: number;
}

/** For each rendered row, the index just past its rendered descendants. One backward pass. */
const subtreeEnds = (rows: readonly LayerTreeRow[]): Int32Array => {
  const ends = new Int32Array(rows.length);
  const open: number[] = [];
  for (let index = rows.length - 1; index >= 0; index -= 1) {
    const depth = rows[index]!.vm.depth;
    while (open.length > 0 && rows[open[open.length - 1]!]!.vm.depth > depth) {
      open.pop();
    }
    ends[index] = open.length > 0 ? open[open.length - 1]! : rows.length;
    open.push(index);
  }
  return ends;
};

/**
 * Project vertical gaps and horizontal depth after removing the dragged block. Reject self/invalid/depth-exceeding
 * targets; model validation owns locks and other refusals. Linear in row count.
 */
export const projectLayerDrop = (input: LayerDropInput): LayerDropTarget | null => {
  const { rows } = input;
  if (rows.length === 0) {
    return null;
  }
  const selected = new Set(input.activeIds);
  const ends = subtreeEnds(rows);
  const moving = new Uint8Array(rows.length);
  const outer: LayerTreeRow[] = [];
  let deepestSubtree = 0;
  let coveredUntil = 0;
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index]!;
    if (index < coveredUntil) {
      moving[index] = 1;
      continue;
    }
    if (!selected.has(row.id)) {
      continue;
    }
    outer.push(row);
    moving[index] = 1;
    coveredUntil = ends[index]!;
    deepestSubtree = Math.max(deepestSubtree, row.vm.subtreeDepth);
  }
  if (outer.length === 0) {
    return null;
  }
  let overIndex = -1;
  const remaining: LayerTreeRow[] = [];
  for (let index = 0; index < rows.length; index += 1) {
    if (moving[index]) {
      continue;
    }
    if (rows[index]!.id === input.overId) {
      overIndex = remaining.length;
    }
    remaining.push(rows[index]!);
  }
  if (overIndex < 0) {
    return null;
  }
  if (input.edge === 'inside') {
    // Drop into the hovered group's top, respecting the entire block's depth limit.
    const over = remaining[overIndex]!;
    if (over.vm.kind !== 'group') {
      return null;
    }
    const depth = over.vm.depth + 1;
    if (depth + deepestSubtree > CANVAS_MAX_NODE_DEPTH) {
      return null;
    }
    const next = remaining[overIndex + 1];
    // Top of the group: the first rendered child, or — for a collapsed group,
    // whose children render nothing — its first model child, so the landing
    // matches the indicator instead of quietly falling to the bottom.
    const firstModelChildId = isGroupNode(over.vm.node) ? (over.vm.node.children[0]?.id ?? null) : null;
    const beforeId = next && next.vm.parentId === over.id ? next.id : over.expanded ? null : firstModelChildId;
    return {
      beforeId,
      beforeRowId: next?.id ?? null,
      depth,
      ids: outer.map((row) => row.id),
      parentId: over.id,
      stack: rows[0]!.vm.stack,
    };
  }
  const insertAt = input.edge === 'above' ? overIndex : overIndex + 1;
  const previous = remaining[insertAt - 1];
  const next = remaining[insertAt];
  const maxDepth = Math.min(
    previous ? previous.vm.depth + (previous.vm.kind === 'group' && previous.expanded ? 1 : 0) : 0,
    CANVAS_MAX_NODE_DEPTH - deepestSubtree
  );
  const minDepth = next ? next.vm.depth : 0;
  if (maxDepth < minDepth) {
    return null;
  }
  const depth = Math.max(minDepth, Math.min(maxDepth, outer[0]!.vm.depth + input.depthOffset));
  let parentId: string | null = null;
  if (depth > 0 && previous) {
    if (previous.vm.depth < depth) {
      parentId = previous.id;
    } else {
      for (let index = insertAt - 1; index >= 0; index -= 1) {
        const candidate = remaining[index]!;
        if (candidate.vm.depth === depth - 1) {
          parentId = candidate.id;
          break;
        }
      }
    }
  }
  const beforeId = next && next.vm.parentId === parentId && next.vm.depth === depth ? next.id : null;
  return {
    beforeId,
    beforeRowId: next?.id ?? null,
    depth,
    ids: outer.map((row) => row.id),
    parentId,
    stack: rows[0]!.vm.stack,
  };
};

/** The selected rendered rows with no selected rendered ancestor: the blocks a drag carries. */
export const outermostRowIds = (rows: readonly LayerTreeRow[], ids: ReadonlySet<string>): string[] => {
  const ends = subtreeEnds(rows);
  const outer: string[] = [];
  let coveredUntil = 0;
  for (let index = 0; index < rows.length; index += 1) {
    if (index < coveredUntil || !ids.has(rows[index]!.id)) {
      continue;
    }
    outer.push(rows[index]!.id);
    coveredUntil = ends[index]!;
  }
  return outer;
};
