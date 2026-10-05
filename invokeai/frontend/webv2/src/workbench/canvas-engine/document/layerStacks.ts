import type {
  CanvasLayerContract,
  CanvasLayerStackKind,
  CanvasNodeContract,
  CanvasStackForests,
} from '@workbench/canvas-engine/contracts';

import { childrenAt, indexStacks, type CanvasDocumentIndex } from './documentIndex';
import { childrenOf, isGroupNode, replaceChildren } from './documentTree';

/**
 * The four stack forests of a v3 document. The compositor draws the stacks bottom to top in
 * {@link LAYER_STACK_ORDER} and the Layers panel lists them top first; within a stack, nodes are
 * ordered by their depth-first preorder position, top first.
 */
export type LayerStackKind = CanvasLayerStackKind;

export type OverlayStackKind = Exclude<LayerStackKind, 'raster'>;

export { LAYER_STACK_ORDER, LAYER_STACKS_TOP_FIRST } from '@workbench/canvas-engine/contracts';

export const layerStackOf = (layer: CanvasLayerContract): LayerStackKind => layer.type;

export const isOverlayStack = (stack: LayerStackKind): stack is OverlayStackKind => stack !== 'raster';

/** Reorders one sibling list; `orderedIds` must be exactly the parent's current children, top first. */
export interface ReorderSiblingsCommand {
  readonly stack: LayerStackKind;
  readonly parentId: string | null;
  readonly orderedIds: readonly string[];
}

export const getSiblingOrder = (
  stacks: CanvasStackForests,
  stack: LayerStackKind,
  parentId: string | null
): ReorderSiblingsCommand => ({
  orderedIds: (childrenOf(stacks, stack, parentId) ?? []).map((node) => node.id),
  parentId,
  stack,
});

const sameIds = (a: readonly string[], b: readonly string[]): boolean =>
  a.length === b.length && a.every((id, index) => id === b[index]);

/** Reorders only the parent's children; rejects ids that do not exactly match them. */
export const reorderSiblings = (
  stacks: CanvasStackForests,
  command: ReorderSiblingsCommand
): CanvasStackForests | null => {
  const children = childrenOf(stacks, command.stack, command.parentId);
  if (!children || children.length !== command.orderedIds.length) {
    return null;
  }
  const byId = new Map(children.map((node) => [node.id, node]));
  if (byId.size !== command.orderedIds.length || new Set(command.orderedIds).size !== command.orderedIds.length) {
    return null;
  }
  const next: CanvasNodeContract[] = [];
  for (const id of command.orderedIds) {
    const node = byId.get(id);
    if (!node) {
      return null;
    }
    next.push(node);
  }
  return next.every((node, index) => node === children[index])
    ? stacks
    : replaceChildren(stacks, command.stack, command.parentId, next);
};

/** Whether both forests hold the same nodes under the same parents in the same order, per stack. */
export const haveSameStructure = (a: CanvasStackForests, b: CanvasStackForests): boolean => {
  if (a === b) {
    return true;
  }
  const left = indexStacks(a).nodes;
  const right = indexStacks(b).nodes;
  return (
    left.length === right.length &&
    left.every((entry, index) => {
      const other = right[index]!;
      return entry.node.id === other.node.id && entry.parentId === other.parentId && entry.stack === other.stack;
    })
  );
};

/** A z-move among siblings; index 0 is the front (top). */
export type LayerStackMoveKind = 'front' | 'forward' | 'backward' | 'back';

const moveSelectedIds = (
  orderedIds: readonly string[],
  selected: ReadonlySet<string>,
  kind: LayerStackMoveKind
): string[] => {
  if (kind === 'front' || kind === 'back') {
    const moving = orderedIds.filter((id) => selected.has(id));
    const remaining = orderedIds.filter((id) => !selected.has(id));
    return kind === 'front' ? [...moving, ...remaining] : [...remaining, ...moving];
  }
  const next = [...orderedIds];
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

/**
 * Moves every selected node among its own siblings, selected members keeping their relative order.
 * A selected node whose ancestor is also selected moves with the ancestor. Returns one command per
 * sibling list whose order changes, in document order.
 */
export const moveNodesWithinSiblings = (
  index: CanvasDocumentIndex,
  selectedIds: readonly string[],
  kind: LayerStackMoveKind
): ReorderSiblingsCommand[] => {
  const selected = new Set(selectedIds);
  const parents: { stack: LayerStackKind; parentId: string | null }[] = [];
  const seen = new Set<string>();
  for (const entry of index.nodes) {
    if (!selected.has(entry.node.id) || entry.path.some((ancestor) => selected.has(ancestor))) {
      continue;
    }
    const key = `${entry.stack}\0${entry.parentId ?? ''}`;
    if (!seen.has(key)) {
      seen.add(key);
      parents.push({ parentId: entry.parentId, stack: entry.stack });
    }
  }
  const commands: ReorderSiblingsCommand[] = [];
  for (const { parentId, stack } of parents) {
    const orderedIds = (childrenAt(index, stack, parentId) ?? []).map((node) => node.id);
    const moved = moveSelectedIds(orderedIds, selected, kind);
    if (!sameIds(moved, orderedIds)) {
      commands.push({ orderedIds: moved, parentId, stack });
    }
  }
  return commands;
};

export { isGroupNode };
