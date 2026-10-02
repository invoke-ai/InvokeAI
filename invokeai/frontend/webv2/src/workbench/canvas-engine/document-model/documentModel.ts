import type {
  CanvasDocumentContractV3,
  CanvasGroupContract,
  CanvasLayerContract,
  CanvasNodeContract,
} from '@workbench/canvas-engine/contracts';
import type { CanvasDocumentIndex, CanvasNodeEntry } from '@workbench/canvas-engine/document/documentIndex';
import type {
  CanvasNodeInsertion,
  CanvasNodeInsertionAnchor,
  CanvasNodeMove,
} from '@workbench/canvas-engine/document/insertionAnchors';
import type { LayerStackKind, ReorderSiblingsCommand } from '@workbench/canvas-engine/document/layerStacks';
import type {
  CanvasLayerBasePatch,
  CanvasLayerConfigPatch,
  CanvasProjectMutation,
} from '@workbench/canvas-engine/mutationContracts';

import { CANVAS_MAX_NODE_COUNT, CANVAS_MAX_NODE_DEPTH } from '@workbench/canvas-engine/contracts';
import { childrenAt, getDocumentIndex, outermostNodes } from '@workbench/canvas-engine/document/documentIndex';
import {
  cloneSubtree,
  collectSubtree,
  collectSubtreeLeaves,
  isGroupNode,
  subtreeDepth,
} from '@workbench/canvas-engine/document/documentTree';
import { captureInsertionAnchor, insertNodesAtAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import {
  isHideableLayer,
  isLayerContributing,
  isNodeHidden,
  isPixelBackedLayer,
} from '@workbench/canvas-engine/document/layerEligibility';
import {
  getSiblingOrder,
  isOverlayStack,
  LAYER_STACKS_TOP_FIRST,
  layerStackOf,
  moveNodesWithinSiblings,
  reorderSiblings,
} from '@workbench/canvas-engine/document/layerStacks';
import { repairSelectedLayerId } from '@workbench/canvas-engine/document/selectionRepair';
import { GROUP_PATCH_KEYS, LOCK_EXEMPT_PATCH_KEYS } from '@workbench/canvas-engine/mutationContracts';

import type {
  DocumentCommand,
  DocumentRefusal,
  MergeDownEligibility,
  PreparedDocumentEdit,
  PrepareEditResult,
} from './documentCommands';
import type { EditPostcondition } from './postconditions';
import type { SemanticLeaf } from './semanticLeaf';
import type { SemanticNode } from './semanticNode';

import { isConfigApplied, isPatchApplied, sameValue } from './postconditions';
import { compileSemanticLeaf, isSemanticLeafCurrent } from './semanticLeaf';
import { compileSemanticNode, isSemanticNodeCurrent } from './semanticNode';

export interface DocumentModelContext {
  readonly projectId: string;
  readonly editRevision: number;
}

/**
 * Pure document lookup, tree facts, semantic leaves and prepared edits. Indexes cache by forest identity; leaves
 * retain identity while layer and ancestor-effective state match. Returned sequences define ordering.
 */
export interface CanvasDocumentModel {
  readonly document: CanvasDocumentContractV3;
  getNode(id: string): CanvasNodeContract | null;
  /** The leaf with `id`; `null` for a group or an absent id. */
  getLayer(id: string): CanvasLayerContract | null;
  getEntry(id: string): CanvasNodeEntry | null;
  getStack(kind: LayerStackKind): readonly CanvasNodeContract[];
  compileLeaves(): readonly SemanticLeaf[];
  /** Every node, leaf or group, stacks top first in preorder, with ancestor-effective state applied. */
  compileNodes(): readonly SemanticNode[];
  canMergeDown(upperId: string): MergeDownEligibility;
  prepare(command: DocumentCommand): PrepareEditResult;
  /** Why `command` would be refused, or `null` when it would prepare or change nothing: the one eligibility authority. */
  refusalFor(command: DocumentCommand): DocumentRefusal | null;
}

export interface DocumentModelDiagnostics {
  readonly leafCompilations: number;
  readonly leavesCompiled: number;
  /** Semantic nodes built rather than reused across compilations. */
  readonly nodesCompiled: number;
  /** Prepared edits built with their inverse and postconditions; eligibility checks never add one. */
  readonly editsMaterialized: number;
}

const diagnostics = { editsMaterialized: 0, leafCompilations: 0, leavesCompiled: 0, nodesCompiled: 0 };

/** Immutable snapshot for deterministic budget tests and diagnostics. */
export const getDocumentModelDiagnostics = (): DocumentModelDiagnostics => ({ ...diagnostics });

export const resetDocumentModelDiagnostics = (): void => {
  diagnostics.editsMaterialized = 0;
  diagnostics.leafCompilations = 0;
  diagnostics.leavesCompiled = 0;
  diagnostics.nodesCompiled = 0;
};

const leavesByIndex = new WeakMap<CanvasDocumentIndex, readonly SemanticLeaf[]>();
const leavesByLayer = new WeakMap<CanvasLayerContract, SemanticLeaf>();
const contributingByIndex = new WeakMap<CanvasDocumentIndex, readonly CanvasLayerContract[]>();
const nodesByIndex = new WeakMap<CanvasDocumentIndex, readonly SemanticNode[]>();
const nodeByContract = new WeakMap<CanvasNodeContract, SemanticNode>();
/** Leaf id → position in `index.leaves`; a derived index shares its previous index's map. */
const leafPositionsByIndex = new WeakMap<CanvasDocumentIndex, ReadonlyMap<string, number>>();

const leafPositions = (index: CanvasDocumentIndex): ReadonlyMap<string, number> => {
  const cached = leafPositionsByIndex.get(index);
  if (cached) {
    return cached;
  }
  const positions = index.derivedFrom
    ? leafPositions(index.derivedFrom.previous)
    : new Map(index.leaves.map((leaf, position) => [leaf.id, position]));
  leafPositionsByIndex.set(index, positions);
  return positions;
};

const compileNodes = (index: CanvasDocumentIndex): readonly SemanticNode[] => {
  const cached = nodesByIndex.get(index);
  if (cached) {
    return cached;
  }
  // After a value edit only the replaced entries compile; every other node is the previous one.
  const inherited = index.derivedFrom ? nodesByIndex.get(index.derivedFrom.previous) : undefined;
  if (index.derivedFrom && inherited) {
    const nodes = inherited.slice();
    for (const id of index.derivedFrom.changedIds) {
      const entry = index.byId.get(id)!;
      diagnostics.nodesCompiled += 1;
      const semantic = compileSemanticNode(entry, inherited[entry.order]);
      nodeByContract.set(entry.node, semantic);
      nodes[entry.order] = semantic;
    }
    nodesByIndex.set(index, nodes);
    return nodes;
  }
  const nodes = index.nodes.map((entry) => {
    const previous = nodeByContract.get(entry.node);
    if (previous && isSemanticNodeCurrent(previous, entry)) {
      return previous;
    }
    diagnostics.nodesCompiled += 1;
    const semantic = compileSemanticNode(entry);
    nodeByContract.set(entry.node, semantic);
    return semantic;
  });
  nodesByIndex.set(index, nodes);
  return nodes;
};

const compileLeaves = (index: CanvasDocumentIndex): readonly SemanticLeaf[] => {
  const cached = leavesByIndex.get(index);
  if (cached) {
    return cached;
  }
  diagnostics.leafCompilations += 1;
  const inherited = index.derivedFrom ? leavesByIndex.get(index.derivedFrom.previous) : undefined;
  if (index.derivedFrom && inherited) {
    const leaves = inherited.slice();
    const positions = leafPositions(index);
    for (const id of index.derivedFrom.changedIds) {
      const position = positions.get(id);
      if (position === undefined) {
        continue;
      }
      const entry = index.byId.get(id)!;
      diagnostics.leavesCompiled += 1;
      const leaf = compileSemanticLeaf(entry.node as CanvasLayerContract, entry);
      leavesByLayer.set(entry.node as CanvasLayerContract, leaf);
      leaves[position] = leaf;
    }
    leavesByIndex.set(index, leaves);
    return leaves;
  }
  const leaves: SemanticLeaf[] = [];
  for (const entry of index.nodes) {
    if (isGroupNode(entry.node)) {
      continue;
    }
    const previous = leavesByLayer.get(entry.node);
    if (previous && isSemanticLeafCurrent(previous, entry.node, entry)) {
      leaves.push(previous);
      continue;
    }
    diagnostics.leavesCompiled += 1;
    const leaf = compileSemanticLeaf(entry.node, entry);
    leavesByLayer.set(entry.node, leaf);
    leaves.push(leaf);
  }
  leavesByIndex.set(index, leaves);
  return leaves;
};

const missing = (ids: readonly string[]): DocumentRefusal => ({ ids, status: 'missing' });

type DocumentView = CanvasDocumentContractV3 | null | undefined;

export const lookupDocumentNode = (document: DocumentView, id: string): CanvasNodeContract | null =>
  document ? (getDocumentIndex(document).byId.get(id)?.node ?? null) : null;

/** The leaf with `id`; `null` for a group or an absent id. */
export const lookupDocumentLayer = (document: DocumentView, id: string): CanvasLayerContract | null => {
  const node = lookupDocumentNode(document, id);
  return node && !isGroupNode(node) ? node : null;
};

/** The document's semantic leaves, stacks top first, each in preorder. */
export const compileDocumentLeaves = (document: CanvasDocumentContractV3): readonly SemanticLeaf[] =>
  compileLeaves(getDocumentIndex(document));

/** Every node with ancestor-effective state applied; the same array while the forests are unchanged. */
export const compileDocumentNodes = (document: Pick<CanvasDocumentContractV3, 'stacks'>): readonly SemanticNode[] =>
  compileNodes(getDocumentIndex(document));

/** The semantic node for `id`, or `null` when absent. */
export const lookupDocumentNodeState = (document: DocumentView, id: string): SemanticNode | null => {
  if (!document) {
    return null;
  }
  const index = getDocumentIndex(document);
  const entry = index.byId.get(id);
  return entry ? (compileNodes(index)[entry.order] ?? null) : null;
};

/** The layers of every contributing leaf, in leaf order; the same array while the forests are unchanged. */
export const compileContributingLayers = (document: CanvasDocumentContractV3): readonly CanvasLayerContract[] => {
  const index = getDocumentIndex(document);
  const cached = contributingByIndex.get(index);
  if (cached) {
    return cached;
  }
  const layers = compileLeaves(index)
    .filter((leaf) => leaf.contributionEnabled)
    .map((leaf) => leaf.layer);
  contributingByIndex.set(index, layers);
  return layers;
};

/** The leaf for `id`, or `null` when the document has no such leaf. */
export const lookupDocumentLeaf = (document: DocumentView, id: string): SemanticLeaf | null => {
  if (!document) {
    return null;
  }
  const index = getDocumentIndex(document);
  const position = leafPositions(index).get(id);
  return position === undefined ? null : (compileLeaves(index)[position] ?? null);
};

/** The sibling directly below `id` when it is a leaf; `null` at the bottom, below a group, or when absent. */
export const lookupLayerBelow = (document: DocumentView, id: string): CanvasLayerContract | null => {
  const entry = document ? getDocumentIndex(document).byId.get(id) : undefined;
  if (!document || !entry) {
    return null;
  }
  const below = childrenAt(getDocumentIndex(document), entry.stack, entry.parentId)?.[entry.siblingIndex + 1];
  return below && !isGroupNode(below) ? below : null;
};

/** Merge-down joins a raster leaf with the raster leaf directly below it under the same parent. */
export const mergeDownEligibility = (document: CanvasDocumentContractV3, upperId: string): MergeDownEligibility => {
  const index = getDocumentIndex(document);
  const upper = index.byId.get(upperId);
  if (!upper) {
    return missing([upperId]);
  }
  if (isGroupNode(upper.node) || upper.stack !== 'raster') {
    return { actual: upper.node.type, expected: ['raster'], status: 'wrong-type' };
  }
  const lower = lookupLayerBelow(document, upperId);
  if (!lower) {
    return { reason: 'no-layer-below', status: 'invalid-target', targetId: upperId };
  }
  const pair = [upper.node, lower];
  const unmergeable = pair.find((layer) => layer.type !== 'raster' || !isPixelBackedLayer(layer));
  if (unmergeable) {
    return { reason: 'not-mergeable', status: 'invalid-target', targetId: unmergeable.id };
  }
  const locked = pair.filter((layer) => layer.isLocked).map((layer) => layer.id);
  if (upper.ancestorsLocked) {
    locked.push(...upper.path.filter((id) => index.byId.get(id)!.node.isLocked));
  }
  if (locked.length > 0) {
    return { ids: locked, status: 'locked' };
  }
  const disabled = pair.find((layer) => !isLayerContributing(layer));
  if (disabled || !upper.ancestorsEnabled) {
    return { reason: 'not-mergeable', status: 'invalid-target', targetId: disabled?.id ?? upperId };
  }
  return { lowerId: lower.id, status: 'eligible', upperId };
};

const unique = <T>(values: readonly T[]): T[] => [...new Set(values)];

/**
 * Whether two patches name the same fields, descending only into the partial containers listed in
 * `containers` as dotted paths; every other value is compared as a whole.
 */
const sameKeys = (
  left: Record<string, unknown>,
  right: Record<string, unknown>,
  containers: readonly string[],
  path = ''
): boolean => {
  const leftKeys = Object.keys(left).sort();
  const rightKeys = Object.keys(right).sort();
  return (
    leftKeys.length === rightKeys.length &&
    leftKeys.every((key, position) => key === rightKeys[position]) &&
    leftKeys.every((key) => {
      const keyPath = path ? `${path}.${key}` : key;
      if (!containers.includes(keyPath)) {
        return true;
      }
      const a = left[key];
      const b = right[key];
      return typeof a === 'object' && a !== null && typeof b === 'object' && b !== null
        ? sameKeys(a as Record<string, unknown>, b as Record<string, unknown>, containers, keyPath)
        : a === undefined && b === undefined;
    })
  );
};

const PATCH_CONTAINERS = ['transform'] as const;
const CONFIG_CONTAINERS = ['adapter', 'mask', 'mask.fill'] as const;

const stacksTopFirst = (stacks: readonly LayerStackKind[]): LayerStackKind[] =>
  LAYER_STACKS_TOP_FIRST.filter((stack) => stacks.includes(stack));

const patchInverse = (node: CanvasNodeContract, patch: CanvasLayerBasePatch): CanvasLayerBasePatch => {
  const inverse: Record<string, unknown> = {};
  for (const key of Object.keys(patch) as (keyof CanvasLayerBasePatch)[]) {
    if (key === 'transform') {
      if (isGroupNode(node)) {
        continue;
      }
      const transform: Record<string, number> = {};
      for (const axis of Object.keys(patch.transform ?? {}) as (keyof CanvasLayerContract['transform'])[]) {
        transform[axis] = node.transform[axis];
      }
      inverse.transform = transform;
    } else {
      inverse[key] = (node as unknown as Record<string, unknown>)[key];
    }
  }
  return inverse as CanvasLayerBasePatch;
};

const stackOf = (node: CanvasNodeContract, index: CanvasDocumentIndex): LayerStackKind | null =>
  isGroupNode(node) ? (index.byId.get(node.id)?.stack ?? null) : layerStackOf(node);

/** Levels a node's child lists reach below it: a group always opens one, even while empty. */
const levelsBelow = (node: CanvasNodeContract): number => (isGroupNode(node) ? Math.max(1, subtreeDepth(node)) : 0);

/** Ids of locked nodes along `path`, root first. */
const lockedAlong = (index: CanvasDocumentIndex, path: readonly string[]): string[] =>
  path.filter((id) => index.byId.get(id)!.node.isLocked);

/** Ids of locked nodes among `entry`'s ancestors, itself, and its subtree. */
const lockedWithin = (index: CanvasDocumentIndex, entry: CanvasNodeEntry): string[] => [
  ...lockedAlong(index, entry.path),
  ...collectSubtree(entry.node)
    .filter((node) => node.isLocked)
    .map((node) => node.id),
];

/** Ids of locked nodes among `entry`'s ancestors and itself: the ones that freeze its content. */
const frozenBy = (index: CanvasDocumentIndex, entry: CanvasNodeEntry): string[] =>
  lockedAlong(index, [...entry.path, entry.node.id]);

/** A contiguous run of outermost nodes under one parent, anchored to its unmoved neighbours. */
interface SiblingRun {
  readonly ids: string[];
  readonly anchor: CanvasNodeInsertionAnchor;
}

/**
 * Splits `entries` (outermost, document order) into maximal runs of adjacent siblings, each anchored
 * to the nearest siblings that are not among `entries`. Applying the runs top first puts every
 * node back exactly where it was, whether they are being restored after a removal or a move.
 */
const captureRestoreRuns = (
  index: CanvasDocumentIndex,
  entries: readonly CanvasNodeEntry[],
  context: DocumentModelContext
): SiblingRun[] => {
  const moving = new Set(entries.map((entry) => entry.node.id));
  const runs: SiblingRun[] = [];
  const seenParents = new Set<string>();
  for (const entry of entries) {
    const parentKey = `${entry.stack}\0${entry.parentId ?? ''}`;
    if (seenParents.has(parentKey)) {
      continue;
    }
    seenParents.add(parentKey);
    const siblings = childrenAt(index, entry.stack, entry.parentId) ?? [];
    let run: string[] = [];
    let afterId: string | null = null;
    const flush = (beforeId: string | null): void => {
      if (run.length > 0) {
        runs.push({
          anchor: {
            afterId,
            beforeId,
            capturedEditRevision: context.editRevision,
            parentPath: entry.path,
            projectId: context.projectId,
            stack: entry.stack,
          },
          ids: run,
        });
        run = [];
      }
    };
    for (const sibling of siblings) {
      if (moving.has(sibling.id)) {
        run.push(sibling.id);
      } else {
        flush(sibling.id);
        afterId = sibling.id;
      }
    }
    flush(null);
  }
  return runs;
};

export const createDocumentModel = (
  document: CanvasDocumentContractV3,
  context: DocumentModelContext
): CanvasDocumentModel => {
  const index = getDocumentIndex(document);
  const { stacks, selectedLayerId } = document;

  const lookup = (ids: readonly string[]): { entries: CanvasNodeEntry[] } | DocumentRefusal => {
    const absent = ids.filter((id) => !index.byId.has(id));
    return absent.length > 0 ? missing(absent) : { entries: ids.map((id) => index.byId.get(id)!) };
  };

  const prepared = (
    forward: CanvasProjectMutation,
    inverse: CanvasProjectMutation,
    detail: Partial<Pick<PreparedDocumentEdit, 'createdIds' | 'history'>> &
      Pick<PreparedDocumentEdit, 'postconditions' | 'selectionAfter' | 'touchedIds' | 'touchedStacks'>
  ): PrepareEditResult => {
    diagnostics.editsMaterialized += 1;
    return {
      edit: {
        createdIds: [],
        history: 'record',
        ...detail,
        expectedRevision: context.editRevision,
        forward,
        inverse,
        projectId: context.projectId,
        rasterWork: null,
        selectionBefore: selectedLayerId,
      },
      status: 'prepared',
    };
  };

  const structural = (
    mutation: Omit<Extract<CanvasProjectMutation, { type: 'applyCanvasLayerStackMutation' }>, 'type' | 'enabledUpdates'>
  ): CanvasProjectMutation => ({ enabledUpdates: [], type: 'applyCanvasLayerStackMutation', ...mutation });

  const limits = (
    added: readonly CanvasNodeContract[],
    parentDepth: number,
    alreadyAdded: number
  ): DocumentRefusal | null => {
    const addedNodes = added.flatMap((node) => collectSubtree(node));
    if (index.byId.size + alreadyAdded + addedNodes.length > CANVAS_MAX_NODE_COUNT) {
      return { reason: 'node-limit', status: 'invalid-target', targetId: added[0]?.id ?? '' };
    }
    const deepest = added.reduce((depth, node) => Math.max(depth, levelsBelow(node)), 0);
    if (parentDepth + 1 + deepest > CANVAS_MAX_NODE_DEPTH + 1) {
      return { reason: 'depth-exceeded', status: 'invalid-target', targetId: added[0]?.id ?? '' };
    }
    return null;
  };

  const siblingOrderOf = (stack: LayerStackKind, parentId: string | null): ReorderSiblingsCommand => ({
    orderedIds: (childrenAt(index, stack, parentId) ?? []).map((node) => node.id),
    parentId,
    stack,
  });

  const siblingOrderAfter = (
    next: CanvasDocumentContractV3['stacks'],
    stack: LayerStackKind,
    parentId: string | null
  ): EditPostcondition => ({ kind: 'sibling-order', ...getSiblingOrder(next, stack, parentId) });

  const prepareInsert = (command: Extract<DocumentCommand, { type: 'insert' }>): PrepareEditResult => {
    if (command.nodes.length === 0) {
      return { operation: 'insert nothing', status: 'unsupported' };
    }
    const ids = command.nodes.flatMap((node) => collectSubtree(node).map((entry) => entry.id));
    const clash = ids.find((id, at) => index.byId.has(id) || ids.indexOf(id) !== at);
    if (clash !== undefined) {
      return { reason: 'id-exists', status: 'invalid-target', targetId: clash };
    }
    const rootIds = command.nodes.map((node) => node.id);
    const selectionAfter = command.selectId === undefined ? rootIds.at(-1)! : command.selectId;
    if (selectionAfter !== null && !ids.includes(selectionAfter) && !index.byId.has(selectionAfter)) {
      return missing([selectionAfter]);
    }
    const stackFor = (node: CanvasNodeContract): LayerStackKind | null =>
      isGroupNode(node)
        ? collectSubtreeLeaves(node)[0]
          ? layerStackOf(collectSubtreeLeaves(node)[0]!)
          : (command.stack ?? null)
        : layerStackOf(node);
    const byStack = new Map<LayerStackKind, CanvasNodeContract[]>();
    for (const node of command.nodes) {
      const stack = stackFor(node);
      if (stack === null) {
        return { operation: 'insert an empty group without a stack', status: 'unsupported' };
      }
      if (isGroupNode(node) && collectSubtreeLeaves(node).some((leaf) => layerStackOf(leaf) !== stack)) {
        return { reason: 'foreign-stack', status: 'invalid-target', targetId: node.id };
      }
      if (
        stack === 'raster' &&
        collectSubtree(node).some((member) => isGroupNode(member) && member.isHidden !== undefined)
      ) {
        return { operation: 'insert a display-hidden group into the raster stack', status: 'unsupported' };
      }
      byStack.set(stack, [...(byStack.get(stack) ?? []), node]);
    }
    const touchedStacks = stacksTopFirst([...byStack.keys()]);
    const add: CanvasNodeInsertion[] = [];
    let expected = stacks;
    let added = 0;
    for (const stack of touchedStacks) {
      const nodes = byStack.get(stack)!;
      const anchor = captureInsertionAnchor(stacks, {
        aboveId: command.aboveId,
        editRevision: context.editRevision,
        insideId: command.insideId,
        projectId: context.projectId,
        stack,
      });
      const locked = lockedAlong(index, anchor.parentPath);
      if (locked.length > 0) {
        return { ids: locked, status: 'locked' };
      }
      const refusal = limits(nodes, anchor.parentPath.length, added);
      if (refusal) {
        return refusal;
      }
      added += nodes.reduce((count, node) => count + collectSubtree(node).length, 0);
      add.push({ anchor, nodes });
      expected = insertNodesAtAnchor(expected, anchor, nodes);
    }
    const expectedIndex = getDocumentIndex({ stacks: expected });
    return prepared(
      structural({ add, selectedLayerId: selectionAfter }),
      structural({ removeIds: rootIds, selectedLayerId }),
      {
        createdIds: ids,
        postconditions: [
          ...add.map((insertion) => {
            const entry = expectedIndex.byId.get(insertion.nodes[0]!.id)!;
            return siblingOrderAfter(expected, entry.stack, entry.parentId);
          }),
          { id: selectionAfter, kind: 'selection' },
        ],
        selectionAfter,
        touchedIds: ids,
        touchedStacks,
      }
    );
  };

  /** Everything `remove` refuses, decided without building the edit. */
  const checkRemove = (
    command: Extract<DocumentCommand, { type: 'remove' }>
  ): DocumentRefusal | { outer: CanvasNodeEntry[] } => {
    const ids = unique(command.ids);
    if (ids.length === 0) {
      return { operation: 'remove nothing', status: 'unsupported' };
    }
    const found = lookup(ids);
    if ('status' in found) {
      return found;
    }
    const outer = outermostNodes(index, ids);
    const locked = unique(outer.flatMap((entry) => lockedWithin(index, entry)));
    if (locked.length > 0) {
      return { ids: locked, status: 'locked' };
    }
    return { outer };
  };

  const prepareRemove = (command: Extract<DocumentCommand, { type: 'remove' }>): PrepareEditResult => {
    const checked = checkRemove(command);
    if ('status' in checked) {
      return checked;
    }
    const { outer } = checked;
    const removedIds = outer.flatMap((entry) => collectSubtree(entry.node).map((node) => node.id));
    const removedSet = new Set(removedIds);
    const remaining = { ...stacks };
    for (const stack of new Set(outer.map((entry) => entry.stack))) {
      remaining[stack] = stacks[stack]
        .filter((node) => !removedSet.has(node.id))
        .map((node) => stripRemoved(node, removedSet));
    }
    const selectionAfter = repairSelectedLayerId(remaining, selectedLayerId, stacks);
    const touchedStacks = stacksTopFirst(outer.map((entry) => entry.stack));
    const add: CanvasNodeInsertion[] = captureRestoreRuns(index, outer, context).map((run) => ({
      anchor: run.anchor,
      nodes: run.ids.map((id) => index.byId.get(id)!.node),
    }));
    return prepared(
      { ids: outer.map((entry) => entry.node.id), type: 'removeCanvasLayers' },
      structural({ add, selectedLayerId }),
      {
        postconditions: [
          { ids: removedIds, kind: 'absent' },
          ...unique(outer.map((entry) => `${entry.stack}\0${entry.parentId ?? ''}`)).map((key) => {
            const [stack, parentId] = key.split('\0') as [LayerStackKind, string];
            return siblingOrderAfter(remaining, stack, parentId === '' ? null : parentId);
          }),
          { id: selectionAfter, kind: 'selection' },
        ],
        selectionAfter,
        touchedIds: removedIds,
        touchedStacks,
      }
    );
  };

  const prepareDuplicate = (command: Extract<DocumentCommand, { type: 'duplicate' }>): PrepareEditResult => {
    const ids = unique(command.ids);
    if (ids.length === 0) {
      return { operation: 'duplicate nothing', status: 'unsupported' };
    }
    const found = lookup(ids);
    if ('status' in found) {
      return found;
    }
    const outer = outermostNodes(index, ids);
    const locked = unique(outer.flatMap((entry) => lockedWithin(index, entry)));
    if (locked.length > 0) {
      return { ids: locked, status: 'locked' };
    }
    // Every refusal is decided before an id is minted, so asking is free of side effects.
    let budgeted = 0;
    for (const entry of outer) {
      const refusal = limits([entry.node], entry.path.length, budgeted);
      if (refusal) {
        return refusal;
      }
      budgeted += collectSubtree(entry.node).length;
    }
    const add: CanvasNodeInsertion[] = [];
    const createdIds: string[] = [];
    const rootIds: string[] = [];
    let expected = stacks;
    for (const entry of outer) {
      const { node } = cloneSubtree(entry.node, command.createId);
      const clone = { ...node, name: `${node.name} copy` };
      const clash = collectSubtree(clone).find((created) => index.byId.has(created.id));
      if (clash) {
        return { reason: 'id-exists', status: 'invalid-target', targetId: clash.id };
      }
      const anchor = captureInsertionAnchor(expected, {
        aboveId: entry.node.id,
        editRevision: context.editRevision,
        projectId: context.projectId,
        stack: entry.stack,
      });
      add.push({ anchor, nodes: [clone] });
      expected = insertNodesAtAnchor(expected, anchor, [clone]);
      createdIds.push(...collectSubtree(clone).map((created) => created.id));
      rootIds.push(clone.id);
    }
    const selectionAfter = rootIds.at(-1)!;
    return prepared(
      structural({ add, selectedLayerId: selectionAfter }),
      structural({ removeIds: rootIds, selectedLayerId }),
      {
        createdIds,
        postconditions: [
          ...outer.map((entry) => siblingOrderAfter(expected, entry.stack, entry.parentId)),
          { id: selectionAfter, kind: 'selection' },
        ],
        selectionAfter,
        touchedIds: createdIds,
        touchedStacks: stacksTopFirst(outer.map((entry) => entry.stack)),
      }
    );
  };

  /** `movingIds` names the nodes a move selected; a raw reorder treats every displaced node as moved. */
  const prepareReorder = (
    orders: readonly ReorderSiblingsCommand[],
    movingIds?: ReadonlySet<string>
  ): PrepareEditResult => {
    if (orders.length === 0) {
      return { operation: 'reorder nothing', status: 'unsupported' };
    }
    if (new Set(orders.map((order) => `${order.stack}\0${order.parentId ?? ''}`)).size !== orders.length) {
      return { operation: 'reorder one sibling list twice', status: 'unsupported' };
    }
    const touchedIds: string[] = [];
    let expected = stacks;
    for (const order of orders) {
      const unknown = order.orderedIds.filter((id) => !index.byId.has(id));
      if (unknown.length > 0) {
        return missing(unknown);
      }
      if (order.parentId !== null) {
        const parent = index.byId.get(order.parentId);
        if (!parent) {
          return missing([order.parentId]);
        }
        if (!isGroupNode(parent.node)) {
          return { reason: 'not-a-group', status: 'invalid-target', targetId: order.parentId };
        }
        if (parent.stack !== order.stack) {
          return { reason: 'foreign-stack', status: 'invalid-target', targetId: order.parentId };
        }
        const locked = frozenBy(index, parent);
        if (locked.length > 0) {
          return { ids: locked, status: 'locked' };
        }
      }
      const current = siblingOrderOf(order.stack, order.parentId).orderedIds;
      const foreign = order.orderedIds.find((id) => {
        const entry = index.byId.get(id)!;
        return entry.stack !== order.stack || entry.parentId !== order.parentId;
      });
      if (foreign !== undefined) {
        return { reason: 'not-siblings', status: 'invalid-target', targetId: foreign };
      }
      const next = reorderSiblings(expected, order);
      if (!next) {
        return { reason: 'not-siblings', status: 'invalid-target', targetId: order.orderedIds[0] ?? current[0]! };
      }
      expected = next;
      touchedIds.push(...order.orderedIds.filter((id, position) => current[position] !== id));
    }
    if (touchedIds.length === 0) {
      return { status: 'unchanged' };
    }
    const moving = movingIds ? touchedIds.filter((id) => movingIds.has(id)) : touchedIds;
    const lockedMoving = unique(moving.flatMap((id) => lockedWithin(index, index.byId.get(id)!)));
    if (lockedMoving.length > 0) {
      return { ids: lockedMoving, status: 'locked' };
    }
    return prepared(
      {
        orders: orders.map((order) => ({ ...order, orderedIds: [...order.orderedIds] })),
        type: 'reorderCanvasSiblings',
      },
      {
        orders: orders.map((order) => siblingOrderOf(order.stack, order.parentId)),
        type: 'reorderCanvasSiblings',
      },
      {
        postconditions: orders.map((order) => ({ kind: 'sibling-order', ...order })),
        selectionAfter: selectedLayerId,
        touchedIds,
        touchedStacks: stacksTopFirst(orders.map((order) => order.stack)),
      }
    );
  };

  const prepareMove = (command: Extract<DocumentCommand, { type: 'move' }>): PrepareEditResult => {
    const ids = unique(command.ids);
    if (ids.length === 0) {
      return { operation: 'move nothing', status: 'unsupported' };
    }
    const found = lookup(ids);
    if ('status' in found) {
      return found;
    }
    const orders = moveNodesWithinSiblings(index, ids, command.kind);
    return orders.length === 0 ? { status: 'unchanged' } : prepareReorder(orders, new Set(ids));
  };

  interface ReparentPlacement {
    outer: CanvasNodeEntry[];
    parent: CanvasNodeEntry | null;
    stack: LayerStackKind;
    moving: Set<string>;
    targetChildren: CanvasNodeContract[];
    position: number;
    orderedIds: string[];
  }

  /** Everything `reparent` refuses, and where the block would land, decided without building the edit. */
  const checkReparent = (
    command: Extract<DocumentCommand, { type: 'reparent' }>
  ): DocumentRefusal | { status: 'unchanged' } | ReparentPlacement => {
    const ids = unique(command.ids);
    if (ids.length === 0) {
      return { operation: 'reparent nothing', status: 'unsupported' };
    }
    const found = lookup(ids);
    if ('status' in found) {
      return found;
    }
    const outer = outermostNodes(index, ids);
    const stack = outer[0]!.stack;
    const foreign = outer.find((entry) => entry.stack !== stack);
    if (foreign) {
      return { reason: 'foreign-stack', status: 'invalid-target', targetId: foreign.node.id };
    }
    const parent = command.parentId === null ? null : index.byId.get(command.parentId);
    if (parent === undefined) {
      return missing([command.parentId!]);
    }
    if (parent && !isGroupNode(parent.node)) {
      return { reason: 'not-a-group', status: 'invalid-target', targetId: parent.node.id };
    }
    if (parent && parent.stack !== stack) {
      return { reason: 'foreign-stack', status: 'invalid-target', targetId: parent.node.id };
    }
    const moving = new Set(outer.map((entry) => entry.node.id));
    if (parent && (moving.has(parent.node.id) || parent.path.some((id) => moving.has(id)))) {
      return { reason: 'cycle', status: 'invalid-target', targetId: parent.node.id };
    }
    const locked = unique([
      ...outer.flatMap((entry) => lockedWithin(index, entry)),
      ...(parent ? frozenBy(index, parent) : []),
    ]);
    if (locked.length > 0) {
      return { ids: locked, status: 'locked' };
    }
    const parentDepth = parent ? parent.path.length + 1 : 0;
    const deepest = outer.reduce((depth, entry) => Math.max(depth, levelsBelow(entry.node)), 0);
    if (parentDepth + deepest > CANVAS_MAX_NODE_DEPTH) {
      return { reason: 'depth-exceeded', status: 'invalid-target', targetId: parent?.node.id ?? outer[0]!.node.id };
    }
    const targetChildren = (childrenAt(index, stack, command.parentId) ?? []).filter((node) => !moving.has(node.id));
    let position = targetChildren.length;
    if (command.beforeId !== null) {
      const before = index.byId.get(command.beforeId);
      if (!before) {
        return missing([command.beforeId]);
      }
      if (before.stack !== stack || before.parentId !== command.parentId || moving.has(before.node.id)) {
        return { reason: 'not-siblings', status: 'invalid-target', targetId: command.beforeId };
      }
      position = targetChildren.findIndex((node) => node.id === command.beforeId);
    }
    const orderedIds = [
      ...targetChildren.slice(0, position).map((node) => node.id),
      ...outer.map((entry) => entry.node.id),
      ...targetChildren.slice(position).map((node) => node.id),
    ];
    if (sameValue(orderedIds, siblingOrderOf(stack, command.parentId).orderedIds)) {
      return { status: 'unchanged' };
    }
    return { moving, orderedIds, outer, parent, position, stack, targetChildren };
  };

  const prepareReparent = (command: Extract<DocumentCommand, { type: 'reparent' }>): PrepareEditResult => {
    const checked = checkReparent(command);
    if ('status' in checked) {
      return checked;
    }
    const { moving, orderedIds, outer, parent, position, stack, targetChildren } = checked;
    const anchor: CanvasNodeInsertionAnchor = {
      afterId: targetChildren[position - 1]?.id ?? null,
      beforeId: targetChildren[position]?.id ?? null,
      capturedEditRevision: context.editRevision,
      parentPath: parent ? [...parent.path, parent.node.id] : [],
      projectId: context.projectId,
      stack,
    };
    const inverseMoves: CanvasNodeMove[] = captureRestoreRuns(index, outer, context);
    const touchedParents = unique([
      `${stack}\0${command.parentId ?? ''}`,
      ...outer.map((entry) => `${entry.stack}\0${entry.parentId ?? ''}`),
    ]);
    return prepared(
      structural({ move: [{ anchor, ids: outer.map((entry) => entry.node.id) }] }),
      structural({ move: inverseMoves }),
      {
        postconditions: [
          { kind: 'sibling-order', orderedIds, parentId: command.parentId, stack },
          ...touchedParents
            .filter((key) => key !== `${stack}\0${command.parentId ?? ''}`)
            .map((key) => {
              const parentId = key.slice(stack.length + 1) || null;
              return {
                kind: 'sibling-order' as const,
                orderedIds: siblingOrderOf(stack, parentId).orderedIds.filter((id) => !moving.has(id)),
                parentId,
                stack,
              };
            }),
        ],
        selectionAfter: selectedLayerId,
        touchedIds: unique([
          ...orderedIds,
          ...outer.flatMap((entry) => collectSubtree(entry.node).map((node) => node.id)),
        ]),
        touchedStacks: [stack],
      }
    );
  };

  /** Everything `group` refuses, decided without building the edit. */
  const checkGroup = (
    command: Extract<DocumentCommand, { type: 'group' }>
  ): DocumentRefusal | { outer: CanvasNodeEntry[]; first: CanvasNodeEntry } => {
    const ids = unique(command.ids);
    if (ids.length === 0) {
      return { operation: 'group nothing', status: 'unsupported' };
    }
    if (index.byId.has(command.groupId)) {
      return { reason: 'id-exists', status: 'invalid-target', targetId: command.groupId };
    }
    const found = lookup(ids);
    if ('status' in found) {
      return found;
    }
    const outer = outermostNodes(index, ids);
    const first = outer[0]!;
    const stranger = outer.find((entry) => entry.stack !== first.stack || entry.parentId !== first.parentId);
    if (stranger) {
      return { reason: 'not-siblings', status: 'invalid-target', targetId: stranger.node.id };
    }
    const locked = unique([...lockedAlong(index, first.path), ...outer.flatMap((entry) => lockedWithin(index, entry))]);
    if (locked.length > 0) {
      return { ids: locked, status: 'locked' };
    }
    const deepest = outer.reduce((depth, entry) => Math.max(depth, levelsBelow(entry.node)), 0);
    if (first.path.length + 1 + deepest > CANVAS_MAX_NODE_DEPTH) {
      return { reason: 'depth-exceeded', status: 'invalid-target', targetId: first.node.id };
    }
    if (index.byId.size + 1 > CANVAS_MAX_NODE_COUNT) {
      return { reason: 'node-limit', status: 'invalid-target', targetId: command.groupId };
    }
    return { first, outer };
  };

  const prepareGroup = (command: Extract<DocumentCommand, { type: 'group' }>): PrepareEditResult => {
    const checked = checkGroup(command);
    if ('status' in checked) {
      return checked;
    }
    const { first, outer } = checked;
    const group: CanvasGroupContract = {
      children: [],
      id: command.groupId,
      isEnabled: true,
      isLocked: false,
      name: command.name,
      type: 'group',
    };
    const groupAnchor = captureInsertionAnchor(stacks, {
      aboveId: first.node.id,
      editRevision: context.editRevision,
      projectId: context.projectId,
      stack: first.stack,
    });
    const inside: CanvasNodeInsertionAnchor = {
      afterId: null,
      beforeId: null,
      capturedEditRevision: context.editRevision,
      parentPath: [...first.path, command.groupId],
      projectId: context.projectId,
      stack: first.stack,
    };
    const movedIds = outer.map((entry) => entry.node.id);
    const moving = new Set(movedIds);
    const siblingsAfter = siblingOrderOf(first.stack, first.parentId).orderedIds.flatMap((id) =>
      id === first.node.id ? [command.groupId] : moving.has(id) ? [] : [id]
    );
    return prepared(
      structural({
        add: [{ anchor: groupAnchor, nodes: [group] }],
        move: [{ anchor: inside, ids: movedIds }],
        selectedLayerId: command.groupId,
      }),
      structural({ move: captureRestoreRuns(index, outer, context), removeIds: [command.groupId], selectedLayerId }),
      {
        createdIds: [command.groupId],
        postconditions: [
          { kind: 'sibling-order', orderedIds: siblingsAfter, parentId: first.parentId, stack: first.stack },
          { kind: 'sibling-order', orderedIds: movedIds, parentId: command.groupId, stack: first.stack },
          { id: command.groupId, kind: 'selection' },
        ],
        selectionAfter: command.groupId,
        touchedIds: [command.groupId, ...movedIds],
        touchedStacks: [first.stack],
      }
    );
  };

  /** Everything `ungroup` refuses, decided without building the edit. */
  const checkUngroup = (
    command: Extract<DocumentCommand, { type: 'ungroup' }>
  ): DocumentRefusal | { groups: CanvasNodeEntry[] } => {
    const ids = unique(command.ids);
    if (ids.length === 0) {
      return { operation: 'ungroup nothing', status: 'unsupported' };
    }
    const found = lookup(ids);
    if ('status' in found) {
      return found;
    }
    const notGroup = found.entries.find((entry) => !isGroupNode(entry.node));
    if (notGroup) {
      return { actual: notGroup.node.type, expected: ['group'], status: 'wrong-type' };
    }
    const named = new Set(ids);
    const groups = index.nodes.filter((entry) => named.has(entry.node.id));
    const locked = unique(groups.flatMap((entry) => [...frozenBy(index, entry), ...lockedWithin(index, entry)]));
    if (locked.length > 0) {
      return { ids: locked, status: 'locked' };
    }
    return { groups };
  };

  const prepareUngroup = (command: Extract<DocumentCommand, { type: 'ungroup' }>): PrepareEditResult => {
    const checked = checkUngroup(command);
    if ('status' in checked) {
      return checked;
    }
    const { groups } = checked;
    const forward: CanvasNodeMove[] = [];
    const inverseAdd: CanvasNodeInsertion[] = [];
    const inverseMove: CanvasNodeMove[] = [];
    const postconditions: EditPostcondition[] = [];
    const touchedIds: string[] = [];
    let expected = stacks;
    const dissolved = new Set(groups.map((entry) => entry.node.id));
    for (const entry of groups) {
      const group = entry.node as CanvasGroupContract;
      const childIds = group.children.map((child) => child.id);
      const siblings = childrenAt(index, entry.stack, entry.parentId) ?? [];
      const base = {
        capturedEditRevision: context.editRevision,
        parentPath: entry.path,
        projectId: context.projectId,
        stack: entry.stack,
      };
      forward.push({
        anchor: { ...base, afterId: siblings[entry.siblingIndex - 1]?.id ?? null, beforeId: group.id },
        ids: childIds,
      });
      inverseAdd.push({
        anchor: {
          ...base,
          afterId: siblings[entry.siblingIndex - 1]?.id ?? null,
          beforeId: siblings[entry.siblingIndex + 1]?.id ?? null,
        },
        nodes: [{ ...group, children: [] }],
      });
      inverseMove.push({
        anchor: { ...base, afterId: null, beforeId: null, parentPath: [...entry.path, group.id] },
        ids: childIds,
      });
      touchedIds.push(group.id, ...childIds);
      expected = {
        ...expected,
        [entry.stack]: dissolve(expected[entry.stack], dissolved),
      };
    }
    for (const key of unique(
      groups
        .filter((entry) => entry.parentId === null || !dissolved.has(entry.parentId))
        .map((entry) => `${entry.stack}\0${entry.parentId ?? ''}`)
    )) {
      const [stack, parentId] = key.split('\0') as [LayerStackKind, string];
      postconditions.push(siblingOrderAfter(expected, stack, parentId === '' ? null : parentId));
    }
    const selectionAfter =
      selectedLayerId !== null && dissolved.has(selectedLayerId)
        ? (collectSubtree(index.byId.get(selectedLayerId)!.node).find((node) => !dissolved.has(node.id))?.id ??
          repairSelectedLayerId(expected, selectedLayerId, stacks))
        : selectedLayerId;
    postconditions.push({ id: selectionAfter, kind: 'selection' });
    return prepared(
      structural({ move: forward, removeIds: groups.map((entry) => entry.node.id), selectedLayerId: selectionAfter }),
      structural({ add: inverseAdd, move: inverseMove, selectedLayerId }),
      {
        postconditions,
        selectionAfter,
        touchedIds,
        touchedStacks: stacksTopFirst(groups.map((entry) => entry.stack)),
      }
    );
  };

  const preparePatch = (command: Extract<DocumentCommand, { type: 'patch' }>): PrepareEditResult => {
    const entry = index.byId.get(command.id);
    if (!entry) {
      return missing([command.id]);
    }
    if (Object.keys(command.patch).length === 0) {
      return { operation: 'patch nothing', status: 'unsupported' };
    }
    if (isGroupNode(entry.node)) {
      const unsupported = (Object.keys(command.patch) as (keyof CanvasLayerBasePatch)[]).find(
        (key) => !GROUP_PATCH_KEYS.includes(key)
      );
      if (unsupported) {
        return {
          actual: 'group',
          expected: ['raster', 'control', 'regional_guidance', 'inpaint_mask'],
          status: 'wrong-type',
        };
      }
      // Opacity/blend describe isolated color composites, so they are invalid for coverage-only overlay groups.
      if ((command.patch.opacity !== undefined || command.patch.blendMode !== undefined) && entry.stack !== 'raster') {
        return { operation: 'blend an overlay-stack group', status: 'unsupported' };
      }
    }
    if (command.before && !sameKeys(command.before, command.patch, PATCH_CONTAINERS)) {
      return { operation: 'patch baseline names other fields', status: 'unsupported' };
    }
    if (
      (Object.keys(command.patch) as (keyof CanvasLayerBasePatch)[]).some(
        (key) => !LOCK_EXEMPT_PATCH_KEYS.includes(key)
      )
    ) {
      const locked = frozenBy(index, entry);
      if (locked.length > 0) {
        return { ids: locked, status: 'locked' };
      }
    }
    const inverse = command.before ?? patchInverse(entry.node, command.patch);
    if (command.before ? sameValue(command.before, command.patch) : isPatchApplied(entry.node, command.patch)) {
      return { status: 'unchanged' };
    }
    return prepared(
      { id: command.id, patch: command.patch, type: 'updateCanvasLayer' },
      { id: command.id, patch: inverse, type: 'updateCanvasLayer' },
      {
        postconditions: [{ id: command.id, kind: 'patched', patch: command.patch }],
        selectionAfter: selectedLayerId,
        touchedIds: [command.id],
        touchedStacks: [entry.stack],
      }
    );
  };

  const configInverse = (layer: CanvasNodeContract, config: CanvasLayerConfigPatch): CanvasLayerConfigPatch => {
    const current = layer as unknown as Record<string, unknown>;
    const inverse: Record<string, unknown> = { layerType: config.layerType };
    for (const [key, value] of Object.entries(config)) {
      if (key === 'layerType') {
        continue;
      }
      const before = current[key];
      inverse[key] =
        (key === 'adapter' || key === 'mask') &&
        typeof value === 'object' &&
        value !== null &&
        typeof before === 'object'
          ? Object.fromEntries(Object.keys(value).map((field) => [field, (before as Record<string, unknown>)[field]]))
          : before;
    }
    return inverse as CanvasLayerConfigPatch;
  };

  /** The unlocked leaf `id` names; content edits are refused inside a locked subtree. */
  const editableLeaf = (id: string): { entry: CanvasNodeEntry; layer: CanvasLayerContract } | DocumentRefusal => {
    const entry = index.byId.get(id);
    if (!entry) {
      return missing([id]);
    }
    if (isGroupNode(entry.node)) {
      return {
        actual: 'group',
        expected: ['raster', 'control', 'regional_guidance', 'inpaint_mask'],
        status: 'wrong-type',
      };
    }
    const locked = frozenBy(index, entry);
    if (locked.length > 0) {
      return { ids: locked, status: 'locked' };
    }
    return { entry, layer: entry.node };
  };

  /** The unlocked config-patch target: a leaf, or a RASTER-stack group through the 'group' arm. */
  const editableConfigNode = (
    id: string,
    layerType: CanvasLayerConfigPatch['layerType']
  ): { entry: CanvasNodeEntry; node: CanvasNodeContract } | DocumentRefusal => {
    if (layerType !== 'group') {
      const found = editableLeaf(id);
      return 'status' in found ? found : { entry: found.entry, node: found.layer };
    }
    const entry = index.byId.get(id);
    if (!entry) {
      return missing([id]);
    }
    if (!isGroupNode(entry.node)) {
      return { actual: entry.node.type, expected: ['group'], status: 'wrong-type' };
    }
    if (entry.stack !== 'raster') {
      return { operation: 'adjust an overlay-stack group', status: 'unsupported' };
    }
    const locked = frozenBy(index, entry);
    if (locked.length > 0) {
      return { ids: locked, status: 'locked' };
    }
    return { entry, node: entry.node };
  };

  const preparePatchConfig = (command: Extract<DocumentCommand, { type: 'patch-config' }>): PrepareEditResult => {
    const found = editableConfigNode(command.id, command.config.layerType);
    if ('status' in found) {
      return found;
    }
    const { entry, node: layer } = found;
    if (layer.type !== command.config.layerType) {
      return { actual: layer.type, expected: [command.config.layerType], status: 'wrong-type' };
    }
    if (Object.keys(command.config).length <= 1) {
      return { operation: 'patch nothing', status: 'unsupported' };
    }
    if (command.before && command.before.layerType !== command.config.layerType) {
      return { operation: 'config baseline names another layer type', status: 'unsupported' };
    }
    if (command.before && !sameKeys(command.before, command.config, CONFIG_CONTAINERS)) {
      return { operation: 'config baseline names other fields', status: 'unsupported' };
    }
    if (command.before ? sameValue(command.before, command.config) : isConfigApplied(layer, command.config)) {
      return { status: 'unchanged' };
    }
    return prepared(
      { config: command.config, id: command.id, type: 'updateCanvasLayerConfig' },
      {
        config: command.before ?? configInverse(layer, command.config),
        id: command.id,
        type: 'updateCanvasLayerConfig',
      },
      {
        postconditions: [{ config: command.config, id: command.id, kind: 'config' }],
        selectionAfter: selectedLayerId,
        touchedIds: [command.id],
        touchedStacks: [entry.stack],
      }
    );
  };

  const preparePatchConfigBatch = (
    command: Extract<DocumentCommand, { type: 'patch-config-batch' }>
  ): PrepareEditResult => {
    if (command.patches.length === 0) {
      return { operation: 'patch nothing', status: 'unsupported' };
    }
    if (unique(command.patches.map((patch) => patch.id)).length !== command.patches.length) {
      return { operation: 'batch patches one layer twice', status: 'unsupported' };
    }
    const forward: { id: string; config: CanvasLayerConfigPatch }[] = [];
    const inverse: { id: string; config: CanvasLayerConfigPatch }[] = [];
    const postconditions: EditPostcondition[] = [];
    const touchedStacks = new Set<LayerStackKind>();
    let changed = false;
    for (const patch of command.patches) {
      const found = editableConfigNode(patch.id, patch.config.layerType);
      if ('status' in found) {
        return found;
      }
      const { entry, node: layer } = found;
      if (layer.type !== patch.config.layerType) {
        return { actual: layer.type, expected: [patch.config.layerType], status: 'wrong-type' };
      }
      if (Object.keys(patch.config).length <= 1) {
        return { operation: 'patch nothing', status: 'unsupported' };
      }
      if (patch.before && patch.before.layerType !== patch.config.layerType) {
        return { operation: 'config baseline names another layer type', status: 'unsupported' };
      }
      if (patch.before && !sameKeys(patch.before, patch.config, CONFIG_CONTAINERS)) {
        return { operation: 'config baseline names other fields', status: 'unsupported' };
      }
      changed ||= patch.before ? !sameValue(patch.before, patch.config) : !isConfigApplied(layer, patch.config);
      forward.push({ config: patch.config, id: patch.id });
      inverse.push({ config: patch.before ?? configInverse(layer, patch.config), id: patch.id });
      postconditions.push({ config: patch.config, id: patch.id, kind: 'config' });
      touchedStacks.add(entry.stack);
    }
    if (!changed) {
      return { status: 'unchanged' };
    }
    return prepared(
      { type: 'updateCanvasLayerConfigs', updates: forward },
      { type: 'updateCanvasLayerConfigs', updates: [...inverse].reverse() },
      {
        postconditions,
        selectionAfter: selectedLayerId,
        touchedIds: command.patches.map((patch) => patch.id),
        touchedStacks: [...touchedStacks],
      }
    );
  };

  const preparePatchSource = (command: Extract<DocumentCommand, { type: 'patch-source' }>): PrepareEditResult => {
    const found = editableLeaf(command.id);
    if ('status' in found) {
      return found;
    }
    const { entry, layer } = found;
    if (layer.type !== 'raster' && layer.type !== 'control') {
      return { actual: layer.type, expected: ['raster', 'control'], status: 'wrong-type' };
    }
    if (layer.source === command.source) {
      return { status: 'unchanged' };
    }
    return prepared(
      { id: command.id, source: command.source, type: 'updateCanvasLayerSource' },
      { id: command.id, source: layer.source, type: 'updateCanvasLayerSource' },
      {
        postconditions: [{ id: command.id, kind: 'source', source: command.source }],
        selectionAfter: selectedLayerId,
        touchedIds: [command.id],
        touchedStacks: [entry.stack],
      }
    );
  };

  const prepareFlags = (
    command: Extract<DocumentCommand, { type: 'set-enabled' | 'set-hidden' | 'set-locked' }>
  ): PrepareEditResult => {
    const ids = unique(command.updates.map((update) => update.id));
    if (ids.length === 0) {
      return { operation: `${command.type} nothing`, status: 'unsupported' };
    }
    const found = lookup(ids);
    if ('status' in found) {
      return found;
    }
    if (command.type === 'set-hidden') {
      const notHideable = found.entries.find((entry) =>
        isGroupNode(entry.node) ? !isOverlayStack(entry.stack) : !isHideableLayer(entry.node)
      );
      if (notHideable) {
        return {
          actual: notHideable.node.type,
          expected: ['control', 'inpaint_mask', 'regional_guidance'],
          status: 'wrong-type',
        };
      }
    }
    const detail = (updated: readonly { id: string }[]) => ({
      selectionAfter: selectedLayerId,
      touchedIds: updated.map((update) => update.id),
      touchedStacks: stacksTopFirst(updated.map((update) => index.byId.get(update.id)!.stack)),
    });
    switch (command.type) {
      case 'set-enabled': {
        const updates = command.updates.filter(
          (update) => index.byId.get(update.id)!.node.isEnabled !== update.isEnabled
        );
        if (updates.length === 0) {
          return { status: 'unchanged' };
        }
        return prepared(
          { type: 'setCanvasLayersEnabled', updates },
          {
            type: 'setCanvasLayersEnabled',
            updates: updates.map((update) => ({ id: update.id, isEnabled: index.byId.get(update.id)!.node.isEnabled })),
          },
          {
            ...detail(updates),
            postconditions: updates.map((update) => ({
              id: update.id,
              kind: 'patched',
              patch: { isEnabled: update.isEnabled },
            })),
          }
        );
      }
      case 'set-hidden': {
        const updates = command.updates.filter(
          (update) => isNodeHidden(index.byId.get(update.id)!.node) !== update.isHidden
        );
        if (updates.length === 0) {
          return { status: 'unchanged' };
        }
        return prepared(
          { type: 'setCanvasLayersHidden', updates },
          {
            type: 'setCanvasLayersHidden',
            updates: updates.map((update) => ({
              id: update.id,
              isHidden: isNodeHidden(index.byId.get(update.id)!.node),
            })),
          },
          {
            ...detail(updates),
            postconditions: updates.map((update) => ({ id: update.id, isHidden: update.isHidden, kind: 'hidden' })),
          }
        );
      }
      case 'set-locked': {
        const updates = command.updates.filter(
          (update) => index.byId.get(update.id)!.node.isLocked !== update.isLocked
        );
        if (updates.length === 0) {
          return { status: 'unchanged' };
        }
        return prepared(
          structural({ lockedUpdates: updates }),
          structural({
            lockedUpdates: updates.map((update) => ({
              id: update.id,
              isLocked: index.byId.get(update.id)!.node.isLocked,
            })),
          }),
          {
            ...detail(updates),
            postconditions: updates.map((update) => ({
              id: update.id,
              kind: 'patched',
              patch: { isLocked: update.isLocked },
            })),
          }
        );
      }
    }
  };

  const prepareTranslate = (command: Extract<DocumentCommand, { type: 'translate' }>): PrepareEditResult => {
    const ids = unique(command.ids);
    if (ids.length === 0) {
      return { operation: 'translate nothing', status: 'unsupported' };
    }
    if (!Number.isFinite(command.dx) || !Number.isFinite(command.dy)) {
      return { operation: 'translate by a non-finite delta', status: 'unsupported' };
    }
    const found = lookup(ids);
    if ('status' in found) {
      return found;
    }
    const outer = outermostNodes(index, ids);
    const locked = unique(outer.flatMap((entry) => lockedWithin(index, entry)));
    if (locked.length > 0) {
      return { ids: locked, status: 'locked' };
    }
    const leaves = outer.flatMap((entry) => collectSubtreeLeaves(entry.node));
    if (leaves.length === 0 || (command.dx === 0 && command.dy === 0)) {
      return { status: 'unchanged' };
    }
    const next = leaves.map((leaf) => ({
      id: leaf.id,
      x: leaf.transform.x + command.dx,
      y: leaf.transform.y + command.dy,
    }));
    return prepared(
      { type: 'setCanvasLayerPositions', updates: next },
      {
        type: 'setCanvasLayerPositions',
        updates: leaves.map((leaf) => ({ id: leaf.id, x: leaf.transform.x, y: leaf.transform.y })),
      },
      {
        postconditions: next.map((update) => ({
          id: update.id,
          kind: 'patched',
          patch: { transform: { x: update.x, y: update.y } },
        })),
        selectionAfter: selectedLayerId,
        touchedIds: leaves.map((leaf) => leaf.id),
        touchedStacks: stacksTopFirst(outer.map((entry) => entry.stack)),
      }
    );
  };

  const prepareSelect = (command: Extract<DocumentCommand, { type: 'select' }>): PrepareEditResult => {
    if (command.id !== null && !index.byId.has(command.id)) {
      return missing([command.id]);
    }
    if (command.id === selectedLayerId) {
      return { status: 'unchanged' };
    }
    return prepared(
      { id: command.id, type: 'setCanvasSelectedLayer' },
      { id: selectedLayerId, type: 'setCanvasSelectedLayer' },
      {
        history: 'none',
        postconditions: [{ id: command.id, kind: 'selection' }],
        selectionAfter: command.id,
        touchedIds: [],
        touchedStacks: [],
      }
    );
  };

  const model: CanvasDocumentModel = {
    canMergeDown: (upperId) => mergeDownEligibility(document, upperId),
    compileLeaves: () => compileLeaves(index),
    compileNodes: () => compileNodes(index),
    document,
    getEntry: (id) => index.byId.get(id) ?? null,
    getLayer: (id) => {
      const node = index.byId.get(id)?.node;
      return node && !isGroupNode(node) ? node : null;
    },
    getNode: (id) => index.byId.get(id)?.node ?? null,
    getStack: (kind) => stacks[kind],
    prepare: (command) => {
      switch (command.type) {
        case 'insert':
          return prepareInsert(command);
        case 'remove':
          return prepareRemove(command);
        case 'duplicate':
          return prepareDuplicate(command);
        case 'move':
          return prepareMove(command);
        case 'reorder':
          return prepareReorder(command.orders);
        case 'reparent':
          return prepareReparent(command);
        case 'group':
          return prepareGroup(command);
        case 'ungroup':
          return prepareUngroup(command);
        case 'patch':
          return preparePatch(command);
        case 'patch-config':
          return preparePatchConfig(command);
        case 'patch-config-batch':
          return preparePatchConfigBatch(command);
        case 'patch-source':
          return preparePatchSource(command);
        case 'set-enabled':
        case 'set-hidden':
        case 'set-locked':
          return prepareFlags(command);
        case 'translate':
          return prepareTranslate(command);
        case 'select':
          return prepareSelect(command);
      }
    },
    refusalFor: (command) => {
      // The hot paths (drag hover, toolbar enablement) validate without materializing an edit.
      const checked =
        command.type === 'reparent'
          ? checkReparent(command)
          : command.type === 'group'
            ? checkGroup(command)
            : command.type === 'ungroup'
              ? checkUngroup(command)
              : command.type === 'remove'
                ? checkRemove(command)
                : null;
      if (checked) {
        return 'status' in checked && checked.status !== 'unchanged' ? (checked as DocumentRefusal) : null;
      }
      const result = model.prepare(command);
      return result.status === 'prepared' || result.status === 'unchanged' ? null : result;
    },
  };
  return model;
};

const stripRemoved = (node: CanvasNodeContract, removed: ReadonlySet<string>): CanvasNodeContract =>
  isGroupNode(node)
    ? {
        ...node,
        children: node.children.filter((child) => !removed.has(child.id)).map((child) => stripRemoved(child, removed)),
      }
    : node;

/** Replaces each dissolved group with its children, in place. */
const dissolve = (nodes: readonly CanvasNodeContract[], dissolved: ReadonlySet<string>): CanvasNodeContract[] =>
  nodes.flatMap((node) => {
    if (!isGroupNode(node)) {
      return [node];
    }
    const children = dissolve(node.children, dissolved);
    return dissolved.has(node.id) ? children : [{ ...node, children }];
  });

export { stackOf as documentStackOf };
