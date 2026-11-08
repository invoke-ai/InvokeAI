import type {
  CanvasDocumentContractV3,
  CanvasGroupContract,
  CanvasLayerContract,
  CanvasLayerStackKind,
  CanvasNodeContract,
  CanvasStackForests,
} from '@workbench/canvas-engine/contracts';

import { LAYER_STACKS_TOP_FIRST } from '@workbench/canvas-engine/contracts';

import { collectSubtree, isGroupNode } from './documentTree';

/** Document facts about one node's place in its forest. */
export interface CanvasNodeEntry {
  readonly node: CanvasNodeContract;
  readonly stack: CanvasLayerStackKind;
  readonly parentId: string | null;
  /** Ancestor ids, root first; `path.length` is the node's depth. */
  readonly path: readonly string[];
  readonly siblingIndex: number;
  /** Position among every node, stacks top first, each stack in preorder. */
  readonly order: number;
  /** Whether every ancestor is enabled, unlocked, or unhidden respectively. */
  readonly ancestorsEnabled: boolean;
  readonly ancestorsLocked: boolean;
  readonly ancestorsHidden: boolean;
}

/** Shared forest index keyed by `stacks`, whose identity survives selection, bbox and geometry-only edits. */
export interface CanvasDocumentIndex {
  readonly stacks: CanvasStackForests;
  readonly byId: ReadonlyMap<string, CanvasNodeEntry>;
  /** Every node, stacks top first, each in preorder. A derived index materializes this on first read. */
  readonly nodes: readonly CanvasNodeEntry[];
  /** Every leaf, in the same order. */
  readonly leaves: readonly CanvasLayerContract[];
  readonly maxDepth: number;
  /** Set on an index derived from a value edit: the index it extends and the ids whose entries changed. */
  readonly derivedFrom?: { readonly previous: CanvasDocumentIndex; readonly changedIds: ReadonlySet<string> };
}

const diagnostics = {
  entriesVisited: 0,
  forestNodesVisited: 0,
  indexBuilds: 0,
  indexDerivations: 0,
  nodesMaterialized: 0,
};

export const getDocumentIndexBuildCount = (): number => diagnostics.indexBuilds;

/** Indexes derived from a previous one after a value edit, without walking the forests. */
export const getDocumentIndexDerivationCount = (): number => diagnostics.indexDerivations;

/** Entries constructed, by a build or a derivation: the work an edit actually costs. */
export const getDocumentIndexVisitCount = (): number => diagnostics.entriesVisited;

/** Times a derived index had to materialize its full `nodes` array for a consumer. */
export const getDocumentIndexMaterializationCount = (): number => diagnostics.nodesMaterialized;

/** Forest nodes a value edit revisited: the edited nodes and their ancestors, never the whole forest. */
export const getValueEditVisitCount = (): number => diagnostics.forestNodesVisited;

export const resetDocumentIndexBuildCount = (): void => {
  diagnostics.entriesVisited = 0;
  diagnostics.forestNodesVisited = 0;
  diagnostics.indexBuilds = 0;
  diagnostics.indexDerivations = 0;
  diagnostics.nodesMaterialized = 0;
};

/** How many derivations may chain before the next one flattens onto the plain root. */
const MAX_DERIVATION_DEPTH = 8;
/** Overrides may cover this fraction of the root before flattening rebuilds a plain index instead. */
const MAX_OVERRIDE_RATIO = 0.25;

/**
 * A map that reads through to the previous index's map for every entry a value edit did not touch.
 * Lookups cost one extra step per derivation; iteration substitutes overrides on the fly.
 */
class LayeredEntryMap implements ReadonlyMap<string, CanvasNodeEntry> {
  readonly [Symbol.toStringTag] = 'LayeredEntryMap';

  constructor(
    private readonly base: ReadonlyMap<string, CanvasNodeEntry>,
    private readonly overrides: ReadonlyMap<string, CanvasNodeEntry>
  ) {}

  get size(): number {
    return this.base.size;
  }

  get(key: string): CanvasNodeEntry | undefined {
    return this.overrides.get(key) ?? this.base.get(key);
  }

  has(key: string): boolean {
    return this.base.has(key);
  }

  forEach(
    callback: (value: CanvasNodeEntry, key: string, map: ReadonlyMap<string, CanvasNodeEntry>) => void,
    thisArg?: unknown
  ): void {
    for (const [key, value] of this) {
      callback.call(thisArg, value, key, this);
    }
  }

  *entries(): MapIterator<[string, CanvasNodeEntry]> {
    for (const [key, value] of this.base) {
      yield [key, this.overrides.get(key) ?? value];
    }
  }

  keys(): MapIterator<string> {
    return this.base.keys();
  }

  *values(): MapIterator<CanvasNodeEntry> {
    for (const [, value] of this) {
      yield value;
    }
  }

  [Symbol.iterator](): MapIterator<[string, CanvasNodeEntry]> {
    return this.entries();
  }
}

/**
 * Value-edit overlay index with predecessor read-through. Every ninth derivation flattens onto the plain root to
 * bound lookup depth and retention.
 */
class DerivedIndex implements CanvasDocumentIndex {
  readonly byId: ReadonlyMap<string, CanvasNodeEntry>;
  readonly derivedFrom: { readonly previous: CanvasDocumentIndex; readonly changedIds: ReadonlySet<string> };
  readonly maxDepth: number;
  private nodesCache: readonly CanvasNodeEntry[] | null = null;
  private leavesCache: readonly CanvasLayerContract[] | null = null;

  constructor(
    readonly stacks: CanvasStackForests,
    private readonly source: CanvasDocumentIndex,
    readonly overrides: ReadonlyMap<string, CanvasNodeEntry>,
    readonly leafChanged: boolean
  ) {
    this.byId = new LayeredEntryMap(source.byId, overrides);
    this.derivedFrom = { changedIds: new Set(overrides.keys()), previous: source };
    this.maxDepth = source.maxDepth;
  }

  get nodes(): readonly CanvasNodeEntry[] {
    if (this.nodesCache === null) {
      diagnostics.nodesMaterialized += 1;
      this.nodesCache = this.patchedNodes();
    }
    return this.nodesCache;
  }

  /** The same content as a plain index; building it is part of flattening, not a consumer read. */
  plain(): CanvasDocumentIndex {
    return {
      byId: new Map(this.byId),
      leaves: this.leaves,
      maxDepth: this.maxDepth,
      nodes: this.nodesCache ?? this.patchedNodes(),
      stacks: this.stacks,
    };
  }

  private patchedNodes(): readonly CanvasNodeEntry[] {
    const nodes = this.source.nodes.slice();
    for (const entry of this.overrides.values()) {
      nodes[entry.order] = entry;
    }
    return nodes;
  }

  get leaves(): readonly CanvasLayerContract[] {
    if (!this.leafChanged) {
      return this.source.leaves;
    }
    if (this.leavesCache === null) {
      this.leavesCache = this.source.leaves.map(
        (leaf) => (this.overrides.get(leaf.id)?.node as CanvasLayerContract | undefined) ?? leaf
      );
    }
    return this.leavesCache;
  }
}

const derivationDepth = (index: CanvasDocumentIndex): number => {
  let depth = 0;
  let current: CanvasDocumentIndex | undefined = index;
  while (current?.derivedFrom) {
    depth += 1;
    current = current.derivedFrom.previous;
  }
  return depth;
};

/** Flatten newest overrides onto the plain root; rebuild a plain index when override coverage becomes too large. */
const flatten = (index: DerivedIndex): CanvasDocumentIndex => {
  const overrides = new Map<string, CanvasNodeEntry>();
  let leafChanged = false;
  let current: CanvasDocumentIndex = index;
  while (current instanceof DerivedIndex) {
    for (const [id, entry] of current.overrides) {
      if (!overrides.has(id)) {
        overrides.set(id, entry);
      }
    }
    leafChanged ||= current.leafChanged;
    current = current.derivedFrom.previous;
  }
  const flat = new DerivedIndex(index.stacks, current, overrides, leafChanged);
  return overrides.size <= current.byId.size * MAX_OVERRIDE_RATIO ? flat : flat.plain();
};

type DocumentView = Pick<CanvasDocumentContractV3, 'stacks'> | null | undefined;

const EMPTY_LEAVES: readonly CanvasLayerContract[] = [];

const indexes = new WeakMap<CanvasStackForests, CanvasDocumentIndex>();

const build = (stacks: CanvasStackForests): CanvasDocumentIndex => {
  diagnostics.indexBuilds += 1;
  const byId = new Map<string, CanvasNodeEntry>();
  const nodes: CanvasNodeEntry[] = [];
  const leaves: CanvasLayerContract[] = [];
  let maxDepth = 0;
  const visit = (
    children: readonly CanvasNodeContract[],
    stack: CanvasLayerStackKind,
    parent: CanvasGroupContract | null,
    path: readonly string[],
    ancestorsEnabled: boolean,
    ancestorsLocked: boolean,
    ancestorsHidden: boolean
  ): void => {
    maxDepth = Math.max(maxDepth, path.length);
    children.forEach((node, siblingIndex) => {
      diagnostics.entriesVisited += 1;
      const entry: CanvasNodeEntry = {
        ancestorsEnabled,
        ancestorsHidden,
        ancestorsLocked,
        node,
        order: nodes.length,
        parentId: parent?.id ?? null,
        path,
        siblingIndex,
        stack,
      };
      byId.set(node.id, entry);
      nodes.push(entry);
      if (isGroupNode(node)) {
        visit(
          node.children,
          stack,
          node,
          [...path, node.id],
          ancestorsEnabled && node.isEnabled,
          ancestorsLocked || node.isLocked,
          ancestorsHidden || node.isHidden === true
        );
      } else {
        leaves.push(node);
      }
    });
  };
  for (const stack of LAYER_STACKS_TOP_FIRST) {
    visit(stacks[stack], stack, null, [], true, false, false);
  }
  return { byId, leaves, maxDepth, nodes, stacks };
};

const flagsChanged = (previous: CanvasNodeContract, next: CanvasNodeContract): boolean =>
  isGroupNode(previous) &&
  isGroupNode(next) &&
  (previous.isEnabled !== next.isEnabled ||
    previous.isLocked !== next.isLocked ||
    (previous.isHidden === true) !== (next.isHidden === true));

/**
 * Derives an index after value-only edits, replacing changed nodes, ancestors and descendants affected by group
 * flags. Unchanged entries read through. Returns null without a prior index or when structure changed, deferring a
 * full build.
 */
const deriveIndexForValueEdit = (
  previousStacks: CanvasStackForests,
  nextStacks: CanvasStackForests,
  changed: ReadonlyMap<string, CanvasNodeContract>
): CanvasDocumentIndex | null => {
  const existing = indexes.get(nextStacks);
  if (existing) {
    return existing;
  }
  const previous = indexes.get(previousStacks);
  if (!previous || previousStacks === nextStacks) {
    return null;
  }
  const replacedNodes = new Map<string, CanvasNodeContract>();
  const reflagged: CanvasGroupContract[] = [];
  for (const [id, node] of changed) {
    const entry = previous.byId.get(id);
    if (!entry || entry.node.id !== node.id || isGroupNode(entry.node) !== isGroupNode(node)) {
      return null;
    }
    replacedNodes.set(id, node);
    if (flagsChanged(entry.node, node) && isGroupNode(node)) {
      reflagged.push(node);
    }
  }
  // Ancestors were rebuilt along each changed path; find their new objects from the new roots.
  for (const id of changed.keys()) {
    const entry = previous.byId.get(id)!;
    let siblings: readonly CanvasNodeContract[] = nextStacks[entry.stack];
    for (const ancestorId of entry.path) {
      const ancestor = replacedNodes.get(ancestorId) ?? siblings[previous.byId.get(ancestorId)!.siblingIndex];
      if (!ancestor || ancestor.id !== ancestorId || !isGroupNode(ancestor)) {
        return null;
      }
      replacedNodes.set(ancestorId, ancestor);
      siblings = ancestor.children;
    }
  }
  diagnostics.indexDerivations += 1;
  const nodeOf = (id: string): CanvasNodeContract => replacedNodes.get(id) ?? previous.byId.get(id)!.node;
  const replaced = new Map<string, CanvasNodeEntry>();
  let leafChanged = false;
  const replaceEntry = (id: string, node: CanvasNodeContract, reflag: boolean): void => {
    const entry = previous.byId.get(id)!;
    diagnostics.entriesVisited += 1;
    if (node !== entry.node && !isGroupNode(node)) {
      leafChanged = true;
    }
    const ancestors = reflag ? entry.path.map(nodeOf) : null;
    replaced.set(id, {
      ...entry,
      ancestorsEnabled: ancestors ? ancestors.every((ancestor) => ancestor.isEnabled) : entry.ancestorsEnabled,
      ancestorsHidden: ancestors
        ? ancestors.some((ancestor) => isGroupNode(ancestor) && ancestor.isHidden === true)
        : entry.ancestorsHidden,
      ancestorsLocked: ancestors ? ancestors.some((ancestor) => ancestor.isLocked) : entry.ancestorsLocked,
      node,
    });
  };
  // Only the subtree under a group whose flags changed re-derives its ancestor-effective flags;
  // a replaced node inside such a subtree is entered there once, with the new ancestors.
  for (const group of reflagged) {
    for (const member of collectSubtree(group)) {
      if (member.id !== group.id) {
        replaceEntry(member.id, member, true);
      }
    }
  }
  for (const [id, node] of replacedNodes) {
    if (!replaced.has(id)) {
      replaceEntry(id, node, false);
    }
  }
  const derived = new DerivedIndex(nextStacks, previous, replaced, leafChanged);
  const registered = derivationDepth(derived) > MAX_DERIVATION_DEPTH ? flatten(derived) : derived;
  indexes.set(nextStacks, registered);
  return registered;
};

/** One node a value edit replaced; ancestors rebuilt only for a new child array are not listed. */
export interface CanvasNodeValueChange {
  readonly before: CanvasNodeContract;
  readonly after: CanvasNodeContract;
}

/** The nodes value edits replaced between two forests that share one structure. */
export type CanvasValueEdit = ReadonlyMap<string, CanvasNodeValueChange>;

interface ValueEditRecord {
  /** Weak, so a chain of edits never retains every superseded forest. */
  readonly from: WeakRef<CanvasStackForests>;
  readonly changes: CanvasValueEdit;
}

const valueEdits = new WeakMap<CanvasStackForests, ValueEditRecord>();

/** How many consecutive value edits a consumer that fell behind may still fold instead of diffing forests. */
const MAX_FOLDED_VALUE_EDITS = 16;
/** Folding up to this many replaced nodes always beats a forest diff, however small the document. */
const MIN_FOLDED_ENTRIES = 64;

/**
 * The value edits that lead from `previous` to `next`, folded oldest first, or null when any step between them
 * restructured the forests or was not recorded, or when folding would touch more than a quarter of the nodes (a
 * forest diff is then cheaper).
 */
export const valueEditBetween = (previous: CanvasStackForests, next: CanvasStackForests): CanvasValueEdit | null => {
  const steps: CanvasValueEdit[] = [];
  let entries = 0;
  let current = next;
  while (current !== previous) {
    const record = valueEdits.get(current);
    const from = record?.from.deref();
    if (!record || !from || steps.length === MAX_FOLDED_VALUE_EDITS) {
      return null;
    }
    steps.push(record.changes);
    entries += record.changes.size;
    if (steps.length > 1 && entries > Math.max(MIN_FOLDED_ENTRIES, indexStacks(next).byId.size / 4)) {
      return null;
    }
    current = from;
  }
  if (steps.length <= 1) {
    return steps[0] ?? new Map();
  }
  const folded = new Map<string, CanvasNodeValueChange>();
  for (let step = steps.length - 1; step >= 0; step -= 1) {
    for (const [id, change] of steps[step]!) {
      const earlier = folded.get(id);
      folded.set(id, earlier ? { after: change.after, before: earlier.before } : change);
    }
  }
  return folded;
};

const rootKey = (stack: CanvasLayerStackKind): string => `\0${stack}`;

/**
 * Rewrites named nodes in place of their old objects, locating them through the index. Only the sibling arrays on
 * each edited path are copied, so cost is the edited paths' sibling counts, not the forest. The next forests inherit
 * a derived index and record which nodes changed for incremental consumers. Returns `stacks` when nothing changed.
 */
export const updateNodeValues = (
  stacks: CanvasStackForests,
  updates: ReadonlyMap<string, (node: CanvasNodeContract) => CanvasNodeContract>
): CanvasStackForests => {
  if (updates.size === 0) {
    return stacks;
  }
  const index = indexStacks(stacks);
  // Sibling positions to revisit, keyed by parent group id or a stack's root key.
  const touched = new Map<string, Set<number>>();
  const mark = (entry: CanvasNodeEntry): boolean => {
    const key = entry.parentId ?? rootKey(entry.stack);
    const positions = touched.get(key) ?? new Set<number>();
    const added = !positions.has(entry.siblingIndex);
    positions.add(entry.siblingIndex);
    touched.set(key, positions);
    return added;
  };
  for (const id of updates.keys()) {
    const entry = index.byId.get(id);
    if (!entry || !mark(entry)) {
      continue;
    }
    for (let depth = entry.path.length - 1; depth >= 0; depth -= 1) {
      if (!mark(index.byId.get(entry.path[depth]!)!)) {
        break;
      }
    }
  }
  const changes = new Map<string, CanvasNodeValueChange>();
  const rebuild = (key: string, list: readonly CanvasNodeContract[]): readonly CanvasNodeContract[] => {
    const positions = touched.get(key);
    if (!positions) {
      return list;
    }
    let next: CanvasNodeContract[] | null = null;
    for (const position of positions) {
      const node = list[position]!;
      diagnostics.forestNodesVisited += 1;
      let current = node;
      if (isGroupNode(current)) {
        const children = rebuild(current.id, current.children);
        if (children !== current.children) {
          current = { ...current, children: children as CanvasNodeContract[] };
        }
      }
      const update = updates.get(node.id);
      if (update) {
        const updated = update(current);
        if (updated !== current) {
          changes.set(node.id, { after: updated, before: node });
          current = updated;
        }
      }
      if (current !== node) {
        next ??= list.slice();
        next[position] = current;
      }
    }
    return next ?? list;
  };
  let result = stacks;
  for (const stack of LAYER_STACKS_TOP_FIRST) {
    const roots = rebuild(rootKey(stack), stacks[stack]);
    if (roots !== stacks[stack]) {
      result = result === stacks ? { ...stacks } : result;
      result[stack] = roots as CanvasNodeContract[];
    }
  }
  if (result !== stacks) {
    const changed = new Map([...changes].map(([id, change]) => [id, change.after]));
    if (deriveIndexForValueEdit(stacks, result, changed)) {
      valueEdits.set(result, { changes, from: new WeakRef(stacks) });
    }
  }
  return result;
};

export const indexStacks = (stacks: CanvasStackForests): CanvasDocumentIndex => {
  const existing = indexes.get(stacks);
  if (existing) {
    return existing;
  }
  const built = build(stacks);
  indexes.set(stacks, built);
  return built;
};

export const getDocumentIndex = (document: Pick<CanvasDocumentContractV3, 'stacks'>): CanvasDocumentIndex =>
  indexStacks(document.stacks);

/** The document's leaves, stacks top first, each in preorder; the same array while `stacks` is unchanged. */
export const getDocumentLeaves = (document: DocumentView): readonly CanvasLayerContract[] =>
  document ? indexStacks(document.stacks).leaves : EMPTY_LEAVES;

export const getDocumentNode = (document: DocumentView, id: string | null | undefined): CanvasNodeContract | null =>
  document && id ? (indexStacks(document.stacks).byId.get(id)?.node ?? null) : null;

/** The leaf with `id`, or `null` when absent, a group, or there is no document. */
export const getDocumentLayer = (document: DocumentView, id: string | null | undefined): CanvasLayerContract | null => {
  const node = getDocumentNode(document, id);
  return node && !isGroupNode(node) ? node : null;
};

export const hasDocumentNode = (document: DocumentView, id: string): boolean =>
  !!document && indexStacks(document.stacks).byId.has(id);

/** True only when a document exists and holds no node with `id`. */
export const isNodeAbsent = (document: DocumentView, id: string): boolean =>
  !!document && !indexStacks(document.stacks).byId.has(id);

/** Whether `ancestorId` is `id` itself or one of its ancestors. */
export const isSelfOrAncestor = (index: CanvasDocumentIndex, id: string, ancestorId: string): boolean =>
  id === ancestorId || (index.byId.get(id)?.path.includes(ancestorId) ?? false);

/** Drops every id whose ancestor is also listed, in document order; absent ids are skipped. */
export const outermostNodes = (index: CanvasDocumentIndex, ids: Iterable<string>): CanvasNodeEntry[] => {
  const selected = new Set(ids);
  const outer: CanvasNodeEntry[] = [];
  for (const id of selected) {
    const entry = index.byId.get(id);
    if (entry && !entry.path.some((ancestor) => selected.has(ancestor))) {
      outer.push(entry);
    }
  }
  return outer.sort((left, right) => left.order - right.order);
};

/** The child list `parentId` names, read through the index; `null` when the parent is not a group of `stack`. */
export const childrenAt = (
  index: CanvasDocumentIndex,
  stack: CanvasLayerStackKind,
  parentId: string | null
): readonly CanvasNodeContract[] | null => {
  if (parentId === null) {
    return index.stacks[stack];
  }
  const parent = index.byId.get(parentId);
  return parent && parent.stack === stack && isGroupNode(parent.node) ? parent.node.children : null;
};
