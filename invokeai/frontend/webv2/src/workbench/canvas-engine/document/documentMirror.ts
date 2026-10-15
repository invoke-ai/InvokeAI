import type {
  CanvasDocumentContractV3,
  CanvasGroupContract,
  CanvasLayerContract,
  CanvasStagingAreaContractV2,
  CanvasStateContractV3,
} from '@workbench/canvas-engine/contracts';

import {
  getDocumentIndex,
  valueEditBetween,
  type CanvasDocumentIndex,
  type CanvasNodeEntry,
  type CanvasValueEdit,
} from './documentIndex';
import { collectSubtreeLeaves, isGroupNode } from './documentTree';

/** The minimal store shape the mirror depends on (a superset of `WorkbenchStore`). */
export interface DocumentMirrorStore {
  getCanvasState?(): CanvasStateContractV3 | null;
  getState?(): { projects: readonly { id: string; canvas: CanvasStateContractV3 }[] };
  subscribe(listener: () => void): () => void;
}

/** Callbacks fired when the mirrored document changes. */
export interface DocumentMirrorCallbacks {
  /**
   * Reports leaf or ancestor-effective changes in document order. `sourceChanged` contains added leaves and changed
   * raster/control sources or mask bitmaps whose caches are stale. Property, transform and ancestor-flag changes
   * retain cached pixels. `restructured` is false for value edits, which add, remove and move nothing.
   */
  onLayersChanged(changed: string[], sourceChanged: string[], restructured: boolean): void;
  /**
   * Leaves changed ONLY by an ancestor group's adjustment stack: recomposite
   * without `onLayersChanged`'s destructive reactions (float and pixel-edit
   * cancellation).
   */
  onLayersRecomposite?(ids: string[]): void;
  /** The forests were restructured without any leaf changing: recomposite with the new order. */
  onLayerOrderChanged(): void;
  /** The document was replaced wholesale (dims/background change, appear/disappear) — full invalidate. */
  onDocumentReplaced(): void;
  /** The generation bounding box changed. */
  onBboxChanged(): void;
  /** The staging area changed. */
  onStagingChanged(): void;
  /**
   * Selection-only changes reuse stacks and bbox, requiring a separate callback for selection chrome and per-layer
   * session cleanup.
   */
  onSelectionChanged?(selectedLayerId: string | null): void;
  /** Any change to the mirrored document, after the callbacks above have reacted to it (even when one threw). */
  onDocumentChanged?(): void;
}

/** The imperative mirror handle. */
export interface DocumentMirror {
  /** The current mirrored document, or `null` if the project is gone. */
  getDocument(): CanvasDocumentContractV3 | null;
  /** Synchronously reconciles from store state when ordinary notification was interrupted. */
  refresh(): void;
  /** Removes the store subscription. */
  dispose(): void;
}

type Bbox = CanvasDocumentContractV3['bbox'];

const bboxEqual = (a: Bbox, b: Bbox): boolean =>
  a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height;

/**
 * Raster invalidation identity is raster/control `source` or mask `bitmap`. Property edits preserve it; mask fill
 * changes must preserve unflushed alpha-cache strokes.
 */
const rasterSourceRef = (layer: CanvasLayerContract): unknown =>
  layer.type === 'raster' || layer.type === 'control' ? layer.source : layer.mask.bitmap;

const effectiveKey = (entry: CanvasNodeEntry): string =>
  `${entry.ancestorsEnabled ? 1 : 0}${entry.ancestorsLocked ? 1 : 0}${entry.ancestorsHidden ? 1 : 0}`;

const diagnostics = { forestDiffs: 0 };

/** Whole-forest diffs: structural edits and replacements; value edits never cost one. */
export const getForestDiffCount = (): number => diagnostics.forestDiffs;

export const resetForestDiffCount = (): void => {
  diagnostics.forestDiffs = 0;
};

interface ForestDiff {
  changed: string[];
  sourceChanged: string[];
  /** Ancestor-adjustment fan-out only; disjoint from `changed`. */
  recompositeOnly: string[];
  restructured: boolean;
}

/**
 * Diffs leaf identity and ancestor-effective enabled, locked and hidden state, including unchanged leaves affected
 * by ancestors.
 */
const diffForests = (prev: CanvasDocumentIndex, next: CanvasDocumentIndex): ForestDiff => {
  diagnostics.forestDiffs += 1;
  const changed = new Set<string>();
  const sourceChanged = new Set<string>();
  const recompositeOnly = new Set<string>();
  // Preorder: a changed group's id is collected before its leaves are visited.
  const adjustedGroups = new Set<string>();
  let restructured = false;
  for (const entry of next.nodes) {
    const before = prev.byId.get(entry.node.id);
    if (!before) {
      restructured = true;
      if (!isGroupNode(entry.node)) {
        changed.add(entry.node.id);
        sourceChanged.add(entry.node.id);
      }
      continue;
    }
    if (before.parentId !== entry.parentId || before.order !== entry.order) {
      restructured = true;
    }
    if (isGroupNode(entry.node)) {
      const beforeGroup = before.node as typeof entry.node;
      if (
        before.node !== entry.node &&
        (beforeGroup.adjustments !== entry.node.adjustments ||
          beforeGroup.opacity !== entry.node.opacity ||
          beforeGroup.blendMode !== entry.node.blendMode)
      ) {
        adjustedGroups.add(entry.node.id);
      }
      continue;
    }
    if (before.node !== entry.node) {
      changed.add(entry.node.id);
      if (rasterSourceRef(before.node as CanvasLayerContract) !== rasterSourceRef(entry.node)) {
        sourceChanged.add(entry.node.id);
      }
    } else if (effectiveKey(before) !== effectiveKey(entry)) {
      changed.add(entry.node.id);
    } else if (entry.path.some((ancestorId) => adjustedGroups.has(ancestorId))) {
      recompositeOnly.add(entry.node.id);
    }
  }
  for (const entry of prev.nodes) {
    if (!next.byId.has(entry.node.id)) {
      restructured = true;
      if (!isGroupNode(entry.node)) {
        changed.add(entry.node.id);
      }
    }
  }
  return {
    changed: [...changed],
    recompositeOnly: [...recompositeOnly].filter((id) => !changed.has(id)),
    restructured,
    sourceChanged: [...sourceChanged],
  };
};

const groupFlagsChanged = (before: CanvasGroupContract, after: CanvasGroupContract): boolean =>
  before.isEnabled !== after.isEnabled ||
  before.isLocked !== after.isLocked ||
  (before.isHidden === true) !== (after.isHidden === true);

/**
 * Diffs a value edit from its recorded node changes: changed leaves, the subtree leaves whose inherited flags
 * moved, and the subtree leaves of groups whose adjustments, opacity or blend changed. Structure is unchanged.
 */
const diffValueEdit = (prev: CanvasDocumentIndex, next: CanvasDocumentIndex, edit: CanvasValueEdit): ForestDiff => {
  const changed = new Set<string>();
  const sourceChanged = new Set<string>();
  const reflagged: string[] = [];
  const adjusted: string[] = [];
  for (const [id, { after, before }] of edit) {
    // Folded steps can hand a node back unchanged.
    if (before === after) {
      continue;
    }
    if (isGroupNode(after) && isGroupNode(before)) {
      if (groupFlagsChanged(before, after)) {
        reflagged.push(id);
      }
      if (
        before.adjustments !== after.adjustments ||
        before.opacity !== after.opacity ||
        before.blendMode !== after.blendMode
      ) {
        adjusted.push(id);
      }
    } else if (!isGroupNode(after) && !isGroupNode(before)) {
      changed.add(id);
      if (rasterSourceRef(before) !== rasterSourceRef(after)) {
        sourceChanged.add(id);
      }
    }
  }
  for (const groupId of reflagged) {
    for (const leaf of collectSubtreeLeaves(next.byId.get(groupId)!.node)) {
      if (!changed.has(leaf.id) && effectiveKey(prev.byId.get(leaf.id)!) !== effectiveKey(next.byId.get(leaf.id)!)) {
        changed.add(leaf.id);
      }
    }
  }
  const recompositeOnly = new Set<string>();
  for (const groupId of adjusted) {
    for (const leaf of collectSubtreeLeaves(next.byId.get(groupId)!.node)) {
      if (!changed.has(leaf.id)) {
        recompositeOnly.add(leaf.id);
      }
    }
  }
  const inDocumentOrder = (ids: Set<string>): string[] =>
    [...ids].sort((left, right) => next.byId.get(left)!.order - next.byId.get(right)!.order);
  return {
    changed: inDocumentOrder(changed),
    recompositeOnly: inDocumentOrder(recompositeOnly),
    restructured: false,
    sourceChanged: inDocumentOrder(sourceChanged),
  };
};

/**
 * Creates a document mirror bound to `projectId`. Subscribes immediately and seeds the last-seen
 * references from the current state, so no spurious callback fires on creation.
 */
export const createDocumentMirror = (
  store: DocumentMirrorStore,
  projectIdOrCallbacks: string | DocumentMirrorCallbacks,
  maybeCallbacks?: DocumentMirrorCallbacks
): DocumentMirror => {
  const projectId = typeof projectIdOrCallbacks === 'string' ? projectIdOrCallbacks : null;
  const callbacks = typeof projectIdOrCallbacks === 'string' ? maybeCallbacks : projectIdOrCallbacks;
  if (!callbacks) {
    throw new Error('DocumentMirror callbacks are required.');
  }
  const selectCanvas = (): CanvasStateContractV3 | null =>
    store.getCanvasState?.() ??
    (projectId === null
      ? null
      : (store.getState?.().projects.find((project) => project.id === projectId)?.canvas ?? null));

  let lastDoc: CanvasDocumentContractV3 | null = selectCanvas()?.document ?? null;
  let lastRevision: number = selectCanvas()?.documentRevision ?? 0;
  let lastStaging: CanvasStagingAreaContractV2 | null = selectCanvas()?.stagingArea ?? null;

  const handleChange = (): void => {
    const canvas = selectCanvas();
    const doc = canvas?.document ?? null;
    const revision = canvas?.documentRevision ?? 0;
    const staging = canvas?.stagingArea ?? null;

    if (doc !== lastDoc) {
      const prevDoc = lastDoc;
      const prevRevision = lastRevision;
      const prevSelectedLayerId = prevDoc?.selectedLayerId ?? null;
      lastDoc = doc;
      lastRevision = revision;
      try {
        if (!prevDoc || !doc) {
          callbacks.onDocumentReplaced();
        } else if (
          revision !== prevRevision ||
          prevDoc.width !== doc.width ||
          prevDoc.height !== doc.height ||
          prevDoc.background !== doc.background
        ) {
          callbacks.onDocumentReplaced();
        } else {
          if (prevDoc.stacks !== doc.stacks) {
            const edit = valueEditBetween(prevDoc.stacks, doc.stacks);
            const prevIndex = getDocumentIndex(prevDoc);
            const nextIndex = getDocumentIndex(doc);
            const diff = edit ? diffValueEdit(prevIndex, nextIndex, edit) : diffForests(prevIndex, nextIndex);
            if (diff.changed.length > 0) {
              callbacks.onLayersChanged(diff.changed, diff.sourceChanged, diff.restructured);
            } else if (diff.restructured) {
              callbacks.onLayerOrderChanged();
            }
            if (diff.recompositeOnly.length > 0) {
              callbacks.onLayersRecomposite?.(diff.recompositeOnly);
            }
          }
          if (!bboxEqual(prevDoc.bbox, doc.bbox)) {
            callbacks.onBboxChanged();
          }
        }

        const selectedLayerId = doc?.selectedLayerId ?? null;
        if (selectedLayerId !== prevSelectedLayerId) {
          callbacks.onSelectionChanged?.(selectedLayerId);
        }
      } finally {
        callbacks.onDocumentChanged?.();
      }
    }

    if (staging !== lastStaging) {
      lastStaging = staging;
      callbacks.onStagingChanged();
    }
  };

  const unsubscribe = store.subscribe(handleChange);

  return {
    dispose: unsubscribe,
    getDocument: () => lastDoc,
    refresh: handleChange,
  };
};
