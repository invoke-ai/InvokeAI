import type { Project } from '@workbench/projectContracts';

import {
  type CanvasDocumentContractV3,
  type CanvasLayerBasePatch,
  type CanvasLayerConfigPatch,
  type CanvasLayerContract,
  type CanvasNodeContract,
  type CanvasNodeInsertion,
  type CanvasNodeInsertionAnchor,
  type CanvasNodeMove,
  type CanvasProjectMutation,
  type CanvasRasterLayerContractV2,
  type CanvasStackForests,
  type CanvasStateContractV3,
  type LayerStackKind,
  type ReorderSiblingsCommand,
  CANVAS_MAX_NODE_COUNT,
  CANVAS_MAX_NODE_DEPTH,
  GROUP_PATCH_KEYS,
  isHideableLayer,
  isNodeHidden,
} from '@workbench/canvas-engine/api';
import {
  childrenAt,
  getDocumentIndex,
  getDocumentLayer,
  getDocumentNode,
  hasDocumentNode,
  indexStacks,
  outermostNodes,
  updateNodeValues,
  type CanvasDocumentIndex,
  type CanvasNodeEntry,
} from '@workbench/canvas-engine/document/documentIndex';
import {
  collectSubtree,
  collectSubtreeLeaves,
  isGroupNode,
  removeNodes,
} from '@workbench/canvas-engine/document/documentTree';
import { insertNodesAtAnchor } from '@workbench/canvas-engine/document/insertionAnchors';
import { isOverlayStack, layerStackOf, reorderSiblings } from '@workbench/canvas-engine/document/layerStacks';
import { repairSelectedLayerId, selectionAfterAcceptedResult } from '@workbench/canvas-engine/document/selectionRepair';
import { prependProjectEvent } from '@workbench/projectEvents';

import { canvasDocumentRequiresFontSchemaV4, normalizeCanvasDocumentContract } from './canvasMigration';
import {
  getCanvasStagingCandidateFingerprint,
  getCanvasStagingSlotCount,
  getCanvasStagingSlots,
} from './canvasStagingView';

export type { CanvasLayerBasePatch, CanvasLayerConfigPatch, CanvasProjectMutation } from '@workbench/canvas-engine/api';

const CANVAS_PROJECT_MUTATION_TYPES: ReadonlySet<string> = new Set<CanvasProjectMutation['type']>([
  'commitStagedImage',
  'rollbackStagedImageCommit',
  'addCanvasLayer',
  'applyCanvasLayerStackMutation',
  'clearCanvasStaging',
  'convertCanvasLayer',
  'cycleStagedImage',
  'deleteCanvasSnapshot',
  'discardAllStagedImages',
  'discardSelectedStagedImage',
  'mergeCanvasLayersDown',
  'removeCanvasLayers',
  'reorderCanvasSiblings',
  'replaceCanvasFontReferences',
  'replaceCanvasDocument',
  'replaceCanvasLayer',
  'resizeCanvasDocument',
  'restoreCanvasSnapshot',
  'saveCanvasSnapshot',
  'setCanvasBbox',
  'setCanvasLayerPositions',
  'setCanvasLayersEnabled',
  'setCanvasLayersHidden',
  'setCanvasSelectedLayer',
  'setCanvasStagingAutoSwitch',
  'setStagedImageIndex',
  'toggleCanvasStagingThumbnailsVisibility',
  'toggleCanvasStagingVisibility',
  'updateCanvasLayer',
  'updateCanvasLayerConfig',
  'updateCanvasLayerConfigs',
  'updateCanvasLayerSource',
]);

export const isCanvasProjectMutation = (value: { type: string }): value is CanvasProjectMutation =>
  CANVAS_PROJECT_MUTATION_TYPES.has(value.type);

const AUTO_LAYER_NAME_PATTERN = /^Layer (\d+)$/;

export const nextLayerName = (existingNames: readonly string[]): string => {
  const used = new Set<number>();
  for (const name of existingNames) {
    const match = AUTO_LAYER_NAME_PATTERN.exec(name.trim());
    if (match) {
      const n = Number(match[1]);
      if (Number.isInteger(n) && n > 0) {
        used.add(n);
      }
    }
  }
  let n = 1;
  while (used.has(n)) {
    n += 1;
  }
  return `Layer ${n}`;
};

const withRepairedSelection = (document: CanvasDocumentContractV3): CanvasDocumentContractV3 => {
  const selectedLayerId = repairSelectedLayerId(document.stacks, document.selectedLayerId);
  return selectedLayerId === document.selectedLayerId ? document : { ...document, selectedLayerId };
};

const setCanvasDocument = (project: Project, document: CanvasDocumentContractV3): Project => {
  const schemaVersion: 3 | 4 =
    project.canvas.version === 4 || document.version === 4 || canvasDocumentRequiresFontSchemaV4(document) ? 4 : 3;
  const nextDocument = document.version === schemaVersion ? document : { ...document, version: schemaVersion };
  return nextDocument === project.canvas.document && schemaVersion === project.canvas.version
    ? project
    : { ...project, canvas: { ...project.canvas, document: nextDocument, version: schemaVersion } };
};

const updateCanvasDocument = (
  project: Project,
  update: (document: CanvasDocumentContractV3) => CanvasDocumentContractV3
): Project => setCanvasDocument(project, update(project.canvas.document));

const setCanvasState = (project: Project, canvas: CanvasStateContractV3): Project => {
  const hasV4Snapshot = canvas.snapshots.some((snapshot) => snapshot.document.version === 4);
  const schemaVersion: 3 | 4 = canvas.version === 4 || canvas.document.version === 4 || hasV4Snapshot ? 4 : 3;
  const nextCanvas = canvas.version === schemaVersion ? canvas : { ...canvas, version: schemaVersion };
  return nextCanvas === project.canvas ? project : { ...project, canvas: nextCanvas };
};

const withStacks = (document: CanvasDocumentContractV3, stacks: CanvasStackForests): CanvasDocumentContractV3 =>
  stacks === document.stacks ? document : { ...document, stacks };

const withinLimits = (stacks: CanvasStackForests): boolean => {
  const index = indexStacks(stacks);
  return index.maxDepth <= CANVAS_MAX_NODE_DEPTH && index.byId.size <= CANVAS_MAX_NODE_COUNT;
};

const mapNode = (
  document: CanvasDocumentContractV3,
  id: string,
  update: (node: CanvasNodeContract) => CanvasNodeContract
): CanvasDocumentContractV3 => withStacks(document, updateNodeValues(document.stacks, new Map([[id, update]])));

const mapLayer = (
  document: CanvasDocumentContractV3,
  id: string,
  update: (layer: CanvasLayerContract) => CanvasLayerContract
): CanvasDocumentContractV3 => mapNode(document, id, (node) => (isGroupNode(node) ? node : update(node)));

const mapNodes = (
  document: CanvasDocumentContractV3,
  updates: Iterable<[string, (node: CanvasNodeContract) => CanvasNodeContract]>
): CanvasDocumentContractV3 => withStacks(document, updateNodeValues(document.stacks, new Map(updates)));

const setCanvasLayersEnabled = (
  document: CanvasDocumentContractV3,
  updates: readonly { id: string; isEnabled: boolean }[]
): CanvasDocumentContractV3 =>
  mapNodes(
    document,
    updates.map(({ id, isEnabled }) => [id, (node) => (node.isEnabled === isEnabled ? node : { ...node, isEnabled })])
  );

const isHideableNode = (index: CanvasDocumentIndex, node: CanvasNodeContract): boolean =>
  isGroupNode(node) ? isOverlayStack(index.byId.get(node.id)!.stack) : isHideableLayer(node);

/** Bulk display-visibility update; raster-stack nodes have no display axis and are skipped. */
const setCanvasLayersHidden = (
  document: CanvasDocumentContractV3,
  updates: readonly { id: string; isHidden: boolean }[]
): CanvasDocumentContractV3 => {
  const index = getDocumentIndex(document);
  return mapNodes(
    document,
    updates.map(({ id, isHidden }) => [
      id,
      (node) => {
        if (!isHideableNode(index, node) || isHidden === isNodeHidden(node)) {
          return node;
        }
        // Absence is the one spelling of "not hidden", so an undo restores the exact node.
        if (!isHidden) {
          const { isHidden: _hidden, ...shown } = node as CanvasNodeContract & { isHidden?: boolean };
          return shown as CanvasNodeContract;
        }
        return { ...node, isHidden };
      },
    ])
  );
};

const setCanvasLayerPositions = (
  document: CanvasDocumentContractV3,
  updates: readonly { id: string; x: number; y: number }[]
): CanvasDocumentContractV3 => {
  const positions = new Map(updates.map((update) => [update.id, update]));
  if (
    positions.size !== updates.length ||
    updates.some(
      (update) => !getDocumentLayer(document, update.id) || !Number.isFinite(update.x) || !Number.isFinite(update.y)
    )
  ) {
    return document;
  }
  return mapNodes(
    document,
    updates.map(({ id, x, y }) => [
      id,
      (node) =>
        isGroupNode(node) || (node.transform.x === x && node.transform.y === y)
          ? node
          : { ...node, transform: { ...node.transform, x, y } },
    ])
  );
};

/**
 * Swaps the leaf `layer.id` names for `layer`. A leaf of another type cannot stay in its forest, so
 * it leaves its group and lands in its new stack at `anchor`, or at the top like a fresh layer.
 */
const replaceLeaf = (
  projectId: string,
  document: CanvasDocumentContractV3,
  layer: CanvasLayerContract,
  anchor?: CanvasNodeInsertionAnchor
): CanvasDocumentContractV3 => {
  const existing = getDocumentLayer(document, layer.id);
  if (!existing) {
    return document;
  }
  if (existing.type === layer.type) {
    return mapLayer(document, layer.id, () => layer);
  }
  const target = layerStackOf(layer);
  if (anchor && (anchor.projectId !== projectId || anchor.stack !== target)) {
    return document;
  }
  const stacks = removeNodes(document.stacks, new Set([layer.id]));
  return withStacks(
    document,
    anchor ? insertNodesAtAnchor(stacks, anchor, [layer]) : { ...stacks, [target]: [layer, ...stacks[target]] }
  );
};

/** The stack every leaf of `nodes` belongs to, or `null` when they disagree; an empty group fits any stack. */
const nodesStack = (nodes: readonly CanvasNodeContract[], fallback: CanvasNodeInsertionAnchor['stack']) => {
  const stacks = new Set(nodes.flatMap((node) => collectSubtreeLeaves(node).map(layerStackOf)));
  return stacks.size === 0 ? fallback : stacks.size === 1 ? [...stacks][0]! : null;
};

const isInsertionValid = (projectId: string, insertion: CanvasNodeInsertion): boolean =>
  insertion.anchor.projectId === projectId &&
  nodesStack(insertion.nodes, insertion.anchor.stack) === insertion.anchor.stack &&
  (insertion.anchor.stack !== 'raster' ||
    insertion.nodes
      .flatMap((root) => collectSubtree(root))
      .every((node) => !isGroupNode(node) || node.isHidden === undefined));

/** The subtrees a move detaches, or `null` when its ids are not distinct, present nodes of the anchor's stack. */
const movedBlock = (index: CanvasDocumentIndex, move: CanvasNodeMove): CanvasNodeEntry[] | null => {
  const outer = outermostNodes(index, move.ids);
  return outer.length === new Set(move.ids).size && outer.every((entry) => entry.stack === move.anchor.stack)
    ? outer
    : null;
};

/** One move at a time; each anchor resolves against the forest the previous move produced. */
const applyMovesInSequence = (stacks: CanvasStackForests, moves: readonly CanvasNodeMove[]) => {
  let next = stacks;
  for (const move of moves) {
    const block = movedBlock(indexStacks(next), move);
    if (!block) {
      return null;
    }
    next = insertNodesAtAnchor(
      removeNodes(next, new Set(move.ids)),
      move.anchor,
      block.map((entry) => entry.node)
    );
  }
  return next;
};

const siblingKey = (stack: CanvasNodeInsertionAnchor['stack'], parentId: string | null): string =>
  `${stack}\0${parentId ?? ''}`;

/**
 * Batch rebuilds must match sequential moves. Use the sequential path when anchors name moving nodes or blocks
 * nest, because intermediate forests affect resolution.
 */
const applyMoves = (stacks: CanvasStackForests, moves: readonly CanvasNodeMove[], projectId: string) => {
  if (moves.length === 0) {
    return stacks;
  }
  if (moves.some((move) => move.anchor.projectId !== projectId)) {
    return null;
  }
  const index = indexStacks(stacks);
  const movingIds = new Set(moves.flatMap((move) => move.ids));
  if (movingIds.size !== moves.reduce((count, move) => count + new Set(move.ids).size, 0)) {
    return null;
  }
  const namesMoving = (id: string | null) => id !== null && movingIds.has(id);
  if (
    outermostNodes(index, movingIds).length !== movingIds.size ||
    moves.some(
      ({ anchor }) => namesMoving(anchor.beforeId) || namesMoving(anchor.afterId) || anchor.parentPath.some(namesMoving)
    )
  ) {
    return applyMovesInSequence(stacks, moves);
  }
  const blocks = moves.map((move) => movedBlock(index, move));
  if (blocks.some((block) => block === null)) {
    return null;
  }
  const removed = removeNodes(stacks, movingIds);
  const removedIndex = indexStacks(removed);
  const lists = new Map<string, CanvasNodeContract[]>();
  const listFor = (stack: CanvasNodeInsertionAnchor['stack'], parentId: string | null): CanvasNodeContract[] => {
    const key = siblingKey(stack, parentId);
    let list = lists.get(key);
    if (!list) {
      list = [...(childrenAt(removedIndex, stack, parentId) ?? [])];
      lists.set(key, list);
    }
    return list;
  };
  const survivingGroup = (stack: CanvasNodeInsertionAnchor['stack'], id: string): boolean => {
    const entry = removedIndex.byId.get(id);
    return !!entry && entry.stack === stack && isGroupNode(entry.node);
  };
  moves.forEach((move, position) => {
    const { anchor } = move;
    const nodes = blocks[position]!.map((entry) => entry.node);
    const before = anchor.beforeId ? removedIndex.byId.get(anchor.beforeId) : undefined;
    const after = anchor.afterId ? removedIndex.byId.get(anchor.afterId) : undefined;
    let list: CanvasNodeContract[];
    let at: number;
    if (before && before.stack === anchor.stack) {
      list = listFor(anchor.stack, before.parentId);
      at = list.findIndex((node) => node.id === anchor.beforeId);
    } else if (after && after.stack === anchor.stack) {
      list = listFor(anchor.stack, after.parentId);
      at = list.findIndex((node) => node.id === anchor.afterId) + 1;
    } else {
      const parentId = [...anchor.parentPath].reverse().find((id) => survivingGroup(anchor.stack, id)) ?? null;
      list = listFor(anchor.stack, parentId);
      at = 0;
    }
    list.splice(at, 0, ...nodes);
  });
  const materialize = (
    nodes: readonly CanvasNodeContract[],
    stack: CanvasNodeInsertionAnchor['stack']
  ): readonly CanvasNodeContract[] => {
    let changed = false;
    const next = nodes.map((node) => {
      if (!isGroupNode(node)) {
        return node;
      }
      const children = materialize(lists.get(siblingKey(stack, node.id)) ?? node.children, stack);
      if (children === node.children) {
        return node;
      }
      changed = true;
      return { ...node, children: [...children] };
    });
    return changed ? next : nodes;
  };
  const next = { ...removed };
  for (const stack of Object.keys(next) as CanvasNodeInsertionAnchor['stack'][]) {
    const roots = materialize(lists.get(siblingKey(stack, null)) ?? removed[stack], stack);
    if (roots !== removed[stack]) {
      next[stack] = [...roots];
    }
  }
  return next;
};

const applyLayerStackMutation = (
  projectId: string,
  document: CanvasDocumentContractV3,
  mutation: Extract<CanvasProjectMutation, { type: 'applyCanvasLayerStackMutation' }>
): CanvasDocumentContractV3 => {
  let stacks = document.stacks;
  const knownIds = new Set(getDocumentIndex(document).byId.keys());
  for (const insertion of mutation.add ?? []) {
    if (!isInsertionValid(projectId, insertion)) {
      return document;
    }
    for (const node of insertion.nodes.flatMap((root) => collectSubtree(root))) {
      if (knownIds.has(node.id)) {
        return document;
      }
      knownIds.add(node.id);
    }
    if (insertion.nodes.length > 0) {
      stacks = insertNodesAtAnchor(stacks, insertion.anchor, insertion.nodes);
    }
  }
  const moved = applyMoves(stacks, mutation.move ?? [], projectId);
  if (!moved) {
    return document;
  }
  const removeIds = new Set(mutation.removeIds ?? []);
  const beforeRemoval = indexStacks(moved);
  if ([...removeIds].some((id) => !beforeRemoval.byId.has(id))) {
    return document;
  }
  stacks = removeNodes(moved, removeIds);
  const index = indexStacks(stacks);
  if (
    mutation.enabledUpdates.some((update) => !index.byId.has(update.id)) ||
    (mutation.lockedUpdates?.some((update) => !index.byId.has(update.id)) ?? false) ||
    (mutation.selectedLayerId !== undefined &&
      mutation.selectedLayerId !== null &&
      !index.byId.has(mutation.selectedLayerId))
  ) {
    return document;
  }
  const enabledById = new Map(mutation.enabledUpdates.map((update) => [update.id, update.isEnabled]));
  const lockedById = new Map(mutation.lockedUpdates?.map((update) => [update.id, update.isLocked]) ?? []);
  stacks = updateNodeValues(
    stacks,
    new Map(
      [...new Set([...enabledById.keys(), ...lockedById.keys()])].map(
        (id): [string, (node: CanvasNodeContract) => CanvasNodeContract] => [
          id,
          (node) => {
            const isEnabled = enabledById.get(id) ?? node.isEnabled;
            const isLocked = lockedById.get(id) ?? node.isLocked;
            return isEnabled === node.isEnabled && isLocked === node.isLocked ? node : { ...node, isEnabled, isLocked };
          },
        ]
      )
    )
  );
  if (stacks !== document.stacks && !withinLimits(stacks)) {
    return document;
  }
  const selectedLayerId =
    mutation.selectedLayerId === undefined
      ? repairSelectedLayerId(stacks, document.selectedLayerId, document.stacks)
      : mutation.selectedLayerId;
  return stacks === document.stacks && selectedLayerId === document.selectedLayerId
    ? document
    : { ...document, selectedLayerId, stacks };
};

const addLayer = (
  projectId: string,
  document: CanvasDocumentContractV3,
  layer: CanvasLayerContract,
  anchor: CanvasNodeInsertionAnchor
): CanvasDocumentContractV3 => {
  if (!isInsertionValid(projectId, { anchor, nodes: [layer] }) || hasDocumentNode(document, layer.id)) {
    return document;
  }
  const stacks = insertNodesAtAnchor(document.stacks, anchor, [layer]);
  return withinLimits(stacks) ? { ...document, selectedLayerId: layer.id, stacks } : document;
};

const removeLayers = (document: CanvasDocumentContractV3, ids: readonly string[]): CanvasDocumentContractV3 => {
  const stacks = removeNodes(document.stacks, new Set(ids));
  if (stacks === document.stacks) {
    return document;
  }
  return {
    ...document,
    selectedLayerId: repairSelectedLayerId(stacks, document.selectedLayerId, document.stacks),
    stacks,
  };
};

const reorderCanvasSiblings = (
  document: CanvasDocumentContractV3,
  orders: readonly ReorderSiblingsCommand[]
): CanvasDocumentContractV3 => {
  if (new Set(orders.map((order) => `${order.stack}\0${order.parentId ?? ''}`)).size !== orders.length) {
    return document;
  }
  let stacks = document.stacks;
  for (const order of orders) {
    const next = reorderSiblings(stacks, order);
    if (!next) {
      return document;
    }
    stacks = next;
  }
  return withStacks(document, stacks);
};

/** Opacity/blend apply only to raster-stack groups, even from unvalidated dispatchers (previews, replays). */
const patchNode = (
  node: CanvasNodeContract,
  patch: CanvasLayerBasePatch,
  stack: LayerStackKind
): CanvasNodeContract => {
  if (isGroupNode(node)) {
    const allowed = Object.fromEntries(
      Object.entries(patch).filter(
        ([key]) =>
          GROUP_PATCH_KEYS.includes(key as keyof CanvasLayerBasePatch) &&
          (stack === 'raster' || (key !== 'opacity' && key !== 'blendMode'))
      )
    );
    return Object.keys(allowed).length === 0 ? node : { ...node, ...allowed };
  }
  const { transform, ...rest } = patch;
  return { ...node, ...rest, transform: transform ? { ...node.transform, ...transform } : node.transform };
};

/** Group configs apply only to raster-stack groups, even from unvalidated dispatchers (previews, replays). */
const isConfigTargetValid = (
  document: CanvasDocumentContractV3,
  id: string,
  config: CanvasLayerConfigPatch
): boolean => {
  const node = getDocumentNode(document, id);
  if (node === null || node.type !== config.layerType) {
    return false;
  }
  return config.layerType !== 'group' || getDocumentIndex(document).byId.get(id)?.stack === 'raster';
};

const patchLayerConfig = (layer: CanvasNodeContract, config: CanvasLayerConfigPatch): CanvasNodeContract => {
  if (layer.type !== config.layerType) {
    return layer;
  }
  if (layer.type === 'group' && config.layerType === 'group') {
    return {
      ...layer,
      ...(Object.hasOwn(config, 'adjustments') ? { adjustments: config.adjustments } : {}),
    };
  }
  if (layer.type === 'raster' && config.layerType === 'raster') {
    const next = {
      ...layer,
      ...(Object.hasOwn(config, 'adjustments') ? { adjustments: config.adjustments } : {}),
      ...(Object.hasOwn(config, 'inpaint') && config.inpaint !== null ? { inpaint: config.inpaint } : {}),
      ...(Object.hasOwn(config, 'isTransparencyLocked') ? { isTransparencyLocked: config.isTransparencyLocked } : {}),
      ...(Object.hasOwn(config, 'filter') ? { filter: config.filter } : {}),
    };
    if (Object.hasOwn(config, 'inpaint') && config.inpaint === null) {
      delete next.inpaint;
    }
    return next;
  }
  if (layer.type === 'control' && config.layerType === 'control') {
    return {
      ...layer,
      ...(config.adapter ? { adapter: { ...layer.adapter, ...config.adapter } } : {}),
      ...(Object.hasOwn(config, 'withTransparencyEffect')
        ? { withTransparencyEffect: config.withTransparencyEffect }
        : {}),
      ...(Object.hasOwn(config, 'filter') ? { filter: config.filter } : {}),
    };
  }
  if (layer.type === 'regional_guidance' && config.layerType === 'regional_guidance') {
    return {
      ...layer,
      ...(config.mask ? { mask: { ...layer.mask, ...config.mask } } : {}),
      ...(Object.hasOwn(config, 'positivePrompt') ? { positivePrompt: config.positivePrompt } : {}),
      ...(Object.hasOwn(config, 'negativePrompt') ? { negativePrompt: config.negativePrompt } : {}),
      ...(Object.hasOwn(config, 'autoNegative') ? { autoNegative: config.autoNegative } : {}),
      ...(Object.hasOwn(config, 'referenceImages') ? { referenceImages: config.referenceImages } : {}),
    };
  }
  if (layer.type === 'inpaint_mask' && config.layerType === 'inpaint_mask') {
    const next = {
      ...layer,
      ...(config.mask ? { mask: { ...layer.mask, ...config.mask } } : {}),
    };
    // `null` removes the modifier; absent leaves it untouched.
    if (Object.hasOwn(config, 'noise')) {
      if (config.noise) {
        next.noise = config.noise;
      } else {
        delete next.noise;
      }
    }
    if (Object.hasOwn(config, 'denoise')) {
      if (config.denoise) {
        next.denoise = config.denoise;
      } else {
        delete next.denoise;
      }
    }
    return next;
  }
  return layer;
};

const mergeLayersDown = (
  document: CanvasDocumentContractV3,
  mutation: Extract<CanvasProjectMutation, { type: 'mergeCanvasLayersDown' }>
): CanvasDocumentContractV3 => {
  const index = getDocumentIndex(document);
  const upper = index.byId.get(mutation.upperLayerId);
  if (!upper || upper.stack !== 'raster' || isGroupNode(upper.node)) {
    return document;
  }
  const below = childrenAt(index, upper.stack, upper.parentId)?.[upper.siblingIndex + 1];
  if (!below || below.type !== 'raster') {
    return document;
  }
  const merged: CanvasRasterLayerContractV2 = {
    blendMode: below.blendMode,
    id: below.id,
    isEnabled: below.isEnabled,
    isLocked: below.isLocked,
    name: below.name,
    opacity: below.opacity,
    source: mutation.source,
    transform: below.transform,
    type: 'raster',
  };
  const stacks = removeNodes(
    updateNodeValues(document.stacks, new Map([[below.id, () => merged]])),
    new Set([upper.node.id])
  );
  return {
    ...document,
    selectedLayerId: repairSelectedLayerId(stacks, document.selectedLayerId, document.stacks),
    stacks,
  };
};

const clampBbox = (bbox: CanvasDocumentContractV3['bbox'], width: number, height: number) => {
  const clampedWidth = Math.min(Math.max(1, Math.round(bbox.width)), width);
  const clampedHeight = Math.min(Math.max(1, Math.round(bbox.height)), height);
  return {
    height: clampedHeight,
    width: clampedWidth,
    x: Math.min(Math.max(0, Math.round(bbox.x)), width - clampedWidth),
    y: Math.min(Math.max(0, Math.round(bbox.y)), height - clampedHeight),
  };
};

const clearStagingArea = (stagingArea: CanvasStateContractV3['stagingArea']) => ({
  ...stagingArea,
  isVisible: false,
  pendingImageIds: [],
  pendingImages: [],
  selectedImageIndex: 0,
  sourceQueueItemId: undefined,
});

const clampStagedImageIndex = (imageIndex: number, slotCount: number): number =>
  Math.min(Math.max(0, slotCount - 1), Math.max(0, imageIndex));

const selectedCandidate = (project: Project) => {
  const slot = getCanvasStagingSlots(project.canvas, project.queue.items)[
    project.canvas.stagingArea.selectedImageIndex
  ];
  return slot?.kind === 'candidate' ? slot.candidate : undefined;
};

export const applyCanvasProjectMutation = (project: Project, mutation: CanvasProjectMutation): Project => {
  switch (mutation.type) {
    case 'commitStagedImage': {
      const stagedImage = selectedCandidate(project);
      if (
        project.canvas.stagingArea.selectedImageIndex !== mutation.selectedImageIndex ||
        !stagedImage ||
        getCanvasStagingCandidateFingerprint(stagedImage) !== mutation.candidateFingerprint
      ) {
        return project;
      }
      const { anchor, layer } = mutation;
      const { document } = project.canvas;
      if (!isInsertionValid(project.id, { anchor, nodes: [layer] }) || hasDocumentNode(document, layer.id)) {
        return project;
      }
      const stacks = insertNodesAtAnchor(document.stacks, anchor, [layer]);
      if (!withinLimits(stacks)) {
        return project;
      }
      const selectedLayerId = mutation.continueStaging
        ? document.selectedLayerId
        : selectionAfterAcceptedResult(document.stacks, document.selectedLayerId, layer.id);
      return {
        ...project,
        canvas: {
          ...project.canvas,
          document: { ...document, selectedLayerId, stacks },
          stagingArea: mutation.continueStaging
            ? project.canvas.stagingArea
            : clearStagingArea(project.canvas.stagingArea),
        },
        events: prependProjectEvent(project.events, mutation.event),
      };
    }
    case 'rollbackStagedImageCommit': {
      const { document } = project.canvas;
      const expectedSelectedLayerId = mutation.continueStaging
        ? mutation.selectedLayerId
        : selectionAfterAcceptedResult(document.stacks, mutation.selectedLayerId, mutation.layer.id);
      const stagingMatchesCommit = mutation.continueStaging
        ? project.canvas.stagingArea === mutation.stagingArea
        : project.canvas.stagingArea.pendingImages.length === 0;
      if (
        document.selectedLayerId !== expectedSelectedLayerId ||
        getDocumentLayer(document, mutation.layer.id) !== mutation.layer ||
        project.events[0] !== mutation.event ||
        !stagingMatchesCommit
      ) {
        return project;
      }
      return {
        ...project,
        canvas: {
          ...project.canvas,
          document: {
            ...document,
            selectedLayerId: mutation.selectedLayerId,
            stacks: removeNodes(document.stacks, new Set([mutation.layer.id])),
          },
          stagingArea: mutation.stagingArea,
        },
        events: project.events.slice(1),
      };
    }
    case 'setStagedImageIndex': {
      const selectedImageIndex = clampStagedImageIndex(
        mutation.imageIndex,
        getCanvasStagingSlotCount(project.canvas, project.queue.items)
      );
      return selectedImageIndex === project.canvas.stagingArea.selectedImageIndex
        ? project
        : {
            ...project,
            canvas: {
              ...project.canvas,
              stagingArea: { ...project.canvas.stagingArea, selectedImageIndex },
            },
          };
    }
    case 'cycleStagedImage': {
      const count = getCanvasStagingSlotCount(project.canvas, project.queue.items);
      const current = project.canvas.stagingArea.selectedImageIndex;
      const selectedImageIndex = count < 2 ? 0 : (current + mutation.direction + count) % count;
      return selectedImageIndex === current
        ? project
        : {
            ...project,
            canvas: {
              ...project.canvas,
              stagingArea: { ...project.canvas.stagingArea, selectedImageIndex },
            },
          };
    }
    case 'discardSelectedStagedImage': {
      const selected = selectedCandidate(project);
      if (!selected) {
        return project;
      }
      const pendingImages = project.canvas.stagingArea.pendingImages.filter(
        (image) => image.sourceQueueItemId !== selected.sourceQueueItemId || image.imageName !== selected.imageName
      );
      const canvas = {
        ...project.canvas,
        stagingArea: {
          ...project.canvas.stagingArea,
          pendingImageIds: pendingImages.map((image) => image.imageName),
          pendingImages,
        },
      };
      const slotCount = getCanvasStagingSlotCount(canvas, project.queue.items);
      return {
        ...project,
        canvas: {
          ...canvas,
          stagingArea: {
            ...canvas.stagingArea,
            isVisible: slotCount > 0 && canvas.stagingArea.isVisible,
            selectedImageIndex: clampStagedImageIndex(canvas.stagingArea.selectedImageIndex, slotCount),
            sourceQueueItemId: pendingImages.length > 0 ? canvas.stagingArea.sourceQueueItemId : undefined,
          },
        },
      };
    }
    case 'discardAllStagedImages': {
      const canvas = { ...project.canvas, stagingArea: clearStagingArea(project.canvas.stagingArea) };
      return {
        ...project,
        canvas: {
          ...canvas,
          stagingArea: {
            ...canvas.stagingArea,
            isVisible: getCanvasStagingSlotCount(canvas, project.queue.items) > 0,
          },
        },
      };
    }
    case 'toggleCanvasStagingVisibility':
      return getCanvasStagingSlotCount(project.canvas, project.queue.items) === 0
        ? project
        : {
            ...project,
            canvas: {
              ...project.canvas,
              stagingArea: {
                ...project.canvas.stagingArea,
                isVisible: !project.canvas.stagingArea.isVisible,
              },
            },
          };
    case 'toggleCanvasStagingThumbnailsVisibility':
      return getCanvasStagingSlotCount(project.canvas, project.queue.items) === 0
        ? project
        : {
            ...project,
            canvas: {
              ...project.canvas,
              stagingArea: {
                ...project.canvas.stagingArea,
                areThumbnailsVisible: !project.canvas.stagingArea.areThumbnailsVisible,
              },
            },
          };
    case 'clearCanvasStaging':
      return { ...project, canvas: { ...project.canvas, stagingArea: clearStagingArea(project.canvas.stagingArea) } };
    case 'addCanvasLayer':
      return updateCanvasDocument(project, (document) =>
        addLayer(project.id, document, mutation.layer, mutation.anchor)
      );
    case 'applyCanvasLayerStackMutation':
      return updateCanvasDocument(project, (document) => applyLayerStackMutation(project.id, document, mutation));
    case 'removeCanvasLayers':
      return updateCanvasDocument(project, (document) => removeLayers(document, mutation.ids));
    case 'reorderCanvasSiblings':
      return updateCanvasDocument(project, (document) => reorderCanvasSiblings(document, mutation.orders));
    case 'updateCanvasLayer':
      return updateCanvasDocument(project, (document) =>
        mapNode(document, mutation.id, (node) =>
          patchNode(node, mutation.patch, getDocumentIndex(document).byId.get(mutation.id)?.stack ?? 'raster')
        )
      );
    case 'replaceCanvasLayer':
      return mutation.layer.id === mutation.layerId
        ? updateCanvasDocument(project, (document) => replaceLeaf(project.id, document, mutation.layer))
        : project;
    case 'setCanvasLayersEnabled':
      return updateCanvasDocument(project, (document) => setCanvasLayersEnabled(document, mutation.updates));
    case 'setCanvasLayerPositions':
      return updateCanvasDocument(project, (document) => setCanvasLayerPositions(document, mutation.updates));
    case 'setCanvasLayersHidden':
      return updateCanvasDocument(project, (document) => setCanvasLayersHidden(document, mutation.updates));
    case 'updateCanvasLayerSource':
      return updateCanvasDocument(project, (document) =>
        mapLayer(document, mutation.id, (layer) =>
          layer.type === 'raster' || layer.type === 'control' ? { ...layer, source: mutation.source } : layer
        )
      );
    case 'updateCanvasLayerConfig':
      return updateCanvasDocument(project, (document) =>
        isConfigTargetValid(document, mutation.id, mutation.config)
          ? mapNode(document, mutation.id, (node) => patchLayerConfig(node, mutation.config))
          : document
      );
    case 'updateCanvasLayerConfigs':
      return updateCanvasDocument(project, (document) => {
        // Reject the whole batch if any target cannot resolve; history replay must be atomic.
        const applicable = mutation.updates.every((update) => isConfigTargetValid(document, update.id, update.config));
        if (!applicable) {
          return document;
        }
        // One value edit for the batch; repeated ids apply their patches in order.
        const patches = new Map<string, (node: CanvasNodeContract) => CanvasNodeContract>();
        for (const { config, id } of mutation.updates) {
          const earlier = patches.get(id);
          patches.set(id, (node) => patchLayerConfig(earlier ? earlier(node) : node, config));
        }
        return mapNodes(document, patches);
      });
    case 'convertCanvasLayer': {
      if (mutation.layer.type !== mutation.targetType) {
        return project;
      }
      const converted = structuredClone(mutation.layer);
      converted.id = mutation.id;
      return updateCanvasDocument(project, (document) => replaceLeaf(project.id, document, converted, mutation.anchor));
    }
    case 'mergeCanvasLayersDown':
      return updateCanvasDocument(project, (document) => mergeLayersDown(document, mutation));
    case 'setCanvasBbox':
      return updateCanvasDocument(project, (document) => ({
        ...document,
        bbox: {
          height: Math.max(1, Math.round(mutation.bbox.height)),
          width: Math.max(1, Math.round(mutation.bbox.width)),
          x: Math.round(mutation.bbox.x),
          y: Math.round(mutation.bbox.y),
        },
      }));
    case 'setCanvasSelectedLayer':
      return updateCanvasDocument(project, (document) =>
        (mutation.id !== null && !hasDocumentNode(document, mutation.id)) || document.selectedLayerId === mutation.id
          ? document
          : { ...document, selectedLayerId: mutation.id }
      );
    case 'resizeCanvasDocument': {
      const width = Math.max(1, Math.round(mutation.width));
      const height = Math.max(1, Math.round(mutation.height));
      const offsetX = mutation.offsetX ?? 0;
      const offsetY = mutation.offsetY ?? 0;
      return updateCanvasDocument(project, (document) => ({
        ...document,
        bbox: clampBbox(
          { ...document.bbox, x: document.bbox.x + offsetX, y: document.bbox.y + offsetY },
          width,
          height
        ),
        height,
        stacks:
          offsetX === 0 && offsetY === 0
            ? document.stacks
            : updateNodeValues(
                document.stacks,
                new Map(
                  getDocumentIndex(document).leaves.map(
                    (leaf): [string, (node: CanvasNodeContract) => CanvasNodeContract] => [
                      leaf.id,
                      (node) =>
                        isGroupNode(node)
                          ? node
                          : {
                              ...node,
                              transform: {
                                ...node.transform,
                                x: node.transform.x + offsetX,
                                y: node.transform.y + offsetY,
                              },
                            },
                    ]
                  )
                )
              ),
        width,
      }));
    }
    case 'replaceCanvasDocument': {
      const document = normalizeCanvasDocumentContract(structuredClone(mutation.document));
      return document
        ? setCanvasState(project, {
            ...project.canvas,
            document: withRepairedSelection(document),
            documentRevision: project.canvas.documentRevision + 1,
            stagingArea: clearStagingArea(project.canvas.stagingArea),
          })
        : project;
    }
    case 'replaceCanvasFontReferences': {
      const document = normalizeCanvasDocumentContract(structuredClone(mutation.document));
      return document
        ? setCanvasState(project, {
            ...project.canvas,
            document: withRepairedSelection(document),
          })
        : project;
    }
    case 'saveCanvasSnapshot': {
      const document = normalizeCanvasDocumentContract(structuredClone(project.canvas.document));
      return document
        ? setCanvasState(project, {
            ...project.canvas,
            snapshots: [
              ...project.canvas.snapshots,
              { createdAt: mutation.createdAt, document, id: mutation.id, name: mutation.name },
            ],
          })
        : project;
    }
    case 'restoreCanvasSnapshot': {
      const snapshot = project.canvas.snapshots.find((entry) => entry.id === mutation.snapshotId);
      const document = snapshot ? normalizeCanvasDocumentContract(structuredClone(snapshot.document)) : null;
      return document
        ? setCanvasState(project, {
            ...project.canvas,
            document: withRepairedSelection(document),
            documentRevision: project.canvas.documentRevision + 1,
          })
        : project;
    }
    case 'deleteCanvasSnapshot': {
      const snapshots = project.canvas.snapshots.filter((entry) => entry.id !== mutation.snapshotId);
      return snapshots.length === project.canvas.snapshots.length
        ? project
        : setCanvasState(project, { ...project.canvas, snapshots });
    }
    case 'setCanvasStagingAutoSwitch':
      return project.canvas.stagingArea.autoSwitchMode === mutation.mode
        ? project
        : {
            ...project,
            canvas: {
              ...project.canvas,
              stagingArea: { ...project.canvas.stagingArea, autoSwitchMode: mutation.mode },
            },
          };
  }
};
