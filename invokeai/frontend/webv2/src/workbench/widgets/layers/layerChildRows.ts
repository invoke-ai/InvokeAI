import type {
  CanvasAdjustmentEntry,
  CanvasDocumentContractV3,
  CanvasInpaintMaskLayerContract,
  DocumentCommand,
  LayerStackKind,
  SemanticNode,
} from '@workbench/canvas-engine/api';

import { getDocumentLayer, getDocumentNode } from '@workbench/canvas-engine/api';
import { layerChildRowKey } from '@workbench/layerPanelState';

export { layerChildRowKey };

/**
 * Project modifier rows from existing layer config without expanding the document node union; edits use the same
 * patch-config seam as Properties.
 */

export type LayerChildRowKind =
  | 'reference-image'
  | 'mask-noise'
  | 'mask-denoise'
  | 'layer-region'
  | 'adjustment-brightness-contrast'
  | 'adjustment-exposure'
  | 'adjustment-levels'
  | 'adjustment-curves'
  | 'adjustment-hsl'
  | 'adjustment-hue'
  | 'adjustment-invert';

type AdjustmentChildKind = Extract<LayerChildRowKind, `adjustment-${string}`>;

/** The row kind projecting an adjustment entry of `type`. */
export const adjustmentChildKind = (type: CanvasAdjustmentEntry['type']): AdjustmentChildKind =>
  ADJUSTMENT_KIND_OF[type];

const ADJUSTMENT_KIND_OF: Record<CanvasAdjustmentEntry['type'], AdjustmentChildKind> = {
  'brightness-contrast': 'adjustment-brightness-contrast',
  curves: 'adjustment-curves',
  exposure: 'adjustment-exposure',
  hsl: 'adjustment-hsl',
  hue: 'adjustment-hue',
  invert: 'adjustment-invert',
  levels: 'adjustment-levels',
};

const CHILD_ROW_NAME_KEYS: Record<Exclude<LayerChildRowKind, 'reference-image'>, string> = {
  'adjustment-brightness-contrast': 'widgets.layers.modifiers.brightnessContrast',
  'adjustment-curves': 'widgets.layers.adjustments.curves',
  'adjustment-exposure': 'widgets.layers.adjustments.exposure',
  'adjustment-hsl': 'widgets.layers.adjustments.saturation',
  'adjustment-hue': 'widgets.layers.adjustments.hue',
  'adjustment-invert': 'widgets.layers.adjustments.invert',
  'adjustment-levels': 'widgets.layers.adjustments.levels',
  'layer-region': 'widgets.layers.modifiers.regenerateRegion',
  'mask-denoise': 'widgets.layers.modifiers.denoise',
  'mask-noise': 'widgets.layers.modifiers.noise',
};

/** The i18n key naming a non-reference child row's kind, for rows and editor subtitles alike. */
export const childRowNameKey = (kind: Exclude<LayerChildRowKind, 'reference-image'>): string =>
  CHILD_ROW_NAME_KEYS[kind];

const signedPercent = (value: number): string => `${value > 0 ? '+' : ''}${Math.round(value * 100)}%`;

/** The compact fact shown beside an adjustment row's name, or `null` when it has none. */
const adjustmentDetail = (entry: CanvasAdjustmentEntry): string | null => {
  switch (entry.type) {
    case 'hsl':
      return signedPercent(entry.saturation);
    case 'hue':
      return `${Math.round(entry.rotation)}°`;
    case 'exposure': {
      const stops = Number(entry.stops.toFixed(2));
      return `${stops > 0 ? '+' : ''}${stops} EV`;
    }
    case 'levels':
      return entry.channel && entry.channel !== 'rgb' ? entry.channel.toUpperCase() : null;
    case 'brightness-contrast':
    case 'curves':
    case 'invert':
      return null;
  }
};

/** Kinds whose list order is document truth; their rows offer Move up/down and Duplicate. */
export const isOrderedChildKind = (kind: LayerChildRowKind): boolean => kind.startsWith('adjustment-');

/** The synthetic item ids of a mask's singleton modifiers. */
export const MASK_NOISE_ITEM_ID = 'noise';
export const MASK_DENOISE_ITEM_ID = 'denoise';
/** The synthetic item id of a raster layer's singleton regenerate region. */
export const LAYER_REGION_ITEM_ID = 'inpaint';

export interface ProjectedChildRow {
  /** Tree key, also the row's DOM id: `child:{layerId}:{itemId}`. */
  readonly key: string;
  readonly kind: LayerChildRowKind;
  readonly layerId: string;
  readonly itemId: string;
  readonly stack: LayerStackKind;
  /** One level below the owning layer. */
  readonly depth: number;
  readonly posInSet: number;
  readonly setSize: number;
  /** Position among the layer's ordered (reorderable) rows only; absent on singleton rows. */
  readonly orderedPosInSet?: number;
  readonly orderedSetSize?: number;
  readonly isEnabled: boolean;
  /** The owning layer and its ancestors are enabled; a gated dot when not. */
  readonly parentContributing: boolean;
  readonly image: { readonly imageName: string; readonly thumbnailUrl: string } | null;
  /** A compact preformatted fact shown beside the name (a percent, degrees), or `null`. */
  readonly detail: string | null;
  /** The user-given name of a renameable item (adjustment entries), or `null`. */
  readonly customName: string | null;
}

export type LayerChildRowAction =
  | { type: 'set-enabled'; isEnabled: boolean }
  | { type: 'remove' }
  | { type: 'move'; direction: -1 | 1 }
  | { type: 'duplicate'; newId: string }
  | { type: 'rename'; name: string | null };

/** A child row's live item facts, resolved from the document; `null` when it is gone. */
export interface LayerChildItem {
  readonly kind: LayerChildRowKind;
  readonly isEnabled: boolean;
}

/** The node whose adjustment stack a child row edits: a raster layer or a raster-stack group. */
const adjustmentOwnerNode = (document: CanvasDocumentContractV3, id: string) => {
  const node = getDocumentNode(document, id);
  return node && (node.type === 'raster' || node.type === 'group') ? node : null;
};

const adjustmentsPatch = (
  ownerType: 'raster' | 'group',
  id: string,
  before: readonly CanvasAdjustmentEntry[],
  next: CanvasAdjustmentEntry[]
): PatchConfigCommand =>
  ownerType === 'group'
    ? {
        before: { adjustments: [...before], layerType: 'group' },
        config: { adjustments: next, layerType: 'group' },
        id,
        type: 'patch-config',
      }
    : {
        before: { adjustments: [...before], layerType: 'raster' },
        config: { adjustments: next, layerType: 'raster' },
        id,
        type: 'patch-config',
      };

const EMPTY_CHILD_ROWS: readonly ProjectedChildRow[] = [];

const childRowsByNode = new WeakMap<SemanticNode, readonly ProjectedChildRow[]>();

const baseRow = (vm: SemanticNode) => ({
  depth: vm.depth + 1,
  layerId: vm.node.id,
  parentContributing: vm.contributionEnabled,
  stack: vm.stack,
});

const maskModifierRows = (vm: SemanticNode, layer: CanvasInpaintMaskLayerContract): ProjectedChildRow[] => {
  const items = [
    layer.noise
      ? {
          detail: `${Math.round(layer.noise.level * 100)}%`,
          isEnabled: layer.noise.isEnabled,
          itemId: MASK_NOISE_ITEM_ID,
          kind: 'mask-noise' as const,
        }
      : null,
    layer.denoise
      ? {
          detail: `${Math.round(layer.denoise.limit * 100)}%`,
          isEnabled: layer.denoise.isEnabled,
          itemId: MASK_DENOISE_ITEM_ID,
          kind: 'mask-denoise' as const,
        }
      : null,
  ].filter((item) => item !== null);
  return items.map((item, index) => ({
    ...baseRow(vm),
    ...item,
    customName: null,
    image: null,
    key: layerChildRowKey(layer.id, item.itemId),
    posInSet: index + 1,
    setSize: items.length,
  }));
};

/** The child rows a layer projects; the identical array while the node is unchanged. */
export const projectLayerChildRows = (vm: SemanticNode): readonly ProjectedChildRow[] => {
  const { node } = vm;
  const cached = childRowsByNode.get(vm);
  if (cached) {
    return cached;
  }
  let rows: readonly ProjectedChildRow[];
  if (node.type === 'regional_guidance' && node.referenceImages.length > 0) {
    const setSize = node.referenceImages.length;
    rows = node.referenceImages.map((ref, index): ProjectedChildRow => ({
      ...baseRow(vm),
      customName: null,
      detail: null,
      image: ref.config.image
        ? { imageName: ref.config.image.imageName, thumbnailUrl: ref.config.image.thumbnailUrl }
        : null,
      isEnabled: ref.isEnabled,
      itemId: ref.id,
      key: layerChildRowKey(node.id, ref.id),
      kind: 'reference-image',
      posInSet: index + 1,
      setSize,
    }));
  } else if (node.type === 'inpaint_mask' && (node.noise || node.denoise)) {
    rows = maskModifierRows(vm, node);
  } else if (
    (node.type === 'raster' || node.type === 'group') &&
    ((node.adjustments && node.adjustments.length > 0) || (node.type === 'raster' && node.inpaint))
  ) {
    const region = node.type === 'raster' ? node.inpaint : undefined;
    // A foreign adjustment claiming the region's reserved item id would alias its row key.
    const adjustments = (node.adjustments ?? []).filter((entry) => !region || entry.id !== LAYER_REGION_ITEM_ID);
    const setSize = adjustments.length + (region ? 1 : 0);
    const regionRows: ProjectedChildRow[] = region
      ? [
          {
            ...baseRow(vm),
            customName: region.name ?? null,
            detail: null,
            image: null,
            isEnabled: region.isEnabled,
            itemId: LAYER_REGION_ITEM_ID,
            key: layerChildRowKey(node.id, LAYER_REGION_ITEM_ID),
            kind: 'layer-region',
            posInSet: 1,
            setSize,
          },
        ]
      : [];
    rows = [
      ...regionRows,
      ...adjustments.map((adjustment, index): ProjectedChildRow => ({
        ...baseRow(vm),
        customName: adjustment.name ?? null,
        detail: adjustmentDetail(adjustment),
        image: null,
        isEnabled: adjustment.isEnabled,
        itemId: adjustment.id,
        key: layerChildRowKey(node.id, adjustment.id),
        kind: ADJUSTMENT_KIND_OF[adjustment.type],
        orderedPosInSet: index + 1,
        orderedSetSize: adjustments.length,
        posInSet: regionRows.length + index + 1,
        setSize,
      })),
    ];
  } else {
    return EMPTY_CHILD_ROWS;
  }
  childRowsByNode.set(vm, rows);
  return rows;
};

/** Resolves a child row's item from the live document; `null` when the layer or item is gone. */
export const getLayerChildItem = (
  document: CanvasDocumentContractV3,
  layerId: string,
  itemId: string
): LayerChildItem | null => {
  const layer = getDocumentLayer(document, layerId);
  if (layer?.type === 'regional_guidance') {
    const ref = layer.referenceImages.find((entry) => entry.id === itemId);
    return ref ? { isEnabled: ref.isEnabled, kind: 'reference-image' } : null;
  }
  if (layer?.type === 'inpaint_mask') {
    if (itemId === MASK_NOISE_ITEM_ID && layer.noise) {
      return { isEnabled: layer.noise.isEnabled, kind: 'mask-noise' };
    }
    if (itemId === MASK_DENOISE_ITEM_ID && layer.denoise) {
      return { isEnabled: layer.denoise.isEnabled, kind: 'mask-denoise' };
    }
  }
  const owner = layer ?? adjustmentOwnerNode(document, layerId);
  if (owner && (owner.type === 'raster' || owner.type === 'group')) {
    if (owner.type === 'raster' && itemId === LAYER_REGION_ITEM_ID && owner.inpaint) {
      return { isEnabled: owner.inpaint.isEnabled, kind: 'layer-region' };
    }
    const entry = owner.adjustments?.find((candidate) => candidate.id === itemId);
    return entry ? { isEnabled: entry.isEnabled, kind: ADJUSTMENT_KIND_OF[entry.type] } : null;
  }
  return null;
};

/** The i18n key naming a child row's rename, for menus and history entries alike. */
export const layerChildRenameLabelKey = (kind: LayerChildRowKind): string =>
  kind === 'layer-region' ? 'widgets.layers.modifiers.renameRegion' : 'widgets.layers.modifiers.renameAdjustment';

/** The i18n key naming a child row's removal, for menus and history entries alike. */
export const layerChildRemoveLabelKey = (kind: LayerChildRowKind): string => {
  switch (kind) {
    case 'reference-image':
      return 'widgets.layers.regionalGuidance.removeReferenceImage';
    case 'mask-noise':
      return 'widgets.layers.modifiers.removeNoise';
    case 'mask-denoise':
      return 'widgets.layers.modifiers.removeDenoise';
    case 'layer-region':
      return 'widgets.layers.modifiers.removeRegion';
    case 'adjustment-brightness-contrast':
    case 'adjustment-exposure':
    case 'adjustment-levels':
    case 'adjustment-curves':
    case 'adjustment-hsl':
    case 'adjustment-hue':
    case 'adjustment-invert':
      return 'widgets.layers.modifiers.removeAdjustment';
  }
};

type PatchConfigCommand = Extract<DocumentCommand, { type: 'patch-config' }>;

/** Where a dragged child row lands: before `beforeItemId` among `layerId`'s items, or at the end. */
export interface LayerChildDropTarget {
  readonly layerId: string;
  readonly beforeItemId: string | null;
}

const insertAt = <T extends { id: string }>(items: readonly T[], item: T, beforeItemId: string | null): T[] => {
  const index = beforeItemId === null ? items.length : items.findIndex((candidate) => candidate.id === beforeItemId);
  return index < 0 ? [...items, item] : [...items.slice(0, index), item, ...items.slice(index)];
};

/**
 * The document command a child-row drag resolves to, or `null` when it is a
 * no-op or the landing is invalid. Adjustment entries reorder within their
 * layer; a reference image also moves to another regional layer as ONE atomic
 * cross-layer edit.
 */
export const layerChildDropCommand = (
  document: CanvasDocumentContractV3,
  child: Pick<ProjectedChildRow, 'kind' | 'layerId' | 'itemId'>,
  target: LayerChildDropTarget
): DocumentCommand | null => {
  if (target.beforeItemId === child.itemId) {
    return null;
  }
  const source = getDocumentLayer(document, child.layerId);
  if (isOrderedChildKind(child.kind)) {
    const sourceOwner =
      source?.type === 'raster' ? source : source ? null : adjustmentOwnerNode(document, child.layerId);
    if (target.layerId !== child.layerId || !sourceOwner) {
      return null;
    }
    const before = sourceOwner.adjustments ?? [];
    const entry = before.find((candidate) => candidate.id === child.itemId);
    if (!entry) {
      return null;
    }
    // The region row always renders above the adjustments; nothing lands before it.
    if (
      target.beforeItemId === LAYER_REGION_ITEM_ID &&
      sourceOwner.type === 'raster' &&
      sourceOwner.inpaint !== undefined
    ) {
      return null;
    }
    const next = insertAt(
      before.filter((candidate) => candidate.id !== child.itemId),
      entry,
      target.beforeItemId
    );
    if (next.every((candidate, index) => candidate === before[index])) {
      return null;
    }
    return adjustmentsPatch(sourceOwner.type, child.layerId, before, next);
  }
  if (child.kind !== 'reference-image' || source?.type !== 'regional_guidance') {
    return null;
  }
  const ref = source.referenceImages.find((candidate) => candidate.id === child.itemId);
  if (!ref) {
    return null;
  }
  if (target.layerId === child.layerId) {
    const before = source.referenceImages;
    const next = insertAt(
      before.filter((candidate) => candidate.id !== child.itemId),
      ref,
      target.beforeItemId
    );
    if (next.every((candidate, index) => candidate === before[index])) {
      return null;
    }
    return {
      before: { layerType: 'regional_guidance', referenceImages: [...before] },
      config: { layerType: 'regional_guidance', referenceImages: next },
      id: child.layerId,
      type: 'patch-config',
    };
  }
  const destination = getDocumentLayer(document, target.layerId);
  if (destination?.type !== 'regional_guidance') {
    return null;
  }
  // A duplicated layer can carry the same item id (cloneSubtree re-mints only
  // node ids); landing there would alias two rows onto one key. Refuse.
  if (destination.referenceImages.some((candidate) => candidate.id === child.itemId)) {
    return null;
  }
  return {
    patches: [
      {
        before: { layerType: 'regional_guidance', referenceImages: [...source.referenceImages] },
        config: {
          layerType: 'regional_guidance',
          referenceImages: source.referenceImages.filter((candidate) => candidate.id !== child.itemId),
        },
        id: child.layerId,
      },
      {
        before: { layerType: 'regional_guidance', referenceImages: [...destination.referenceImages] },
        config: {
          layerType: 'regional_guidance',
          referenceImages: insertAt(destination.referenceImages, ref, target.beforeItemId),
        },
        id: target.layerId,
      },
    ],
    type: 'patch-config-batch',
  };
};

/** Resolve child actions to whole-value patches, or null for missing/no-op targets. */
export const layerChildRowCommand = (
  document: CanvasDocumentContractV3,
  target: Pick<ProjectedChildRow, 'layerId' | 'itemId'>,
  action: LayerChildRowAction
): PatchConfigCommand | null => {
  const layer = getDocumentLayer(document, target.layerId);
  if (layer?.type === 'regional_guidance') {
    if (action.type === 'move' || action.type === 'duplicate' || action.type === 'rename') {
      return null;
    }
    if (!layer.referenceImages.some((ref) => ref.id === target.itemId)) {
      return null;
    }
    const before = layer.referenceImages;
    const next =
      action.type === 'remove'
        ? before.filter((ref) => ref.id !== target.itemId)
        : before.map((ref) => (ref.id === target.itemId ? { ...ref, isEnabled: action.isEnabled } : ref));
    if (action.type === 'set-enabled' && before.every((ref, index) => next[index]!.isEnabled === ref.isEnabled)) {
      return null;
    }
    return {
      before: { layerType: 'regional_guidance', referenceImages: [...before] },
      config: { layerType: 'regional_guidance', referenceImages: next },
      id: target.layerId,
      type: 'patch-config',
    };
  }
  if (layer?.type === 'inpaint_mask') {
    if (action.type === 'move' || action.type === 'duplicate' || action.type === 'rename') {
      return null;
    }
    const field =
      target.itemId === MASK_NOISE_ITEM_ID ? 'noise' : target.itemId === MASK_DENOISE_ITEM_ID ? 'denoise' : null;
    const current = field ? layer[field] : undefined;
    if (!field || !current) {
      return null;
    }
    const next = action.type === 'remove' ? null : { ...current, isEnabled: action.isEnabled };
    if (next && next.isEnabled === current.isEnabled) {
      return null;
    }
    return {
      before: { [field]: current, layerType: 'inpaint_mask' },
      config: { [field]: next, layerType: 'inpaint_mask' },
      id: target.layerId,
      type: 'patch-config',
    };
  }
  const owner = layer?.type === 'raster' ? layer : layer ? null : adjustmentOwnerNode(document, target.layerId);
  if (owner?.type === 'raster' && target.itemId === LAYER_REGION_ITEM_ID && owner.inpaint) {
    const current = owner.inpaint;
    if (action.type === 'move' || action.type === 'duplicate') {
      return null;
    }
    let next: typeof current | null;
    if (action.type === 'remove') {
      next = null;
    } else if (action.type === 'set-enabled') {
      if (current.isEnabled === action.isEnabled) {
        return null;
      }
      next = { ...current, isEnabled: action.isEnabled };
    } else {
      if ((current.name ?? null) === action.name) {
        return null;
      }
      const { name: _cleared, ...rest } = current;
      next = action.name === null ? rest : { ...current, name: action.name };
    }
    return {
      before: { inpaint: current, layerType: 'raster' },
      config: { inpaint: next, layerType: 'raster' },
      id: target.layerId,
      type: 'patch-config',
    };
  }
  if (owner) {
    const before = owner.adjustments ?? [];
    const index = before.findIndex((entry) => entry.id === target.itemId);
    if (index < 0) {
      return null;
    }
    let next: CanvasAdjustmentEntry[];
    switch (action.type) {
      case 'set-enabled': {
        if (before[index]!.isEnabled === action.isEnabled) {
          return null;
        }
        next = before.map((entry, i) => (i === index ? { ...entry, isEnabled: action.isEnabled } : entry));
        break;
      }
      case 'remove':
        next = before.filter((_, i) => i !== index);
        break;
      case 'move': {
        const to = index + action.direction;
        if (to < 0 || to >= before.length) {
          return null;
        }
        next = [...before];
        [next[index], next[to]] = [next[to]!, next[index]!];
        break;
      }
      case 'duplicate': {
        const source = before[index]!;
        // The copy must not alias the source's nested curve arrays.
        const copy =
          source.type === 'curves'
            ? { ...source, curves: structuredClone(source.curves), id: action.newId }
            : { ...source, id: action.newId };
        next = [...before.slice(0, index + 1), copy, ...before.slice(index + 1)];
        break;
      }
      case 'rename': {
        const current = before[index]!;
        if ((current.name ?? null) === action.name) {
          return null;
        }
        next = before.map((entry, i) => {
          if (i !== index) {
            return entry;
          }
          const { name: _cleared, ...rest } = entry;
          return action.name === null ? rest : { ...entry, name: action.name };
        });
        break;
      }
    }
    return adjustmentsPatch(owner.type, target.layerId, before, next);
  }
  return null;
};
