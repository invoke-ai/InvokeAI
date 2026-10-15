import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasNodeContract,
  CanvasStackForests,
  CanvasTextFontRef,
  CanvasTextFontStyle,
  CanvasTextFontVariations,
} from './contracts';

type CanvasSourceLayerContract = Extract<CanvasLayerContract, { type: 'raster' | 'control' }>;

/** The range of one OpenType variation axis exposed by a replacement face. */
export interface CanvasFontAxisRange {
  tag: string;
  minimum: number;
  maximum: number;
  default: number;
}

/** File metadata and axis ranges used to reset typography when replacing a face. */
export interface CanvasFontReplacementTarget {
  fontRef: CanvasTextFontRef;
  axes: readonly CanvasFontAxisRange[];
  style: CanvasTextFontStyle;
  weight: number;
}

/** One exact persisted reference and the text layers that use it. */
export interface CanvasFontReferenceGroup {
  readonly fontRef: CanvasTextFontRef;
  readonly count: number;
  readonly layerIds: readonly string[];
}

/** Details returned by a replace-all operation before it is added to history. */
export interface CanvasFontReplacementSummary {
  readonly document: CanvasDocumentContractV3;
  readonly from: CanvasTextFontRef;
  readonly to: CanvasTextFontRef;
  readonly replacedLayerIds: readonly string[];
  readonly replacedCount: number;
  /** Coordinates retained on the target face after clamping to its ranges. */
  readonly clampedAxisTags: readonly string[];
  /** Coordinates present on the source face but absent from the target face. */
  readonly resetAxisTags: readonly string[];
}

export const sameCanvasTextFontRef = (left: CanvasTextFontRef, right: CanvasTextFontRef): boolean =>
  left.id === right.id &&
  left.contentHash === right.contentHash &&
  left.family === right.family &&
  left.label === right.label;

const referenceKey = (reference: CanvasTextFontRef): string =>
  JSON.stringify([reference.id, reference.contentHash, reference.family, reference.label]);

const visit = (nodes: readonly CanvasNodeContract[], callback: (layer: CanvasLayerContract) => void): void => {
  for (const node of nodes) {
    if (node.type === 'group') {
      visit(node.children, callback);
    } else {
      callback(node);
    }
  }
};

const visitStacks = (stacks: CanvasStackForests, callback: (layer: CanvasLayerContract) => void): void => {
  visit(stacks.raster, callback);
  visit(stacks.control, callback);
  visit(stacks.regional_guidance, callback);
  visit(stacks.inpaint_mask, callback);
};

const hasSource = (layer: CanvasLayerContract): layer is CanvasSourceLayerContract =>
  layer.type === 'raster' || layer.type === 'control';

/** Collects custom text references in deterministic stack/tree order. */
export const collectCanvasFontReferences = (
  document: CanvasDocumentContractV3
): readonly CanvasFontReferenceGroup[] => {
  const groups = new Map<string, { fontRef: CanvasTextFontRef; layerIds: string[] }>();
  visitStacks(document.stacks, (layer) => {
    if (!hasSource(layer) || layer.source.type !== 'text' || !layer.source.fontRef) {
      return;
    }
    const key = referenceKey(layer.source.fontRef);
    const group = groups.get(key);
    if (group) {
      group.layerIds.push(layer.id);
      return;
    }
    groups.set(key, { fontRef: { ...layer.source.fontRef }, layerIds: [layer.id] });
  });
  return [...groups.values()].map(({ fontRef, layerIds }) => ({
    count: layerIds.length,
    fontRef,
    layerIds,
  }));
};

const clamp = (value: number, range: CanvasFontAxisRange): number =>
  Math.min(range.maximum, Math.max(range.minimum, value));

const replaceVariations = (
  variations: CanvasTextFontVariations | undefined,
  targetAxes: ReadonlyMap<string, CanvasFontAxisRange>,
  clampedAxisTags: Set<string>,
  resetAxisTags: Set<string>
): CanvasTextFontVariations | undefined => {
  const entries = Object.entries(variations ?? {});
  const next: Record<string, number> = Object.fromEntries([...targetAxes].map(([tag, range]) => [tag, range.default]));
  for (const [tag, value] of entries) {
    const range = targetAxes.get(tag);
    if (!range || !Number.isFinite(value)) {
      resetAxisTags.add(tag);
      continue;
    }
    const bounded = clamp(value, range);
    if (bounded !== value) {
      clampedAxisTags.add(tag);
    }
    next[tag] = bounded;
  }
  return Object.keys(next).length > 0 ? next : undefined;
};

const replaceNodes = (
  nodes: readonly CanvasNodeContract[],
  from: CanvasTextFontRef,
  target: CanvasFontReplacementTarget,
  targetAxes: ReadonlyMap<string, CanvasFontAxisRange>,
  replacedLayerIds: string[],
  clampedAxisTags: Set<string>,
  resetAxisTags: Set<string>
): readonly CanvasNodeContract[] => {
  let changed = false;
  const next = nodes.map((node) => {
    if (node.type === 'group') {
      const children = replaceNodes(
        node.children,
        from,
        target,
        targetAxes,
        replacedLayerIds,
        clampedAxisTags,
        resetAxisTags
      );
      if (children === node.children) {
        return node;
      }
      changed = true;
      return { ...node, children: [...children] };
    }
    if (
      !hasSource(node) ||
      node.source.type !== 'text' ||
      !node.source.fontRef ||
      !sameCanvasTextFontRef(node.source.fontRef, from)
    ) {
      return node;
    }
    const variations = replaceVariations(node.source.fontVariations, targetAxes, clampedAxisTags, resetAxisTags);
    const style =
      variations?.ital !== undefined
        ? variations.ital >= 0.5
          ? 'italic'
          : 'normal'
        : variations?.slnt !== undefined
          ? variations.slnt !== 0
            ? 'oblique'
            : 'normal'
          : target.style;
    replacedLayerIds.push(node.id);
    changed = true;
    return {
      ...node,
      source: {
        ...node.source,
        fontFamily: target.fontRef.family,
        fontRef: { ...target.fontRef },
        fontStyle: style,
        fontWeight: variations?.wght ?? target.weight,
        fontVariations: variations,
      },
    };
  });
  return changed ? next : nodes;
};

/**
 * Replaces exact font refs with structural sharing. Static targets have no axes, so reset and report all source
 * variations.
 */
export const replaceCanvasFontReferences = (
  document: CanvasDocumentContractV3,
  from: CanvasTextFontRef,
  target: CanvasFontReplacementTarget
): CanvasFontReplacementSummary => {
  const { fontRef: to, axes: targetAxes } = target;
  if (sameCanvasTextFontRef(from, to)) {
    return {
      clampedAxisTags: [],
      document,
      from: { ...from },
      replacedCount: 0,
      replacedLayerIds: [],
      resetAxisTags: [],
      to: { ...to },
    };
  }
  const axes = new Map(targetAxes.map((axis) => [axis.tag, axis]));
  const replacedLayerIds: string[] = [];
  const clampedAxisTags = new Set<string>();
  const resetAxisTags = new Set<string>();
  let changed = false;
  const stacks = { ...document.stacks };
  for (const stack of ['raster', 'control', 'regional_guidance', 'inpaint_mask'] as const) {
    const nodes = replaceNodes(
      document.stacks[stack],
      from,
      target,
      axes,
      replacedLayerIds,
      clampedAxisTags,
      resetAxisTags
    );
    if (nodes !== document.stacks[stack]) {
      stacks[stack] = [...nodes];
      changed = true;
    }
  }
  const nextDocument = changed ? { ...document, stacks, version: 4 as const } : document;
  return {
    clampedAxisTags: [...clampedAxisTags].sort(),
    document: nextDocument,
    from: { ...from },
    replacedCount: replacedLayerIds.length,
    replacedLayerIds,
    resetAxisTags: [...resetAxisTags].sort(),
    to: { ...to },
  };
};
