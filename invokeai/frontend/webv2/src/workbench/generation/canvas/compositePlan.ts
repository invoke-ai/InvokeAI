/**
 * Pure composite planner.
 *
 * {@link planComposites} turns a canvas document + generation bbox into a
 * {@link CompositePlan} — a pixel-free description of what the executor must
 * composite and upload — without touching any pixels or the engine. Phase 4.2
 * emits a single `base-raster` entry: the enabled raster layers (image/paint
 * sources) in z-order, each with its transform / opacity / blend mode.
 *
 * ## Stable identity key
 *
 * Each entry carries a `key` derived from exactly the inputs that determine its
 * pixels: the bbox rect plus, for each contributing layer in z-order, its
 * `(id, sourceRef, transform, opacity, blendMode)`. Only *enabled* raster
 * layers contribute, so a layer's enablement is encoded by its presence in the
 * list (toggling it on/off changes the set, and thus the key). This makes the
 * key:
 * - **stable**: same document + bbox → byte-identical key;
 * - **sensitive** to reorder, opacity, transform, source swap, and bbox moves;
 * - **insensitive** to irrelevant edits: `selectedLayerId`, background, and any
 *   change to a disabled layer (it never appears in the list).
 *
 * The key is a plain deterministic string (no hashing needed) so it stays cheap
 * and debuggable; the executor uses it as a map key to skip redundant work.
 */

import type {
  CanvasControlLayerContract,
  CanvasDocumentContractV2,
  CanvasInpaintMaskLayerContract,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasRasterLayerContractV2,
  CanvasRegionalGuidanceLayerContract,
} from '@workbench/types';

import { adjustmentsKey, isIdentityAdjustments } from '@workbench/canvas-engine/render/adjustments';

import type { CompositeEntry, CompositeLayerRef, CompositeMaskLayerRef, CompositePlan, Rect } from './types';

import { DEFAULT_MASK_DENOISE_LIMIT } from './types';

/** True when a layer is an enabled raster layer with rasterizable, non-empty pixels. */
const isBaseRasterLayer = (layer: CanvasLayerContract): layer is CanvasRasterLayerContractV2 => {
  if (!layer.isEnabled || layer.type !== 'raster') {
    return false;
  }
  if (layer.source.type === 'image') {
    return true;
  }
  // A paint layer only contributes once it holds pixels. An empty (`bitmap:
  // null`) paint layer — e.g. an auto-created layer left blank after the stroke
  // was undone — carries no pixels; including it would force a doc-sized
  // transparent rect into the composite, so its full-bbox coverage plus scanned
  // transparency reads as `outpaint` (unsupported) instead of `txt2img`. The
  // orchestrator flushes pending paint uploads before planning, so a bitmap
  // still `null` here really is empty rather than merely unpersisted.
  return layer.source.type === 'paint' && layer.source.bitmap !== null;
};

/** A stable string identifying a source's pixels (its asset name, or an empty sentinel). */
const sourceRefOf = (source: CanvasLayerSourceContract): string => {
  switch (source.type) {
    case 'image':
      return `image:${source.image.imageName}`;
    case 'paint':
      return source.bitmap ? `paint:${source.bitmap.imageName}` : 'paint:empty';
    default:
      // Only image/paint layers reach the planner (filtered by isBaseRasterLayer).
      return `${source.type}:unsupported`;
  }
};

/** The native (unscaled) content rect of a base-raster layer's source (layer-local). */
const contentRectOf = (
  layer: CanvasRasterLayerContractV2,
  doc: CanvasDocumentContractV2
): { width: number; height: number; x: number; y: number } => {
  const { source } = layer;
  if (source.type === 'image') {
    return { height: source.image.height, width: source.image.width, x: 0, y: 0 };
  }
  // Paint layers are content-sized: the persisted bitmap dims at its offset. Only
  // paint layers WITH a bitmap reach here (empty paint is filtered upstream), but
  // default defensively to the document rect when absent.
  if (source.type === 'paint' && source.bitmap) {
    const offset = source.offset ?? { x: 0, y: 0 };
    return { height: source.bitmap.height, width: source.bitmap.width, x: offset.x, y: offset.y };
  }
  return { height: doc.height, width: doc.width, x: 0, y: 0 };
};

/** Projects a document layer into its frozen composite contribution. */
const toLayerRef = (layer: CanvasRasterLayerContractV2, doc: CanvasDocumentContractV2): CompositeLayerRef => {
  const rect = contentRectOf(layer, doc);
  const hasAdjustments = !isIdentityAdjustments(layer.adjustments);
  return {
    blendMode: layer.blendMode,
    contentOffset: { x: rect.x, y: rect.y },
    contentSize: { height: rect.height, width: rect.width },
    id: layer.id,
    opacity: layer.opacity,
    sourceRef: sourceRefOf(layer.source),
    // Bake non-destructive adjustments into the generated pixels (and the key).
    ...(hasAdjustments && layer.adjustments ? { adjustments: layer.adjustments } : {}),
    transform: {
      rotation: layer.transform.rotation,
      scaleX: layer.transform.scaleX,
      scaleY: layer.transform.scaleY,
      x: layer.transform.x,
      y: layer.transform.y,
    },
  };
};

/** Serializes a rect into a compact, deterministic key fragment. */
const rectKey = (rect: Rect): string => `${rect.x},${rect.y},${rect.width},${rect.height}`;

/** Serializes a layer ref's pixel-determining fields into a key fragment. */
const layerKey = (ref: CompositeLayerRef): string => {
  const t = ref.transform;
  const o = ref.contentOffset;
  return [
    ref.id,
    ref.sourceRef,
    o.x,
    o.y,
    t.x,
    t.y,
    t.scaleX,
    t.scaleY,
    t.rotation,
    ref.opacity,
    ref.blendMode,
    ref.adjustments ? adjustmentsKey(ref.adjustments) : '-',
  ].join(':');
};

/** Derives the stable identity key for a base-raster entry. */
const deriveKey = (kind: string, bbox: Rect, layers: CompositeLayerRef[]): string =>
  `${kind}|${rectKey(bbox)}|${layers.map(layerKey).join('|')}`;

/** True when a layer is an enabled inpaint mask with persisted (non-empty) alpha. */
const isActiveInpaintMaskLayer = (layer: CanvasLayerContract): layer is CanvasInpaintMaskLayerContract =>
  layer.isEnabled && layer.type === 'inpaint_mask' && layer.mask.bitmap !== null;

/** The native content rect of a mask layer's persisted bitmap (layer-local, at its offset). */
const maskContentRect = (
  layer: CanvasInpaintMaskLayerContract
): { width: number; height: number; x: number; y: number } => {
  const { bitmap, offset } = layer.mask;
  if (bitmap) {
    const o = offset ?? { x: 0, y: 0 };
    return { height: bitmap.height, width: bitmap.width, x: o.x, y: o.y };
  }
  return { height: 0, width: 0, x: 0, y: 0 };
};

/** Projects a mask layer into a grayscale-composite contribution with its resolved attribute value. */
const toMaskLayerRef = (layer: CanvasInpaintMaskLayerContract, attributeValue: number): CompositeMaskLayerRef => {
  const rect = maskContentRect(layer);
  return {
    attributeValue,
    contentOffset: { x: rect.x, y: rect.y },
    contentSize: { height: rect.height, width: rect.width },
    id: layer.id,
    sourceRef: layer.mask.bitmap ? `mask:${layer.mask.bitmap.imageName}` : 'mask:empty',
    transform: {
      rotation: layer.transform.rotation,
      scaleX: layer.transform.scaleX,
      scaleY: layer.transform.scaleY,
      x: layer.transform.x,
      y: layer.transform.y,
    },
  };
};

/** Serializes a mask layer ref (adds `attributeValue` to the pixel-determining key). */
const maskLayerKey = (ref: CompositeMaskLayerRef): string => {
  const t = ref.transform;
  const o = ref.contentOffset;
  return [ref.id, ref.sourceRef, o.x, o.y, t.x, t.y, t.scaleX, t.scaleY, t.rotation, ref.attributeValue].join(':');
};

/** Derives the stable identity key for a grayscale mask entry. */
const deriveMaskKey = (kind: string, bbox: Rect, layers: CompositeMaskLayerRef[]): string =>
  `${kind}|${rectKey(bbox)}|${layers.map(maskLayerKey).join('|')}`;

/**
 * Plans the composites required to invoke `document` over `bbox`:
 * - one `base-raster` entry (the initial image);
 * - one `inpaint-mask` entry (grayscale denoise-limit mask) when enabled inpaint
 *   masks with content exist — an undefined `denoiseLimit` resolves to the legacy
 *   default (1.0, full denoise);
 * - one `noise-mask` entry when at least one such mask defines a `noiseLevel`
 *   (masks with an undefined `noiseLevel` are excluded, mirroring legacy — they
 *   must NOT be treated as noise 0).
 */
export const planComposites = (document: CanvasDocumentContractV2, bbox: Rect): CompositePlan => {
  const layers = document.layers.filter(isBaseRasterLayer).map((layer) => toLayerRef(layer, document));

  const baseEntry: CompositeEntry = {
    bbox,
    key: deriveKey('base-raster', bbox, layers),
    kind: 'base-raster',
    layers,
  };

  const entries: CompositeEntry[] = [baseEntry];

  const maskLayers = document.layers.filter(isActiveInpaintMaskLayer);

  if (maskLayers.length > 0) {
    const denoiseRefs = maskLayers.map((layer) =>
      toMaskLayerRef(layer, layer.denoiseLimit ?? DEFAULT_MASK_DENOISE_LIMIT)
    );
    entries.push({
      bbox,
      key: deriveMaskKey('inpaint-mask', bbox, denoiseRefs),
      kind: 'inpaint-mask',
      layers: [],
      maskLayers: denoiseRefs,
    });

    const noiseRefs = maskLayers
      .filter((layer) => layer.noiseLevel !== undefined)
      .map((layer) => toMaskLayerRef(layer, layer.noiseLevel as number));

    if (noiseRefs.length > 0) {
      entries.push({
        bbox,
        key: deriveMaskKey('noise-mask', bbox, noiseRefs),
        kind: 'noise-mask',
        layers: [],
        maskLayers: noiseRefs,
      });
    }
  }

  return { bbox, entries };
};

/** True when a control layer is enabled and holds rasterizable (non-empty) content. */
const hasControlContent = (layer: CanvasLayerContract): layer is CanvasControlLayerContract => {
  if (!layer.isEnabled || layer.type !== 'control') {
    return false;
  }
  if (layer.source.type === 'image') {
    return true;
  }
  return layer.source.type === 'paint' && layer.source.bitmap !== null;
};

/** The native (unscaled) content rect of a control layer's source (layer-local). */
const controlContentRect = (
  layer: CanvasControlLayerContract,
  doc: CanvasDocumentContractV2
): { width: number; height: number; x: number; y: number } => {
  const { source } = layer;
  if (source.type === 'image') {
    return { height: source.image.height, width: source.image.width, x: 0, y: 0 };
  }
  if (source.type === 'paint' && source.bitmap) {
    const offset = source.offset ?? { x: 0, y: 0 };
    return { height: source.bitmap.height, width: source.bitmap.width, x: offset.x, y: offset.y };
  }
  return { height: doc.height, width: doc.width, x: 0, y: 0 };
};

/**
 * Projects a control layer into a standalone composite contribution. Opacity is
 * forced to 1 and blend mode to `normal` (the layer's DISPLAY opacity/blend and
 * transparency effect never alter the control image sent to the backend — legacy
 * rasterizes control at opacity 1, no filters), so display tweaks don't churn the
 * entry key.
 */
const toControlLayerRef = (layer: CanvasControlLayerContract, doc: CanvasDocumentContractV2): CompositeLayerRef => {
  const rect = controlContentRect(layer, doc);
  return {
    blendMode: 'normal',
    contentOffset: { x: rect.x, y: rect.y },
    contentSize: { height: rect.height, width: rect.width },
    id: layer.id,
    opacity: 1,
    sourceRef: sourceRefOf(layer.source),
    transform: {
      rotation: layer.transform.rotation,
      scaleX: layer.transform.scaleX,
      scaleY: layer.transform.scaleY,
      x: layer.transform.x,
      y: layer.transform.y,
    },
  };
};

/** One control layer's separate composite (never blended with other control layers). */
export interface ControlCompositeEntry {
  layerId: string;
  entry: CompositeEntry;
}

/**
 * Plans one composite per enabled control layer WITH content — each composited
 * separately over `bbox` (legacy parity: control images are never blended
 * together). Empty control layers (no source / blank paint) are excluded (they
 * carry no control content and are rejected upstream with a "no control" reason).
 * Control layers never contribute to the `base-raster` composite, so they never
 * paint into the img2img/inpaint source.
 */
export const planControlComposites = (document: CanvasDocumentContractV2, bbox: Rect): ControlCompositeEntry[] =>
  document.layers.filter(hasControlContent).map((layer) => {
    const ref = toControlLayerRef(layer, document);
    return {
      entry: {
        bbox,
        key: `control-layer|${layer.id}|${rectKey(bbox)}|${layerKey(ref)}`,
        kind: 'control-layer',
        layerId: layer.id,
        layers: [ref],
      },
      layerId: layer.id,
    };
  });

/** True when a regional-guidance layer is enabled and holds a persisted (non-empty) mask. */
const hasRegionalMaskContent = (layer: CanvasLayerContract): layer is CanvasRegionalGuidanceLayerContract =>
  layer.isEnabled && layer.type === 'regional_guidance' && layer.mask.bitmap !== null;

/** The native content rect of a regional mask's persisted bitmap (layer-local, at its offset). */
const regionalMaskContentRect = (
  layer: CanvasRegionalGuidanceLayerContract
): { width: number; height: number; x: number; y: number } => {
  const { bitmap, offset } = layer.mask;
  if (bitmap) {
    const o = offset ?? { x: 0, y: 0 };
    return { height: bitmap.height, width: bitmap.width, x: o.x, y: o.y };
  }
  return { height: 0, width: 0, x: 0, y: 0 };
};

/**
 * Projects a regional-guidance layer into a standalone ALPHA composite
 * contribution (opacity 1, normal blend) — the executor composites the mask's
 * alpha coverage over the bbox and uploads it, so `alpha_mask_to_tensor` can read
 * the region's alpha. Display opacity/blend never alter the mask sent to the
 * backend (mirrors control layers).
 */
const toRegionalMaskRef = (layer: CanvasRegionalGuidanceLayerContract): CompositeLayerRef => {
  const rect = regionalMaskContentRect(layer);
  return {
    blendMode: 'normal',
    contentOffset: { x: rect.x, y: rect.y },
    contentSize: { height: rect.height, width: rect.width },
    id: layer.id,
    opacity: 1,
    sourceRef: layer.mask.bitmap ? `mask:${layer.mask.bitmap.imageName}` : 'mask:empty',
    transform: {
      rotation: layer.transform.rotation,
      scaleX: layer.transform.scaleX,
      scaleY: layer.transform.scaleY,
      x: layer.transform.x,
      y: layer.transform.y,
    },
  };
};

/** One regional-guidance region's separate alpha-mask composite. */
export interface RegionalMaskCompositeEntry {
  layerId: string;
  entry: CompositeEntry;
}

/**
 * Plans one alpha-mask composite per enabled regional-guidance layer WITH mask
 * content — each composited separately over `bbox` (a region's mask feeds its own
 * `alpha_mask_to_tensor`; regions are never combined). Regions without mask
 * content carry no region and are skipped (they'd be rejected upstream with a
 * "no region" reason). Regional layers never contribute to `base-raster`.
 */
export const planRegionalMaskComposites = (
  document: CanvasDocumentContractV2,
  bbox: Rect
): RegionalMaskCompositeEntry[] =>
  document.layers.filter(hasRegionalMaskContent).map((layer) => {
    const ref = toRegionalMaskRef(layer);
    return {
      entry: {
        bbox,
        key: `regional-mask|${layer.id}|${rectKey(bbox)}|${layerKey(ref)}`,
        kind: 'regional-mask',
        layerId: layer.id,
        layers: [ref],
      },
      layerId: layer.id,
    };
  });
