import type {
  CanvasControlLayerContract,
  CanvasDocumentContractV2,
  CanvasInpaintMaskLayerContract,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasRegionalGuidanceLayerContract,
} from '@workbench/canvas-engine/api';

import {
  adjustmentsKey,
  getBaseRasterContentBounds,
  getCompositeLayerBounds,
  planBaseRasterComposite,
} from '@workbench/canvas-engine/api';

import type {
  CompositeEntry,
  CompositeLayerRef,
  CompositeMaskLayerRef,
  CompositePlan,
  Rect,
} from './generationContracts';

import { DEFAULT_MASK_DENOISE_LIMIT } from './generationContracts';

/** A stable string identifying a source's pixels (its asset name, or an empty sentinel). */
const sourceRefOf = (source: CanvasLayerSourceContract): string => {
  switch (source.type) {
    case 'image':
      return `image:${source.image.imageName}`;
    case 'paint':
      return source.bitmap ? `paint:${source.bitmap.imageName}` : 'paint:empty';
    default:
      return `${source.type}:unsupported`;
  }
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

export { getBaseRasterContentBounds, getCompositeLayerBounds, planBaseRasterComposite };

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
  const entries: CompositeEntry[] = [planBaseRasterComposite(document, bbox)];

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
