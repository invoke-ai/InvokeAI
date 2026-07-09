/**
 * Shared helpers for the layers panel: id minting, the paint-layer factory, the
 * blend-mode list, merge-down eligibility, and the single seam that routes a
 * structural edit either through the engine's canvas history (when an engine is
 * attached) or as a plain reducer dispatch (when it is not).
 */

import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { Rect } from '@workbench/canvas-engine/types';
import type { GenerateReferenceImage } from '@workbench/generation/types';
import type {
  CanvasBlendMode,
  CanvasControlAdapterContract,
  CanvasControlLayerContract,
  CanvasDocumentContractV2,
  CanvasInpaintMaskLayerContract,
  CanvasLayerBaseContract,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasMaskContract,
  CanvasRasterLayerContractV2,
  CanvasRegionalGuidanceLayerContract,
} from '@workbench/types';
import type { WorkbenchAction } from '@workbench/workbenchState';
import type { Dispatch } from 'react';

import { getSourceContentRect, isMergeableRasterLayer } from '@workbench/canvas-engine/document/sources';

type LayerTransform = CanvasLayerBaseContract['transform'];

type ControlConfigPatch = { layerType: 'control'; withTransparencyEffect?: boolean };
type RegionalGuidanceConfigPatch = {
  layerType: 'regional_guidance';
  positivePrompt?: string | null;
  negativePrompt?: string | null;
  autoNegative?: boolean;
  referenceImages?: CanvasRegionalGuidanceLayerContract['referenceImages'];
};
type InpaintMaskConfigPatch = { layerType: 'inpaint_mask'; noiseLevel?: number; denoiseLimit?: number };
type PatchPair<T> = { forward: T; inverse: T };

/**
 * Re-exported so existing imports of `isMergeableRasterLayer` from this module
 * keep working; the canonical definition lives in the engine's `document/sources`
 * (the single source of truth the engine's own `mergeLayerDown` guard uses too —
 * see the finding this fixed: the engine used to gate on the broader
 * `isRenderableLayer`, which let a mask merge-down corrupt data).
 */
export { isMergeableRasterLayer };

/** All 16 document blend modes, in the contract's declared order. */
export const CANVAS_BLEND_MODES: readonly CanvasBlendMode[] = [
  'normal',
  'multiply',
  'screen',
  'overlay',
  'darken',
  'lighten',
  'color-dodge',
  'color-burn',
  'hard-light',
  'soft-light',
  'difference',
  'exclusion',
  'hue',
  'saturation',
  'color',
  'luminosity',
];

/** Mints a fresh layer id (matches the engine's `createLayerId` shape). */
export const createLayerId = (): string => `layer-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

/** Builds an empty, document-sized paint layer with sensible defaults. */
export const createEmptyPaintLayer = (name: string, id: string = createLayerId()): CanvasRasterLayerContractV2 => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name,
  opacity: 1,
  source: { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster',
});

/**
 * Legacy-default inpaint mask fill: a diagonal hatch (legacy's
 * `getInpaintMaskState` default `style: 'diagonal'`) in the first cycled mask
 * fill colour (legacy `rgb(224,117,117)`). Multiple inpaint masks are allowed.
 */
export const DEFAULT_INPAINT_MASK_FILL = { color: '#e07575', style: 'diagonal' } as const;

/** The next free "Inpaint Mask N" name given the existing layer names (N ≥ 1, first gap). */
export const nextInpaintMaskName = (existingNames: readonly string[]): string => {
  const used = new Set<number>();
  for (const name of existingNames) {
    const match = /^Inpaint Mask (\d+)$/.exec(name.trim());
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
  return `Inpaint Mask ${n}`;
};

/** Builds an empty inpaint mask layer with the legacy-default fill (no bitmap yet). */
export const createInpaintMaskLayer = (name: string, id: string = createLayerId()): CanvasInpaintMaskLayerContract => ({
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: null, fill: { ...DEFAULT_INPAINT_MASK_FILL } },
  name,
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'inpaint_mask',
});

/**
 * Legacy's regional-guidance mask fill palette (`DEFAULT_RG_MASK_FILL_COLORS`),
 * as hex. New regions cycle through these so overlapping regions stay visually
 * distinct.
 */
export const REGIONAL_GUIDANCE_FILL_COLORS: readonly string[] = [
  '#799ddb',
  '#83d683',
  '#fae150',
  '#dc9065',
  '#e07575',
  '#d58bca',
  '#a178d6',
];

/**
 * The next cycled regional-guidance fill colour, derived purely from the document
 * (the count of existing regional-guidance layers) rather than session-global
 * state. Legacy's cycler was seeded at index 0 and pre-incremented, so the first
 * new region (0 existing) gets index 1 (`#83d683`, green); each subsequent region
 * advances by one, wrapping modulo the palette length. Being a pure function of
 * the document, colours no longer depend on session history or leak across
 * projects.
 */
export const nextRegionalGuidanceFillColor = (existingRegionalGuidanceCount: number): string =>
  REGIONAL_GUIDANCE_FILL_COLORS[(existingRegionalGuidanceCount + 1) % REGIONAL_GUIDANCE_FILL_COLORS.length];

/** The next free "Regional Guidance N" name given the existing layer names (N ≥ 1, first gap). */
export const nextRegionalGuidanceName = (existingNames: readonly string[]): string => {
  const used = new Set<number>();
  for (const name of existingNames) {
    const match = /^Regional Guidance (\d+)$/.exec(name.trim());
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
  return `Regional Guidance ${n}`;
};

/**
 * Builds an empty regional-guidance layer with legacy defaults: a solid fill in
 * the next cycled palette colour (derived from `existingRegionalGuidanceCount`,
 * the number of regional-guidance layers already in the document), opacity 0.5,
 * no prompts, autoNegative off, and no reference images (no mask bitmap yet — the
 * user paints the region).
 */
export const createRegionalGuidanceLayer = (
  name: string,
  existingRegionalGuidanceCount: number,
  id: string = createLayerId()
): CanvasRegionalGuidanceLayerContract => ({
  autoNegative: false,
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  mask: {
    bitmap: null,
    fill: { color: nextRegionalGuidanceFillColor(existingRegionalGuidanceCount), style: 'solid' },
  },
  name,
  negativePrompt: null,
  opacity: 0.5,
  positivePrompt: null,
  referenceImages: [],
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'regional_guidance',
});

/** Mints a fresh regional-guidance reference-image id. */
export const createReferenceImageId = (): string =>
  `rgref-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

/**
 * A fresh regional reference image, minting the config kind the region's base can
 * actually consume: FLUX regions use FLUX Redux (the only kind resolved for FLUX
 * in `resolveRegionalReferenceImages`), everything else uses an IP-Adapter (legacy
 * `initialRegionalGuidanceIPAdapter`). The model is chosen by the user; the image
 * is assigned via drop/upload. Shared by `RegionalGuidanceSettings` (the in-popover
 * "add reference image" button) and the header add-layer menu's "Regional Reference
 * Image" item, so both mint the same shape.
 */
export const createRegionalReferenceImage = (
  base: string | null,
  id: string = createReferenceImageId()
): GenerateReferenceImage => {
  if (base === 'flux') {
    return {
      config: { image: null, imageInfluence: 'highest', model: null, type: 'flux_redux' },
      id,
      isEnabled: true,
    };
  }
  return {
    config: {
      beginEndStepPct: [0, 1],
      clipVisionModel: 'ViT-H',
      image: null,
      method: 'full',
      model: null,
      type: 'ip_adapter',
      weight: 1,
    },
    id,
    isEnabled: true,
  };
};

/**
 * Builds a regional-guidance layer pre-seeded with ONE empty reference image
 * (legacy's "regional guidance with a reference image" add-layer path). Identical
 * to `createRegionalGuidanceLayer` except the region starts with a single ref-image
 * slot the user then assigns a model + image to via the region's settings popover.
 */
export const createRegionalGuidanceLayerWithRefImage = (
  name: string,
  existingRegionalGuidanceCount: number,
  base: string | null,
  id: string = createLayerId()
): CanvasRegionalGuidanceLayerContract => ({
  ...createRegionalGuidanceLayer(name, existingRegionalGuidanceCount, id),
  referenceImages: [createRegionalReferenceImage(base)],
});

/**
 * The legacy-default control adapter for a freshly created control layer
 * (`features/controlLayers/store/util.ts` `initialControlNet`): a ControlNet with
 * weight 1, an active range of the first 75% of steps, `balanced` control mode,
 * and no model selected yet.
 */
export const DEFAULT_CONTROL_ADAPTER: CanvasControlAdapterContract = {
  beginEndStepPct: [0, 0.75],
  controlMode: 'balanced',
  kind: 'controlnet',
  model: null,
  weight: 1,
};

/** The next free "Control Layer N" name given the existing layer names (N ≥ 1, first gap). */
export const nextControlLayerName = (existingNames: readonly string[]): string => {
  const used = new Set<number>();
  for (const name of existingNames) {
    const match = /^Control Layer (\d+)$/.exec(name.trim());
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
  return `Control Layer ${n}`;
};

/**
 * Builds an empty control layer with the legacy-default ControlNet adapter and
 * transparency effect ON (legacy `withTransparencyEffect: true`). No source /
 * filter yet — the user paints or drops an image, then optionally previews a
 * filter.
 */
export const createControlLayer = (name: string, id: string = createLayerId()): CanvasControlLayerContract => ({
  adapter: { ...DEFAULT_CONTROL_ADAPTER, beginEndStepPct: [...DEFAULT_CONTROL_ADAPTER.beginEndStepPct] },
  blendMode: 'normal',
  id,
  isEnabled: true,
  isLocked: false,
  name,
  opacity: 1,
  source: { bitmap: null, type: 'paint' },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'control',
  withTransparencyEffect: true,
});

/** Returns the transform that fits the layer's unrotated local content into the bbox. */
export const fitLayerTransformToBbox = (
  layer: CanvasLayerContract,
  bbox: Rect,
  documentRect: Rect = bbox
): LayerTransform | null => {
  const doc = { height: documentRect.height, width: documentRect.width } as CanvasDocumentContractV2;
  const contentRect = getSourceContentRect(layer, doc);
  if (contentRect.width <= 0 || contentRect.height <= 0) {
    return null;
  }
  const scaleX = bbox.width / contentRect.width;
  const scaleY = bbox.height / contentRect.height;
  return {
    rotation: 0,
    scaleX,
    scaleY,
    x: bbox.x - contentRect.x * scaleX,
    y: bbox.y - contentRect.y * scaleY,
  };
};

export const getControlTransparencyEffectPatch = (
  layer: CanvasControlLayerContract
): PatchPair<ControlConfigPatch> => ({
  forward: { layerType: 'control', withTransparencyEffect: !layer.withTransparencyEffect },
  inverse: { layerType: 'control', withTransparencyEffect: layer.withTransparencyEffect },
});

export const getRegionalGuidancePositivePromptPatch = (
  layer: CanvasRegionalGuidanceLayerContract
): PatchPair<RegionalGuidanceConfigPatch> => ({
  forward: { layerType: 'regional_guidance', positivePrompt: layer.positivePrompt ?? '' },
  inverse: { layerType: 'regional_guidance', positivePrompt: layer.positivePrompt },
});

export const getRegionalGuidanceNegativePromptPatch = (
  layer: CanvasRegionalGuidanceLayerContract
): PatchPair<RegionalGuidanceConfigPatch> => ({
  forward: { layerType: 'regional_guidance', negativePrompt: layer.negativePrompt ?? '' },
  inverse: { layerType: 'regional_guidance', negativePrompt: layer.negativePrompt },
});

export const getRegionalGuidanceAutoNegativePatch = (
  layer: CanvasRegionalGuidanceLayerContract
): PatchPair<RegionalGuidanceConfigPatch> => ({
  forward: { autoNegative: !layer.autoNegative, layerType: 'regional_guidance' },
  inverse: { autoNegative: layer.autoNegative, layerType: 'regional_guidance' },
});

export const getRegionalGuidanceReferenceImagePatch = (
  layer: CanvasRegionalGuidanceLayerContract,
  base: string | null
): PatchPair<RegionalGuidanceConfigPatch> => ({
  forward: {
    layerType: 'regional_guidance',
    referenceImages: [...layer.referenceImages, createRegionalReferenceImage(base)],
  },
  inverse: { layerType: 'regional_guidance', referenceImages: layer.referenceImages },
});

export const getInpaintNoisePatch = (layer: CanvasInpaintMaskLayerContract): PatchPair<InpaintMaskConfigPatch> => ({
  forward: { layerType: 'inpaint_mask', noiseLevel: layer.noiseLevel === undefined ? 0.25 : undefined },
  inverse: { layerType: 'inpaint_mask', noiseLevel: layer.noiseLevel },
});

export const getInpaintDenoiseLimitPatch = (
  layer: CanvasInpaintMaskLayerContract
): PatchPair<InpaintMaskConfigPatch> => ({
  forward: { denoiseLimit: layer.denoiseLimit === undefined ? 0.8 : undefined, layerType: 'inpaint_mask' },
  inverse: { denoiseLimit: layer.denoiseLimit, layerType: 'inpaint_mask' },
});

/**
 * True when `layer` can be converted to a control layer (and vice-versa): only a
 * raster layer with a rasterizable image/paint source, or a control layer,
 * qualifies — parametric raster sources (shape/text/gradient) and mask layers do
 * not (legacy only offers raster↔control conversion for pixel-backed layers).
 */
export const canConvertRasterControl = (layer: CanvasLayerContract): boolean => {
  if (layer.type === 'control') {
    return true;
  }
  return layer.type === 'raster' && (layer.source.type === 'image' || layer.source.type === 'paint');
};

type PixelLayer = CanvasRasterLayerContractV2 | CanvasControlLayerContract;
type PixelSource = Extract<CanvasLayerSourceContract, { type: 'image' | 'paint' }>;

const cloneTransform = (layer: CanvasLayerContract): LayerTransform => ({ ...layer.transform });

const clonePixelSource = (layer: PixelLayer): PixelSource | null => {
  if (layer.source.type === 'image') {
    return { image: { ...layer.source.image }, type: 'image' };
  }
  if (layer.source.type === 'paint') {
    return {
      bitmap: layer.source.bitmap ? { ...layer.source.bitmap } : null,
      offset: layer.source.offset ? { ...layer.source.offset } : undefined,
      type: 'paint',
    };
  }
  return null;
};

const pixelLayerToMask = (layer: PixelLayer): CanvasMaskContract | null => {
  const source = clonePixelSource(layer);
  if (!source) {
    return null;
  }
  if (source.type === 'image') {
    return { bitmap: source.image, fill: { ...DEFAULT_INPAINT_MASK_FILL } };
  }
  return {
    bitmap: source.bitmap,
    fill: { ...DEFAULT_INPAINT_MASK_FILL },
    offset: source.offset,
  };
};

const cloneMask = (mask: CanvasMaskContract): CanvasMaskContract => ({
  bitmap: mask.bitmap ? { ...mask.bitmap } : null,
  fill: { ...mask.fill },
  offset: mask.offset ? { ...mask.offset } : undefined,
});

const destinationBase = (layer: CanvasLayerContract, id: string, isCopy: boolean): CanvasLayerBaseContract => ({
  blendMode: layer.blendMode,
  id,
  isEnabled: layer.isEnabled,
  isLocked: layer.isLocked,
  name: isCopy ? `${layer.name} copy` : layer.name,
  opacity: layer.opacity,
  transform: cloneTransform(layer),
});

const pixelLayerToControl = (layer: PixelLayer, id: string, isCopy: boolean): CanvasControlLayerContract | null => {
  const source = clonePixelSource(layer);
  if (!source) {
    return null;
  }
  return {
    ...destinationBase(layer, id, isCopy),
    adapter: { ...DEFAULT_CONTROL_ADAPTER, beginEndStepPct: [...DEFAULT_CONTROL_ADAPTER.beginEndStepPct] },
    source,
    type: 'control',
    withTransparencyEffect: true,
  };
};

const pixelLayerToInpaintMask = (
  layer: PixelLayer,
  id: string,
  isCopy: boolean
): CanvasInpaintMaskLayerContract | null => {
  const mask = pixelLayerToMask(layer);
  return mask ? { ...destinationBase(layer, id, isCopy), mask, type: 'inpaint_mask' } : null;
};

const pixelLayerToRegionalGuidance = (
  layer: PixelLayer,
  id: string,
  isCopy: boolean
): CanvasRegionalGuidanceLayerContract | null => {
  const mask = pixelLayerToMask(layer);
  return mask
    ? {
        ...destinationBase(layer, id, isCopy),
        autoNegative: false,
        mask,
        negativePrompt: null,
        positivePrompt: null,
        referenceImages: [],
        type: 'regional_guidance',
      }
    : null;
};

export const copyRasterToControl = (
  layer: CanvasRasterLayerContractV2,
  id: string
): CanvasControlLayerContract | null => pixelLayerToControl(layer, id, true);

export const copyRasterToInpaintMask = (
  layer: CanvasRasterLayerContractV2,
  id: string
): CanvasInpaintMaskLayerContract | null => pixelLayerToInpaintMask(layer, id, true);

export const copyRasterToRegionalGuidance = (
  layer: CanvasRasterLayerContractV2,
  id: string
): CanvasRegionalGuidanceLayerContract | null => pixelLayerToRegionalGuidance(layer, id, true);

export const convertRasterToControl = (layer: CanvasRasterLayerContractV2): CanvasControlLayerContract | null =>
  pixelLayerToControl(layer, layer.id, false);

export const convertRasterToInpaintMask = (layer: CanvasRasterLayerContractV2): CanvasInpaintMaskLayerContract | null =>
  pixelLayerToInpaintMask(layer, layer.id, false);

export const convertRasterToRegionalGuidance = (
  layer: CanvasRasterLayerContractV2
): CanvasRegionalGuidanceLayerContract | null => pixelLayerToRegionalGuidance(layer, layer.id, false);

export const copyControlToRaster = (
  layer: CanvasControlLayerContract,
  id: string
): CanvasRasterLayerContractV2 | null => {
  const source = clonePixelSource(layer);
  return source ? { ...destinationBase(layer, id, true), source, type: 'raster' } : null;
};

export const copyControlToInpaintMask = (
  layer: CanvasControlLayerContract,
  id: string
): CanvasInpaintMaskLayerContract | null => pixelLayerToInpaintMask(layer, id, true);

export const copyControlToRegionalGuidance = (
  layer: CanvasControlLayerContract,
  id: string
): CanvasRegionalGuidanceLayerContract | null => pixelLayerToRegionalGuidance(layer, id, true);

export const copyMaskToRegionalGuidance = (
  layer: CanvasInpaintMaskLayerContract,
  id: string
): CanvasRegionalGuidanceLayerContract => ({
  ...destinationBase(layer, id, true),
  autoNegative: false,
  mask: cloneMask(layer.mask),
  negativePrompt: null,
  positivePrompt: null,
  referenceImages: [],
  type: 'regional_guidance',
});

export const copyRegionalGuidanceToInpaintMask = (
  layer: CanvasRegionalGuidanceLayerContract,
  id: string
): CanvasInpaintMaskLayerContract => ({
  ...destinationBase(layer, id, true),
  mask: cloneMask(layer.mask),
  type: 'inpaint_mask',
});

/**
 * Converts a raster layer to a control layer (default adapter, transparency on)
 * or a control layer back to a raster layer, preserving the pixel source, id,
 * name, transform, and base props. Returns `null` when the layer cannot convert
 * (parametric/mask sources) or already has the target type.
 */
export const convertRasterControlLayer = (
  layer: CanvasLayerContract,
  targetType: 'raster' | 'control'
): CanvasLayerContract | null => {
  if (layer.type === targetType || !canConvertRasterControl(layer)) {
    return null;
  }
  if (layer.type === 'raster' && targetType === 'control') {
    return convertRasterToControl(layer);
  }
  if (layer.type === 'control' && targetType === 'raster') {
    const source = clonePixelSource(layer);
    return source ? { ...destinationBase(layer, layer.id, false), source, type: 'raster' } : null;
  }
  return null;
};

/**
 * Whether the layer at `index` can be merged down: an engine must be attached
 * (merge is pixel work), a layer must exist directly below it, and both that
 * layer and the one below must be mergeable raster layers.
 */
export const canMergeLayerDown = (
  layers: readonly CanvasLayerContract[],
  index: number,
  hasEngine: boolean
): boolean => {
  if (!hasEngine) {
    return false;
  }
  const layer = layers[index];
  const below = layers[index + 1];
  return !!layer && !!below && isMergeableRasterLayer(layer) && isMergeableRasterLayer(below);
};

/**
 * Applies a structural document edit. With an engine attached it goes through
 * `commitStructural`, joining the canvas undo stack; without one it is a plain
 * dispatch (no undo entry — acceptable when the canvas is not mounted).
 */
export const applyStructural = (
  engine: CanvasEngine | null,
  dispatch: Dispatch<WorkbenchAction>,
  label: string,
  forward: WorkbenchAction,
  inverse: WorkbenchAction
): void => {
  if (engine) {
    engine.commitStructural(label, forward, inverse);
  } else {
    dispatch(forward);
  }
};
