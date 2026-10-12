import type {
  CanvasAdjustmentEntry,
  CanvasBlendMode,
  CanvasControlLayerContract,
  CanvasDocumentContractV3,
  CanvasImageRef,
  CanvasInpaintMaskLayerContract,
  CanvasLayerBaseContract,
  CanvasLayerContract,
  CanvasLayerSourceContract,
  CanvasMaskContract,
  CanvasMaskFillContract,
  CanvasRasterLayerContractV2,
  CanvasRegionalGuidanceLayerContract,
  RegionalGuidanceReferenceImage,
  Rect,
} from '@workbench/canvas-engine/api';

export type { CanvasStructuralEngine } from '@workbench/canvas-engine/api';

import { getRegionalGuidanceSupport } from '@features/generation/graph';
import { getSourceContentRect, isMergeableRasterLayer, mergeDownEligibility } from '@workbench/canvas-engine/api';
import { CONTROL_ADAPTER_DEFAULTS, createDefaultControlAdapter } from '@workbench/controlAdapters';

type LayerTransform = CanvasLayerBaseContract['transform'];

type ControlConfigPatch = { layerType: 'control'; withTransparencyEffect?: boolean };
type RegionalGuidanceConfigPatch = {
  layerType: 'regional_guidance';
  positivePrompt?: string | null;
  negativePrompt?: string | null;
  autoNegative?: boolean;
  referenceImages?: CanvasRegionalGuidanceLayerContract['referenceImages'];
};
/**
 * Re-export the engine's canonical mergeability predicate so UI and engine exclude unsafe mask merges
 * consistently.
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

/** Use the legacy diagonal inpaint hatch and first fill color; multiple masks are allowed. */
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

/** The next free "Group N" name given the existing node names (N ≥ 1, first gap). */
export const nextGroupName = (existingNames: readonly string[]): string => {
  const used = new Set<number>();
  for (const name of existingNames) {
    const match = /^Group (\d+)$/.exec(name.trim());
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
  return `Group ${n}`;
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

/** Cycle legacy regional fill colors to distinguish overlaps. */
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
 * Derive colors from document region count with legacy pre-increment ordering: the first region uses palette index
 * 1. Avoid session-global cross-project state.
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

export interface CreateMaskLayerFromImageInput {
  image: CanvasImageRef;
  rect: Rect;
  id: string;
  name: string;
  fill: CanvasMaskFillContract;
}

export const createInpaintMaskFromImage = (input: CreateMaskLayerFromImageInput): CanvasInpaintMaskLayerContract => ({
  ...createInpaintMaskLayer(input.name, input.id),
  mask: {
    bitmap: { ...input.image },
    fill: { ...input.fill },
    offset: { x: input.rect.x, y: input.rect.y },
  },
});

export const createRegionalGuidanceFromImage = (
  input: CreateMaskLayerFromImageInput
): CanvasRegionalGuidanceLayerContract => ({
  ...createRegionalGuidanceLayer(input.name, 0, input.id),
  mask: {
    bitmap: { ...input.image },
    fill: { ...input.fill },
    offset: { x: input.rect.x, y: input.rect.y },
  },
});

/** Mints a fresh regional-guidance reference-image id. */
export const createReferenceImageId = (): string =>
  `rgref-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

/** Mints a fresh raster adjustment-entry id. */
export const createAdjustmentId = (): string =>
  `adj-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

/** A fresh adjustment entry of `type` at its identity values (invert has none). */
export const createIdentityAdjustment = (type: CanvasAdjustmentEntry['type']): CanvasAdjustmentEntry => {
  const id = createAdjustmentId();
  switch (type) {
    case 'brightness-contrast':
      return { brightness: 0, contrast: 0, id, isEnabled: true, type };
    case 'exposure':
      return { id, isEnabled: true, stops: 0, type };
    case 'levels':
      return { gamma: 1, id, inBlack: 0, inWhite: 255, isEnabled: true, outBlack: 0, outWhite: 255, type };
    case 'curves':
      return { curves: {}, id, isEnabled: true, type };
    case 'hsl':
      return { id, isEnabled: true, saturation: 0, type };
    case 'hue':
      return { id, isEnabled: true, rotation: 0, type };
    case 'invert':
      return { id, isEnabled: true, type };
  }
};

/** A new regional IP-Adapter reference's weight. */
export const DEFAULT_REGIONAL_REFERENCE_WEIGHT = 1;

/**
 * Create shared regional-reference defaults: FLUX uses Redux, other bases IP-Adapter. Users choose models and
 * assign images via drop/upload.
 */
export const createRegionalReferenceImage = (
  base: string | null,
  id: string = createReferenceImageId()
): RegionalGuidanceReferenceImage => {
  if (getRegionalGuidanceSupport(base)?.referenceImages === 'flux_redux') {
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
      weight: DEFAULT_REGIONAL_REFERENCE_WEIGHT,
    },
    id,
    isEnabled: true,
  };
};

/** Whether a regional reference image can be added; with no model selected every option stays open. */
export const canAddRegionalReferenceImage = (base: string | null): boolean =>
  base === null || Boolean(getRegionalGuidanceSupport(base)?.referenceImages);

export const createRegionalGuidanceLayerWithRefImage = (
  name: string,
  existingRegionalGuidanceCount: number,
  base: string | null,
  id: string = createLayerId()
): CanvasRegionalGuidanceLayerContract => ({
  ...createRegionalGuidanceLayer(name, existingRegionalGuidanceCount, id),
  referenceImages: [createRegionalReferenceImage(base)],
});

export { CONTROL_ADAPTER_DEFAULTS };

export const CONTROL_WEIGHT_BOUNDS = {
  inputMax: 2,
  inputMin: -1,
  sliderMax: 2,
  sliderMin: 0,
  step: 0.05,
} as const;

export const DEFAULT_CONTROL_ADAPTER = CONTROL_ADAPTER_DEFAULTS.controlnet;

/** A new mask modifier's magnitude (legacy defaults): noise level and denoise limit. */
export const MASK_MODIFIER_DEFAULTS = { denoise: 0.8, noise: 0.25 } as const;

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

export const createControlLayer = (
  name: string,
  id: string = createLayerId(),
  base?: string | null,
  model?: string | null
): CanvasControlLayerContract => {
  return {
    adapter: createDefaultControlAdapter(base, model ?? null),
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
  };
};

/** Returns the transform that fits the layer's unrotated local content into the bbox. */
export const fitLayerTransformToBbox = (
  layer: CanvasLayerContract,
  bbox: Rect,
  documentRect: Rect = bbox
): LayerTransform | null => {
  const doc = { height: documentRect.height, width: documentRect.width } as CanvasDocumentContractV3;
  const contentRect = getSourceContentRect(layer, doc);
  if (contentRect.width <= 0 || contentRect.height <= 0) {
    return null;
  }
  const scale = Math.min(bbox.width / contentRect.width, bbox.height / contentRect.height);
  return {
    rotation: 0,
    scaleX: scale,
    scaleY: scale,
    x: bbox.x + (bbox.width - contentRect.width * scale) / 2 - contentRect.x * scale,
    y: bbox.y + (bbox.height - contentRect.height * scale) / 2 - contentRect.y * scale,
  };
};

export const getControlTransparencyEffectPatch = (layer: CanvasControlLayerContract): ControlConfigPatch => ({
  layerType: 'control',
  withTransparencyEffect: !layer.withTransparencyEffect,
});

export const getRegionalGuidanceAutoNegativePatch = (
  layer: CanvasRegionalGuidanceLayerContract
): RegionalGuidanceConfigPatch => ({ autoNegative: !layer.autoNegative, layerType: 'regional_guidance' });

/** Allow raster/control conversion only for pixel-backed image/paint sources, excluding parametric and mask layers. */
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
  ...(layer.colorLabel ? { colorLabel: layer.colorLabel } : {}),
  id,
  isEnabled: layer.isEnabled,
  isLocked: layer.isLocked,
  name: isCopy ? `${layer.name} copy` : layer.name,
  opacity: layer.opacity,
  transform: cloneTransform(layer),
});

const pixelLayerToControl = (
  layer: PixelLayer,
  id: string,
  isCopy: boolean,
  base?: string | null,
  model?: string | null
): CanvasControlLayerContract | null => {
  const source = clonePixelSource(layer);
  if (!source) {
    return null;
  }
  return {
    ...destinationBase(layer, id, isCopy),
    adapter: createDefaultControlAdapter(base, model ?? null),
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
  id: string,
  base?: string | null,
  model?: string | null
): CanvasControlLayerContract | null => pixelLayerToControl(layer, id, true, base, model);

export const copyRasterToInpaintMask = (
  layer: CanvasRasterLayerContractV2,
  id: string
): CanvasInpaintMaskLayerContract | null => pixelLayerToInpaintMask(layer, id, true);

export const copyRasterToRegionalGuidance = (
  layer: CanvasRasterLayerContractV2,
  id: string
): CanvasRegionalGuidanceLayerContract | null => pixelLayerToRegionalGuidance(layer, id, true);

export const convertRasterToControl = (
  layer: CanvasRasterLayerContractV2,
  base?: string | null,
  model?: string | null
): CanvasControlLayerContract | null => pixelLayerToControl(layer, layer.id, false, base, model);

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
 * Convert raster/control types while preserving source, identity, transform, and base properties; reject
 * unsupported or unchanged targets.
 */
export const convertRasterControlLayer = (
  layer: CanvasLayerContract,
  targetType: 'raster' | 'control',
  base?: string | null,
  model?: string | null
): CanvasLayerContract | null => {
  if (layer.type === targetType || !canConvertRasterControl(layer)) {
    return null;
  }
  if (layer.type === 'raster' && targetType === 'control') {
    return convertRasterToControl(layer, base, model);
  }
  if (layer.type === 'control' && targetType === 'raster') {
    const source = clonePixelSource(layer);
    return source ? { ...destinationBase(layer, layer.id, false), source, type: 'raster' } : null;
  }
  return null;
};

/** Whether the layers panel may offer merge-down for `layerId`; merging is pixel work, so it needs an engine. */
export const canMergeLayerDown = (document: CanvasDocumentContractV3, layerId: string, hasEngine: boolean): boolean =>
  hasEngine && mergeDownEligibility(document, layerId).status === 'eligible';
