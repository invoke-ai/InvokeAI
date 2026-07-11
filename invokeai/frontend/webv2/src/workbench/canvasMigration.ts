import { z } from 'zod';

/**
 * Canvas v1 -> v2 document migration.
 *
 * `CanvasStateContract` (v1) only ever produced a flat stack of single-image
 * raster layers positioned by an absolute `placement` rect. `CanvasStateContractV2`
 * (see `types.ts`) generalizes the document into a typed layer union (raster,
 * control, regional guidance, inpaint mask) positioned by a `transform`, with
 * bitmaps referenced by `imageName` rather than by resolved URL.
 *
 * `migrateCanvasStateToV2` accepts genuinely unknown input (as read from
 * localStorage) so it doubles as both the v1->v2 converter and the
 * garbage/undefined-input fallback used when normalizing a loaded project.
 * Already-v2 input passes through normalized, not double-migrated.
 */
import type {
  CanvasDocumentContractV2,
  CanvasImageRef,
  CanvasInpaintMaskLayerContract,
  CanvasLayerBaseContract,
  CanvasRasterLayerContractV2,
  CanvasStagingAreaContractV2,
  CanvasStateContractV2,
} from './types';

import { normalizeControlAdapter } from './controlAdapters';

export const DEFAULT_CANVAS_DOCUMENT_WIDTH = 1024;
export const DEFAULT_CANVAS_DOCUMENT_HEIGHT = 1024;

const createMigrationId = (prefix: string): string =>
  `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

const zFiniteNumber = z.number().finite();
const zCoordinate = z.object({ x: zFiniteNumber, y: zFiniteNumber });
const zImageRef = z.object({
  contentHash: z.string().optional(),
  height: zFiniteNumber.nonnegative(),
  imageName: z.string(),
  width: zFiniteNumber.nonnegative(),
});
const zPaintSource = z.object({
  bitmap: zImageRef.nullable(),
  offset: zCoordinate.optional(),
  type: z.literal('paint'),
});
const zLayerSource = z.discriminatedUnion('type', [
  zPaintSource,
  z.object({ image: zImageRef, type: z.literal('image') }),
  z.object({
    align: z.enum(['left', 'center', 'right']),
    color: z.string(),
    content: z.string(),
    fontFamily: z.string(),
    fontSize: zFiniteNumber,
    fontWeight: zFiniteNumber,
    lineHeight: zFiniteNumber,
    type: z.literal('text'),
  }),
  z.object({
    fill: z.string().nullable(),
    height: zFiniteNumber,
    kind: z.enum(['rect', 'ellipse', 'polygon']),
    points: z.array(zCoordinate).optional(),
    stroke: z.string().nullable(),
    strokeWidth: zFiniteNumber,
    type: z.literal('shape'),
    width: zFiniteNumber,
  }),
  z.object({
    angle: zFiniteNumber,
    height: zFiniteNumber.positive().optional(),
    kind: z.enum(['linear', 'radial']),
    stops: z.array(z.object({ color: z.string(), offset: zFiniteNumber })),
    type: z.literal('gradient'),
    width: zFiniteNumber.positive().optional(),
  }),
]);
const zTransform = z.object({
  rotation: zFiniteNumber,
  scaleX: zFiniteNumber,
  scaleY: zFiniteNumber,
  x: zFiniteNumber,
  y: zFiniteNumber,
});
const zFilter = z.object({ settings: z.record(z.string(), z.unknown()), type: z.string() });
const zCurve = z.array(z.tuple([zFiniteNumber, zFiniteNumber]));
const zAdjustments = z.object({
  brightness: zFiniteNumber,
  contrast: zFiniteNumber,
  curves: z.object({ b: zCurve, g: zCurve, r: zCurve }).optional(),
  saturation: zFiniteNumber,
});
const zControlAdapter = z
  .object({
    beginEndStepPct: z.tuple([zFiniteNumber, zFiniteNumber]),
    controlMode: z.enum(['balanced', 'more_prompt', 'more_control', 'unbalanced']).nullable(),
    kind: z.enum(['controlnet', 't2i_adapter', 'control_lora', 'z_image_control']),
    model: z.string().nullable(),
    weight: zFiniteNumber,
  })
  .refine(({ beginEndStepPct }) => {
    const [begin, end] = beginEndStepPct;
    return begin >= 0 && end <= 1 && begin < end;
  })
  .refine(({ kind, weight }) => weight >= (kind === 'z_image_control' ? 0 : -1) && weight <= 2);
const zMask = z.object({
  bitmap: zImageRef.nullable(),
  fill: z.object({
    color: z.string(),
    style: z.enum(['solid', 'grid', 'crosshatch', 'diagonal', 'horizontal', 'vertical']),
  }),
  offset: zCoordinate.optional(),
});
const zGeneratedImage = z.object({
  height: zFiniteNumber.nonnegative(),
  imageName: z.string(),
  imageUrl: z.string(),
  queuedAt: z.string(),
  sourceQueueItemId: z.string(),
  thumbnailUrl: z.string(),
  width: zFiniteNumber.nonnegative(),
});
const zModelIdentifier = z.object({ base: z.string(), key: z.string(), name: z.string(), type: z.string() });
const zReferenceImage = z.object({
  config: z.discriminatedUnion('type', [
    z.object({
      beginEndStepPct: z.tuple([zFiniteNumber, zFiniteNumber]),
      clipVisionModel: z.enum(['ViT-H', 'ViT-G', 'ViT-L']),
      image: zGeneratedImage.nullable(),
      method: z.enum(['full', 'style', 'composition', 'style_strong', 'style_precise']),
      model: zModelIdentifier.nullable(),
      type: z.literal('ip_adapter'),
      weight: zFiniteNumber,
    }),
    z.object({
      image: zGeneratedImage.nullable(),
      imageInfluence: z.enum(['lowest', 'low', 'medium', 'high', 'highest']),
      model: zModelIdentifier.nullable(),
      type: z.literal('flux_redux'),
    }),
    z.object({
      image: zGeneratedImage.nullable(),
      model: zModelIdentifier.nullable(),
      type: z.literal('flux_kontext_reference_image'),
    }),
    z.object({ image: zGeneratedImage.nullable(), type: z.literal('flux2_reference_image') }),
    z.object({ image: zGeneratedImage.nullable(), type: z.literal('qwen_image_reference_image') }),
    z.object({ image: zGeneratedImage.nullable(), type: z.literal('external_reference_image') }),
  ]),
  id: z.string(),
  isEnabled: z.boolean(),
});
const zLayerBase = z.object({
  blendMode: z.enum([
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
  ]),
  id: z.string(),
  isEnabled: z.boolean(),
  isLocked: z.boolean(),
  name: z.string(),
  opacity: zFiniteNumber.min(0).max(1),
  transform: zTransform,
});
const zCanvasLayer = z.discriminatedUnion('type', [
  zLayerBase.extend({
    adjustments: zAdjustments.optional(),
    filter: zFilter.optional(),
    isTransparencyLocked: z.boolean().optional(),
    source: zLayerSource,
    type: z.literal('raster'),
  }),
  zLayerBase.extend({
    adapter: zControlAdapter,
    filter: zFilter.optional(),
    source: zLayerSource,
    type: z.literal('control'),
    withTransparencyEffect: z.boolean(),
  }),
  zLayerBase.extend({
    autoNegative: z.boolean(),
    mask: zMask,
    negativePrompt: z.string().nullable(),
    positivePrompt: z.string().nullable(),
    referenceImages: z.array(zReferenceImage),
    type: z.literal('regional_guidance'),
  }),
  zLayerBase.extend({
    denoiseLimit: zFiniteNumber.optional(),
    mask: zMask,
    noiseLevel: zFiniteNumber.optional(),
    type: z.literal('inpaint_mask'),
  }),
]);

const asNumber = (value: unknown, fallback: number): number => (typeof value === 'number' ? value : fallback);

const asString = (value: unknown, fallback: string): string => (typeof value === 'string' ? value : fallback);

const asPositiveNumber = (value: unknown, fallback: number): number => {
  const numeric = asNumber(value, fallback);

  return numeric > 0 ? numeric : fallback;
};

export const createEmptyCanvasStateV2 = (
  width = DEFAULT_CANVAS_DOCUMENT_WIDTH,
  height = DEFAULT_CANVAS_DOCUMENT_HEIGHT
): CanvasStateContractV2 => ({
  document: createEmptyCanvasDocumentV2(width, height),
  documentRevision: 0,
  snapshots: [],
  stagingArea: createDefaultStagingAreaV2(),
  version: 2,
});

export const createEmptyCanvasDocumentV2 = (
  width = DEFAULT_CANVAS_DOCUMENT_WIDTH,
  height = DEFAULT_CANVAS_DOCUMENT_HEIGHT
): CanvasDocumentContractV2 => ({
  background: 'transparent',
  bbox: { height, width, x: 0, y: 0 },
  height,
  layers: [],
  selectedLayerId: null,
  version: 2,
  width,
});

/**
 * A brand-new project's default inpaint mask: one empty mask (no bitmap/strokes)
 * with the legacy-default diagonal-hatch fill in the first cycled mask colour
 * (legacy `rgb(224,117,117)`). Mirrors `createInpaintMaskLayer` /
 * `DEFAULT_INPAINT_MASK_FILL` in `widgets/layers/layerOps` — duplicated here so
 * the pure reducer/migration module doesn't pull in the layers-panel/engine
 * module graph; `canvasMigration.test.ts` locks the shape against that factory.
 */
const createInitialInpaintMaskLayer = (): CanvasInpaintMaskLayerContract => ({
  blendMode: 'normal',
  id: createMigrationId('layer'),
  isEnabled: true,
  isLocked: false,
  mask: { bitmap: null, fill: { color: '#e07575', style: 'diagonal' } },
  name: 'Inpaint Mask 1',
  opacity: 1,
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'inpaint_mask',
});

/**
 * A fresh canvas state for a newly created project: an empty document that
 * already carries one empty inpaint mask (selected), matching legacy, which seeds
 * a canvas session with an inpaint mask present. The mask has no content, so it
 * does NOT flip generation-mode detection to inpaint (see `detectCanvasMode`).
 *
 * This is the NEW-canvas path only. The migration / garbage-fallback path
 * (`migrateCanvasStateToV2`) and `createEmptyCanvasStateV2` stay empty, so
 * existing and migrated documents are left untouched.
 */
export const createNewCanvasStateV2 = (
  width = DEFAULT_CANVAS_DOCUMENT_WIDTH,
  height = DEFAULT_CANVAS_DOCUMENT_HEIGHT
): CanvasStateContractV2 => {
  const base = createEmptyCanvasStateV2(width, height);
  const mask = createInitialInpaintMaskLayer();

  return {
    ...base,
    document: { ...base.document, layers: [mask], selectedLayerId: mask.id },
  };
};

const createDefaultStagingAreaV2 = (): CanvasStagingAreaContractV2 => ({
  areThumbnailsVisible: true,
  autoSwitchMode: 'off',
  isVisible: false,
  pendingImageIds: [],
  pendingImages: [],
  selectedImageIndex: 0,
});

/**
 * Converts a v1 `{x,y,width,height}` placement rect, plus the native size of the image it
 * places, into a v2 `transform`. Shared by the migration path and the live "accept staged
 * image into a raster layer" reducer, which still works from a v1-shaped placement.
 */
export const placementToTransform = (
  placement: { x: number; y: number; width: number; height: number },
  imageWidth: number,
  imageHeight: number
): CanvasLayerBaseContract['transform'] => ({
  rotation: 0,
  scaleX: imageWidth > 0 ? placement.width / imageWidth : 1,
  scaleY: imageHeight > 0 ? placement.height / imageHeight : 1,
  x: placement.x,
  y: placement.y,
});

/** Migrates a single v1 `CanvasRasterLayerContract` (or bare imageName string) into a v2 raster layer. */
const migrateLayerToV2 = (rawLayer: unknown, index: number): CanvasRasterLayerContractV2 => {
  if (typeof rawLayer === 'string') {
    return {
      blendMode: 'normal',
      id: rawLayer || createMigrationId('layer'),
      isEnabled: true,
      isLocked: false,
      name: rawLayer || `Layer ${index + 1}`,
      opacity: 1,
      source: { image: { height: 0, imageName: rawLayer, width: 0 }, type: 'image' },
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
      type: 'raster',
    };
  }

  const layer = isRecord(rawLayer) ? rawLayer : {};
  const image: CanvasImageRef = {
    height: asNumber(layer.height, 0),
    imageName: asString(layer.imageName, ''),
    width: asNumber(layer.width, 0),
  };
  const placement = isRecord(layer.placement) ? layer.placement : {};
  const placementX = asNumber(placement.x, 0);
  const placementY = asNumber(placement.y, 0);
  const placementWidth = asNumber(placement.width, image.width);
  const placementHeight = asNumber(placement.height, image.height);
  const placementOpacity = typeof placement.opacity === 'number' ? placement.opacity : 1;

  return {
    blendMode: 'normal',
    id: asString(layer.id, createMigrationId('layer')),
    isEnabled: true,
    isLocked: false,
    name: asString(layer.label, `Layer ${index + 1}`),
    opacity: placementOpacity,
    source: { image, type: 'image' },
    transform: placementToTransform(
      { height: placementHeight, width: placementWidth, x: placementX, y: placementY },
      image.width,
      image.height
    ),
    type: 'raster',
  };
};

const migrateDocumentToV2 = (rawCanvas: Record<string, unknown>): CanvasDocumentContractV2 => {
  const rawDocument = isRecord(rawCanvas.document) ? rawCanvas.document : rawCanvas;
  const width = asPositiveNumber(rawDocument.width, DEFAULT_CANVAS_DOCUMENT_WIDTH);
  const height = asPositiveNumber(rawDocument.height, DEFAULT_CANVAS_DOCUMENT_HEIGHT);
  const rawLayers = Array.isArray(rawDocument.layers) ? rawDocument.layers : [];

  return {
    background: 'transparent',
    bbox: { height, width, x: 0, y: 0 },
    height,
    layers: rawLayers.map((rawLayer, index) => migrateLayerToV2(rawLayer, index)),
    selectedLayerId: null,
    version: 2,
    width,
  };
};

const AUTO_SWITCH_MODES: CanvasStagingAreaContractV2['autoSwitchMode'][] = ['off', 'latest', 'progress'];

const asAutoSwitchMode = (value: unknown): CanvasStagingAreaContractV2['autoSwitchMode'] =>
  AUTO_SWITCH_MODES.includes(value as CanvasStagingAreaContractV2['autoSwitchMode'])
    ? (value as CanvasStagingAreaContractV2['autoSwitchMode'])
    : 'off';

/**
 * Normalizes a v1 or v2 staging area. v1 never had `autoSwitchMode`, so it's absent from raw
 * input and defaults to `'off'`; already-v2 input keeps its existing value.
 */
const migrateStagingAreaToV2 = (rawCanvas: Record<string, unknown>): CanvasStagingAreaContractV2 => {
  const rawStagingArea = isRecord(rawCanvas.stagingArea) ? rawCanvas.stagingArea : {};
  const defaults = createDefaultStagingAreaV2();

  return {
    areThumbnailsVisible:
      typeof rawStagingArea.areThumbnailsVisible === 'boolean'
        ? rawStagingArea.areThumbnailsVisible
        : defaults.areThumbnailsVisible,
    autoSwitchMode: asAutoSwitchMode(rawStagingArea.autoSwitchMode),
    isVisible: typeof rawStagingArea.isVisible === 'boolean' ? rawStagingArea.isVisible : defaults.isVisible,
    pendingImageIds: Array.isArray(rawStagingArea.pendingImageIds)
      ? (rawStagingArea.pendingImageIds as CanvasStagingAreaContractV2['pendingImageIds'])
      : defaults.pendingImageIds,
    pendingImages: Array.isArray(rawStagingArea.pendingImages)
      ? (rawStagingArea.pendingImages as CanvasStagingAreaContractV2['pendingImages'])
      : defaults.pendingImages,
    selectedImageIndex: asNumber(rawStagingArea.selectedImageIndex, defaults.selectedImageIndex),
    ...(typeof rawStagingArea.selectedLayerId === 'string' ? { selectedLayerId: rawStagingArea.selectedLayerId } : {}),
    ...(typeof rawStagingArea.sourceQueueItemId === 'string'
      ? { sourceQueueItemId: rawStagingArea.sourceQueueItemId }
      : {}),
  };
};

const isCanvasStateV2 = (canvas: unknown): canvas is Record<string, unknown> & { version: 2 } =>
  isRecord(canvas) && canvas.version === 2;

const normalizeCanvasLayer = (value: unknown): CanvasDocumentContractV2['layers'][number] | null => {
  if (!isRecord(value)) {
    return null;
  }
  let candidate: Record<string, unknown> = value;
  if (value.type === 'control') {
    candidate = {
      ...value,
      adapter: normalizeControlAdapter(value.adapter),
      withTransparencyEffect: value.withTransparencyEffect === undefined ? true : value.withTransparencyEffect,
    };
  } else if (value.type === 'regional_guidance') {
    candidate = {
      ...value,
      autoNegative: value.autoNegative === undefined ? false : value.autoNegative,
      negativePrompt: value.negativePrompt === undefined ? null : value.negativePrompt,
      positivePrompt: value.positivePrompt === undefined ? null : value.positivePrompt,
      referenceImages: value.referenceImages === undefined ? [] : value.referenceImages,
    };
  }
  const parsed = zCanvasLayer.safeParse(candidate);
  return parsed.success ? (candidate as unknown as CanvasDocumentContractV2['layers'][number]) : null;
};

export const normalizeCanvasDocumentControlAdapters = (
  document: CanvasDocumentContractV2
): CanvasDocumentContractV2 => ({
  ...document,
  layers: document.layers.flatMap((layer) => {
    const normalized = normalizeCanvasLayer(layer);
    return normalized ? [normalized] : [];
  }),
});

const normalizeCanvasDocumentV2 = (value: unknown, requireValidLayers: boolean): CanvasDocumentContractV2 | null => {
  const rawDocument = isRecord(value) ? value : {};
  const rawLayers = Array.isArray(rawDocument.layers) ? rawDocument.layers : [];
  const layers = rawLayers.flatMap((layer) => {
    const normalized = normalizeCanvasLayer(layer);
    return normalized ? [normalized] : [];
  });
  if (requireValidLayers && (!Array.isArray(rawDocument.layers) || layers.length !== rawLayers.length)) {
    return null;
  }
  const width = asPositiveNumber(rawDocument.width, DEFAULT_CANVAS_DOCUMENT_WIDTH);
  const height = asPositiveNumber(rawDocument.height, DEFAULT_CANVAS_DOCUMENT_HEIGHT);
  const bbox = isRecord(rawDocument.bbox) ? rawDocument.bbox : {};
  return {
    background:
      rawDocument.background === 'transparent' || isRecord(rawDocument.background)
        ? (rawDocument.background as CanvasDocumentContractV2['background'])
        : 'transparent',
    bbox: {
      height: asPositiveNumber(bbox.height, height),
      width: asPositiveNumber(bbox.width, width),
      x: asNumber(bbox.x, 0),
      y: asNumber(bbox.y, 0),
    },
    height,
    layers,
    selectedLayerId: typeof rawDocument.selectedLayerId === 'string' ? rawDocument.selectedLayerId : null,
    version: 2,
    width,
  };
};

const normalizeCanvasSnapshot = (value: unknown): CanvasStateContractV2['snapshots'][number] | null => {
  if (
    !isRecord(value) ||
    typeof value.id !== 'string' ||
    typeof value.name !== 'string' ||
    typeof value.createdAt !== 'string'
  ) {
    return null;
  }
  const document = normalizeCanvasDocumentV2(value.document, true);
  return document ? ({ ...value, document } as CanvasStateContractV2['snapshots'][number]) : null;
};

/** Defensively re-normalizes an already-v2 canvas state (fills in anything missing without re-deriving layers). */
const normalizeCanvasStateV2 = (canvas: Record<string, unknown>): CanvasStateContractV2 => {
  const document = normalizeCanvasDocumentV2(canvas.document, false) ?? createEmptyCanvasDocumentV2();

  return {
    document,
    documentRevision: asNumber(canvas.documentRevision, 0),
    snapshots: Array.isArray(canvas.snapshots)
      ? canvas.snapshots.flatMap((snapshot) => {
          const normalized = normalizeCanvasSnapshot(snapshot);
          return normalized ? [normalized] : [];
        })
      : [],
    stagingArea: migrateStagingAreaToV2(canvas),
    version: 2,
  };
};

/**
 * Converts unknown persisted canvas state (v1, v2, or garbage) into `CanvasStateContractV2`.
 * Already-v2 input is normalized (defaults filled in) but not re-derived from placements.
 */
export const migrateCanvasStateToV2 = (canvas: unknown): CanvasStateContractV2 => {
  if (isCanvasStateV2(canvas)) {
    return normalizeCanvasStateV2(canvas);
  }

  const rawCanvas = isRecord(canvas) ? canvas : {};

  return {
    document: migrateDocumentToV2(rawCanvas),
    documentRevision: 0,
    snapshots: [],
    stagingArea: migrateStagingAreaToV2(rawCanvas),
    version: 2,
  };
};
