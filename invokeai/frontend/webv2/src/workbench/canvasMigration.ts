import type {
  CanvasDocumentContractV3,
  CanvasGroupContract,
  CanvasInpaintMaskLayerContract,
  CanvasLayerBaseContract,
  CanvasLayerContract,
  CanvasNodeContract,
  CanvasSnapshotContract,
  CanvasStackForests,
  CanvasStagingAreaContractV2,
  CanvasStateContractV3,
  LayerStackKind,
} from '@workbench/canvas-engine/api';

import { CANVAS_MAX_NODE_COUNT, CANVAS_MAX_NODE_DEPTH, LAYER_STACK_ORDER } from '@workbench/canvas-engine/api';
import { repairSelectedLayerId } from '@workbench/canvas-engine/document/selectionRepair';
import { z } from 'zod';

import type { CanvasLoadDiagnostic, CanvasLoadResult, CanvasVersionScope } from './canvasLoadContracts';

import { MAX_SUPPORTED_CANVAS_SCHEMA_VERSION, MIN_SUPPORTED_CANVAS_SCHEMA_VERSION } from './canvasSchemaVersion';
import { normalizeControlAdapter } from './controlAdapters';

export const DEFAULT_CANVAS_DOCUMENT_WIDTH = 1024;
export const DEFAULT_CANVAS_DOCUMENT_HEIGHT = 1024;

const createMigrationId = (prefix: string): string =>
  `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

export const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

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
const zTextFontRef = z.object({
  contentHash: z.string().regex(/^[a-f0-9]{64}$/u),
  family: z.string(),
  id: z.string().min(1),
  label: z.string(),
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
    fontRef: zTextFontRef.optional(),
    fontStyle: z.enum(['normal', 'italic', 'oblique']).optional(),
    fontVariations: z.record(z.string().min(1), zFiniteNumber).optional(),
    lineHeight: zFiniteNumber,
    type: z.literal('text'),
  }),
  z.object({
    fill: z.string().nullable(),
    height: zFiniteNumber,
    kind: z.enum(['rect', 'ellipse', 'triangle', 'star', 'polygon']),
    points: z.array(zCoordinate).optional(),
    stroke: z.string().nullable(),
    strokeWidth: zFiniteNumber,
    type: z.literal('shape'),
    width: zFiniteNumber,
  }),
  z.object({
    angle: zFiniteNumber,
    center: zCoordinate.optional(),
    height: zFiniteNumber.positive().optional(),
    kind: z.enum(['linear', 'radial']),
    span: zFiniteNumber.positive().optional(),
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
const zAdjustmentBase = { id: z.string(), isEnabled: z.boolean(), name: z.string().optional() };
const zAdjustmentEntry = z.discriminatedUnion('type', [
  z.object({
    ...zAdjustmentBase,
    brightness: zFiniteNumber,
    contrast: zFiniteNumber,
    type: z.literal('brightness-contrast'),
  }),
  z.object({ ...zAdjustmentBase, stops: zFiniteNumber, type: z.literal('exposure') }),
  z.object({ ...zAdjustmentBase, saturation: zFiniteNumber, type: z.literal('hsl') }),
  z.object({
    ...zAdjustmentBase,
    curves: z.object({ b: zCurve.optional(), g: zCurve.optional(), r: zCurve.optional() }),
    type: z.literal('curves'),
  }),
  z
    .object({
      ...zAdjustmentBase,
      channel: z.enum(['rgb', 'r', 'g', 'b']).optional(),
      gamma: zFiniteNumber,
      inBlack: zFiniteNumber,
      inWhite: zFiniteNumber,
      outBlack: zFiniteNumber,
      outWhite: zFiniteNumber,
      type: z.literal('levels'),
    })
    .refine((entry) => entry.inBlack < entry.inWhite && entry.gamma > 0),
  z.object({ ...zAdjustmentBase, rotation: zFiniteNumber, type: z.literal('hue') }),
  z.object({ ...zAdjustmentBase, type: z.literal('invert') }),
]);
const zAdjustments = z.array(zAdjustmentEntry);
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
const zMaskFill = z.object({
  color: z.string(),
  style: z.enum(['solid', 'grid', 'crosshatch', 'diagonal', 'horizontal', 'vertical']),
});
const zMask = z.object({
  bitmap: zImageRef.nullable(),
  fill: zMaskFill,
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
const zModelIdentifier = z
  .object({ base: z.string(), key: z.string(), name: z.string(), type: z.string() })
  .passthrough();
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
const zBlendMode = z.enum([
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
]);
const zColorLabel = z.enum(['red', 'orange', 'yellow', 'green', 'blue', 'violet', 'gray']);
const zLayerBase = z.object({
  blendMode: zBlendMode,
  colorLabel: zColorLabel.optional().catch(undefined),
  id: z.string(),
  isEnabled: z.boolean(),
  isLocked: z.boolean(),
  name: z.string(),
  opacity: zFiniteNumber.min(0).max(1),
  transform: zTransform,
});
const zCanvasLayer = z.discriminatedUnion('type', [
  zLayerBase.extend({
    // A pre-stack adjustments OBJECT is dropped rather than failing the document.
    adjustments: zAdjustments.optional().catch(undefined),
    inpaint: z
      .object({ fill: zMaskFill, isEnabled: z.boolean(), name: z.string().optional() })
      .optional()
      .catch(undefined),
    filter: zFilter.optional(),
    isTransparencyLocked: z.boolean().optional(),
    source: zLayerSource,
    type: z.literal('raster'),
  }),
  zLayerBase.extend({
    adapter: zControlAdapter,
    filter: zFilter.optional(),
    // Missing display-only visibility means visible for older documents; only overlay types carry it.
    isHidden: z.boolean().optional(),
    source: zLayerSource,
    type: z.literal('control'),
    withTransparencyEffect: z.boolean(),
  }),
  zLayerBase.extend({
    autoNegative: z.boolean(),
    isHidden: z.boolean().optional(),
    mask: zMask,
    negativePrompt: z.string().nullable(),
    positivePrompt: z.string().nullable(),
    referenceImages: z.array(zReferenceImage),
    type: z.literal('regional_guidance'),
  }),
  zLayerBase.extend({
    denoise: z.object({ isEnabled: z.boolean(), limit: zFiniteNumber }).optional(),
    isHidden: z.boolean().optional(),
    mask: zMask,
    noise: z.object({ isEnabled: z.boolean(), level: zFiniteNumber }).optional(),
    type: z.literal('inpaint_mask'),
  }),
]);

const asNumber = (value: unknown, fallback: number): number => (typeof value === 'number' ? value : fallback);

const asPositiveNumber = (value: unknown, fallback: number): number => {
  const numeric = asNumber(value, fallback);

  return numeric > 0 ? numeric : fallback;
};

/** A positive whole-pixel dimension, shared by creation and every load path. */
const asPositiveInteger = (value: unknown, fallback: number): number =>
  Math.max(1, Math.round(asPositiveNumber(value, fallback)));

/** The canonical whole-pixel geometry for a live or persisted canvas document. */
const normalizeCanvasDocumentGeometry = (rawWidth: unknown, rawHeight: unknown, rawBbox?: unknown) => {
  const width = asPositiveInteger(rawWidth, DEFAULT_CANVAS_DOCUMENT_WIDTH);
  const height = asPositiveInteger(rawHeight, DEFAULT_CANVAS_DOCUMENT_HEIGHT);
  const bbox = isRecord(rawBbox) ? rawBbox : {};
  return {
    bbox: {
      height: asPositiveInteger(bbox.height, height),
      width: asPositiveInteger(bbox.width, width),
      x: Math.round(asNumber(bbox.x, 0)),
      y: Math.round(asNumber(bbox.y, 0)),
    },
    height,
    width,
  };
};

const createEmptyStacks = (): CanvasStackForests => ({
  control: [],
  inpaint_mask: [],
  raster: [],
  regional_guidance: [],
});

export const createEmptyCanvasState = (
  width = DEFAULT_CANVAS_DOCUMENT_WIDTH,
  height = DEFAULT_CANVAS_DOCUMENT_HEIGHT
): CanvasStateContractV3 => ({
  document: createEmptyCanvasDocument(width, height),
  documentRevision: 0,
  snapshots: [],
  stagingArea: createDefaultStagingArea(),
  version: 3,
});

export const createEmptyCanvasDocument = (
  width = DEFAULT_CANVAS_DOCUMENT_WIDTH,
  height = DEFAULT_CANVAS_DOCUMENT_HEIGHT
): CanvasDocumentContractV3 => ({
  background: 'transparent',
  ...normalizeCanvasDocumentGeometry(width, height),
  selectedLayerId: null,
  stacks: createEmptyStacks(),
  version: 3,
});

/** Mirror the default inpaint mask without importing the Layers panel into this pure module. */
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

/** A fresh canvas state for a newly created project: an empty document with one selected empty inpaint mask. */
export const createNewCanvasState = (
  width = DEFAULT_CANVAS_DOCUMENT_WIDTH,
  height = DEFAULT_CANVAS_DOCUMENT_HEIGHT
): CanvasStateContractV3 => {
  const base = createEmptyCanvasState(width, height);
  const mask = createInitialInpaintMaskLayer();
  return {
    ...base,
    document: { ...base.document, selectedLayerId: mask.id, stacks: { ...base.document.stacks, inpaint_mask: [mask] } },
  };
};

const createDefaultStagingArea = (): CanvasStagingAreaContractV2 => ({
  areThumbnailsVisible: true,
  autoSwitchMode: 'off',
  isVisible: false,
  pendingImageIds: [],
  pendingImages: [],
  selectedImageIndex: 0,
});

/** v4 is adopted only when a document actually carries custom-font typography. */
const textSourceRequiresV4 = (source: unknown): boolean =>
  isRecord(source) &&
  source.type === 'text' &&
  ('fontRef' in source || 'fontStyle' in source || 'fontVariations' in source);

const nodeRequiresV4 = (node: unknown): boolean =>
  isRecord(node) &&
  (textSourceRequiresV4(node.source) || (Array.isArray(node.children) && node.children.some(nodeRequiresV4)));

/** Returns whether a document uses any v4-only custom-font source fields. */
export const canvasDocumentRequiresFontSchemaV4 = (document: Pick<CanvasDocumentContractV3, 'stacks'>): boolean =>
  Object.values(document.stacks).some((roots) => roots.some(nodeRequiresV4));

const writableDocumentVersion = (declaredVersion: unknown, stacks: CanvasStackForests): 3 | 4 =>
  declaredVersion === 4 || canvasDocumentRequiresFontSchemaV4({ stacks }) ? 4 : 3;

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

const AUTO_SWITCH_MODES: CanvasStagingAreaContractV2['autoSwitchMode'][] = ['off', 'latest', 'progress'];

const asAutoSwitchMode = (value: unknown): CanvasStagingAreaContractV2['autoSwitchMode'] =>
  AUTO_SWITCH_MODES.includes(value as CanvasStagingAreaContractV2['autoSwitchMode'])
    ? (value as CanvasStagingAreaContractV2['autoSwitchMode'])
    : 'off';

/** Fills the known optional defaults of a staging area. */
const normalizeStagingArea = (rawCanvas: Record<string, unknown>): CanvasStagingAreaContractV2 => {
  const rawStagingArea = isRecord(rawCanvas.stagingArea) ? rawCanvas.stagingArea : {};
  const defaults = createDefaultStagingArea();

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

type Refusal =
  | { status: 'unsupported-version'; scope: CanvasVersionScope; version: number }
  | { status: 'invalid'; scope: CanvasVersionScope; diagnostics: readonly CanvasLoadDiagnostic[] };

type LoadStep<T> = Extract<CanvasLoadResult<T>, { status: 'loaded' }> | Refusal;

type DeclaredVersion = { kind: 'current' } | { kind: 'other'; version: number } | { kind: 'malformed' };

const classifyVersion = (value: unknown): DeclaredVersion => {
  if (value === MAX_SUPPORTED_CANVAS_SCHEMA_VERSION) {
    return { kind: 'current' };
  }
  if (
    typeof value === 'number' &&
    Number.isInteger(value) &&
    value >= MIN_SUPPORTED_CANVAS_SCHEMA_VERSION &&
    value < MAX_SUPPORTED_CANVAS_SCHEMA_VERSION
  ) {
    return { kind: 'current' };
  }
  if (typeof value === 'number' && Number.isInteger(value) && value >= 1) {
    return { kind: 'other', version: value };
  }
  return { kind: 'malformed' };
};

const describeVersion = (value: unknown): string => (typeof value === 'string' ? JSON.stringify(value) : String(value));

const invalid = (scope: CanvasVersionScope, diagnostics: readonly CanvasLoadDiagnostic[]): Refusal => ({
  diagnostics,
  scope,
  status: 'invalid',
});

const unsupported = (scope: CanvasVersionScope, version: number): Refusal => ({
  scope,
  status: 'unsupported-version',
  version,
});

/** Refuses unsupported declared versions before anything is defaulted or parsed. */
const checkVersion = (value: Record<string, unknown>, scope: CanvasVersionScope, path: string): Refusal | null => {
  const version = classifyVersion(value.version);
  switch (version.kind) {
    case 'current':
      return null;
    case 'other':
      return unsupported(scope, version.version);
    case 'malformed':
      return invalid(scope, [
        { message: `canvas version ${describeVersion(value.version)} is not recognized`, path: `${path}version` },
      ]);
  }
};

const describeIssue = (value: Record<string, unknown>, path: string, issues: readonly z.core.$ZodIssue[]) => {
  const issue = issues[0];
  const detail = issue ? `${issue.path.length > 0 ? `${issue.path.join('.')}: ` : ''}${issue.message}` : 'invalid node';
  const type = typeof value.type === 'string' ? value.type : undefined;
  return { message: type ? `${type} node is invalid (${detail})` : `node is invalid (${detail})`, path };
};

const zGroupShell = z.object({
  // Like the raster arm: a malformed stack drops without failing the document.
  adjustments: zAdjustments.optional().catch(undefined),
  blendMode: zBlendMode.optional().catch(undefined),
  colorLabel: zColorLabel.optional().catch(undefined),
  id: z.string(),
  isEnabled: z.boolean(),
  isHidden: z.boolean().optional(),
  isLocked: z.boolean(),
  name: z.string(),
  opacity: z.number().min(0).max(1).optional().catch(undefined),
  type: z.literal('group'),
});

interface ParseContext {
  stack: CanvasLayerContract['type'];
  readonly ids: Set<string>;
  readonly diagnostics: CanvasLoadDiagnostic[];
  count: number;
}

/** Fills the known optional defaults on a leaf, then validates it strictly. */
const parseLeaf = (value: Record<string, unknown>, path: string, context: ParseContext): CanvasLayerContract | null => {
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
  if (!parsed.success) {
    context.diagnostics.push(describeIssue(value, path, parsed.error.issues));
    return null;
  }
  const layer = parsed.data as CanvasLayerContract;
  if (layer.type !== context.stack) {
    context.diagnostics.push({ message: `${layer.type} layer does not belong to the ${context.stack} stack`, path });
    return null;
  }
  return layer;
};

const parseNode = (value: unknown, path: string, depth: number, context: ParseContext): CanvasNodeContract | null => {
  if (!isRecord(value)) {
    context.diagnostics.push({ message: 'node is not an object', path });
    return null;
  }
  context.count += 1;
  if (context.count > CANVAS_MAX_NODE_COUNT) {
    context.diagnostics.push({ message: `document exceeds ${CANVAS_MAX_NODE_COUNT} nodes`, path });
    return null;
  }
  if (typeof value.id === 'string') {
    if (context.ids.has(value.id)) {
      context.diagnostics.push({ message: `node id ${JSON.stringify(value.id)} is not unique`, path: `${path}.id` });
      return null;
    }
    context.ids.add(value.id);
  }
  if (value.type !== 'group') {
    return parseLeaf(value, path, context);
  }
  const shell = zGroupShell.safeParse(value);
  if (!shell.success) {
    context.diagnostics.push(describeIssue(value, path, shell.error.issues));
    return null;
  }
  if (context.stack === 'raster' && shell.data.isHidden === true) {
    context.diagnostics.push({ message: 'a raster group has no display-only hidden state', path: `${path}.isHidden` });
    return null;
  }
  if (context.stack === 'raster' || shell.data.isHidden === false) {
    delete shell.data.isHidden;
  }
  // Overlay groups carry coverage, not color. Drop raster-only properties silently because diagnostics reject the
  // whole document.
  if (context.stack !== 'raster') {
    delete shell.data.adjustments;
    delete shell.data.blendMode;
    delete shell.data.opacity;
  }
  if (depth >= CANVAS_MAX_NODE_DEPTH) {
    context.diagnostics.push({ message: `group nests deeper than ${CANVAS_MAX_NODE_DEPTH} levels`, path });
    return null;
  }
  if (!Array.isArray(value.children)) {
    context.diagnostics.push({ message: 'group children is not an array', path: `${path}.children` });
    return null;
  }
  const children = value.children.map((child, index) =>
    parseNode(child, `${path}.children[${index}]`, depth + 1, context)
  );
  if (children.some((child) => child === null)) {
    return null;
  }
  const group: CanvasGroupContract = { ...shell.data, children: children as CanvasNodeContract[] };
  return group;
};

type ParsedStacks = { stacks: CanvasStackForests } | { diagnostics: CanvasLoadDiagnostic[] };

const parseStacks = (value: unknown, path: string): ParsedStacks => {
  if (!isRecord(value)) {
    return { diagnostics: [{ message: 'stacks is not an object', path }] };
  }
  const stacks = createEmptyStacks();
  const context: ParseContext = { count: 0, diagnostics: [], ids: new Set(), stack: 'raster' };
  for (const key of Object.keys(value)) {
    if (!LAYER_STACK_ORDER.includes(key as LayerStackKind)) {
      context.diagnostics.push({ message: `unknown stack ${JSON.stringify(key)}`, path: `${path}.${key}` });
    }
  }
  for (const stack of LAYER_STACK_ORDER) {
    const roots = value[stack] === undefined ? [] : value[stack];
    if (!Array.isArray(roots)) {
      context.diagnostics.push({ message: `${stack} stack is not an array`, path: `${path}.${stack}` });
      continue;
    }
    context.stack = stack;
    const parsed = roots.map((root, index) => parseNode(root, `${path}.${stack}[${index}]`, 0, context));
    stacks[stack] = parsed.filter((node): node is CanvasNodeContract => node !== null);
  }
  return context.diagnostics.length > 0 ? { diagnostics: context.diagnostics } : { stacks };
};

/** Re-validates an in-memory document, filling known optional defaults; `null` when any node is invalid. */
export const normalizeCanvasDocumentContract = (
  document: CanvasDocumentContractV3
): CanvasDocumentContractV3 | null => {
  const parsed = parseStacks(document.stacks, 'stacks');
  return 'stacks' in parsed
    ? {
        ...document,
        ...normalizeCanvasDocumentGeometry(document.width, document.height, document.bbox),
        stacks: parsed.stacks,
        version: writableDocumentVersion(document.version, parsed.stacks),
      }
    : null;
};

const loadCanvasDocument = (
  value: unknown,
  scope: CanvasVersionScope,
  path: string
): LoadStep<CanvasDocumentContractV3> => {
  if (!isRecord(value)) {
    return invalid(scope, [{ message: 'document is not an object', path }]);
  }
  const refusal = value.version === undefined ? null : checkVersion(value, scope, `${path}.`);
  if (refusal) {
    return refusal;
  }
  const stacks = parseStacks(value.stacks, `${path}.stacks`);
  if ('diagnostics' in stacks) {
    return invalid(scope, stacks.diagnostics);
  }
  const declaredSelection = typeof value.selectedLayerId === 'string' ? value.selectedLayerId : null;
  const selectedLayerId = repairSelectedLayerId(stacks.stacks, declaredSelection);
  return {
    diagnostics:
      declaredSelection !== null && selectedLayerId !== declaredSelection
        ? [
            {
              message: `selected layer ${JSON.stringify(declaredSelection)} is not in the document`,
              path: `${path}.selectedLayerId`,
            },
          ]
        : [],
    status: 'loaded',
    value: {
      background:
        value.background === 'transparent' || isRecord(value.background)
          ? (value.background as CanvasDocumentContractV3['background'])
          : 'transparent',
      ...normalizeCanvasDocumentGeometry(value.width, value.height, value.bbox),
      selectedLayerId,
      stacks: stacks.stacks,
      version: writableDocumentVersion(value.version, stacks.stacks),
    },
  };
};

const loadCanvasSnapshot = (value: unknown, path: string): LoadStep<CanvasSnapshotContract> => {
  if (!isRecord(value)) {
    return invalid('snapshot', [{ message: 'snapshot is not an object', path }]);
  }
  const missing = (['id', 'name', 'createdAt'] as const).filter((key) => typeof value[key] !== 'string');
  if (missing.length > 0) {
    return invalid(
      'snapshot',
      missing.map((key) => ({ message: `snapshot ${key} is missing`, path: `${path}.${key}` }))
    );
  }
  const document = loadCanvasDocument(value.document, 'snapshot', `${path}.document`);
  return document.status === 'loaded'
    ? {
        diagnostics: document.diagnostics,
        status: 'loaded',
        value: { ...value, document: document.value } as CanvasSnapshotContract,
      }
    : document;
};

const loadCanvasStateStep = (canvas: unknown): LoadStep<CanvasStateContractV3> => {
  if (canvas === undefined || canvas === null) {
    return { diagnostics: [], status: 'loaded', value: createEmptyCanvasState() };
  }
  if (!isRecord(canvas) || Array.isArray(canvas)) {
    return invalid('state', [{ message: 'canvas state is not an object', path: '' }]);
  }
  const refusal = checkVersion(canvas, 'state', '');
  if (refusal) {
    return refusal;
  }
  const document = loadCanvasDocument(canvas.document === undefined ? {} : canvas.document, 'document', 'document');
  if (document.status !== 'loaded') {
    return document;
  }
  const rawSnapshots = canvas.snapshots === undefined ? [] : canvas.snapshots;
  if (!Array.isArray(rawSnapshots)) {
    return invalid('state', [{ message: 'snapshots is not an array', path: 'snapshots' }]);
  }
  const snapshots: CanvasSnapshotContract[] = [];
  const diagnostics = [...document.diagnostics];
  for (const [index, rawSnapshot] of rawSnapshots.entries()) {
    const snapshot = loadCanvasSnapshot(rawSnapshot, `snapshots[${index}]`);
    if (snapshot.status !== 'loaded') {
      return snapshot;
    }
    snapshots.push(snapshot.value);
    diagnostics.push(...snapshot.diagnostics);
  }
  return {
    diagnostics,
    status: 'loaded',
    value: {
      document: document.value,
      documentRevision: asNumber(canvas.documentRevision, 0),
      snapshots,
      stagingArea: normalizeStagingArea(canvas),
      version:
        canvas.version === MAX_SUPPORTED_CANVAS_SCHEMA_VERSION ||
        document.value.version === MAX_SUPPORTED_CANVAS_SCHEMA_VERSION ||
        snapshots.some((snapshot) => snapshot.document.version === MAX_SUPPORTED_CANVAS_SCHEMA_VERSION)
          ? MAX_SUPPORTED_CANVAS_SCHEMA_VERSION
          : 3,
    },
  };
};

/**
 * Check versions before defaulting: absent state is fresh, current state is validated, and other versions retain
 * raw payloads for recovery.
 */
export const loadCanvasState = (canvas: unknown): CanvasLoadResult<CanvasStateContractV3> => {
  const step = loadCanvasStateStep(canvas);
  return step.status === 'loaded' ? step : { ...step, raw: canvas };
};
