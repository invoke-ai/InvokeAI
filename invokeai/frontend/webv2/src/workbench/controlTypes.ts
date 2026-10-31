import type { ModelConfig } from '@features/models';
import type { ControlType } from '@workbench/canvas-engine/api';

/**
 * What a control map encodes. A control type groups the families its detectors produce; a control model is trained
 * on one family (or, for union models, several).
 */
export type ControlFamily =
  | 'openpose'
  | 'face'
  | 'depth'
  | 'normal'
  | 'canny'
  | 'mlsd'
  | 'softedge'
  | 'lineart'
  | 'lineart_anime'
  | 'scribble'
  | 'color'
  | 'shuffle'
  | 'tile';

/** A detector a control type offers: a filter type from `CONTROL_FILTERS`, with settings it fixes. */
export interface ControlDetectorSpec {
  type: string;
  family: ControlFamily;
  /** Settings this entry requires; a detector input matches it only when every one is equal. */
  settings?: Record<string, unknown>;
}

export interface ControlTypeSpec {
  id: ControlType;
  detectors: readonly ControlDetectorSpec[];
  /** The detector a new image starts with; null when the image is usually already the map. */
  defaultDetector: string | null;
  /** The family of a map that skipped detection. */
  primaryFamily: ControlFamily;
  /** FLUX InstantX Union `instantx_control_mode` per family; absent families run without a mode. */
  instantxMode: Partial<Record<ControlFamily, number>>;
}

export const CONTROL_TYPE_SPECS: Readonly<Record<ControlType, ControlTypeSpec>> = {
  color: {
    defaultDetector: 'color_map',
    detectors: [
      { family: 'color', type: 'color_map' },
      { family: 'shuffle', type: 'content_shuffle' },
    ],
    id: 'color',
    instantxMode: {},
    primaryFamily: 'color',
  },
  depth: {
    defaultDetector: 'depth_anything_depth_estimation',
    detectors: [
      { family: 'depth', type: 'depth_anything_depth_estimation' },
      { family: 'normal', type: 'normal_map' },
    ],
    id: 'depth',
    instantxMode: { depth: 2 },
    primaryFamily: 'depth',
  },
  edges: {
    defaultDetector: 'canny_edge_detection',
    detectors: [
      { family: 'canny', type: 'canny_edge_detection' },
      { family: 'mlsd', type: 'mlsd_detection' },
    ],
    id: 'edges',
    instantxMode: { canny: 0 },
    primaryFamily: 'canny',
  },
  lineart: {
    defaultDetector: 'lineart_edge_detection',
    detectors: [
      { family: 'lineart', type: 'lineart_edge_detection' },
      { family: 'lineart_anime', type: 'lineart_anime_edge_detection' },
    ],
    id: 'lineart',
    instantxMode: {},
    primaryFamily: 'lineart',
  },
  pose: {
    defaultDetector: 'dw_openpose_detection',
    detectors: [
      { family: 'openpose', type: 'dw_openpose_detection' },
      { family: 'face', type: 'mediapipe_face_detection' },
    ],
    id: 'pose',
    instantxMode: { openpose: 4 },
    primaryFamily: 'openpose',
  },
  // Sketches are usually drawn as the map already, so a new sketch layer skips detection.
  sketch: {
    defaultDetector: null,
    detectors: [
      { family: 'scribble', type: 'scribble_edge_detection' },
      { family: 'scribble', settings: { scribble: true }, type: 'pidi_edge_detection' },
    ],
    id: 'sketch',
    instantxMode: {},
    primaryFamily: 'scribble',
  },
  softedge: {
    defaultDetector: 'hed_edge_detection',
    detectors: [
      { family: 'softedge', type: 'hed_edge_detection' },
      { family: 'softedge', type: 'pidi_edge_detection' },
    ],
    id: 'softedge',
    instantxMode: {},
    primaryFamily: 'softedge',
  },
  tile: {
    defaultDetector: null,
    detectors: [],
    id: 'tile',
    instantxMode: { tile: 1 },
    primaryFamily: 'tile',
  },
};

const settingsMatch = (required: Record<string, unknown>, actual: Record<string, unknown> | undefined): boolean =>
  Object.entries(required).every(([key, value]) => actual?.[key] === value);

/** The family a type's detector produces; no detector (or one the type does not list) means its primary family. */
export const getDetectorFamily = (type: ControlType, detectorType: string | null): ControlFamily => {
  const spec = CONTROL_TYPE_SPECS[type];
  return spec.detectors.find((detector) => detector.type === detectorType)?.family ?? spec.primaryFamily;
};

/**
 * The control type a detector run belongs to, for maps made outside a control layer (a raster filter saved as
 * control). An entry that fixes settings beats a generic one, so PiDi with `scribble` is a sketch.
 */
export const controlTypeForDetector = (detector: {
  type: string;
  settings?: Record<string, unknown>;
}): ControlType | null => {
  let generic: ControlType | null = null;
  for (const spec of Object.values(CONTROL_TYPE_SPECS)) {
    for (const entry of spec.detectors) {
      if (entry.type !== detector.type) {
        continue;
      }
      if (entry.settings) {
        if (settingsMatch(entry.settings, detector.settings)) {
          return spec.id;
        }
      } else {
        generic ??= spec.id;
      }
    }
  }
  return generic;
};

/**
 * Model `default_settings.preprocessor` values mapped to the family they produce: the legacy `*_image_processor`
 * ids older installs carry, the filter node ids the model manager saves today, and `tile`.
 */
export const PREPROCESSOR_FAMILY: Readonly<Record<string, ControlFamily>> = {
  canny_edge_detection: 'canny',
  canny_image_processor: 'canny',
  color_map: 'color',
  color_map_image_processor: 'color',
  content_shuffle: 'shuffle',
  content_shuffle_image_processor: 'shuffle',
  depth_anything_depth_estimation: 'depth',
  depth_anything_image_processor: 'depth',
  dw_openpose_detection: 'openpose',
  dw_openpose_image_processor: 'openpose',
  hed_edge_detection: 'softedge',
  hed_image_processor: 'softedge',
  lineart_anime_edge_detection: 'lineart_anime',
  lineart_anime_image_processor: 'lineart_anime',
  lineart_edge_detection: 'lineart',
  lineart_image_processor: 'lineart',
  mediapipe_face_detection: 'face',
  mediapipe_face_processor: 'face',
  midas_depth_image_processor: 'depth',
  mlsd_detection: 'mlsd',
  mlsd_image_processor: 'mlsd',
  normal_map: 'normal',
  normalbae_image_processor: 'normal',
  openpose_image_processor: 'openpose',
  pidi_edge_detection: 'softedge',
  pidi_image_processor: 'softedge',
  scribble_edge_detection: 'scribble',
  tile: 'tile',
  zoe_depth_image_processor: 'depth',
};

const FAMILY_PURPOSE: Readonly<Record<ControlFamily, ControlType>> = {
  canny: 'edges',
  color: 'color',
  depth: 'depth',
  face: 'pose',
  lineart: 'lineart',
  lineart_anime: 'lineart',
  mlsd: 'edges',
  normal: 'depth',
  openpose: 'pose',
  scribble: 'sketch',
  shuffle: 'color',
  softedge: 'softedge',
  tile: 'tile',
};

/** What a control model was trained for, as far as its name and saved preprocessor tell. */
export interface ControlModelClass {
  /** The control type it is made for; null for union models and models nothing identifies. */
  purpose: ControlType | null;
  family: ControlFamily | null;
  union: boolean;
  /** Families a union model handles without an explicit pick. */
  covers: readonly ControlFamily[];
}

/**
 * The Fun-ControlNet-Union models these bases install as `controlnet`. Both coverage lists follow the model cards'
 * advertised conditions and still need confirming against them.
 */
const FUN_UNION_BASES: ReadonlySet<string> = new Set(['z-image', 'qwen-image', 'flux2']);
const FUN_UNION_COVERS: readonly ControlFamily[] = ['canny', 'mlsd', 'softedge', 'depth', 'openpose'];
/** The InstantX Qwen-Image ControlNet-Union, which also installs as a Qwen-Image `controlnet`: no MLSD. */
const INSTANTX_QWEN_UNION_COVERS: readonly ControlFamily[] = ['canny', 'softedge', 'depth', 'openpose'];
/** InstantX FLUX.1 Union modes the control types map onto. */
const FLUX_UNION_COVERS: readonly ControlFamily[] = ['canny', 'tile', 'depth', 'openpose'];
/** The xinsir SDXL Union (ProMax) conditions; upscale also takes an SDXL union as its tile ControlNet. */
const SDXL_UNION_COVERS: readonly ControlFamily[] = [
  'openpose',
  'depth',
  'canny',
  'mlsd',
  'lineart',
  'lineart_anime',
  'scribble',
  'softedge',
  'normal',
  'tile',
];
const UNION_COVERS_BY_BASE: Readonly<Record<string, readonly ControlFamily[]>> = {
  flux: FLUX_UNION_COVERS,
  sdxl: SDXL_UNION_COVERS,
};

/** Lowercase name words, split at punctuation and camelCase boundaries. */
const nameTokens = (name: string): string[] =>
  name
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);

/** Prefix matching catches `normalbae` and `softedge` without matching inside unrelated words. */
const hasToken = (tokens: readonly string[], ...keywords: string[]): boolean =>
  tokens.some((token) => keywords.some((keyword) => token.startsWith(keyword)));

/** Names that outrank a saved preprocessor: the backend guesses `lineart` for scribble models, for one. */
const strongNameFamily = (tokens: readonly string[]): ControlFamily | null => {
  if (hasToken(tokens, 'tile')) {
    return 'tile';
  }
  if (hasToken(tokens, 'scribble', 'sketch')) {
    return 'scribble';
  }
  if (hasToken(tokens, 'lineart') && hasToken(tokens, 'anime')) {
    return 'lineart_anime';
  }
  if (hasToken(tokens, 'mediapipe', 'face')) {
    return 'face';
  }
  return null;
};

const WEAK_NAME_FAMILIES: readonly (readonly [ControlFamily, readonly string[]])[] = [
  ['canny', ['canny']],
  ['depth', ['depth']],
  ['openpose', ['openpose', 'pose']],
  ['softedge', ['hed', 'softedge', 'soft']],
  ['lineart', ['lineart']],
  ['mlsd', ['mlsd']],
  ['normal', ['normal']],
  ['color', ['color']],
  ['shuffle', ['shuffle']],
];

const weakNameFamily = (tokens: readonly string[]): ControlFamily | null =>
  WEAK_NAME_FAMILIES.find(([, keywords]) => hasToken(tokens, ...keywords))?.[0] ?? null;

const preprocessorFamily = (model: Pick<ModelConfig, 'default_settings'>): ControlFamily | null => {
  const preprocessor = model.default_settings?.preprocessor;
  return preprocessor && Object.hasOwn(PREPROCESSOR_FAMILY, preprocessor) ? PREPROCESSOR_FAMILY[preprocessor]! : null;
};

type ClassifiableModel = Pick<ModelConfig, 'name' | 'base' | 'type' | 'default_settings' | 'control_architecture'>;

const classify = (model: ClassifiableModel): ControlModelClass => {
  const tokens = nameTokens(model.name);
  if (model.type === 'controlnet' && FUN_UNION_BASES.has(model.base)) {
    const covers = model.control_architecture === 'instantx_union' ? INSTANTX_QWEN_UNION_COVERS : FUN_UNION_COVERS;
    return { covers, family: null, purpose: null, union: true };
  }
  if (hasToken(tokens, 'union')) {
    // Other bases' union models need the backend to expose their control types; until then they are explicit picks.
    return { covers: UNION_COVERS_BY_BASE[model.base] ?? [], family: null, purpose: null, union: true };
  }
  const family = strongNameFamily(tokens) ?? preprocessorFamily(model) ?? weakNameFamily(tokens);
  return { covers: [], family, purpose: family ? FAMILY_PURPOSE[family] : null, union: false };
};

/**
 * Resolution, row renders and readiness ticks classify the same model configs over and over. Configs are replaced,
 * never mutated, so a config object keys its class for as long as anything holds it.
 */
const classCache = new WeakMap<ClassifiableModel, ControlModelClass>();

/**
 * Classify a control model: union models first (a base's Fun union; a FLUX or SDXL union, covering its known
 * conditions; any other `union` name, covering nothing), then strong name words, the saved preprocessor, and the
 * remaining name words.
 */
export const classifyControlModel = (model: ClassifiableModel): ControlModelClass => {
  let modelClass = classCache.get(model);
  if (!modelClass) {
    modelClass = classify(model);
    classCache.set(model, modelClass);
  }
  return modelClass;
};
